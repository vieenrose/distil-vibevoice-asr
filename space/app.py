"""zh-TW / English meeting transcription + speaker diarization demo.

Runs Luigi/moss-transcribe-diarize-zhtw (0.9B, fine-tuned from
OpenMOSS-Team/MOSS-Transcribe-Diarize) over arbitrarily long meetings by
splitting the audio into windows, transcribing each window on ZeroGPU, and
linking speakers across windows with ECAPA-TDNN embeddings + global
agglomerative clustering on CPU.

Long example clips ship with precomputed results (generated offline with this
same pipeline) so they display instantly without spending GPU quota.
"""
from __future__ import annotations

import html as html_mod
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import spaces  # noqa: F401  (must be imported before torch on ZeroGPU)
import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = os.environ.get("MOSS_MODEL_ID", "Luigi/moss-transcribe-diarize-zhtw")
SR = 16000
WINDOW_S = 300.0          # transcription window (5 min)
AHC_THRESHOLD = 0.45      # cosine distance for global speaker linking
MIN_CORE_DUR_S = 3.0      # only segments this long participate in clustering
MAX_NEW_TOKENS = 3072     # ~5 min of dense speech fits comfortably

PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

# Out-of-domain (far-field) windows can flip whole windows to Simplified
# (base-model prior); s2tw is a pure script conversion that never touches
# phrases, so proper nouns survive.
from opencc import OpenCC  # noqa: E402

_S2TW = OpenCC("s2tw")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype="auto"
).to(torch.bfloat16).to(device).eval()
PROC = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

ECAPA = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/ecapa",
    run_opts={"device": "cpu"},
)
ECAPA.eval()

EXAMPLES_DIR = Path(__file__).parent / "examples"


# --------------------------------------------------------------------------- #
# lenient transcript parser: accepts [start][Sxx]text[end] AND [start]text[end]
# (the model sometimes drops the speaker tag on far-field audio); untagged
# segments inherit the previous speaker.
# --------------------------------------------------------------------------- #
_SEG_RE = re.compile(
    r"\[(?P<start>\d{1,7}(?:\.\d{1,3})?)\]"
    r"(?:\[(?P<spk>S\d{1,3})\])?"
    r"(?P<text>[^\[\]]+?)"
    r"\[(?P<end>\d{1,7}(?:\.\d{1,3})?)\]"
)


@dataclass(slots=True)
class Seg:
    start: float
    end: float
    speaker: str
    text: str


def parse_transcript_lenient(text: str) -> list[Seg]:
    segs: list[Seg] = []
    prev_spk = "S01"
    pos = 0
    while True:
        m = _SEG_RE.search(text, pos)
        if not m:
            break
        start, end = float(m.group("start")), float(m.group("end"))
        spk = m.group("spk") or prev_spk
        body = m.group("text").strip()
        if body and end >= start:
            segs.append(Seg(start, end, spk, body))
            prev_spk = spk
        pos = m.end()
    return segs


# --------------------------------------------------------------------------- #
# GPU: transcribe one window
# --------------------------------------------------------------------------- #
def _gpu_duration(wav: np.ndarray) -> int:
    # ~0.12x realtime generation + fixed overhead, capped for tier limits
    return int(min(120, 25 + 0.12 * len(wav) / SR))


@spaces.GPU(duration=_gpu_duration)
def transcribe_window(wav: np.ndarray) -> str:
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": "x.wav"},
        {"type": "text", "text": PROMPT}]}]
    text = PROC.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    inputs = PROC(text=text, audio=[wav], return_tensors="pt").to(
        MODEL.device, MODEL.dtype)
    with torch.no_grad():
        out = MODEL.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False)
    return PROC.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# CPU: audio decode, speaker embedding, global linking, rendering
# --------------------------------------------------------------------------- #
def decode_audio(path: str) -> np.ndarray:
    """Any container/codec -> 16 kHz mono float32 (ffmpeg)."""
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        raw = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1",
             "-ar", str(SR), "-f", "f32le", raw],
            check=True, capture_output=True)
        return np.fromfile(raw, dtype=np.float32)
    finally:
        os.unlink(raw)


def embed_segment(wav: np.ndarray, start: float, end: float) -> np.ndarray | None:
    a = wav[int(start * SR):int(end * SR)]
    if len(a) < int(0.3 * SR):
        return None
    x = torch.from_numpy(np.ascontiguousarray(a))[None, :]
    with torch.no_grad():
        emb = ECAPA.encode_batch(x)
    v = emb.squeeze().cpu().numpy().reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else None


def link_speakers(segs: list[dict]) -> list[dict]:
    """Global speaker labels via core-segment AHC + nearest-centroid assign.

    Validated on real 2h meetings: only segments >= MIN_CORE_DUR_S s join the
    clustering (short segments have noisy embeddings and fragment clusters);
    the rest snap to the nearest cluster centroid. Segments without a usable
    embedding inherit the previous segment's label.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    embs = {i: s["emb"] for i, s in enumerate(segs) if s["emb"] is not None}
    core = [i for i in embs
            if segs[i]["end"] - segs[i]["start"] >= MIN_CORE_DUR_S]
    lab_of: dict[int, int] = {}
    if len(core) >= 2:
        X = np.stack([embs[i] for i in core])
        labs = fcluster(linkage(pdist(X, "cosine"), "average"),
                        t=AHC_THRESHOLD, criterion="distance")
        lab_of = {i: int(l) for i, l in zip(core, labs)}
        cents: dict[int, list[np.ndarray]] = {}
        for i, l in lab_of.items():
            cents.setdefault(l, []).append(embs[i])
        keys = list(cents)
        M = np.stack([np.mean(cents[k], axis=0) for k in keys])
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        for i in embs:
            if i not in lab_of:
                lab_of[i] = keys[int(np.argmax(M @ embs[i]))]
    elif embs:
        lab_of = {i: 1 for i in embs}

    out = [dict(s) for s in segs]
    canon: dict[int, str] = {}
    prev = "S01"
    for i, s in enumerate(out):
        l = lab_of.get(i)
        if l is None:
            s["speaker"] = prev
        else:
            if l not in canon:
                canon[l] = f"S{len(canon) + 1:02d}"
            s["speaker"] = canon[l]
        prev = s["speaker"]
    return out


_PALETTE = ["#4f7cff", "#e05563", "#2aa876", "#c78c2c", "#9761d8",
            "#3aa6b9", "#d0679d", "#7a9a01", "#b05c45", "#5c7285"]


def fmt_ts(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def render_html(segs: list[dict], done: bool) -> str:
    speakers = sorted({s["speaker"] for s in segs})
    color = {spk: _PALETTE[i % len(_PALETTE)] for i, spk in enumerate(speakers)}
    rows = []
    for s in segs:
        c = color[s["speaker"]]
        rows.append(
            f'<div style="margin:6px 0;line-height:1.5">'
            f'<span style="color:#888;font-size:0.85em">'
            f'{fmt_ts(s["start"])}–{fmt_ts(s["end"])}</span> '
            f'<b style="color:{c}">{s["speaker"]}</b> '
            f'{html_mod.escape(s["text"])}'
            f"</div>")
    banner = "" if done else (
        '<div style="color:#c78c2c;font-weight:bold">⏳ transcribing… '
        "speaker labels refine as more windows finish</div>")
    return (banner + '<div style="max-height:520px;overflow-y:auto;'
            'padding:4px 8px">' + "".join(rows) + "</div>")


def export_files(segs: list[dict], stem: str) -> tuple[str, str]:
    def srt_ts(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{int((t % 1) * 1000):03d}"

    tmpdir = tempfile.mkdtemp()
    srt_path = os.path.join(tmpdir, f"{stem}.srt")
    json_path = os.path.join(tmpdir, f"{stem}.json")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{srt_ts(s['start'])} --> {srt_ts(s['end'])}\n"
                    f"[{s['speaker']}] {s['text']}\n\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{k: s[k] for k in ("start", "end", "speaker", "text")}
                   for s in segs], f, ensure_ascii=False, indent=1)
    return srt_path, json_path


def load_cached(name: str) -> list[dict] | None:
    p = EXAMPLES_DIR / (Path(name).stem + ".json")
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return [dict(s, emb=None) for s in data["segments"]]


# --------------------------------------------------------------------------- #
# handler
# --------------------------------------------------------------------------- #
def run(audio_path: str | None, max_minutes: float,
        progress=gr.Progress()):
    if not audio_path:
        yield "<i>Please provide an audio file.</i>", "", None, None
        return

    stem = Path(audio_path).stem
    cached = load_cached(Path(audio_path).name) if max_minutes == 0 else None
    if cached is not None:
        srt, js = export_files(cached, stem)
        dur = max(s["end"] for s in cached)
        n_spk = len({s["speaker"] for s in cached})
        stats = (f"**{fmt_ts(dur)}** of audio · **{len(cached)}** segments · "
                 f"**{n_spk}** speakers · ⚡ precomputed offline with this "
                 f"exact pipeline (no GPU quota used)")
        yield render_html(cached, done=True), stats, srt, js
        return

    progress(0, desc="decoding audio")
    wav = decode_audio(audio_path)
    if max_minutes and max_minutes > 0:
        wav = wav[: int(max_minutes * 60 * SR)]
    total_s = len(wav) / SR
    if total_s < 1:
        yield "<i>Audio too short.</i>", "", None, None
        return

    offsets = [o for o in np.arange(0.0, total_s, WINDOW_S)
               if total_s - o >= 2.0]
    segs: list[dict] = []
    t0 = time.time()
    note = ""
    for wi, off in enumerate(offsets):
        progress(wi / len(offsets),
                 desc=f"window {wi + 1}/{len(offsets)} "
                      f"({fmt_ts(off)}–{fmt_ts(min(off + WINDOW_S, total_s))})")
        piece = wav[int(off * SR): int(min(off + WINDOW_S, total_s) * SR)]
        try:
            gen = transcribe_window(piece)
        except Exception as e:  # noqa: BLE001 — quota / scheduling errors
            note = (f"\n\n⚠️ stopped at window {wi + 1}/{len(offsets)}: "
                    f"`{type(e).__name__}` (likely GPU quota). "
                    "Partial results below — sign in / try later for more.")
            break
        for s in parse_transcript_lenient(gen):
            end = min(s.end, WINDOW_S + 5.0)
            segs.append({
                "start": round(off + s.start, 2),
                "end": round(off + end, 2),
                "speaker": s.speaker,
                "text": _S2TW.convert(s.text),
                "emb": embed_segment(piece, s.start, end),
            })
        linked = link_speakers(segs)
        done_s = min(off + WINDOW_S, total_s)
        stats = (f"{fmt_ts(done_s)} / {fmt_ts(total_s)} processed · "
                 f"{len(linked)} segments · "
                 f"{len({s['speaker'] for s in linked})} speakers so far")
        yield render_html(linked, done=False), stats + note, None, None

    if not segs:
        yield ("<i>No speech recognized." + note + "</i>"), note, None, None
        return
    linked = link_speakers(segs)
    wall = time.time() - t0
    srt, js = export_files(linked, stem)
    done_s = min(offsets[len(offsets) - 1] + WINDOW_S, total_s) if not note \
        else segs[-1]["end"]
    stats = (f"**{fmt_ts(done_s)}** of audio · **{len(linked)}** segments · "
             f"**{len({s['speaker'] for s in linked})}** speakers · "
             f"wall {wall:.0f}s · {done_s / max(wall, 1e-6):.1f}× realtime"
             + note)
    yield render_html(linked, done=True), stats, srt, js


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
DESCRIPTION = """
# 🎙️ zh-TW / English meeting transcription + speaker diarization

**One pipeline, hours-long meetings**: [Luigi/moss-transcribe-diarize-zhtw](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw)
(0.9B, fine-tuned from [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) for
Traditional-Chinese / English code-switched meetings) transcribes 5-minute windows with
speaker labels + timestamps, then ECAPA-TDNN embeddings + global clustering link the
speakers across windows — so a 2+ hour meeting gets one consistent set of speakers.

The long example below is a real Taiwan Legislative Yuan meeting
(立法院 IVOD open data, CC-BY-4.0) with **precomputed results** — click it to see the
full-meeting output instantly. Upload your own audio to run the pipeline live on ZeroGPU.
"""

FOOTER = """
---
Example audio: 立法院 IVOD (Taiwan Legislative Yuan) open data, **CC-BY-4.0**, via
[openfun/tw-ly-ivod](https://huggingface.co/datasets/openfun/tw-ly-ivod).
Model + pipeline: fine-tune and long-form chunking design from the
[distil-vibevoice-asr](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw) project.
"""


def build_examples() -> list[list]:
    rows = []
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("*")):
            if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
                rows.append([str(p), 0])
    return rows


with gr.Blocks(title="zh-TW Meeting Transcriber") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="Meeting audio", type="filepath",
                                sources=["upload"])
            max_min = gr.Slider(
                0, 180, value=0, step=5,
                label="Process only first N minutes (0 = whole file)",
                info="Live GPU runs cost quota (~30 s GPU per 5 min of audio); "
                     "examples at 0 use precomputed results.")
            btn = gr.Button("Transcribe", variant="primary")
            ex = build_examples()
            if ex:
                gr.Examples(examples=ex, inputs=[audio_in, max_min],
                            cache_examples=False,
                            label="Real 2+ hour meetings (precomputed)")
        with gr.Column(scale=2):
            stats_out = gr.Markdown()
            html_out = gr.HTML()
            with gr.Row():
                srt_out = gr.File(label="SRT")
                json_out = gr.File(label="JSON")
    gr.Markdown(FOOTER)
    btn.click(run, inputs=[audio_in, max_min],
              outputs=[html_out, stats_out, srt_out, json_out])

if __name__ == "__main__":
    demo.launch()
