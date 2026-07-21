#!/usr/bin/env python
"""v6.1: marker-density FT on top of v6-stream.

MOTIVATION (measured 2026-07-18, 2h council audit): on dense continuous
speech the deployed model emits ONE leading [0.00] and then NO time markers
for minutes — the demo's whole time axis for that region collapses, cursor
advancement falls back to a 20 s crawl, and pre-fix builds lost the text
outright. Root cause is supervision: the IVOD manifest's median segment is
18.7 s / 90 chars (never < ~15 s markers apart), so sentence-cadence marker
emission was never taught.

FIX: densify training targets — split every long manifest segment at
sentence punctuation into ~<=8 s pieces with character-proportional
interpolated timestamps, so the model learns to emit a marker at every
sentence boundary even with no acoustic pause. Everything else = the
v6-stream recipe (bounded audio-KV eviction mask + frozen full-attention
teacher KL + speaker-tag-weighted CE).

(original v6-stream header follows)

Streaming FT: teach MOSS to decode with a BOUNDED audio-KV window.

MOTIVATION (measured, this project): monotonic audio-KV eviction at inference
(RS_AUDIO_KV_WINDOW=45 in RapidSpeech.cpp) makes the model drift to Simplified
Chinese and stop early — MOSS was trained with FULL attention over all audio
tokens, so chopping the audio KV is out-of-distribution. This script closes
that gap: fine-tune with a 4D attention mask that EMULATES the eviction
(text tokens of a segment starting at time T cannot attend audio tokens older
than T - W seconds), while a frozen full-attention teacher (the same
checkpoint) KL-distills its distribution into the windowed student.

Keeps the v5 protections: speaker-tag-weighted CE (8x) so diarization does not
collapse, and the teacher KL doubles as the anchor-to-base defence.

Deployment target: RapidSpeech.cpp's env-gated eviction path (positions are
PRESERVED under eviction there — logical_pos keeps counting — which matches
training with a mask and ordinary consecutive position_ids).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

SPK_RE = re.compile(r"\[S\d+\]")
AUDIO_TOK_DT = 0.08          # decoder-side audio tokens: 12.5 / s


def window_sample(rec: dict, max_s: float, rng: random.Random):
    dur = float(rec.get("duration") or max(s["end"] for s in rec["segments"]))
    inside = []
    for _ in range(8):
        off = 0.0 if dur <= max_s else rng.uniform(0.0, dur - max_s)
        inside = [s for s in rec["segments"]
                  if s["start"] >= off and s["end"] <= off + max_s]
        if len(inside) >= 2 or dur <= max_s:
            break
    if not inside:
        return None, None
    spk_map: dict = {}
    segs = []
    for s in sorted(inside, key=lambda x: x["start"]):
        if s["speaker"] not in spk_map:
            spk_map[s["speaker"]] = len(spk_map) + 1
        segs.append({"start": s["start"] - off, "end": s["end"] - off,
                     "speaker": spk_map[s["speaker"]], "text": s["text"]})
    return off, segs


SENT_SPLIT_RE = re.compile(r"[^。！？!?;；]+[。！？!?;；]?")
CLAUSE_SPLIT_RE = re.compile(r"[^，,、]+[，,、]?")


def _split_text(txt, chars_per_piece):
    """Cascade: sentence punctuation -> clause punctuation -> hard char count.
    The pseudo-label text is mostly unpunctuated (measured ~1 sentence ender
    per 130 chars), so all three tiers matter."""
    pieces = []
    for m in SENT_SPLIT_RE.finditer(txt):
        sent = m.group(0)
        if len(sent) <= chars_per_piece * 1.5:
            pieces.append(sent)
            continue
        for cm in CLAUSE_SPLIT_RE.finditer(sent):
            cl = cm.group(0)
            while len(cl) > chars_per_piece * 1.5:
                pieces.append(cl[:chars_per_piece])
                cl = cl[chars_per_piece:]
            pieces.append(cl)
    # greedy re-merge of tiny fragments
    merged = []
    for x in pieces:
        if merged and len(merged[-1]) + len(x) <= chars_per_piece:
            merged[-1] += x
        else:
            merged.append(x)
    return [x for x in merged if x]


def densify_segments(segments, max_s=8.0, min_chars=20):
    """Split long segments into ~max_s pieces; interpolate timestamps
    proportionally to character count. Supervision noise ~+-1-2 s — fine for
    teaching marker CADENCE (exact times stay anchored at real segment
    boundaries)."""
    out = []
    for s in segments:
        dur = s["end"] - s["start"]
        txt = s["text"]
        if dur <= max_s or len(txt) < 2 * min_chars:
            out.append(s)
            continue
        # chars per max_s-sized piece at this segment's own speaking rate
        cpp = max(min_chars, int(len(txt) * max_s / dur))
        pieces = _split_text(txt, cpp)
        if len(pieces) <= 1:
            out.append(s)
            continue
        total = sum(len(x) for x in pieces)
        t = s["start"]
        for x in pieces:
            t2 = t + dur * len(x) / total
            out.append({**s, "start": round(t, 2), "end": round(t2, 2),
                        "text": x})
            t = t2
        out[-1]["end"] = s["end"]
    return out


def seg_text(s) -> str:
    return f"[{s['start']:.2f}][S{s['speaker']:02d}]{s['text']}[{s['end']:.2f}]"


def target_text(segments) -> str:
    return "".join(seg_text(s) for s in segments)


def speaker_token_weights(tok, assistant: str, label_ids: list[int],
                          eos_id: int, w_spk: float, w_change: float = None):
    """Per-token weight aligned to the non-masked (assistant + eos) labels.
    None on BPE drift -> caller falls back to uniform for that step.

    v7.1: a [Sxx] token that CHANGES speaker gets `w_change`; one that repeats
    the previous segment's speaker gets `w_spk`. v7 applied a single 8x weight
    to every speaker token -- but 91% of consecutive training segments keep the
    same speaker (measured on ivod_ft_v4 at 120 s windows, and marker
    densification made it worse by splitting one segment into many same-speaker
    pieces). So that uniform boost overwhelmingly rewarded CONTINUITY. v6.1
    survived only because the teacher's speaker-position KL (weight 2.0) pushed
    back; v7 had to drop all KL (it fights fake-quant), leaving the continuity
    bias unopposed -- and v7 collapsed to one speaker where base/v5/v6-stream/
    v6.1 all resolve three."""
    if w_change is None:
        w_change = w_spk
    enc = tok(assistant, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if ids + [eos_id] != list(label_ids):
        return None
    spans, prev = [], None
    for m in SPK_RE.finditer(assistant):
        tag = m.group(0)
        spans.append((m.start(), m.end(), w_change if (prev is not None and tag != prev) else w_spk))
        prev = tag
    w = []
    for a, b in offs:
        hit = 1.0
        if a != b:
            for st, en, wt in spans:
                if a < en and b > st:
                    hit = max(hit, wt)
        w.append(hit)
    w.append(1.0)  # eos
    return w


def eviction_bias(input_ids, audio_token_id, n_prompt, tok, segs,
                  window_s: float, dtype, device):
    """(1,1,L,L) additive attention bias emulating monotonic audio-KV eviction.

    Causal everywhere; ADDITIONALLY, target-text tokens belonging to a segment
    that starts at time T cannot see audio tokens with timestamp < T - W (they
    were evicted by the time the decoder got there). Prompt/audio/suffix
    positions keep plain causal attention (prefill happens before eviction).

    Returns None if the per-segment tokenization doesn't concatenate to the
    exact label ids (BPE drift) -> caller trains that step with full attention
    rather than a misaligned mask.
    """
    L = input_ids.shape[1]
    ids0 = input_ids[0]
    audio_pos = (ids0 == audio_token_id).nonzero().flatten()
    if audio_pos.numel() == 0:
        return None
    a0 = audio_pos[0].item()
    audio_t = (audio_pos - a0).to(torch.float32) * AUDIO_TOK_DT

    # Map target tokens -> segments via offset_mapping on the WHOLE assistant
    # string (per-seg tokenization does NOT concatenate: BPE merges across the
    # "[end][start]" boundary). A token straddling a boundary goes to the
    # EARLIER segment (smaller cutoff = less eviction = conservative).
    assistant = "".join(seg_text(s) for s in segs)
    enc = tok(assistant, return_offsets_mapping=True, add_special_tokens=False)
    if n_prompt + len(enc["input_ids"]) + 1 != L:          # + eos
        return None
    if enc["input_ids"] != ids0[n_prompt:L - 1].tolist():  # BPE drift guard
        return None
    starts, cum = [], 0
    for s in segs:
        starts.append((cum, float(s["start"])))
        cum += len(seg_text(s))
    spans = []           # (token_start, token_end, seg_start_time), contiguous
    cur_seg, run_start = 0, n_prompt
    for i, (a, _b) in enumerate(enc["offset_mapping"]):
        seg = cur_seg
        while seg + 1 < len(starts) and a >= starts[seg + 1][0]:
            seg += 1
        if seg != cur_seg:
            spans.append((run_start, n_prompt + i, starts[cur_seg][1]))
            cur_seg, run_start = seg, n_prompt + i
    spans.append((run_start, L, starts[cur_seg][1]))       # tail + eos

    allow = torch.ones(L, L, dtype=torch.bool, device=device).tril_()
    for qs, qe, t_start in spans:
        cutoff = t_start - window_s
        if cutoff <= 0:
            continue
        dead = audio_pos[audio_t < cutoff]
        if dead.numel():
            allow[qs:qe, dead] = False
    bias = torch.zeros(L, L, dtype=dtype, device=device)
    bias.masked_fill_(~allow, torch.finfo(dtype).min)
    return bias.unsqueeze(0).unsqueeze(0)


def _unwrap_qat(model):
    """Replace every QATLinear with its underlying nn.Linear in place, so
    save_pretrained stores standard param names (q_proj.weight, not
    q_proj.lin.weight). The saved weights are the full-precision latents
    trained to be q4-robust; the GGUF q4_K_M export applies the real rounding.
    Same weight tensors, so the optimizer's references stay valid across a
    save→re-wrap cycle. (Mirrors scripts/40 and 49.)"""
    for _, module in list(model.named_modules()):
        for cn, child in list(module.named_children()):
            if child.__class__.__name__ == "QATLinear":
                setattr(module, cn, child.lin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/moss_ft_zhtw_v6_1")
    ap.add_argument("--kl-base", default="models/moss_ft_zhtw_v5_kl_qat_fixed",
                    help="frozen FULL-ATTENTION teacher for the windowed student")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="KL(student||teacher) on ALL label positions")
    ap.add_argument("--spk-kl-weight", type=float, default=2.0,
                    help="EXTRA KL on [Sxx] speaker-tag positions (v5 lesson: "
                         "all-position KL alone lets diarization drift)")
    ap.add_argument("--window-s", type=float, default=45.0,
                    help="audio-KV window the student must live with")
    ap.add_argument("--window-anneal", default="120:300,75:800",
                    help="'W:until,...' curriculum before settling on --window-s")
    ap.add_argument("--spk-weight", type=float, default=2.0,
                    help="weight on a [Sxx] token that REPEATS the previous "
                         "speaker (v7 used 8.0 for every speaker token)")
    ap.add_argument("--margin-weight", type=float, default=1.0,
                    help="weight of the structural-token hinge margin loss")
    ap.add_argument("--margin-neg-weight", type=float, default=2.0,
                    help="weight of the NEGATIVE hinge: at text positions the "
                         "gold token must beat '[' by margin-neg-target")
    ap.add_argument("--margin-neg-target", type=float, default=5.0,
                    help="required gap of gold text token over '['")
    ap.add_argument("--margin-target", type=float, default=5.0,
                    help="target logit gap for structural tokens; base MOSS "
                         "sits at ~4.9 median, our v7.1 at ~0.98")
    ap.add_argument("--spk-change-weight", type=float, default=16.0,
                    help="weight on a [Sxx] token that CHANGES speaker")
    ap.add_argument("--p-multispk", type=float, default=0.6,
                    help="fraction of steps forced to use a window containing "
                         ">=2 distinct speakers")
    ap.add_argument("--tts-manifests", nargs="+",
                    default=["data/pseudo/tts_all.jsonl",
                             "data/pseudo/tts_v3.jsonl.shard0"])
    ap.add_argument("--ivod-manifest", default="data/pseudo/ivod_ft_v4.jsonl")
    ap.add_argument("--p-ivod", type=float, default=0.5,
                    help="long real meetings exercise the window most")
    ap.add_argument("--p-aug", type=float, default=0.43)
    ap.add_argument("--p-silence-tail", type=float, default=0.0,
                    help="prob of trimming the wav at the last segment end and "
                         "appending 15-60s of near-silence (target unchanged: "
                         "the model must NOT invent speech after the talk ends)")
    ap.add_argument("--rir-dir", default="data/aug/RIRS_NOISES")
    ap.add_argument("--musan-dir", default="data/aug/musan")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--no-densify", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-audio-s", type=float, default=300.0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out", default="models/moss_ft_zhtw_v6_2")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
    from distil_vibevoice.data.augment import augment_wav

    anneal = []
    if args.window_anneal:
        for part in args.window_anneal.split(","):
            w, until = part.split(":")
            anneal.append((int(until), float(w)))
        anneal.sort()

    def window_at(step):
        for until, w in anneal:
            if step < until:
                return w
        return args.window_s

    dev = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    eos_id = tok.eos_token_id
    audio_token_id = model.config.audio_token_id

    teacher = AutoModelForCausalLM.from_pretrained(
        args.kl_base, trust_remote_code=True, dtype="auto"
    ).to(torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher (full attention, frozen): {args.kl_base} "
          f"kl-weight={args.kl_weight}", flush=True)

    # v6.2 = v6.1 recipe + QAT. STE int4 fake-quant on the decoder Linears that
    # become q4 in the GGUF export, so the weights re-adapt to 4-bit block-32
    # rounding. The bf16 streaming/marker FTs (v6-stream, v6.1) trained AWAY the
    # q4-robustness that v5-kl-QAT had, which is what lets q4 fall into the
    # repetition attractor on long windows (f16 is clean). The teacher stays
    # full precision (unwrapped) so the KL target is the clean distribution.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from distil_vibevoice.quant.fakequant import wrap_decoder_linears
    wrapped = wrap_decoder_linears(model, bits=4)
    print(f"QAT: wrapped {len(wrapped)} decoder Linears with int4 fake-quant",
          flush=True)

    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    tts_rows = []
    for mp in args.tts_manifests:
        p = ROOT / mp
        if p.exists():
            tts_rows += [json.loads(l) for l in p.open()]
    ivod_rows = []
    p = ROOT / args.ivod_manifest
    if p.exists():
        ivod_rows = [json.loads(l) for l in p.open()]
    if not args.no_densify:
        nb = sum(len(r["segments"]) for r in ivod_rows + tts_rows)
        for r in ivod_rows + tts_rows:
            r["segments"] = densify_segments(r["segments"])
        na = sum(len(r["segments"]) for r in ivod_rows + tts_rows)
        print(f"marker densification: {nb} -> {na} segments", flush=True)
    if not ivod_rows:
        print("WARNING: no IVOD rows; TTS-only (+aug)")
        args.p_ivod = 0.0
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rng.shuffle(tts_rows)
    rng.shuffle(ivod_rows)
    print(f"TTS {len(tts_rows)} | IVOD {len(ivod_rows)}; {args.steps} steps @ "
          f"{args.max_audio_s:.0f}s; W={args.window_s}s anneal={args.window_anneal}",
          flush=True)

    tgt_sr = proc.feature_extractor.sampling_rate
    losses, n_skip, n_uniform, n_fullattn = [], 0, 0, 0
    # Structural tokens whose margin we defend: the '[' that opens a timestamp
    # or a speaker tag, plus the ']' that closes one. These are the decisions
    # that flip under ggml/quant noise; ordinary text tokens are not at risk.
    struct_ids = []
    for piece in ("[", "]"):
        try:
            ids = tok.encode(piece, add_special_tokens=False)
            if len(ids) == 1:
                struct_ids.append(ids[0])
        except Exception:
            pass
    struct_ids = sorted(set(struct_ids))
    _br = tok.encode("[", add_special_tokens=False)
    bracket_id = _br[0] if len(_br) == 1 else None
    print(f"margin loss: struct ids {struct_ids} w={args.margin_weight} "
          f"target={args.margin_target} | NEG '[' id={bracket_id} "
          f"w={args.margin_neg_weight} target={args.margin_neg_target}", flush=True)

    ti = ii = 0
    for step in range(args.steps):
        use_ivod = ivod_rows and rng.random() < args.p_ivod
        if use_ivod:
            rec = ivod_rows[ii % len(ivod_rows)]; ii += 1
        else:
            rec = tts_rows[ti % len(tts_rows)]; ti += 1

        off, segs = window_sample(rec, args.max_audio_s, rng)
        # Oversample multi-speaker windows. window_sample only guarantees >=2
        # SEGMENTS, not >=2 SPEAKERS: at 120 s, 57.5% of sampled windows are
        # single-speaker, so most steps teach nothing about switching.
        if rng.random() < args.p_multispk:
            for _ in range(24):
                if segs and len({s["speaker"] for s in segs}) >= 2:
                    break
                rec = (ivod_rows[ii % len(ivod_rows)] if use_ivod
                       else tts_rows[ti % len(tts_rows)])
                if use_ivod: ii += 1
                else: ti += 1
                off, segs = window_sample(rec, args.max_audio_s, rng)
        if not segs:
            n_skip += 1
            continue
        try:
            wav, sr = sf.read(rec["audio_path"], start=int(off * 24000),
                              frames=int(args.max_audio_s * 24000))
        except Exception:
            wav, sr = sf.read(rec["audio_path"])
            wav = wav[int(off * sr): int((off + args.max_audio_s) * sr)]
        wav = np.asarray(wav if np.ndim(wav) == 1 else wav.mean(1),
                         dtype=np.float32)
        if len(wav) < sr:
            n_skip += 1
            continue
        if (not use_ivod) and rng.random() < args.p_aug:
            wav = augment_wav(wav, sr, rir_dir=args.rir_dir,
                              musan_dir=args.musan_dir, rng=np_rng)
        if sr != tgt_sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, tgt_sr)
            wav = resample_poly(wav, tgt_sr // g, sr // g).astype(np.float32)

        if rng.random() < args.p_silence_tail:
            cut = int(min(len(wav), (segs[-1]["end"] + rng.uniform(0.3, 2.0))
                          * tgt_sr))
            tail = int(rng.uniform(15.0, 60.0) * tgt_sr)
            noise = (np_rng.standard_normal(tail) * 1e-4).astype(np.float32)
            wav = np.concatenate([wav[:cut], noise])

        assistant = target_text(segs)
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": "x.wav"},
            {"type": "text", "text": DEFAULT_PROMPT}]}]
        prompt_text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + assistant + tok.eos_token

        enc = proc(text=full_text, audio=[wav], max_length=args.max_len,
                   return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        n_prompt = proc(text=prompt_text, audio=[wav], max_length=args.max_len,
                        return_tensors="pt")["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100

        label_ids = input_ids[0, n_prompt:].tolist()
        w_assist = speaker_token_weights(tok, assistant, label_ids, eos_id,
                                         args.spk_weight, args.spk_change_weight)
        weight_vec = torch.ones_like(input_ids, dtype=torch.float32)
        if w_assist is not None and len(w_assist) == len(label_ids):
            weight_vec[0, n_prompt:] = torch.tensor(w_assist, device=dev)
        else:
            n_uniform += 1

        W = window_at(step)
        bias = eviction_bias(input_ids, audio_token_id, n_prompt, tok, segs,
                             W, torch.bfloat16, dev)
        if bias is None:
            n_fullattn += 1          # BPE drift: train full-attention this step

        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in enc.items()}
        if bias is not None:
            batch["attention_mask"] = bias

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        out = model(**batch)
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_w = weight_vec[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none"
        ).reshape(shift_labels.shape).float()
        mask = (shift_labels != -100).float()
        denom = (shift_w * mask).sum().clamp_min(1.0)
        loss = (ce * shift_w * mask).sum() / denom

        # ---- structural-token MARGIN loss (v7.2) --------------------------
        # Measured root cause of the ggml marker collapse: the decision to open
        # a timestamp is a NEAR-TIE. On dense180 the winning '[' beat the
        # runner-up text token by 0.18 logits, and over all marker positions our
        # margin median is +0.98 vs base MOSS's +4.90 (23% of ours sit under
        # 0.5). Any implementation difference -- ggml kernels, accumulation
        # order, f16 rounding over 28 layers, or q4 noise -- flips it, and a
        # single flip drops the model out of marker mode for the whole window.
        # CE alone does not fix this: it maximises P(correct), which saturates
        # long before the MARGIN is comfortable (that is why --spk-weight 8/16
        # raised likelihood but left the gap thin). Hinge the gap explicitly.
        margin_val = 0.0
        if args.margin_weight > 0 and struct_ids:
            with torch.no_grad():
                is_struct = torch.zeros_like(shift_labels, dtype=torch.bool)
                for tid in struct_ids:
                    is_struct |= (shift_labels == tid)
                is_struct &= (shift_labels != -100)
            if is_struct.any():
                sl = shift_logits[is_struct].float()          # (N, V)
                gold = shift_labels[is_struct]                # (N,)
                z_gold = sl.gather(1, gold[:, None]).squeeze(1)
                sl_masked = sl.scatter(1, gold[:, None], float("-inf"))
                z_other = sl_masked.max(dim=1).values
                gap = z_gold - z_other
                margin_loss = F.relu(args.margin_target - gap).mean()
                margin_val = float(margin_loss.detach())
                loss = loss + args.margin_weight * margin_loss

            # ---- NEGATIVE side (v7.4) --------------------------------------
            # v7.3 lesson: hinging ONLY where a marker belongs has a degenerate
            # optimum -- raise '[' everywhere. Gradient descent found it: margin
            # median 0.98 -> 7.15 (past base's 4.90) but 114 marker positions on
            # audio with 12 ground-truth utterances, and through ggml the model
            # emitted 49 chars total. Nothing penalised '[' being high where a
            # marker does NOT belong. Penalise it: at ordinary text positions the
            # gold token must beat '[' by margin_neg_target. Satisfying both
            # sides is only possible by being confident about WHERE boundaries
            # are, which is the property we actually want.
            if args.margin_neg_weight > 0 and bracket_id is not None:
                with torch.no_grad():
                    is_text = (shift_labels != -100) & ~is_struct
                if is_text.any():
                    tl = shift_logits[is_text].float()
                    tgold = shift_labels[is_text]
                    z_t = tl.gather(1, tgold[:, None]).squeeze(1)
                    z_br = tl[:, bracket_id]
                    neg_loss = F.relu(args.margin_neg_target - (z_t - z_br)).mean()
                    loss = loss + args.margin_neg_weight * neg_loss

        # Full-attention teacher -> windowed student, ALL label positions:
        # carries both the transcription distribution the student can no longer
        # read off evicted audio, and the v5 speaker-discrimination anchor.
        kl_val = 0.0
        if args.kl_weight > 0:
            lbl_pos = shift_labels != -100
            with torch.no_grad():
                t_batch = {k: v for k, v in batch.items()
                           if k != "attention_mask"}
                t_logits = teacher(**t_batch).logits[:, :-1, :]
            s_lp = F.log_softmax(shift_logits[lbl_pos].float(), dim=-1)
            t_p = F.softmax(t_logits[lbl_pos].float(), dim=-1)
            kl = F.kl_div(s_lp, t_p, reduction="batchmean")
            loss = loss + args.kl_weight * kl
            kl_val = float(kl.detach())
            # v5 lesson: without a DEDICATED speaker-position KL the [Sxx]
            # distribution drifts (DER 0.128 -> 0.190 measured on the first
            # streaming FT) even with 8x CE weight + all-position KL.
            spk_pos = (shift_w > 1.0) & lbl_pos
            if args.spk_kl_weight > 0 and spk_pos.any():
                s_spk = F.log_softmax(shift_logits[spk_pos].float(), dim=-1)
                t_spk = F.softmax(t_logits[spk_pos].float(), dim=-1)
                loss = loss + args.spk_kl_weight * F.kl_div(
                    s_spk, t_spk, reduction="batchmean")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(loss.detach().item())
        if step % 10 == 0:
            print(f"  step {step}: loss={losses[-1]:.3f} lr={lr_at(step):.2e} "
                  f"W={W:.0f}s len={input_ids.shape[1]} kl={kl_val:.3f}",
                  flush=True)
        if args.save_every and step and step % args.save_every == 0:
            ck = ROOT / (args.out + f"_ckpt{step}")
            _unwrap_qat(model)              # standard names, not q_proj.lin.*
            model.save_pretrained(ck)
            proc.save_pretrained(ck)
            wrap_decoder_linears(model, bits=4)   # re-arm QAT for next steps
            print(f"  ckpt -> {ck}", flush=True)

    _unwrap_qat(model)                      # final: save q4-robust bf16 latents
    outdir = ROOT / args.out
    model.save_pretrained(outdir)
    proc.save_pretrained(outdir)
    print(f"\nloss {losses[0]:.3f} -> {sum(losses[-20:])/20:.3f} "
          f"(skip {n_skip}, uniform {n_uniform}, fullattn-fallback {n_fullattn})"
          f" | saved -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
