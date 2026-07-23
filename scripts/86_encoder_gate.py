#!/usr/bin/env python3
"""Checkpoint gate for encoder-distillation candidates. NEVER gates on cosine.

v8's post-mortem is the whole reason this exists: the whisper-small swap hit
student-teacher cosine 0.9846 and then scored MER 0.878 on English -- cosine
(and any reconstruction metric) predicts nothing, because a student's residual
concentrates in exactly the decoder-relevant directions it failed to learn.
So a checkpoint passes or fails ONLY on the real battery:

  1. golden 5-min clips (zh+en): text agreement vs the byte-validated f32
     reference transcripts (single-pass, the stage-1 contract surface)
  2. windowed 16-min en (AMI ES2004a): WER + speaker accuracy vs word-level
     ground truth, through the REAL pipeline (90s windows + CAM++ linking)
  3. windowed 16-min zh: utterance-count structure (312-utterance baseline;
     collapse here is the historical failure mode of everything that ever
     trained near this model)

Pipeline per checkpoint: HF dir -> f32 GGUF (upstream converter) -> the same
CUDA binary and scoring scripts every prior decision in this project used.

Pass bands (vs the frozen-teacher baseline measured with the same tooling):
  golden agreement >= 97% zh / >= 95% en, en WER <= 0.20 (teacher 0.164),
  speaker acc >= 97% (teacher 99.0), zh utterances within 250-380.
Bands are deliberately generous for EARLY checkpoints -- the point of gating
every checkpoint is watching the TREND, not shipping the first pass.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = Path("/tmp/claude-1001/golden")
CONVERTER = Path("/tmp/claude-1001/ref/moss-transcribe.cpp/scripts/convert_moss_transcribe_to_gguf.py")
OFFICIAL_PKG = "/tmp/claude-1001/ref/MOSS-Transcribe-Diarize"
BIN = Path("/home/luigi/rs-pure/build-cuda/rs-moss-td")
CAMPPLUS_GLOB = "/home/luigi/.cache/huggingface/hub/models--Luigi--moss-transcribe-diarize-zhtw-gguf/snapshots/*/campplus.gguf"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def transcribe(gguf: Path, wav: Path, max_new: int = 16384) -> str:
    r = sh([str(BIN), "transcribe", str(gguf), str(wav), "--max-new", str(max_new)])
    return r.stdout.rstrip("\n")


def strip_marks(s: str) -> str:
    return re.sub(r"\[[0-9.]+\]|\[S\d+\]", "", s)


def agreement(a: str, b: str) -> float:
    xa, xb = list(strip_marks(a)), list(strip_marks(b))
    # autojunk=False is load-bearing: with the default heuristic, popular chars
    # (space/e/t...) become junk on >200-element English sequences, and a
    # NEAR-identical transcript scores ~10% while an identical one scores 100.
    sm = difflib.SequenceMatcher(a=xa, b=xb, autojunk=False)
    return 100.0 * sum(bl.size for bl in sm.get_matching_blocks()) / max(len(xa), len(xb), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="HF model dir saved by the distill script")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--keep-gguf", action="store_true")
    ap.add_argument("--cuda-device", default="0",
                    help="gate GPU (training owns the other one)")
    args = ap.parse_args()

    ck = Path(args.checkpoint)
    tag = args.tag or ck.name
    work = Path(f"/tmp/claude-1001/gate_{tag}")
    work.mkdir(parents=True, exist_ok=True)
    gguf = work / "model_f32.gguf"
    report = {"tag": tag, "checkpoint": str(ck), "ts": time.strftime("%F %T")}

    # 1. convert ------------------------------------------------------------
    t0 = time.time()
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(CONVERTER), str(ck), "-o", str(gguf)],
        capture_output=True, text=True,
        env={"PYTHONPATH": OFFICIAL_PKG, "PATH": "/usr/bin:/bin"})
    if r.returncode != 0 or not gguf.exists():
        print(f"[gate:{tag}] CONVERT FAILED\n{r.stderr[-1500:]}")
        return 2
    report["convert_s"] = round(time.time() - t0, 1)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # 2. golden 5-min clips -------------------------------------------------
    for c in ("zh", "en"):
        out = transcribe(gguf, G / f"golden_{c}_5min.wav", max_new=5120)
        ref = (G / f"ref_{c}_f32.txt").read_text().rstrip("\n")
        report[f"golden_{c}_agree"] = round(agreement(ref, out), 2)
        (work / f"golden_{c}.txt").write_text(out)

    # 3. windowed 16-min en, scored vs AMI ----------------------------------
    import glob
    camp = glob.glob(CAMPPLUS_GLOB)[0]
    r = sh([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/85_window_sweep.py"),
            str(G / "stage2_en_16min.wav"), "--windows", "90",
            "--gguf", str(gguf), "--speaker-gguf", camp,
            "--link-threshold", "0.50",
            "--out-prefix", str(work / "en16")])
    en_txt = work / "en16_w90.txt"
    if en_txt.exists():
        r2 = sh([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/81_score_vs_ami.py"),
                 str(en_txt), "--meeting", "ES2004a"])
        m = re.search(r"WER (\d+\.\d+)", r2.stdout)
        report["en_wer"] = float(m.group(1)) if m else None
        m = re.search(r"speaker accuracy (\d+\.\d+)%", r2.stdout)
        report["en_spk"] = float(m.group(1)) if m else None

    # 4. windowed 16-min zh structure ---------------------------------------
    r = sh([str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/85_window_sweep.py"),
            str(G / "stage2_zh_16min.wav"), "--windows", "90",
            "--gguf", str(gguf), "--speaker-gguf", camp,
            "--link-threshold", "0.50",
            "--out-prefix", str(work / "zh16")])
    zh_txt = work / "zh16_w90.txt"
    if zh_txt.exists():
        t = zh_txt.read_text()
        segs = [c.strip() for a, b, c in
                re.findall(r"\[(\d+\.\d+)\]\s*(?:\[(S\d+)\])?\s*([^\[]*)", t) if c.strip()]
        report["zh_utts"] = len(segs)
        report["zh_chars"] = sum(len(s) for s in segs)

    # verdict ---------------------------------------------------------------
    ok = (report.get("golden_zh_agree", 0) >= 97 and
          report.get("golden_en_agree", 0) >= 95 and
          (report.get("en_wer") or 9) <= 0.20 and
          (report.get("en_spk") or 0) >= 97 and
          250 <= (report.get("zh_utts") or 0) <= 380)
    report["verdict"] = "PASS" if ok else "FAIL"

    line = json.dumps(report, ensure_ascii=False)
    print(line)
    with open("/tmp/claude-1001/encoder_gate_log.jsonl", "a") as f:
        f.write(line + "\n")
    if not args.keep_gguf:
        gguf.unlink(missing_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
