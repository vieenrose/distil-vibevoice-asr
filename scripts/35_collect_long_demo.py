#!/usr/bin/env python
"""Collect ONE long (2.5h-3.5h) real IVOD meeting as the HF Space demo clip.

Reuses the 01b collector: picks the shortest catalog record above --min-minutes
(full meetings, transcript required, so we get a pyannote/whisperx reference),
downloads the whole thing past the dead-air skip, and writes a manifest row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "c01b", str(ROOT / "scripts/01b_collect_ivod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--min-minutes", type=float, default=150.0)
    ap.add_argument("--max-minutes", type=float, default=210.0, help="download cap")
    ap.add_argument("--out", default="data/raw/ivod_demo")
    args = ap.parse_args()

    c = load_collector()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cands = []
    for rec in c.iter_records(c.get_catalog(args.year)):
        ln = rec.get("影片長度") or ""
        try:
            h, mi, s = [int(x) for x in ln.split(":")]
        except ValueError:
            continue
        mins = h * 60 + mi + s / 60
        if not (args.min_minutes <= mins <= args.max_minutes):
            continue
        if "full" not in str(rec.get("影片種類", "")).lower():
            continue
        tr = c._maybe_dict(rec.get("transcript"))
        if not (isinstance(tr, dict) and tr.get("whisperx") and tr.get("pyannote")):
            continue
        cands.append((mins, rec))
    if not cands:
        print("no candidate found")
        return 1
    cands.sort(key=lambda x: x[0])
    mins, rec = cands[0]
    ivod_id = str(rec.get("IVOD_ID"))
    print(f"chosen: IVOD {ivod_id}, {mins:.0f} min, {rec.get('會議名稱', '')[:70]}",
          flush=True)

    out_wav = out_dir / f"ivod_{args.year}_{ivod_id}.wav"
    tr = c._maybe_dict(rec.get("transcript")) or {}
    skip_s = c.robust_speech_start(tr.get("whisperx") or [])
    if not out_wav.exists():
        if not c.download_audio(rec["video_url"], out_wav, args.max_minutes,
                                skip_seconds=skip_s):
            print("download failed")
            return 1
    ok, dur, srate, ch = c.validate_wav(out_wav)
    print(f"wav ok={ok} dur={dur/60:.1f}min sr={srate}", flush=True)
    if not ok:
        return 1

    meta = c.build_meta(rec)
    row = {
        "audio_path": str(out_wav.resolve()),
        "duration": round(dur, 3),
        "samplerate": srate,
        "channels": ch,
        "title": rec.get("會議名稱"),
        "date": rec.get("日期"),
        "committee": meta.get("committee"),
        "source": "ivod",
        "license": "CC-BY-4.0",
        "attribution": c.ATTRIBUTION,
        "ivod_id": ivod_id,
        "meta": meta,
    }
    with (out_dir / "manifest_long.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("manifest_long.jsonl written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
