#!/usr/bin/env python
"""Pseudo-label a directory of audio with the 8B VibeVoice-ASR teacher.

Globs audio under --audio-dir, labels in batches with TeacherLabeler,
filters by mean logprob, normalizes text to zh-TW (OpenCC s2twp), and writes
data/pseudo/<source>_manifest.jsonl. Resumable: paths already present in the
output manifest are skipped. --two-pass labels everything twice and keeps
only records whose passes agree (cpWER <= threshold).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.manifest import MeetingRecord, read_manifest, write_manifest
from distil_vibevoice.data.normalize_zhtw import normalize_record
from distil_vibevoice.data.pseudo_label import (
    TeacherLabeler,
    filter_by_confidence,
    two_pass_agreement,
)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def label_paths(labeler: TeacherLabeler, paths: list[str], hotwords: list[str] | None,
                two_pass: bool, max_cpwer: float) -> list[MeetingRecord]:
    recs = labeler.label_batch(paths, hotwords=hotwords)
    if not two_pass:
        return recs
    recs_b = labeler.label_batch(paths, hotwords=hotwords)
    kept = [a for a, b in zip(recs, recs_b) if two_pass_agreement(a, b, max_cpwer=max_cpwer)]
    print(f"  two-pass agreement kept {len(kept)}/{len(recs)}")
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--glob", default="**/*", help="glob under --audio-dir")
    ap.add_argument("--source", required=True, help="source tag, names the output manifest")
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--hotwords-file", default=None, help="one hotword/name per line")
    ap.add_argument("--two-pass", action="store_true", help="two_pass_agreement filtering")
    ap.add_argument("--min-logprob", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pl_cfg = load_cfg(Path(args.config)).get("pseudo_label") or {}
    model_path = str(ROOT / pl_cfg.get("teacher_path", "models/teacher"))
    device = args.device or pl_cfg.get("device", "cuda:0")
    min_lp = args.min_logprob if args.min_logprob is not None \
        else float(pl_cfg.get("min_mean_logprob", -0.5))
    max_cpwer = float(pl_cfg.get("two_pass_agreement_max_cpwer", 0.05))
    hotwords = None
    if args.hotwords_file:
        hotwords = [w for w in Path(args.hotwords_file).read_text().splitlines() if w.strip()]

    out = ROOT / "data/pseudo" / f"{args.source}_manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done_records = read_manifest(out) if out.exists() else []
    done = {r.audio_path for r in done_records}

    all_paths = sorted(str(p) for p in Path(args.audio_dir).glob(args.glob)
                       if p.suffix.lower() in AUDIO_EXTS)
    todo = [p for p in all_paths if p not in done]
    print(f"{len(all_paths)} audio files, {len(done)} already labeled, {len(todo)} to do")
    if not todo:
        return 0

    labeler = TeacherLabeler(model_path=model_path, device=device,
                             dtype=pl_cfg.get("dtype", "bfloat16"))
    kept_total = 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        recs = label_paths(labeler, batch, hotwords, args.two_pass, max_cpwer)
        recs = filter_by_confidence(recs, min_mean_logprob=min_lp)
        recs = [normalize_record(r) for r in recs]
        kept_total += len(recs)
        done_records.extend(recs)
        write_manifest(done_records, out)  # flush after every batch -> resumable
        print(f"  [{i + len(batch)}/{len(todo)}] kept {kept_total} records -> {out}")
    print(f"done: {len(done_records)} total records in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
