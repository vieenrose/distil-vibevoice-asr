#!/usr/bin/env python
"""Merge per-source manifests into train/val manifests per data.yaml mix ratios.

Steps: load every source manifest, dedupe records against the eval-set audio
fingerprint index, subsample sources to match ``mix_ratios`` (hour-based),
stratified train/val split per source, write data/manifests/{train,val}.jsonl
and print an hours-per-source table (rich if available).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.dedupe import build_eval_index, filter_against_index
from distil_vibevoice.data.manifest import MeetingRecord, read_manifest, write_manifest

# Default mapping of mix bucket -> source manifests produced by earlier stages.
DEFAULT_BUCKETS: dict[str, list[str]] = {
    "tts_synthetic": ["data/tts_aug/manifest.jsonl"],
    "pseudo_labeled": ["data/pseudo/*_manifest.jsonl"],
    "simulated_mixtures": ["data/simulated/manifest.jsonl"],
    "gold": ["data/gold/manifest.jsonl"],
}


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def hours(records: list[MeetingRecord]) -> float:
    return sum(r.duration_s for r in records) / 3600.0


def load_bucket(patterns: list[str]) -> list[MeetingRecord]:
    records: list[MeetingRecord] = []
    for pat in patterns:
        for p in sorted(ROOT.glob(pat)) or ([ROOT / pat] if (ROOT / pat).exists() else []):
            records.extend(read_manifest(p))
    return records


def print_table(rows: list[tuple[str, float, float, int, int]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title="Training mix")
        for col in ("bucket", "available h", "selected h", "train recs", "val recs"):
            t.add_column(col)
        for r in rows:
            t.add_row(r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", str(r[3]), str(r[4]))
        Console().print(t)
    except ImportError:
        print(f"{'bucket':<22}{'avail_h':>9}{'sel_h':>9}{'train':>8}{'val':>7}")
        for r in rows:
            print(f"{r[0]:<22}{r[1]:>9.1f}{r[2]:>9.1f}{r[3]:>8}{r[4]:>7}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths_cfg = cfg.get("paths") or {}
    ratios: dict[str, float] = cfg.get("mix_ratios") or {
        "tts_synthetic": 0.4, "pseudo_labeled": 0.4, "simulated_mixtures": 0.15, "gold": 0.05}
    out_dir = ROOT / paths_cfg.get("manifests", "data/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = {name: load_bucket(DEFAULT_BUCKETS.get(name, [])) for name in ratios}
    eval_manifests = [str(ROOT / p) for p in paths_cfg.get("eval_manifests", [])
                      if (ROOT / p).exists()]
    if eval_manifests and not args.no_dedupe:
        index = build_eval_index(eval_manifests)
        print(f"eval fingerprint index: {len(index)} entries from {len(eval_manifests)} manifests")
        buckets = {n: filter_against_index(recs, index, audio_root=str(ROOT))
                   for n, recs in buckets.items()}

    # Hour-budget: the scarcest bucket relative to its ratio caps the total.
    nonempty = {n: h for n in ratios if (h := hours(buckets[n])) > 0}
    total_h = min(nonempty[n] / ratios[n] for n in nonempty) if nonempty else 0.0
    rng = random.Random(args.seed)
    train: list[MeetingRecord] = []
    val: list[MeetingRecord] = []
    rows = []
    for name, ratio in ratios.items():
        recs = buckets[name]
        rng.shuffle(recs)
        budget_s, sel, acc = total_h * ratio * 3600.0, [], 0.0
        for r in recs:
            if acc >= budget_s:
                break
            sel.append(r)
            acc += r.duration_s
        n_val = max(1, round(len(sel) * args.val_fraction)) if sel else 0
        val += sel[:n_val]
        train += sel[n_val:]
        rows.append((name, hours(recs), hours(sel), len(sel) - n_val, n_val))

    rng.shuffle(train)
    write_manifest(train, out_dir / "train.jsonl")
    write_manifest(val, out_dir / "val.jsonl")
    print_table(rows)
    print(f"train: {len(train)} recs ({hours(train):.1f} h) -> {out_dir / 'train.jsonl'}")
    print(f"val:   {len(val)} recs ({hours(val):.1f} h) -> {out_dir / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
