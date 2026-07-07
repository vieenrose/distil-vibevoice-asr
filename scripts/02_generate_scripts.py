#!/usr/bin/env python
"""Generate zh-TW/en code-switched meeting dialogue scripts.

Wraps ``distil_vibevoice.data.dialogue_scripts.generate_scripts`` and writes
one JSON object per line (dataclasses.asdict of DialogueScript) to
data/scripts/scripts.jsonl. Counts/domains/seed default from
configs/data.yaml ``dialogue_scripts:``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.dialogue_scripts import generate_scripts


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("-n", "--num-scripts", type=int, default=None)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "data/scripts/scripts.jsonl"))
    args = ap.parse_args()

    ds_cfg = load_cfg(Path(args.config)).get("dialogue_scripts") or {}
    n = args.num_scripts if args.num_scripts is not None else int(ds_cfg.get("n_scripts", 20000))
    domains = args.domains if args.domains is not None else ds_cfg.get("domains")
    seed = args.seed if args.seed is not None else int(ds_cfg.get("seed", 0))

    print(f"generating {n} scripts (domains={domains or 'default'}, seed={seed}) ...")
    scripts = generate_scripts(n, domains=domains, seed=seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in scripts:
            f.write(json.dumps(dataclasses.asdict(s), ensure_ascii=False) + "\n")
    print(f"wrote {len(scripts)} scripts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
