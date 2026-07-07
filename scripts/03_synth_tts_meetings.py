#!/usr/bin/env python
"""Synthesize TTS meetings from dialogue scripts, then augment them.

Pass 1: MeetingSynthesizer renders each DialogueScript from scripts.jsonl to
data/tts_raw/meet_NNNNN.wav (+ manifest.jsonl). Pass 2: augment_wav applies
RIR/MUSAN/codec augmentation into data/tts_aug/ (+ manifest.jsonl). Both
passes are resumable: existing output wavs are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.augment import augment_wav
from distil_vibevoice.data.dialogue_scripts import DialogueScript, Turn
from distil_vibevoice.data.manifest import MeetingRecord, read_manifest, write_manifest
from distil_vibevoice.data.tts_synth import MeetingSynthesizer


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def load_scripts(path: Path) -> list[DialogueScript]:
    scripts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            scripts.append(DialogueScript(
                speakers=d["speakers"], turns=[Turn(**t) for t in d["turns"]],
                domain=d["domain"], language=d.get("language", "zh-TW-en")))
    return scripts


def synth_pass(scripts: list[DialogueScript], out_dir: Path, cfg: dict,
               tts_model: str, device: str, limit: int | None) -> list[MeetingRecord]:
    bank = str(ROOT / (cfg.get("paths") or {}).get("speaker_bank", "data/speaker_bank"))
    synth = MeetingSynthesizer(tts_model_path=tts_model, speaker_bank_dir=bank, device=device)
    manifest_path = out_dir / "manifest.jsonl"
    records = read_manifest(manifest_path) if manifest_path.exists() else []
    done = {r.audio_path for r in records}
    for i, script in enumerate(scripts[:limit]):
        wav = out_dir / f"meet_{i:05d}.wav"
        if str(wav) in done or wav.exists():
            continue
        records.append(synth.synth(script, str(wav)))
        if len(records) % 50 == 0:
            write_manifest(records, manifest_path)
            print(f"  synthesized {len(records)}/{len(scripts[:limit])}")
    write_manifest(records, manifest_path)
    return records


def augment_pass(records: list[MeetingRecord], out_dir: Path, cfg: dict, seed: int) -> None:
    import numpy as np
    import soundfile as sf
    aug_cfg = cfg.get("augmentation") or {}
    paths = cfg.get("paths") or {}
    rir = str(ROOT / paths.get("rir_dir", "data/aug/RIRS_NOISES"))
    musan = str(ROOT / paths.get("musan_dir", "data/aug/musan"))
    rng = np.random.default_rng(seed)
    out_records = []
    for rec in records:
        out_wav = out_dir / Path(rec.audio_path).name
        if not out_wav.exists():
            wav, sr = sf.read(rec.audio_path, dtype="float32")
            aug = augment_wav(wav, sr, rir_dir=rir, musan_dir=musan,
                              codec_prob=float(aug_cfg.get("codec_prob", 0.3)),
                              snr_db_range=tuple(aug_cfg.get("snr_db_range", (5.0, 25.0))),
                              rng=rng)
            sf.write(out_wav, aug, sr)
        out_records.append(dataclasses.replace(
            rec, audio_path=str(out_wav), meta={**rec.meta, "augmented": True}))
    write_manifest(out_records, out_dir / "manifest.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--scripts", default=str(ROOT / "data/scripts/scripts.jsonl"))
    ap.add_argument("--tts-model", default="aoi-ot/VibeVoice-Large",
                    help="TTS weights (mirror of withdrawn microsoft/VibeVoice-Large)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="synthesize at most N scripts")
    ap.add_argument("--skip-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    raw_dir, aug_dir = ROOT / "data/tts_raw", ROOT / "data/tts_aug"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)

    scripts = load_scripts(Path(args.scripts))
    print(f"loaded {len(scripts)} scripts")
    records = synth_pass(scripts, raw_dir, cfg, args.tts_model, args.device, args.limit)
    print(f"synth pass done: {len(records)} meetings -> {raw_dir}/manifest.jsonl")
    if not args.skip_augment:
        augment_pass(records, aug_dir, cfg, args.seed)
        print(f"augment pass done -> {aug_dir}/manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
