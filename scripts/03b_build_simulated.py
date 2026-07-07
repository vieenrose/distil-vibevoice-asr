#!/usr/bin/env python
"""Build SIMULATED zh-TW multi-speaker meetings from real Common Voice clips.

This exercises the *unblocked* real-data path (no `vibevoice` package needed):
real single-speaker zh-TW utterances -> synthetic overlapping meeting mixtures
with EXACT Segment labels, via
``distil_vibevoice.data.simulate_meetings.simulate_meeting``.

Pipeline per meeting:
  1. pick 2-6 speakers from Common Voice ``client_id``s (real speaker identity),
  2. take a few real utterances per speaker (mp3 -> 24 kHz mono float32),
  3. interleave them round-robin and mix with random silence + overlap_ratio,
  4. write the mixture wav to ``data/simulated/`` and a
     :class:`distil_vibevoice.data.manifest.MeetingRecord` line to
     ``data/manifests/simulated.jsonl`` (start/end/speaker/text are exact).

An augment pass (:func:`distil_vibevoice.data.augment.augment_wav`) is run on a
few meetings to confirm the codec/gain chain works even with RIR/MUSAN absent
(dirs passed as None).

Source clips default to the extracted Common Voice 22.0 zh-TW *test* shard
downloaded by ``scripts/01_download_data.py --only common_voice_zhtw``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.augment import augment_wav
from distil_vibevoice.data.manifest import (
    MeetingRecord,
    Segment,
    read_manifest,
    write_manifest,
)
from distil_vibevoice.data.simulate_meetings import simulate_meeting


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _resample_to(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Mono float32 resample via scipy polyphase (matches augment._resample)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr_from == sr_to:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def build_client_index(tsv: Path, clips_dir: Path, min_clips: int) -> dict[str, list[tuple[str, str]]]:
    """Map client_id -> [(abs_mp3_path, sentence), ...] for clips present on disk.

    Only clients with at least ``min_clips`` utterances are kept so every
    simulated speaker contributes several real turns.
    """
    on_disk = {os.path.basename(p): p for p in glob.glob(str(clips_dir / "*.mp3"))}
    by_client: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with tsv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            p = on_disk.get(row["path"])
            sent = (row.get("sentence") or "").strip()
            if p and sent:
                by_client[row["client_id"]].append((p, sent))
    return {c: v for c, v in by_client.items() if len(v) >= min_clips}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument(
        "--clips-dir",
        default=str(ROOT / "data/raw/common_voice_zhtw/extracted/zh-TW_test_0"),
        help="dir of source .mp3 utterances (Common Voice zh-TW shard)",
    )
    ap.add_argument(
        "--tsv",
        default=str(ROOT / "data/raw/common_voice_zhtw/transcript/zh-TW/test.tsv"),
        help="Common Voice transcript tsv (path/client_id/sentence)",
    )
    ap.add_argument("--out-audio", default=str(ROOT / "data/simulated"))
    ap.add_argument("--out-manifest", default=str(ROOT / "data/manifests/simulated.jsonl"))
    ap.add_argument("-n", "--n-meetings", type=int, default=30)
    ap.add_argument("--sr", type=int, default=None, help="target sample rate (default data.yaml audio.sample_rate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speakers-min", type=int, default=2)
    ap.add_argument("--speakers-max", type=int, default=6)
    ap.add_argument("--utts-min", type=int, default=3, help="min utterances per speaker per meeting")
    ap.add_argument("--utts-max", type=int, default=6, help="max utterances per speaker per meeting")
    ap.add_argument("--overlap-ratio", type=float, default=None)
    ap.add_argument("--n-augment", type=int, default=3, help="how many meetings to also augment")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    sr = args.sr or int((cfg.get("audio") or {}).get("sample_rate", 24000))
    sim_cfg = cfg.get("simulate") or {}
    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else float(sim_cfg.get("overlap_ratio", 0.10))
    silence_range = tuple(sim_cfg.get("silence_range", [0.2, 1.5]))
    aug_cfg = cfg.get("augmentation") or {}
    codec_prob = float(aug_cfg.get("codec_prob", 0.3))
    snr_db_range = tuple(aug_cfg.get("snr_db_range", [5.0, 25.0]))

    clips_dir = Path(args.clips_dir)
    tsv = Path(args.tsv)
    if not clips_dir.is_dir():
        ap.error(f"clips dir not found: {clips_dir} (run scripts/01_download_data.py + extract the tar)")
    if not tsv.is_file():
        ap.error(f"transcript tsv not found: {tsv}")

    by_client = build_client_index(tsv, clips_dir, min_clips=args.utts_min)
    clients = sorted(by_client)  # deterministic order before shuffle
    if len(clients) < args.speakers_max:
        ap.error(f"not enough speakers with >={args.utts_min} clips: {len(clients)}")
    print(f"source: {len(clients)} zh-TW speakers (client_ids) with >={args.utts_min} clips")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(clients)

    out_audio = Path(args.out_audio)
    out_audio.mkdir(parents=True, exist_ok=True)
    aug_dir = out_audio / "augmented"

    # A pool of speakers we draw from without replacement across meetings; refill
    # (reshuffle) when exhausted so every meeting has fresh real speakers.
    pool: list[str] = []

    def draw_speakers(k: int) -> list[str]:
        nonlocal pool
        if len(pool) < k:
            pool = list(clients)
            rng.shuffle(pool)
        picked, pool = pool[:k], pool[k:]
        return picked

    # cache decoded+resampled clips so a reused speaker isn't re-decoded
    wav_cache: dict[str, np.ndarray] = {}

    def load_clip(path: str) -> np.ndarray:
        if path not in wav_cache:
            x, in_sr = sf.read(path, dtype="float32", always_2d=False)
            wav_cache[path] = _resample_to(np.asarray(x), int(in_sr), sr)
        return wav_cache[path]

    records: list[MeetingRecord] = []
    spk_dist: collections.Counter = collections.Counter()
    total_samples = 0
    n_clips_used = 0

    for mi in range(args.n_meetings):
        n_spk = int(rng.integers(args.speakers_min, args.speakers_max + 1))
        spk_clients = draw_speakers(n_spk)

        # Per speaker: an ordered queue of that speaker's real utterances.
        queues: list[list[tuple[str, str]]] = []
        for c in spk_clients:
            utts = list(by_client[c])
            rng.shuffle(utts)
            k = int(rng.integers(args.utts_min, min(args.utts_max, len(utts)) + 1))
            queues.append(utts[:k])

        # Round-robin interleave so speakers alternate (realistic turn-taking).
        utterances: list[tuple[np.ndarray, str, str]] = []
        idx = 0
        while any(queues):
            spk = idx % n_spk
            if queues[spk]:
                path, sent = queues[spk].pop(0)
                wav = load_clip(path)
                if wav.size:
                    utterances.append((wav, str(spk), sent))
                    n_clips_used += 1
            idx += 1
            if idx > n_spk * (args.utts_max + 2):
                break

        if not utterances:
            continue
        mix, segments = simulate_meeting(
            utterances, sr, overlap_ratio=overlap_ratio, silence_range=silence_range, rng=rng
        )
        wav_path = out_audio / f"sim_meeting_{mi:04d}.wav"
        sf.write(str(wav_path), mix, sr, subtype="PCM_16")
        total_samples += mix.shape[0]
        spk_dist[n_spk] += 1

        records.append(
            MeetingRecord(
                audio_path=str(wav_path),
                duration_s=round(mix.shape[0] / sr, 3),
                sample_rate=sr,
                language="zh-TW",
                source="common_voice_zhtw_simulated",
                split="train",
                segments=segments,
                meta={
                    "n_speakers": n_spk,
                    "client_ids": spk_clients,
                    "overlap_ratio": overlap_ratio,
                    "sim_seed": args.seed,
                    "origin": "cv22_zh-TW_test",
                },
            )
        )

    write_manifest(records, Path(args.out_manifest))
    print(f"wrote {len(records)} meetings -> {args.out_manifest}")
    print(f"       {n_clips_used} real CV clips used; audio in {out_audio}/")

    # ---- augment pass (RIR/MUSAN absent -> None; exercises codec + gain) ----
    augment_ok = False
    aug_written = 0
    if args.n_augment > 0 and records:
        aug_dir.mkdir(parents=True, exist_ok=True)
        arng = np.random.default_rng(args.seed + 999)
        for rec in records[: args.n_augment]:
            x, xsr = sf.read(rec.audio_path, dtype="float32", always_2d=False)
            y = augment_wav(
                np.asarray(x),
                int(xsr),
                rir_dir=None,
                musan_dir=None,
                codec_prob=1.0,  # force codec so the chain is actually exercised
                snr_db_range=snr_db_range,
                rng=arng,
            )
            assert y.shape == np.asarray(x).shape and y.dtype == np.float32
            ap_out = aug_dir / (Path(rec.audio_path).stem + "_aug.wav")
            sf.write(str(ap_out), y, int(xsr), subtype="PCM_16")
            aug_written += 1
        augment_ok = aug_written == min(args.n_augment, len(records))
        print(f"augment: wrote {aug_written} augmented copies -> {aug_dir}/ (codec+gain, rir/musan=None)")

    # ---- verify manifest reloads and segments are well-formed ----
    reloaded = read_manifest(args.out_manifest)
    manifest_ok = len(reloaded) == len(records)
    seg_problems = 0
    total_hours = total_samples / sr / 3600.0
    for rec in reloaded:
        segs = rec.segments
        # sorted by start
        if any(segs[i].start > segs[i + 1].start + 1e-6 for i in range(len(segs) - 1)):
            seg_problems += 1
            continue
        for s in segs:
            if not (0.0 <= s.start < s.end <= rec.duration_s + 1e-3):
                seg_problems += 1
                break
    manifest_ok = manifest_ok and seg_problems == 0

    print("\n=== SUMMARY ===")
    print(f"meetings          : {len(records)}")
    print(f"total hours       : {total_hours:.4f}")
    print(f"real CV clips used: {n_clips_used}")
    print(f"speaker-count dist: {dict(sorted(spk_dist.items()))}")
    print(f"manifest reloads  : {manifest_ok} ({len(reloaded)} records, {seg_problems} seg problems)")
    print(f"augment ok        : {augment_ok}")
    return 0 if (manifest_ok and (augment_ok or args.n_augment == 0)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
