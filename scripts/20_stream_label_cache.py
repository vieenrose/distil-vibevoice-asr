#!/usr/bin/env python
"""Stream-and-discard: pseudo-label + cache latents + free audio, in one teacher load.

Per audio file: (1) teacher transcribes -> segments -> OpenCC s2twp -> pseudo-label
manifest; (2) frozen encoders -> cached 7.5 Hz latents (fp16 npz); (3) optionally
delete the wav. Keeps disk bounded (latents ~10 MB/hour) so a 1000-hour corpus fits
on the working disk without an external drive.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 24000
CHUNK_S = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/teacher")
    ap.add_argument("--audio-glob", default="data/raw/ivod/*.wav")
    ap.add_argument("--latents-out", default="data/latents/ivod")
    ap.add_argument("--manifest", default="data/pseudo/ivod_stream_manifest.jsonl")
    ap.add_argument("--delete-audio", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max files this run (0=all)")
    ap.add_argument("--max-audio-sec", type=float, default=480.0, help="cap audio fed to teacher (VRAM guard)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard id (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="total workers; each takes files where idx %% num_shards == shard")
    args = ap.parse_args()

    import soundfile as sf
    import torch

    from distil_vibevoice.data.manifest import Segment, MeetingRecord
    from distil_vibevoice.data.normalize_zhtw import normalize_record
    from distil_vibevoice.data.pseudo_label import TeacherLabeler

    latdir = Path(ROOT / args.latents_out); latdir.mkdir(parents=True, exist_ok=True)
    manifest = Path(ROOT / args.manifest); manifest.parent.mkdir(parents=True, exist_ok=True)
    done_stems = set()
    if manifest.exists():
        for line in manifest.open():
            try:
                done_stems.add(Path(json.loads(line)["audio_path"]).stem)
            except Exception:
                pass

    labeler = TeacherLabeler(args.teacher, device=args.device)
    model = labeler.model
    acoustic = model.model.acoustic_tokenizer
    semantic = model.model.semantic_tokenizer
    dtype = next(model.parameters()).dtype

    import zlib
    paths = sorted(glob.glob(str(ROOT / args.audio_glob)))
    if args.num_shards > 1:
        # stable hash of stem -> shard (survives deletions; no cross-shard collision)
        paths = [p for p in paths if zlib.crc32(Path(p).stem.encode()) % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(paths)} files")
    n_proc = 0
    tot_h = 0.0
    with manifest.open("a", encoding="utf-8") as mf:
        for p in paths:
            stem = Path(p).stem
            if stem in done_stems:
                continue
            if args.limit and n_proc >= args.limit:
                break
            # cap audio length (VRAM guard on a shared GPU); write a temp clip to label
            wav, sr = sf.read(p)
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(1)
            max_n = int(args.max_audio_sec * SR)
            if len(wav) > max_n:
                wav = wav[:max_n]
            dur_h = len(wav) / SR / 3600
            label_path = p
            tmp_clip = None
            if len(wav) < sf.info(p).frames:  # truncated -> label the truncated clip
                tmp_clip = str(Path(p).with_suffix(".clip.wav"))
                sf.write(tmp_clip, wav, SR); label_path = tmp_clip
            try:
                # (1) pseudo-label
                rec = labeler.label_file(label_path)
                rec = normalize_record(rec)  # OpenCC s2twp
                # (2) cache latents (frozen encoders)
                ac_means, se_means = [], []
                with torch.no_grad():
                    step = int(CHUNK_S * SR)
                    for s0 in range(0, len(wav), step):
                        chunk = torch.tensor(wav[s0:s0 + step], device=args.device, dtype=dtype).unsqueeze(0).unsqueeze(1)
                        if chunk.shape[-1] < SR // 10:
                            continue
                        ac_means.append(acoustic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
                        se_means.append(semantic.encode(chunk).mean.squeeze(0).to(torch.float16).cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tmp_clip and os.path.exists(tmp_clip):
                    os.remove(tmp_clip)
                print(f"  OOM on {stem} ({dur_h*60:.1f}min) — skipped")
                continue
            if tmp_clip and os.path.exists(tmp_clip):
                os.remove(tmp_clip)
            if not ac_means:
                continue
            latfile = latdir / f"{stem}.npz"
            np.savez_compressed(latfile, acoustic=np.concatenate(ac_means), semantic=np.concatenate(se_means), sr=SR, hop=3200)
            # (3) manifest: latents_path in META so read_manifest preserves it
            rec.meta["latents_path"] = str(latfile.relative_to(ROOT))
            row = {**asdict(rec), "segments": [asdict(s) for s in rec.segments]}
            mf.write(json.dumps(row, ensure_ascii=False) + "\n"); mf.flush()
            if args.delete_audio:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            n_proc += 1; tot_h += dur_h
            print(f"  [{n_proc}] {stem}: {dur_h*60:.1f}min, {len(rec.segments)} segs -> "
                  f"latents {latfile.stat().st_size/1e6:.1f}MB{' (audio deleted)' if args.delete_audio else ''}")

    print(f"\nprocessed {n_proc} files, {tot_h:.2f} h this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
