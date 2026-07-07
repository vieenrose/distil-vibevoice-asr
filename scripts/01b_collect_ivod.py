#!/usr/bin/env python
"""Collect real zh-TW parliamentary MEETING audio from the Taiwan Legislative
Yuan IVOD open-data catalog (HuggingFace dataset ``openfun/tw-ly-ivod``).

LICENSE / ATTRIBUTION
--------------------
The IVOD catalog and the underlying video/audio are Taiwan Legislative Yuan
open data, released under **CC-BY-4.0** (see the dataset card at
https://huggingface.co/datasets/openfun/tw-ly-ivod and https://data.ly.gov.tw ).
You MUST attribute the source when you redistribute or publish anything derived
from this audio, e.g.:

    Source: 立法院 IVOD (Taiwan Legislative Yuan) open data, CC-BY-4.0,
    aggregated by openfun/tw-ly-ivod.

This attribution string is also written into every manifest record (``license``)
and printed at the end of each run.

CATALOG SCHEMA (per JSONL line, one IVOD video)
----------------------------------------------
    IVOD_ID       str   e.g. "16441"
    IVOD_URL      str   human page, e.g. https://ivod.ly.gov.tw/Play/Full/1M/16441
    日期          str   meeting date "YYYY-MM-DD"
    會議資料      str   (stringified dict) committee / session metadata
    影片種類      str   "Full" (whole meeting) or "Clip" (single legislator turn)
    開始時間      str   ISO8601 video start
    結束時間      str   ISO8601 video end
    影片長度      str   "HH:MM:SS" video duration
    支援功能      str   e.g. "['ai-transcript']"
    video_url     str   HLS playlist .m3u8  <-- what we fetch audio from
    會議時間      str   ISO8601 meeting time
    會議名稱      str   full meeting title (committee, session, agenda)
    委員名稱      str   legislator name, or "完整會議" for a Full-meeting video
    委員發言時間  str   "HH:MM:SS - HH:MM:SS" speaking window
    transcript    dict  {"pyannote":[{speaker,start,end}...],            # diarization
                         "whisperx":[{start,end,text}...]}               # ASR text
                        (may be empty/absent for older records)

We prefer 影片種類=="Full" (委員名稱=="完整會議") records because they are genuine
multi-speaker meetings (chair + legislators + officials) which is exactly the
diarization+timestamp target for this project. Each download is capped with
``ffmpeg -t`` so we only pull the first --max-minutes of segments (polite: we do
not fetch the whole 10-hour stream). Audio is written as 24 kHz mono WAV.

USAGE
-----
    python scripts/01b_collect_ivod.py --year 2025 --limit 3 --max-minutes 5
    python scripts/01b_collect_ivod.py --year 2025 --limit 3 --kind any --out data/raw/ivod

Idempotent: existing valid WAVs are skipped.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:  # optional; only used for validation
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

from huggingface_hub import hf_hub_download

DATASET_REPO = "openfun/tw-ly-ivod"
ATTRIBUTION = (
    "Source: 立法院 IVOD (Taiwan Legislative Yuan) open data, CC-BY-4.0, "
    "aggregated by HuggingFace dataset openfun/tw-ly-ivod."
)
TARGET_SR = 24000


def log(msg: str) -> None:
    print(f"[collect_ivod] {msg}", flush=True)


def get_catalog(year: int) -> Path:
    """Download (cached) the per-year catalog JSONL from the HF dataset."""
    fname = f"ivod-{year}.jsonl"
    log(f"fetching catalog {fname} from {DATASET_REPO} ...")
    path = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset")
    return Path(path)


def iter_records(catalog: Path):
    with catalog.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _maybe_dict(val):
    """Some fields are python-repr strings; parse leniently, else return raw."""
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def robust_speech_start(wx) -> float:
    """First SUSTAINED speech time (start of the first cluster with >=3 whisperx
    segments in the following 60s). Robust to spurious early false-positive
    detections that a plain min(starts) would seek to (landing in dead air)."""
    starts = sorted(w["start"] for w in wx if isinstance(w.get("start"), (int, float)))
    if not starts:
        return 0.0
    for s in starts:
        if sum(1 for t in starts if s <= t < s + 60) >= 3:
            return max(0.0, s - 5.0)
    return max(0.0, starts[0] - 5.0)


def _has_whisperx(rec) -> bool:
    tr = _maybe_dict(rec.get("transcript")) or {}
    wx = tr.get("whisperx") if isinstance(tr, dict) else None
    return bool(wx) and any(isinstance(w.get("start"), (int, float)) for w in wx)


def select_records(catalog: Path, kind: str, limit: int, require_transcript: bool = False):
    """Pick up to `limit` records. kind: 'full' (multi-speaker meetings),
    'clip' (single legislator), or 'any'. require_transcript keeps only records
    with a catalog whisperx transcript (needed for the dead-air skip; avoids
    collecting standby-noise meetings that pseudo-label as pure [Noise])."""
    picked = []
    for rec in iter_records(catalog):
        if not rec.get("video_url"):
            continue
        is_full = rec.get("委員名稱") == "完整會議" or rec.get("影片種類") == "Full"
        if kind == "full" and not is_full:
            continue
        if kind == "clip" and is_full:
            continue
        if require_transcript and not _has_whisperx(rec):
            continue
        picked.append(rec)
        if len(picked) >= limit:
            break
    return picked


def download_audio(video_url: str, out_wav: Path, max_minutes: float, skip_seconds: float = 0.0) -> bool:
    """Pull max_minutes of the HLS stream (after skip_seconds) -> 24k mono WAV.

    IVOD videos open with 20-40 min of pre-meeting standby (dead air), so we seek
    past it via -ss (before -i = fast keyframe seek) to the catalog's first-speech
    time; -t then caps how much actual meeting we capture."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_suffix(".part.wav")
    if tmp.exists():
        tmp.unlink()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-user_agent", "Mozilla/5.0"]
    if skip_seconds > 0:
        cmd += ["-ss", str(int(skip_seconds))]
    cmd += [
        "-i", video_url,
        "-t", str(int(max_minutes * 60)),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
        str(tmp),
    ]
    log(f"ffmpeg -> {out_wav.name} (skip {skip_seconds:.0f}s, cap {max_minutes} min)")
    try:
        subprocess.run(cmd, check=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        log(f"  ffmpeg failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        log("  ffmpeg timed out")
        return False
    if not tmp.exists() or tmp.stat().st_size == 0:
        return False
    tmp.replace(out_wav)
    return True


def validate_wav(path: Path):
    """Return (ok, duration_sec, samplerate, channels) using soundfile."""
    if sf is None:
        return (path.exists() and path.stat().st_size > 0, 0.0, 0, 0)
    try:
        info = sf.info(str(path))
        ok = info.samplerate == TARGET_SR and info.channels == 1 and info.frames > 0
        return (ok, info.frames / info.samplerate, info.samplerate, info.channels)
    except Exception as e:  # pragma: no cover
        log(f"  soundfile validate error: {e}")
        return (False, 0.0, 0, 0)


def build_meta(rec: dict) -> dict:
    """Extract useful catalog metadata + teacher transcript into a meta blob."""
    transcript = _maybe_dict(rec.get("transcript")) or {}
    meeting = _maybe_dict(rec.get("會議資料"))
    meta = {
        "ivod_id": rec.get("IVOD_ID"),
        "ivod_url": rec.get("IVOD_URL"),
        "video_url": rec.get("video_url"),
        "committee": meeting if isinstance(meeting, dict) else rec.get("會議資料"),
        "legislator": rec.get("委員名稱"),
        "video_length": rec.get("影片長度"),
        "speaking_window": rec.get("委員發言時間"),
        "features": rec.get("支援功能"),
    }
    # keep the diarization + ASR transcript (may be long) for later use as
    # a weak reference / hotword mining; store compactly.
    if isinstance(transcript, dict):
        meta["transcript"] = {
            "pyannote": transcript.get("pyannote"),
            "whisperx": transcript.get("whisperx"),
        }
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2025, help="catalog year (2005-2025)")
    ap.add_argument("--limit", type=int, default=3, help="number of meetings to fetch")
    ap.add_argument("--out", type=str, default="data/raw/ivod", help="output dir")
    ap.add_argument("--max-minutes", type=float, default=5.0,
                    help="cap each download to first N minutes")
    ap.add_argument("--kind", choices=["full", "clip", "any"], default="full",
                    help="'full'=multi-speaker whole meeting (default), "
                         "'clip'=single legislator, 'any'")
    ap.add_argument("--done-manifest", default="", help="skip meetings already in this label manifest (stem dedup)")
    ap.add_argument("--require-transcript", action="store_true",
                    help="only collect meetings with a catalog whisperx transcript (enables dead-air skip; avoids noise-only meetings)")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="polite delay (s) between downloads")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    log(ATTRIBUTION)

    catalog = get_catalog(args.year)
    records = select_records(catalog, args.kind, args.limit, require_transcript=args.require_transcript)
    log(f"selected {len(records)} record(s) (kind={args.kind}) from {catalog.name}")
    if not records:
        log("no matching records; nothing to do.")
        return 1

    # load existing manifest ids (idempotent)
    existing = {}
    if manifest_path.exists():
        for r in iter_records(manifest_path):
            existing[str(r.get("ivod_id"))] = r

    written = []
    # skip meetings already pseudo-labeled (wavs are deleted after labeling, so
    # out_wav.exists() alone can't dedup — check the label manifest by stem).
    done_stems = set()
    dm = Path(args.done_manifest) if args.done_manifest else None
    if dm and dm.exists():
        import json as _json
        for l in dm.read_text().splitlines():
            try:
                done_stems.add(Path(_json.loads(l)["audio_path"]).stem.replace(".clip", ""))
            except Exception:
                pass

    for i, rec in enumerate(records, 1):
        ivod_id = str(rec.get("IVOD_ID"))
        out_wav = out_dir / f"ivod_{args.year}_{ivod_id}.wav"
        if f"ivod_{args.year}_{ivod_id}" in done_stems:
            continue  # already labeled in a previous cycle
        log(f"[{i}/{len(records)}] IVOD {ivod_id}  {rec.get('會議名稱','')[:50]}")

        ok, dur, srate, ch = (False, 0.0, 0, 0)
        if out_wav.exists():
            ok, dur, srate, ch = validate_wav(out_wav)
            if ok:
                log(f"  exists & valid ({dur:.1f}s); skip download")
        if not (out_wav.exists() and ok):
            # seek past leading dead air to the catalog's first-speech time
            skip_s = 0.0
            tr = _maybe_dict(rec.get("transcript")) or {}
            wx = tr.get("whisperx") if isinstance(tr, dict) else None
            if wx:
                skip_s = robust_speech_start(wx)  # first sustained-speech cluster, not min()
            if not download_audio(rec["video_url"], out_wav, args.max_minutes, skip_seconds=skip_s):
                log("  download failed; skipping record")
                continue
            ok, dur, srate, ch = validate_wav(out_wav)
            if not ok:
                log(f"  produced WAV invalid (sr={srate}, ch={ch}); skipping")
                continue
            if args.sleep:
                time.sleep(args.sleep)

        meta = build_meta(rec)
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
            "attribution": ATTRIBUTION,
            "ivod_id": ivod_id,
            "meta": meta,
        }
        existing[ivod_id] = row
        written.append(row)

    # rewrite manifest (idempotent merge)
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_min = sum(r["duration"] for r in written) / 60.0
    log(f"wrote/updated {len(written)} record(s); manifest -> {manifest_path}")
    log(f"downloaded this run: {len(written)} meeting(s), {total_min:.2f} min total")
    log("REMINDER: " + ATTRIBUTION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
