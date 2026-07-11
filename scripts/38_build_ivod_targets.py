#!/usr/bin/env python
"""Fuse IVOD catalog whisperx (text+timestamps) x pyannote (speakers) into
MOSS-format training targets for real far-field fine-tuning.

v4 label-quality revision (v3 gate finding: fused segments spanning speaker
turns taught the model to emit long merged turns -> 123-min cluster purity
dropped 0.912->0.816). A 15 s length cap is NOT viable (median whisperx segment
is 18 s; the cap kept only 15% of speech). Instead the winning pyannote speaker
must cover >=70% of the segment span and >=75% of overlapped time — this drops
exactly the boundary-crossing segments while keeping clean single-speaker
turns.

For each whisperx segment (shifted by the collector's dead-air skip):
  - speaker = pyannote turn with max temporal overlap
  - drop if the winning speaker covers <60% of the overlapped time (ambiguous /
    overlapped speech) or if there is no overlap at all
  - text -> OpenCC s2tw (pure script conversion; whisperx zh is usually already
    Traditional, this is a safety net that never corrupts proper nouns)

Speakers are stored as meeting-level integer indices (order of first
appearance). Window-level renumbering (MOSS convention: S01 = first voice in
the WINDOW) happens in the FT dataloader, not here.

Output rows (same shape as the TTS manifests):
  {audio_path, duration, segments: [{start, end, speaker, text}], source}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "c01b", str(ROOT / "scripts/01b_collect_ivod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fuse_meeting(rec: dict, c01b, cc, min_dominance: float = 0.75,
                 min_dur: float = 0.4, max_dur: float = 30.0,
                 min_coverage: float = 0.7):
    """Return (segments, stats) for one collected meeting."""
    tr = rec.get("meta", {}).get("transcript") or {}
    wx = tr.get("whisperx") or []
    py = tr.get("pyannote") or []
    if not wx or not py:
        return [], {"reason": "no wx/py"}
    skip_s = c01b.robust_speech_start(wx)
    dur = float(rec["duration"])

    turns = sorted(
        [(p["start"] - skip_s, p["end"] - skip_s, str(p["speaker"]))
         for p in py if p.get("end") is not None],
        key=lambda t: t[0])

    spk_index: dict[str, int] = {}
    segs = []
    n_drop_overlap = n_drop_range = n_drop_dur = 0
    for w in wx:
        s, e = w["start"] - skip_s, w["end"] - skip_s
        text = (w.get("text") or "").strip()
        if not text:
            continue
        if s < 0 or e > dur or e <= s:
            n_drop_range += 1
            continue
        if not (min_dur <= e - s <= max_dur):
            n_drop_dur += 1
            continue
        # overlap per pyannote speaker
        ov: dict[str, float] = {}
        for ts, te, spk in turns:
            if te <= s:
                continue
            if ts >= e:
                break
            ov[spk] = ov.get(spk, 0.0) + min(e, te) - max(s, ts)
        total = sum(ov.values())
        if total <= 0:
            n_drop_overlap += 1
            continue
        best_spk, best = max(ov.items(), key=lambda kv: kv[1])
        if best / total < min_dominance or best / (e - s) < min_coverage:
            n_drop_overlap += 1
            continue
        if best_spk not in spk_index:
            spk_index[best_spk] = len(spk_index)
        segs.append({"start": round(s, 2), "end": round(e, 2),
                     "speaker": spk_index[best_spk],
                     "text": cc.convert(text)})
    segs.sort(key=lambda x: x["start"])
    stats = {"kept": len(segs), "drop_overlap": n_drop_overlap,
             "drop_range": n_drop_range, "drop_dur": n_drop_dur,
             "speakers": len(spk_index)}
    return segs, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+",
                    default=["data/raw/ivod_ft/manifest.jsonl",
                             "data/raw/ivod_eval/manifest.jsonl",
                             "data/raw/ivod_demo/manifest_long.jsonl"])
    ap.add_argument("--exclude-ids", nargs="+",
                    default=["15361", "15362", "15857"],
                    help="held-out eval/demo meetings; NEVER train on these")
    ap.add_argument("--out", default="data/pseudo/ivod_ft.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="print N sample segments per meeting and exit (QC)")
    ap.add_argument("--scan-dir", default="",
                    help="also reconstruct rows from catalog for ivod_<year>_"
                         "<id>.wav files in this dir (collector writes its "
                         "manifest only at the END of a run)")
    ap.add_argument("--year", type=int, default=2024)
    args = ap.parse_args()

    from opencc import OpenCC
    cc = OpenCC("s2tw")
    c01b = load_collector()

    rows = []
    seen = set()
    for mp in args.manifests:
        p = ROOT / mp
        if not p.exists():
            continue
        for line in p.open():
            rec = json.loads(line)
            if str(rec.get("ivod_id")) in seen:
                continue
            seen.add(str(rec.get("ivod_id")))
            rows.append(rec)

    if args.scan_dir:
        wavs = {p.stem.split("_")[-1]: p
                for p in (ROOT / args.scan_dir).glob(f"ivod_{args.year}_*.wav")}
        missing = {i for i in wavs if i not in seen}
        if missing:
            cat = {}
            for r in c01b.iter_records(c01b.get_catalog(args.year)):
                rid = str(r.get("IVOD_ID"))
                if rid in missing:
                    cat[rid] = r
            for rid in sorted(missing):
                r = cat.get(rid)
                if r is None:
                    continue
                wav = wavs[rid]
                ok, dur, srate, ch = c01b.validate_wav(wav)
                if not ok:
                    continue
                rows.append({"audio_path": str(wav.resolve()),
                             "duration": round(dur, 3),
                             "ivod_id": rid, "meta": c01b.build_meta(r)})
                seen.add(rid)
            print(f"scan-dir: reconstructed {len(rows)} total rows "
                  f"({len(missing)} from catalog)")

    out_rows, total_h, total_segs = [], 0.0, 0
    for rec in rows:
        ivod_id = str(rec.get("ivod_id"))
        if ivod_id in args.exclude_ids:
            print(f"  {ivod_id}: EXCLUDED (held-out eval)")
            continue
        if not Path(rec["audio_path"]).exists():
            continue
        segs, stats = fuse_meeting(rec, c01b, cc)
        speech = sum(s["end"] - s["start"] for s in segs)
        if len(segs) < 10 or stats.get("speakers", 0) < 2 or speech < 600:
            print(f"  {ivod_id}: skipped ({stats}, {speech/60:.0f} min)")
            continue
        print(f"  {ivod_id}: {stats['kept']} segs, {stats['speakers']} spk, "
              f"{speech/60:.0f} min speech "
              f"(drops: ov={stats['drop_overlap']} rng={stats['drop_range']} "
              f"dur={stats['drop_dur']})")
        if args.sample:
            for s in segs[:args.sample]:
                print(f"      [{s['start']:.1f}][S{s['speaker']+1:02d}]"
                      f"{s['text'][:50]}[{s['end']:.1f}]")
        out_rows.append({"audio_path": rec["audio_path"],
                         "duration": rec["duration"],
                         "segments": segs, "source": "ivod_wx_py"})
        total_h += speech / 3600
        total_segs += len(segs)

    if args.sample:
        print(f"\nQC only ({len(out_rows)} meetings usable, "
              f"{total_h:.1f}h speech, {total_segs} segs) — not written")
        return 0
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{outp}: {len(out_rows)} meetings, {total_h:.1f}h labeled speech, "
          f"{total_segs} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
