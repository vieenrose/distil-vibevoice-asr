#!/usr/bin/env python
"""Run the student over an eval manifest and check the release gates (CI-able).

Transcribes every reference record's audio with ChunkedTranscriber (student
checkpoint wrapped in TeacherLabeler), writes a hypothesis manifest (resumable:
already-transcribed audio paths are skipped), then runs run_gates() with the
pooled thresholds from configs/eval_gates.yaml plus each per-slice threshold
set (slices select records by manifest-field match; filtered temp manifests go
under <hyp-dir>/slices/). Pretty-prints every GateReport and exits 1 if any
gate fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distil_vibevoice.data.manifest import MeetingRecord, read_manifest, write_manifest
from distil_vibevoice.data.pseudo_label import TeacherLabeler
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.gates import GateReport, run_gates
from distil_vibevoice.runtime.chunked_inference import ChunkedTranscriber


def speaker_consistency_over_manifest(
    refs: list[MeetingRecord], hyp_by_path: dict[str, MeetingRecord]
) -> float | None:
    """Duration-weighted mean global-speaker consistency over paired records."""
    total_w = 0.0
    acc = 0.0
    for ref in refs:
        hyp = hyp_by_path.get(ref.audio_path)
        if hyp is None:
            continue
        w = max(ref.duration_s, 1e-9)
        acc += speaker_consistency(ref.segments, hyp.segments) * w
        total_w += w
    return acc / total_w if total_w > 0.0 else None


def transcribe_all(model: str, refs: list[MeetingRecord], hyp_path: Path, device: str,
                   dtype: str, window_s: float, overlap_s: float,
                   hotwords: list[str] | None, consolidate: bool = False,
                   registry_state: str | None = None) -> list[MeetingRecord]:
    done = {r.audio_path: r for r in read_manifest(hyp_path)} if hyp_path.exists() else {}
    todo = [r for r in refs if r.audio_path not in done]
    print(f"{len(refs)} eval records, {len(done)} already transcribed, {len(todo)} to do")
    hyps = [done[r.audio_path] for r in refs if r.audio_path in done]
    if todo:
        # A registry (persistent global-speaker identity) is only engaged when
        # consolidation or a state path is requested; otherwise the legacy
        # stitch-only path is used unchanged.
        embedder = None
        if consolidate or registry_state:
            from distil_vibevoice.runtime.embeddings import load_embedder
            embedder = load_embedder("mfcc")
        transcriber = ChunkedTranscriber(
            TeacherLabeler(model_path=model, device=device, dtype=dtype),
            window_s=window_s, overlap_s=overlap_s,
            embedder=embedder, consolidate_on_finish=consolidate,
            registry_state=registry_state)
        for i, ref in enumerate(todo, 1):
            hw = hotwords or ref.meta.get("hotwords")
            hyps.append(transcriber.transcribe(ref.audio_path, hotwords=hw))
            write_manifest(hyps, hyp_path)  # flush -> resumable
            print(f"  [{i}/{len(todo)}] {ref.audio_path}")
    return hyps


def matches(rec: MeetingRecord, select: dict) -> bool:
    return all(getattr(rec, k, None) == v or rec.meta.get(k) == v for k, v in select.items())


def slice_report(name: str, spec: dict, refs: list[MeetingRecord],
                 hyp_by_path: dict[str, MeetingRecord], work: Path) -> GateReport | None:
    sub_ref = [r for r in refs if matches(r, spec.get("select") or {})]
    sub_hyp = [hyp_by_path[r.audio_path] for r in sub_ref if r.audio_path in hyp_by_path]
    if not sub_ref:
        print(f"[warn] slice '{name}': no matching records — skipped", file=sys.stderr)
        return None
    ref_p, hyp_p = work / f"{name}_ref.jsonl", work / f"{name}_hyp.jsonl"
    write_manifest(sub_ref, ref_p)
    write_manifest(sub_hyp, hyp_p)
    return run_gates(str(ref_p), str(hyp_p), spec.get("thresholds") or {})


def print_report(name: str, rep: GateReport) -> None:
    status = "PASS" if rep.passed else "FAIL"
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(title=f"{name} — {status}")
        t.add_column("metric")
        t.add_column("value")
        for k, v in rep.metrics.items():
            t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        Console().print(t)
        for f in rep.failures:
            Console().print(f"[red]  FAIL: {f}[/red]")
    except ImportError:
        print(f"\n== {name}: {status}")
        for k, v in rep.metrics.items():
            print(f"  {k:<20}{v}")
        for f in rep.failures:
            print(f"  FAIL: {f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="student checkpoint dir (VibeVoice-style)")
    ap.add_argument("--config", default=str(ROOT / "configs/eval_gates.yaml"))
    ap.add_argument("--ref", default=None, help="override manifests.ref")
    ap.add_argument("--hyp-out", default=None, help="override manifests.hyp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--window-s", type=float, default=900.0)
    ap.add_argument("--overlap-s", type=float, default=45.0)
    ap.add_argument("--hotwords-file", default=None, help="one hotword/name per line")
    ap.add_argument("--gates-only", action="store_true",
                    help="skip transcription; hyp manifest must already exist")
    ap.add_argument("--consolidate", action="store_true",
                    help="engage the persistent speaker registry + end-of-meeting "
                         "consolidation for stable global speaker identity")
    ap.add_argument("--registry-state", default=None,
                    help="registry state path (base name; .json + .npz sidecar) to "
                         "persist/resume global speaker identities across runs")
    args = ap.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    man = cfg.get("manifests") or {}
    ref_path = ROOT / (args.ref or man.get("ref", "data/manifests/eval_gold.jsonl"))
    hyp_path = ROOT / (args.hyp_out or man.get("hyp", "runs/eval/hyp.jsonl"))
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    hotwords = None
    if args.hotwords_file:
        hotwords = [w for w in Path(args.hotwords_file).read_text().splitlines() if w.strip()]

    refs = read_manifest(ref_path)
    if args.gates_only:
        hyps = read_manifest(hyp_path)
    else:
        hyps = transcribe_all(args.model, refs, hyp_path, args.device, args.dtype,
                              args.window_s, args.overlap_s, hotwords,
                              consolidate=args.consolidate,
                              registry_state=args.registry_state)

    reports: dict[str, GateReport] = {
        "overall": run_gates(str(ref_path), str(hyp_path), cfg.get("thresholds") or {})}
    hyp_by_path = {h.audio_path: h for h in hyps}
    # Global speaker-identity consistency (not gated): the registry/consolidation
    # design is meant to keep one global id per speaker over the whole meeting.
    sc = speaker_consistency_over_manifest(refs, hyp_by_path)
    if sc is not None:
        reports["overall"].metrics["speaker_consistency"] = sc
    slice_dir = hyp_path.parent / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in (cfg.get("slices") or {}).items():
        rep = slice_report(name, spec, refs, hyp_by_path, slice_dir)
        if rep is not None:
            reports[name] = rep

    for name, rep in reports.items():
        print_report(name, rep)
    failed = [n for n, r in reports.items() if not r.passed]
    if failed:
        print(f"\nGATES FAILED: {failed}", file=sys.stderr)
        return 1
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
