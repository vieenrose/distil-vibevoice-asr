"""Release gates: aggregate metrics over manifests and check thresholds.

``run_gates`` pairs reference and hypothesis :class:`MeetingRecord`s by
``audio_path``, computes per-record MER (concatenated text), cpWER, DER and
timestamp MAE, aggregates duration-weighted means, and compares them against
a thresholds dict.  Slice thresholds of the form ``"<slice>_<metric>"`` (e.g.
``"codeswitch_mer"``) are evaluated over records whose ``meta['slice']``
equals ``<slice>``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from distil_vibevoice.eval.cpwer import cpwer
from distil_vibevoice.eval.der import der
from distil_vibevoice.eval.mer import mer
from distil_vibevoice.eval.timestamps import timestamp_mae

if TYPE_CHECKING:  # pragma: no cover - typing only
    from distil_vibevoice.data.manifest import MeetingRecord, Segment

logger = logging.getLogger(__name__)

__all__ = ["GateReport", "run_gates"]

_BASE_METRICS = ("mer", "cpwer", "der", "timestamp_mae")


@dataclass
class GateReport:
    """Outcome of a gate run: overall pass flag, metrics and failure strings."""

    passed: bool
    metrics: dict
    failures: list[str] = field(default_factory=list)


def _concat_text(segments: list["Segment"]) -> str:
    return " ".join(
        seg.text for seg in sorted(segments, key=lambda s: (s.start, s.end))
    )


def _record_metrics(ref: "MeetingRecord", hyp_segments: list["Segment"]) -> dict:
    """MER/cpWER/DER/timestamp-MAE for one reference record vs hyp segments."""
    return {
        "mer": mer(_concat_text(ref.segments), _concat_text(hyp_segments)),
        "cpwer": cpwer(ref.segments, hyp_segments),
        "der": der(ref.segments, hyp_segments),
        "timestamp_mae": timestamp_mae(ref.segments, hyp_segments),
    }


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0.0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _aggregate(per_record: list[tuple["MeetingRecord", dict]]) -> dict:
    """Duration-weighted mean of each base metric over (record, metrics) pairs."""
    weights = [max(rec.duration_s, 0.0) or 1.0 for rec, _ in per_record]
    return {
        m: _weighted_mean([met[m] for _, met in per_record], weights)
        for m in _BASE_METRICS
    }


def _slice_key(key: str) -> tuple[str, str] | None:
    """Split a threshold key like ``codeswitch_mer`` into (slice, metric)."""
    for m in sorted(_BASE_METRICS, key=len, reverse=True):
        suffix = "_" + m
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], m
    return None


def run_gates(ref_manifest: str, hyp_manifest: str, thresholds: dict) -> GateReport:
    """Evaluate hyp manifest against ref manifest and check thresholds.

    Records are paired by ``audio_path``; reference records with no hypothesis
    counterpart are scored against an empty hypothesis (worst case).  Metrics
    are duration-weighted means of per-record values.  ``thresholds`` maps
    metric names (``mer``, ``cpwer``, ``der``, ``timestamp_mae``, or slice
    variants like ``codeswitch_mer``) to maximum allowed values; every
    violation appends a failure string like ``'der 0.15 > 0.12'``.
    """
    from distil_vibevoice.data.manifest import read_manifest

    ref_records = read_manifest(ref_manifest)
    hyp_by_path = {rec.audio_path: rec for rec in read_manifest(hyp_manifest)}
    if not ref_records:
        return GateReport(
            passed=False, metrics={}, failures=["no records in reference manifest"]
        )

    per_record: list[tuple["MeetingRecord", dict]] = []
    n_missing = 0
    for ref in ref_records:
        hyp = hyp_by_path.get(ref.audio_path)
        if hyp is None:
            n_missing += 1
            logger.warning("run_gates: no hypothesis for %s", ref.audio_path)
        hyp_segments = hyp.segments if hyp is not None else []
        per_record.append((ref, _record_metrics(ref, hyp_segments)))

    metrics = _aggregate(per_record)
    metrics["n_records"] = len(ref_records)
    metrics["n_missing_hyp"] = n_missing

    failures: list[str] = []
    for key, limit in thresholds.items():
        if key in _BASE_METRICS:
            value = metrics[key]
        else:
            parsed = _slice_key(key)
            if parsed is None:
                logger.warning("run_gates: unknown threshold key %r ignored", key)
                continue
            slice_name, metric = parsed
            sliced = [
                (rec, met)
                for rec, met in per_record
                if rec.meta.get("slice") == slice_name
            ]
            if not sliced:
                logger.warning(
                    "run_gates: no records with meta['slice']==%r; skipping %s",
                    slice_name,
                    key,
                )
                continue
            value = _aggregate(sliced)[metric]
            metrics[key] = value
        if value > limit:
            failures.append(f"{key} {value:.4g} > {limit:.4g}")

    return GateReport(passed=not failures, metrics=metrics, failures=failures)
