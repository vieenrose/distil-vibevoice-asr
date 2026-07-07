"""Evaluation metrics for zh-TW/en code-switched meeting transcription.

Pure python/numpy (scipy only for the Hungarian assignment).  This is the
release-gate code: metrics are duration-weighted and exercised by unit tests.
"""
from distil_vibevoice.eval.consistency import speaker_consistency
from distil_vibevoice.eval.cpwer import cpwer, optimal_speaker_map
from distil_vibevoice.eval.der import der
from distil_vibevoice.eval.gates import GateReport, run_gates
from distil_vibevoice.eval.mer import levenshtein, mer, tokenize_mixed
from distil_vibevoice.eval.timestamps import timestamp_mae, timestamp_report

__all__ = [
    "mer",
    "tokenize_mixed",
    "levenshtein",
    "cpwer",
    "optimal_speaker_map",
    "der",
    "speaker_consistency",
    "timestamp_mae",
    "timestamp_report",
    "GateReport",
    "run_gates",
]
