"""Data plumbing for distil-vibevoice: manifests, normalization, pseudo-labels.

Only lightweight symbols are re-exported here; heavier modules
(``pseudo_label``, ``dedupe``, ...) should be imported explicitly.
"""
from distil_vibevoice.data.manifest import (
    MeetingRecord,
    Segment,
    format_target,
    iter_manifest,
    parse_teacher_output,
    read_manifest,
    write_manifest,
)

__all__ = [
    "Segment",
    "MeetingRecord",
    "read_manifest",
    "write_manifest",
    "iter_manifest",
    "parse_teacher_output",
    "format_target",
]
