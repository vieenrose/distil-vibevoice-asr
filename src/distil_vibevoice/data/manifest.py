"""Manifest dataclasses and JSONL IO for meeting-transcription data.

Also implements the (de)serialization of the VibeVoice-ASR structured output
format: a JSON-array string of segments with keys ``Start``/``End`` (seconds),
``Speaker`` (integer id) and ``Content`` (text). The original (non-HF) package
prompt uses the alternate key spellings ``Start time`` / ``End time`` /
``Speaker ID`` / ``Content``; both spellings are accepted by
:func:`parse_teacher_output`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "Segment",
    "MeetingRecord",
    "read_manifest",
    "write_manifest",
    "iter_manifest",
    "parse_teacher_output",
    "format_target",
]


@dataclass
class Segment:
    """A single speaker turn with absolute timestamps in seconds."""

    start: float
    end: float
    speaker: str
    text: str


@dataclass
class MeetingRecord:
    """One meeting-audio example plus its segment-level annotation."""

    audio_path: str
    duration_s: float
    sample_rate: int
    language: str
    source: str
    split: str
    segments: list[Segment]
    meta: dict = field(default_factory=dict)


def _record_from_dict(d: dict[str, Any]) -> MeetingRecord:
    """Build a MeetingRecord from a plain dict (as read from JSONL)."""
    segments = [
        Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker=str(s["speaker"]),
            text=str(s["text"]),
        )
        for s in d.get("segments", [])
    ]
    return MeetingRecord(
        audio_path=str(d["audio_path"]),
        duration_s=float(d["duration_s"]),
        sample_rate=int(d["sample_rate"]),
        language=str(d["language"]),
        source=str(d["source"]),
        split=str(d["split"]),
        segments=segments,
        meta=dict(d.get("meta", {})),
    )


def read_manifest(path: str | Path) -> list[MeetingRecord]:
    """Read a JSONL manifest into a list of MeetingRecords.

    Malformed lines are skipped with a logged warning.
    """
    return list(iter_manifest(path))


def iter_manifest(path: str | Path) -> Iterator[MeetingRecord]:
    """Lazily iterate MeetingRecords from a JSONL manifest.

    Malformed lines (bad JSON or missing required keys) are skipped with a
    logged warning instead of raising.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield _record_from_dict(d)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning("Skipping malformed manifest line %s:%d: %s", path, lineno, e)


def write_manifest(records: Iterable[MeetingRecord], path: str | Path) -> None:
    """Write MeetingRecords to a JSONL manifest (UTF-8, one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# VibeVoice-ASR structured-output (de)serialization
# ---------------------------------------------------------------------------

# Both key spellings used by the teacher: HF variant emits "Start"/"End"/
# "Speaker"/"Content"; the original package prompt asks for "Start time"/
# "End time"/"Speaker ID"/"Content". snake_case accepted for robustness.
_START_KEYS = ("Start", "Start time", "start", "start_time")
_END_KEYS = ("End", "End time", "end", "end_time")
_SPEAKER_KEYS = ("Speaker", "Speaker ID", "speaker", "speaker_id")
_CONTENT_KEYS = ("Content", "content", "Text", "text")

_CHAT_WRAPPER_RE = re.compile(r"<\|im_start\|>\s*assistant\s*|<\|im_end\|>|<\|endoftext\|>")


def _first_key(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} present in segment dict")


def _first_key_default(d: dict, keys: tuple[str, ...], default: Any) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _segment_from_dict(d: dict) -> Segment:
    # Speaker is optional: the real teacher emits non-speech marker segments
    # (e.g. {"Start":..,"End":..,"Content":"[Silence]"} or "[Music]") that carry
    # no Speaker field. These are kept with an empty speaker label to match the
    # reference processor's post_process_transcription (which retains them).
    return Segment(
        start=float(_first_key(d, _START_KEYS)),
        end=float(_first_key(d, _END_KEYS)),
        speaker=str(_first_key_default(d, _SPEAKER_KEYS, "")),
        text=str(_first_key(d, _CONTENT_KEYS)),
    )


def _scan_json_objects(text: str) -> list[dict]:
    """Recover top-level JSON objects from possibly truncated/dirty text."""
    decoder = json.JSONDecoder()
    objs: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("{", i)
        if j == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        i = end
    return objs


def parse_teacher_output(text: str) -> list[Segment]:
    """Parse a VibeVoice-ASR generation into Segments.

    Accepts the raw decoded string (chat-template wrappers such as
    ``<|im_start|>assistant`` / ``<|im_end|>`` are stripped), a bare JSON
    array, or dirty/truncated output — in the latter case individual JSON
    objects are recovered one by one. Malformed segment dicts are skipped
    with a logged warning.
    """
    cleaned = _CHAT_WRAPPER_RE.sub("", text).strip()
    raw_items: list[Any]
    try:
        parsed = json.loads(cleaned)
        raw_items = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # Try the outermost [...] slice, then fall back to per-object scan.
        lo, hi = cleaned.find("["), cleaned.rfind("]")
        raw_items = []
        if lo != -1 and hi > lo:
            try:
                parsed = json.loads(cleaned[lo : hi + 1])
                if isinstance(parsed, list):
                    raw_items = parsed
            except json.JSONDecodeError:
                raw_items = []
        if not raw_items:
            raw_items = list(_scan_json_objects(cleaned))
            if not raw_items and cleaned:
                logger.warning("parse_teacher_output: no JSON segments found in output")

    segments: list[Segment] = []
    for item in raw_items:
        if not isinstance(item, dict):
            logger.warning("parse_teacher_output: skipping non-dict item %r", item)
            continue
        try:
            segments.append(_segment_from_dict(item))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("parse_teacher_output: skipping malformed segment %r (%s)", item, e)
    return segments


def format_target(segments: list[Segment]) -> str:
    """Render Segments as a training-target string in the teacher's format.

    Inverse of :func:`parse_teacher_output`: emits a compact JSON array with
    keys ``Start``/``End``/``Speaker``/``Content`` matching the HF-variant raw
    output. Purely-numeric speaker labels are emitted as integers (matching
    the teacher's integer speaker ids); other labels are kept as strings.
    """
    items = []
    for seg in segments:
        s = str(seg.speaker).strip()
        item: dict[str, Any] = {"Start": seg.start, "End": seg.end}
        # An empty speaker label denotes a non-speech marker segment
        # ([Silence]/[Music]); the teacher emits no Speaker key for these, so
        # omit it here to keep parse_teacher_output an exact inverse.
        if s:
            item["Speaker"] = int(s) if re.fullmatch(r"-?\d+", s) else seg.speaker
        item["Content"] = seg.text
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
