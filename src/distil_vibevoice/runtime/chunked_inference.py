"""Chunked long-form transcription with speaker-roster carry-over.

Splits a long recording into overlapping windows, labels each window with any
duck-typed labeler exposing ``label_file(path, hotwords=...)`` (e.g.
``distil_vibevoice.data.pseudo_label.TeacherLabeler``), carries the
accumulated speaker roster + last turns into the next window as context, and
stitches the per-window segments into one globally-labeled
:class:`~distil_vibevoice.data.manifest.MeetingRecord` via
:mod:`distil_vibevoice.runtime.speaker_stitch`.

Context injection is duck-typed: if the labeler's ``label_file`` accepts an
explicit ``context`` keyword it is used (mirrors VibeVoice-ASR's
``context_info`` / ``prompt`` mechanism); otherwise the context lines are
prepended to the hotword list.
"""
from __future__ import annotations

import inspect
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from distil_vibevoice.data.manifest import MeetingRecord, Segment
from distil_vibevoice.runtime.speaker_stitch import stitch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from distil_vibevoice.runtime.embeddings import BaseEmbedder
    from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

logger = logging.getLogger(__name__)

__all__ = ["ChunkedTranscriber"]

#: Number of trailing turns from the previous windows injected as context.
_CONTEXT_TAIL_TURNS = 2
#: Max characters of a speaker's first utterance kept as a roster snippet.
_ROSTER_SNIPPET_CHARS = 60
#: Max seconds of a speaker's audio pooled per window into its voice embedding.
_EMBED_POOL_S = 10.0


def _require_soundfile() -> Any:
    try:
        import soundfile
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "soundfile is required for chunked inference: pip install soundfile"
        ) from e
    return soundfile


def _accepts_context_kwarg(labeler: Any) -> bool:
    """True if ``labeler.label_file`` declares an explicit ``context`` parameter."""
    fn = getattr(labeler, "label_file", None)
    if fn is None:
        raise TypeError(f"labeler {labeler!r} has no label_file method")
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "context" in sig.parameters


class ChunkedTranscriber:
    """Transcribe arbitrarily long audio in overlapping windows.

    Parameters
    ----------
    labeler:
        Any object with ``label_file(audio_path, hotwords=...) -> MeetingRecord``
        returning window-local segment times.
    window_s / overlap_s:
        Window length and inter-window overlap in seconds (overlap < window).
    max_roster:
        Maximum number of speakers carried in the context roster.
    embedder / registry:
        Optional voice-embedding backend and persistent
        :class:`~distil_vibevoice.runtime.speaker_registry.SpeakerRegistry`.
        When either is supplied the transcriber anchors every window-local
        speaker to a *global* identity (star-topology matching) instead of
        relying solely on pairwise stitch continuity: a speaker who is silent
        during an overlap is re-identified by voice, so one bad boundary no
        longer poisons the rest of the meeting.  Omitting both keeps the legacy
        behavior byte-for-byte.  A missing counterpart is auto-created (an
        ``mfcc`` embedder / an empty registry sized to the embedder).
    consolidate_on_finish:
        Run the end-of-meeting agglomerative consolidation pass
        (:func:`~distil_vibevoice.runtime.consolidate.consolidate`) to
        retroactively merge global ids that turned out to share a voice.  Only
        meaningful when a registry is active.
    registry_state:
        Path (base name; a ``.json`` + ``.npz`` sidecar pair) used to persist
        the registry.  If the file already exists and no explicit ``registry``
        is given it is loaded to *resume* a paused meeting; on finish the
        registry is saved back to it.
    """

    def __init__(
        self,
        labeler: Any,
        window_s: float = 900.0,
        overlap_s: float = 45.0,
        max_roster: int = 12,
        embedder: "BaseEmbedder | None" = None,
        registry: "SpeakerRegistry | None" = None,
        consolidate_on_finish: bool = True,
        registry_state: "str | None" = None,
        consolidate_mode: str = "merge",
        recluster_threshold: float = 0.7,
        normalize_zhtw: bool = False,
    ) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if not 0 <= overlap_s < window_s:
            raise ValueError(f"need 0 <= overlap_s < window_s, got {overlap_s} vs {window_s}")
        if max_roster < 1:
            raise ValueError(f"max_roster must be >= 1, got {max_roster}")
        self.labeler = labeler
        self.window_s = float(window_s)
        self.overlap_s = float(overlap_s)
        self.max_roster = int(max_roster)
        self._accepts_context = _accepts_context_kwarg(labeler)

        self.normalize_zhtw = bool(normalize_zhtw)
        self.consolidate_on_finish = bool(consolidate_on_finish)
        self.consolidate_mode = str(consolidate_mode)
        self.recluster_threshold = float(recluster_threshold)
        self.registry_state = str(registry_state) if registry_state else None
        self._embedder = embedder
        self._registry = registry
        use_registry = (
            embedder is not None or registry is not None or self.registry_state is not None
        )
        if use_registry:
            if self._embedder is None:
                from distil_vibevoice.runtime.embeddings import load_embedder

                self._embedder = load_embedder("mfcc")
            if self._registry is None and self.registry_state:
                if Path(self.registry_state).with_suffix(".json").exists():
                    from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

                    self._registry = SpeakerRegistry.load(self.registry_state)
            if self._registry is None:
                from distil_vibevoice.runtime.speaker_registry import SpeakerRegistry

                self._registry = SpeakerRegistry(embed_dim=int(self._embedder.dim))

    # ------------------------------------------------------------------ #

    def _window_starts(self, duration_s: float) -> list[float]:
        """Window start times covering [0, duration_s] with the configured overlap."""
        if duration_s <= 0:
            return []
        starts = [0.0]
        step = self.window_s - self.overlap_s
        while starts[-1] + self.window_s < duration_s - 1e-6:
            starts.append(starts[-1] + step)
        return starts

    def _context_lines(
        self, roster: dict[str, str], stitched: list[Segment]
    ) -> list[str]:
        """Roster snippets + last turns of the transcript so far, one per line."""
        lines = [f"{spk}: {snippet}" for spk, snippet in roster.items()]
        for seg in stitched[-_CONTEXT_TAIL_TURNS:]:
            lines.append(f"[{seg.start:.1f}s] {seg.speaker}: {seg.text}")
        return lines

    def _update_roster(self, stitched: list[Segment]) -> dict[str, str]:
        """First-utterance snippet per global speaker, capped at max_roster."""
        roster: dict[str, str] = {}
        for seg in stitched:
            if seg.speaker in roster:
                continue
            if len(roster) >= self.max_roster:
                break
            roster[seg.speaker] = seg.text[:_ROSTER_SNIPPET_CHARS]
        return roster

    def _label_window(
        self,
        wav_path: str,
        hotwords: list[str] | None,
        context_lines: list[str],
    ) -> MeetingRecord:
        """Call the labeler, injecting context via kwarg or hotword prefix."""
        if self._accepts_context:
            context = "\n".join(context_lines) if context_lines else None
            return self.labeler.label_file(wav_path, hotwords=hotwords, context=context)
        merged = list(context_lines) + list(hotwords or [])
        return self.labeler.label_file(wav_path, hotwords=merged or None)

    # ---------------------- registry anchoring ------------------------- #

    def _registry_context_lines(self) -> list[str]:
        """Roster prompt drawn from the persistent registry, one line per speaker."""
        assert self._registry is not None
        prompt = self._registry.roster_prompt(self.max_roster)
        return prompt.splitlines() if prompt else []

    def _registry_roster(self) -> dict[str, str]:
        """`{global_id: latest snippet}` for the most-recently-active speakers."""
        assert self._registry is not None
        roster: dict[str, str] = {}
        for line in self._registry.roster_prompt(self.max_roster).splitlines():
            gid, _, snippet = line.partition(": ")
            roster[gid] = snippet
        return roster

    def _pool_embed(self, data: "np.ndarray", sr: int, clips: list[slice]) -> "np.ndarray":
        """Embed up to ``_EMBED_POOL_S`` seconds of a speaker's window audio."""
        import numpy as np

        assert self._embedder is not None
        budget = int(round(_EMBED_POOL_S * sr))
        parts: list[np.ndarray] = []
        total = 0
        for sl in clips:
            parts.append(data[sl])
            total += sl.stop - sl.start
            if total >= budget:
                break
        if not parts:
            return np.zeros(int(self._embedder.dim), dtype=np.float32)
        audio = np.concatenate(parts)[:budget]
        return self._embedder.embed(audio, sr)

    def _anchor_window(
        self,
        data: "np.ndarray",
        sr: int,
        t0: float,
        stitched: list[Segment],
        stitch_to_global: dict[str, str],
    ) -> list[Segment]:
        """Anchor this window's stitch labels to persistent global identities.

        For every stitch-global label that first speaks *inside this window*
        (``start >= t0``) the speaker's window audio is pooled and embedded.  A
        label already anchored keeps its global id (stitch's text continuity is
        the primary boundary signal) and refreshes the centroid; a freshly
        appearing label is anchored by voice via :meth:`SpeakerRegistry.match`
        (catching a speaker who was silent through the overlap and got a new
        stitch label) or, failing that, allocated as a new global speaker.

        Returns the full ``stitched`` transcript relabeled to global ids.
        """
        reg = self._registry
        assert reg is not None
        window_end = t0 + self.window_s

        order: list[str] = []
        pooled: dict[str, dict[str, Any]] = {}
        for seg in sorted(stitched, key=lambda s: (s.start, s.end)):
            if seg.start < t0 - 1e-6 or seg.start >= window_end:
                continue  # prev-window overlap leftover, or beyond this window
            lo = max(seg.start, t0)
            hi = min(seg.end, window_end)
            f0 = max(int(round((lo - t0) * sr)), 0)
            f1 = min(int(round((hi - t0) * sr)), len(data))
            if f1 <= f0:
                continue
            info = pooled.get(seg.speaker)
            if info is None:
                info = {"clips": [], "start": seg.start, "end": seg.end, "snippet": seg.text}
                pooled[seg.speaker] = info
                order.append(seg.speaker)
            info["clips"].append(slice(f0, f1))
            info["start"] = min(info["start"], seg.start)
            info["end"] = max(info["end"], seg.end)

        for local in order:
            info = pooled[local]
            emb = self._pool_embed(data, sr, info["clips"])
            snippet = info["snippet"][:_ROSTER_SNIPPET_CHARS]
            t_end = float(info["end"])
            gid = stitch_to_global.get(local)
            if gid is not None:
                reg.update(gid, emb, snippet, t_end)
            else:
                gid, _score = reg.match(emb)
                if gid is None:
                    gid = reg.new_speaker(emb, snippet, t_end)
                else:
                    reg.update(gid, emb, snippet, t_end)
                stitch_to_global[local] = gid
            reg.add_segment(gid, float(info["start"]), t_end, emb)

        return [
            replace(seg, speaker=stitch_to_global.get(seg.speaker, seg.speaker))
            for seg in stitched
        ]

    # ------------------------------------------------------------------ #

    def transcribe(self, audio_path: str, hotwords: list[str] | None = None) -> MeetingRecord:
        """Transcribe ``audio_path`` into a single globally-stitched MeetingRecord."""
        sf = _require_soundfile()
        info = sf.info(str(audio_path))
        sr = int(info.samplerate)
        total_frames = int(info.frames)
        duration_s = total_frames / sr if sr > 0 else 0.0

        starts = self._window_starts(duration_s)
        chunks: list[list[Segment]] = []
        offsets: list[float] = []
        stitched: list[Segment] = []
        roster: dict[str, str] = {}
        language = "zh-TW-en"
        reg = self._registry
        stitch_to_global: dict[str, str] = {}
        anchored: list[Segment] = []

        with tempfile.TemporaryDirectory(prefix="chunked_asr_") as tmpdir:
            for i, t0 in enumerate(starts):
                f0 = int(round(t0 * sr))
                f1 = min(int(round((t0 + self.window_s) * sr)), total_frames)
                data, _ = sf.read(str(audio_path), start=f0, stop=f1, dtype="float32")
                wav_path = str(Path(tmpdir) / f"window_{i:04d}.wav")
                sf.write(wav_path, data, sr)

                if reg is not None:
                    context_lines = self._registry_context_lines()
                else:
                    context_lines = self._context_lines(roster, stitched)
                rec = self._label_window(wav_path, hotwords, context_lines)
                if i == 0 and rec.language:
                    language = rec.language

                chunks.append(
                    [
                        Segment(seg.start + t0, seg.end + t0, seg.speaker, seg.text)
                        for seg in rec.segments
                    ]
                )
                offsets.append(t0)
                stitched = stitch(chunks, offsets, self.overlap_s)
                if reg is not None:
                    anchored = self._anchor_window(data, sr, t0, stitched, stitch_to_global)
                    roster = self._registry_roster()
                else:
                    roster = self._update_roster(stitched)
                logger.debug(
                    "window %d/%d [%.1fs, %.1fs): %d segments, roster size %d",
                    i + 1, len(starts), t0, t0 + self.window_s, len(chunks[-1]), len(roster),
                )

        meta: dict[str, Any] = {
            "num_windows": len(starts),
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "roster": dict(roster),
        }
        segments = stitched
        if reg is not None:
            segments = anchored
            mapping: dict[str, str] = {}
            if self.consolidate_on_finish:
                from distil_vibevoice.runtime.consolidate import consolidate

                segments, mapping = consolidate(
                    reg, segments,
                    mode=self.consolidate_mode,
                    recluster_threshold=self.recluster_threshold,
                )
            if self.registry_state:
                reg.save(self.registry_state)
            meta["registry_speakers"] = len(reg.speakers)
            meta["consolidated"] = mapping

        if self.normalize_zhtw:
            from dataclasses import replace

            from distil_vibevoice.data.normalize_zhtw import to_zhtw

            segments = [replace(s, text=to_zhtw(s.text)) for s in segments]
            meta["normalized_zhtw"] = True

        return MeetingRecord(
            audio_path=str(audio_path),
            duration_s=duration_s,
            sample_rate=sr,
            language=language,
            source="chunked_inference",
            split="inference",
            segments=segments,
            meta=meta,
        )
