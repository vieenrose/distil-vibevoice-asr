"""Simulate multi-speaker meeting mixtures with exact overlap labels.

Utterances are placed on a timeline: sequential with random silence gaps,
and with probability ``overlap_ratio`` the next utterance starts before the
previous one ends (overlap window proportional to ``overlap_ratio``).
"""

from __future__ import annotations

import numpy as np

from distil_vibevoice.data.manifest import Segment

__all__ = ["simulate_meeting"]


def simulate_meeting(
    utterances: list[tuple["np.ndarray", str, str]],
    sr: int,
    overlap_ratio: float = 0.1,
    silence_range: tuple = (0.2, 1.5),
    rng=None,
) -> tuple["np.ndarray", list[Segment]]:
    """Mix single-speaker utterances into one meeting waveform.

    Args:
        utterances: list of (mono float wav, speaker_id, text) tuples.
        sr: sample rate shared by all utterances and the output.
        overlap_ratio: probability that an utterance overlaps the previous
            one; also scales the maximum overlap duration. 0 disables
            overlap entirely.
        silence_range: (lo, hi) seconds of uniform-random silence between
            non-overlapping utterances.
        rng: numpy Generator for determinism; a fresh default_rng() if None.

    Returns:
        (wav, segments): peak-normalized float32 mixture and one Segment per
        utterance with exact start/end times, sorted by start time.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError("overlap_ratio must be in [0, 1]")
    if not utterances:
        return np.zeros(0, dtype=np.float32), []

    placements: list[tuple[int, np.ndarray, str, str]] = []  # (start_sample, wav, spk, text)
    cursor = 0  # end of the previous utterance, in samples
    prev_len = 0
    for i, (wav, speaker, text) in enumerate(utterances):
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"utterance {i}: expected mono shape (T,), got {x.shape}")
        if i == 0:
            start = 0
        elif overlap_ratio > 0 and rng.random() < overlap_ratio:
            # Start before the previous utterance ends; the overlap window
            # is proportional to overlap_ratio and bounded by both durations.
            max_overlap = int(overlap_ratio * min(prev_len, x.shape[0]))
            overlap = int(rng.integers(1, max_overlap + 1)) if max_overlap >= 1 else 0
            start = max(0, cursor - overlap)
        else:
            gap = float(rng.uniform(silence_range[0], silence_range[1]))
            start = cursor + int(round(gap * sr))
        placements.append((start, x, speaker, text))
        cursor = max(cursor, start + x.shape[0])
        prev_len = x.shape[0]

    total = max(start + x.shape[0] for start, x, _, _ in placements)
    mix = np.zeros(total, dtype=np.float32)
    segments: list[Segment] = []
    for start, x, speaker, text in placements:
        mix[start : start + x.shape[0]] += x
        segments.append(
            Segment(
                start=start / sr,
                end=(start + x.shape[0]) / sr,
                speaker=speaker,
                text=text,
            )
        )

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0:
        mix = (mix / peak) * 0.95
    segments.sort(key=lambda s: (s.start, s.end))
    return mix.astype(np.float32), segments
