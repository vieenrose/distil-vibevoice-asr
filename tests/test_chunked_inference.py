"""Tests for ChunkedTranscriber with mock labelers (CPU, no network, no models)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from distil_vibevoice.data.manifest import MeetingRecord, Segment
from distil_vibevoice.runtime.chunked_inference import ChunkedTranscriber

sf = pytest.importorskip("soundfile")

SR = 8000
DURATION_S = 26.0


def _write_wav(path: Path, seconds: float = DURATION_S, sr: int = SR) -> Path:
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    wav = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), wav, sr)
    return path


def _scripts() -> list[list[Segment]]:
    """Window-local segments for 3 windows (window_s=10, overlap_s=2, step=8).

    Window global spans: [0,10), [8,18), [16,26). Overlap texts repeat so the
    stitcher can align the permuted per-window speaker labels.
    """
    w0 = [
        Segment(0.5, 3.0, "0", "hello everyone welcome to the sync"),
        Segment(4.0, 7.0, "1", "thanks happy to join today"),
        Segment(8.4, 9.6, "0", "first item is the budget review"),
    ]
    w1 = [  # global 8-18; "a" is the same person as w0's "0"
        Segment(0.4, 1.6, "a", "first item is the budget review"),
        Segment(3.0, 6.0, "b", "the budget looks fine to me"),
        Segment(8.3, 9.7, "a", "second item is the hiring plan"),
    ]
    w2 = [  # global 16-26; "x" is the same person as w1's "a"
        Segment(0.3, 1.7, "x", "second item is the hiring plan"),
        Segment(3.0, 5.0, "x", "we plan to hire two engineers"),
        Segment(6.0, 8.0, "y", "great that works for my team"),
    ]
    return [w0, w1, w2]


class ScriptedLabeler:
    """Hotwords-only labeler: returns the scripted segments per call index."""

    def __init__(self, scripts: list[list[Segment]]) -> None:
        self.scripts = scripts
        self.calls: list[dict] = []

    def label_file(self, audio_path: str, hotwords: list[str] | None = None) -> MeetingRecord:
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate
        self.calls.append(
            {"path": str(audio_path), "hotwords": list(hotwords or []), "duration": duration}
        )
        segments = [replace(s) for s in self.scripts[len(self.calls) - 1]]
        return MeetingRecord(
            audio_path=str(audio_path),
            duration_s=duration,
            sample_rate=int(info.samplerate),
            language="zh-TW-en",
            source="mock",
            split="test",
            segments=segments,
        )


class ContextLabeler(ScriptedLabeler):
    """Labeler advertising an explicit ``context`` kwarg (duck-typed path)."""

    def label_file(
        self,
        audio_path: str,
        hotwords: list[str] | None = None,
        context: str | None = None,
    ) -> MeetingRecord:
        rec = super().label_file(audio_path, hotwords=hotwords)
        self.calls[-1]["context"] = context
        return rec


@pytest.fixture()
def wav_path(tmp_path: Path) -> Path:
    return _write_wav(tmp_path / "meeting.wav")


def _run(labeler, wav_path: Path, hotwords: list[str] | None = None) -> MeetingRecord:
    transcriber = ChunkedTranscriber(labeler, window_s=10.0, overlap_s=2.0, max_roster=12)
    return transcriber.transcribe(str(wav_path), hotwords=hotwords)


def test_three_windows_full_slices_and_record_shape(wav_path):
    labeler = ScriptedLabeler(_scripts())
    rec = _run(labeler, wav_path)

    assert isinstance(rec, MeetingRecord)
    assert rec.duration_s == pytest.approx(DURATION_S)
    assert rec.sample_rate == SR
    assert rec.audio_path == str(wav_path)
    assert rec.meta["num_windows"] == 3
    assert len(labeler.calls) == 3
    # Each temp slice is a full 10 s window: [0,10), [8,18), [16,26).
    for call in labeler.calls:
        assert call["duration"] == pytest.approx(10.0, abs=1e-3)


def test_global_timestamps_and_overlap_dedup(wav_path):
    rec = _run(ScriptedLabeler(_scripts()), wav_path)
    texts = [seg.text for seg in rec.segments]

    # 9 scripted segments, 2 overlap duplicates dropped.
    assert len(rec.segments) == 7
    assert texts.count("first item is the budget review") == 1
    assert texts.count("second item is the hiring plan") == 1

    by_text = {seg.text: seg for seg in rec.segments}
    hire = by_text["we plan to hire two engineers"]  # window 2 local 3.0-5.0
    assert hire.start == pytest.approx(16.0 + 3.0)
    assert hire.end == pytest.approx(16.0 + 5.0)
    assert by_text["first item is the budget review"].start == pytest.approx(8.4)
    starts = [seg.start for seg in rec.segments]
    assert starts == sorted(starts)


def test_roster_continuity_across_three_windows(wav_path):
    rec = _run(ScriptedLabeler(_scripts()), wav_path)
    by_text = {seg.text: seg for seg in rec.segments}

    host = by_text["hello everyone welcome to the sync"].speaker
    # Same person tracked through the two overlaps despite permuted local labels.
    assert by_text["first item is the budget review"].speaker == host
    assert by_text["second item is the hiring plan"].speaker == host
    assert by_text["we plan to hire two engineers"].speaker == host

    guest = by_text["thanks happy to join today"].speaker
    assert guest != host
    speakers = {seg.speaker for seg in rec.segments}
    assert host in speakers and guest in speakers
    assert all(spk.startswith("SPEAKER_") for spk in speakers)
    assert rec.meta["roster"][host].startswith("hello everyone")


def test_context_prepended_to_hotwords_for_plain_labeler(wav_path):
    labeler = ScriptedLabeler(_scripts())
    _run(labeler, wav_path, hotwords=["taipei"])

    # First window: no context yet, only the user hotwords.
    assert labeler.calls[0]["hotwords"] == ["taipei"]
    # Later windows: roster/context lines prepended, user hotwords preserved.
    for call in labeler.calls[1:]:
        assert call["hotwords"][-1] == "taipei"
        assert any("hello everyone" in line for line in call["hotwords"])
        assert any("SPEAKER_0" in line for line in call["hotwords"])


def test_context_kwarg_ducktyping_keeps_hotwords_clean(wav_path):
    labeler = ContextLabeler(_scripts())
    _run(labeler, wav_path, hotwords=["taipei"])

    assert labeler.calls[0]["context"] is None
    for call in labeler.calls:
        assert call["hotwords"] == ["taipei"]
    for call in labeler.calls[1:]:
        assert "SPEAKER_0" in call["context"]
        assert "hello everyone" in call["context"]


def test_short_audio_uses_single_window(wav_path):
    labeler = ScriptedLabeler([[
        Segment(0.5, 3.0, "0", "hello everyone welcome to the sync"),
    ]])
    transcriber = ChunkedTranscriber(labeler, window_s=30.0, overlap_s=5.0)
    rec = transcriber.transcribe(str(wav_path))

    assert len(labeler.calls) == 1
    assert labeler.calls[0]["hotwords"] == []
    assert rec.duration_s == pytest.approx(DURATION_S)
    assert len(rec.segments) == 1
    assert rec.segments[0].speaker == "SPEAKER_0"


def test_invalid_window_config_raises():
    labeler = ScriptedLabeler([])
    with pytest.raises(ValueError):
        ChunkedTranscriber(labeler, window_s=10.0, overlap_s=10.0)
    with pytest.raises(ValueError):
        ChunkedTranscriber(labeler, window_s=0.0)
    with pytest.raises(TypeError):
        ChunkedTranscriber(object())


def test_normalize_zhtw_flag_converts_simplified(wav_path):
    """normalize_zhtw=True applies OpenCC s2twp to final segment text; English protected."""
    import pytest as _pytest
    _pytest.importorskip("opencc")

    simp = [[Segment(0.5, 3.0, "0", "这个软件的视频质量"),
             Segment(4.0, 7.0, "1", "我们要 align 一下 timeline")]]
    labeler = ScriptedLabeler(simp)
    t = ChunkedTranscriber(labeler, window_s=30.0, overlap_s=5.0, normalize_zhtw=True)
    rec = t.transcribe(str(wav_path))
    texts = [s.text for s in rec.segments]
    assert "軟體" in texts[0] and "影片" in texts[0]  # Simplified -> Traditional/Taiwan
    assert "align" in texts[1] and "timeline" in texts[1]  # English spans untouched
    assert rec.meta.get("normalized_zhtw") is True


def test_normalize_zhtw_default_off_is_unchanged(wav_path):
    simp = [[Segment(0.5, 3.0, "0", "这个软件")]]
    rec = ChunkedTranscriber(ScriptedLabeler(simp), window_s=30.0, overlap_s=5.0).transcribe(str(wav_path))
    assert rec.segments[0].text == "这个软件"  # default: no conversion
    assert "normalized_zhtw" not in rec.meta
