"""Tests for distil_vibevoice.data.dedupe (fingerprinting + eval-set filtering)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from distil_vibevoice.data.dedupe import (
    audio_fingerprint,
    build_eval_index,
    filter_against_index,
)
from distil_vibevoice.data.manifest import MeetingRecord, Segment, write_manifest

SR = 16000


def _speechy_wav(seed: int, seconds: float = 3.0, sr: int = SR) -> np.ndarray:
    """Deterministic pseudo-speech: modulated multi-tone + a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    wav = np.zeros_like(t)
    for _ in range(4):
        f0 = rng.uniform(80, 2000)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t + rng.uniform(0, 6.28))
        wav += env * np.sin(2 * np.pi * f0 * t)
    wav += 0.02 * rng.standard_normal(t.shape)
    return (0.9 * wav / np.max(np.abs(wav))).astype(np.float32)


def _record(audio_path: str) -> MeetingRecord:
    return MeetingRecord(
        audio_path=audio_path,
        duration_s=3.0,
        sample_rate=SR,
        language="zh-TW-en",
        source="test",
        split="train",
        segments=[Segment(0.0, 3.0, "0", "測試")],
    )


class TestAudioFingerprint:
    def test_deterministic_and_identity(self):
        wav = _speechy_wav(0)
        fp1 = audio_fingerprint(wav, SR)
        fp2 = audio_fingerprint(wav.copy(), SR)
        assert fp1 == fp2
        assert isinstance(fp1, str) and len(fp1) == 40  # sha1 hexdigest

    def test_different_content_differs(self):
        assert audio_fingerprint(_speechy_wav(1), SR) != audio_fingerprint(_speechy_wav(2), SR)

    def test_noise_added_differs_ok(self):
        wav = _speechy_wav(3)
        rng = np.random.default_rng(99)
        noisy = wav + 0.1 * rng.standard_normal(wav.shape).astype(np.float32)
        # Exact-match key: an audible edit is expected to change the hash.
        assert audio_fingerprint(noisy, SR) != audio_fingerprint(wav, SR)

    def test_stereo_and_short_and_silent_inputs(self):
        stereo = np.stack([_speechy_wav(4), _speechy_wav(4)], axis=1)  # (frames, ch)
        assert audio_fingerprint(stereo, SR) == audio_fingerprint(_speechy_wav(4), SR)
        short = _speechy_wav(5)[:100]  # shorter than one STFT window
        assert len(audio_fingerprint(short, SR)) == 40
        silent = np.zeros(SR, dtype=np.float32)
        assert len(audio_fingerprint(silent, SR)) == 40


class TestEvalIndexFiltering:
    def test_build_and_filter(self, tmp_path):
        sf = pytest.importorskip("soundfile")

        eval_wav = _speechy_wav(10)
        train_wav = _speechy_wav(11)
        eval_path = tmp_path / "eval.wav"
        dup_path = tmp_path / "dup.wav"  # same audio as eval, different file
        train_path = tmp_path / "train.wav"
        sf.write(str(eval_path), eval_wav, SR)
        sf.write(str(dup_path), eval_wav, SR)
        sf.write(str(train_path), train_wav, SR)

        eval_manifest = tmp_path / "eval.jsonl"
        write_manifest([_record(str(eval_path))], eval_manifest)

        index = build_eval_index([str(eval_manifest)])
        assert len(index) == 1

        records = [_record(str(dup_path)), _record(str(train_path))]
        kept = filter_against_index(records, index)
        assert [r.audio_path for r in kept] == [str(train_path)]

    def test_missing_audio_handling(self, tmp_path):
        pytest.importorskip("soundfile")

        # Manifest referencing a missing file -> skipped in the index.
        eval_manifest = tmp_path / "eval.jsonl"
        write_manifest([_record(str(tmp_path / "nope.wav"))], eval_manifest)
        assert build_eval_index([str(eval_manifest)]) == set()

        # Missing training audio -> record kept (cannot be checked).
        rec = _record(str(tmp_path / "also_missing.wav"))
        assert filter_against_index([rec], {"deadbeef"}) == [rec]

    def test_audio_root_join(self, tmp_path):
        sf = pytest.importorskip("soundfile")

        wav = _speechy_wav(12)
        (tmp_path / "audio").mkdir()
        sf.write(str(tmp_path / "audio" / "x.wav"), wav, SR)
        index = {audio_fingerprint(wav, SR)}

        rec = _record("audio/x.wav")  # relative path
        # Note: soundfile round-trip may quantize samples; recompute fp from disk.
        disk_wav, disk_sr = sf.read(str(tmp_path / "audio" / "x.wav"), dtype="float32")
        index.add(audio_fingerprint(disk_wav, disk_sr))
        assert filter_against_index([rec], index, audio_root=str(tmp_path)) == []
