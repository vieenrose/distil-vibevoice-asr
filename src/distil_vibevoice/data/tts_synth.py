"""Synthesize meeting audio from dialogue scripts with VibeVoice TTS.

Wraps the community-maintained VibeVoice inference package
(github.com/vibevoice-community/VibeVoice, ``pip install -e .``) for
voice-cloned multi-speaker synthesis. Each script speaker is bound to a
reference wav sampled deterministically (by script hash) from a speaker
bank; turns are synthesized one by one so cumulative timestamps are exact.

This module is importable without torch/vibevoice installed — heavy imports
happen inside methods.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from distil_vibevoice.data.dialogue_scripts import DialogueScript
from distil_vibevoice.data.manifest import MeetingRecord, Segment

__all__ = ["MeetingSynthesizer"]

_TTS_SAMPLE_RATE = 24000
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def _script_hash(script: DialogueScript) -> str:
    """Stable content hash of a script (drives all per-script randomness)."""
    h = hashlib.sha256()
    h.update(script.domain.encode("utf-8"))
    h.update(script.language.encode("utf-8"))
    for s in script.speakers:
        h.update(s.encode("utf-8"))
    for t in script.turns:
        h.update(t.speaker.encode("utf-8"))
        h.update(t.text.encode("utf-8"))
    return h.hexdigest()


def _write_wav(path: str, wav: "Any", sr: int) -> None:
    """Write float waveform to disk; soundfile preferred, scipy fallback."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(path, x, sr)
    except ImportError:
        from scipy.io import wavfile

        wavfile.write(path, sr, (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16))


class MeetingSynthesizer:
    """Voice-cloning meeting synthesizer over VibeVoice TTS.

    Args:
        tts_model_path: HF id or local path of the VibeVoice TTS model
            (e.g. ``aoi-ot/VibeVoice-Large``).
        speaker_bank_dir: directory of reference speaker wavs (searched
            recursively); one voice is assigned per script speaker.
        device: torch device for the TTS model.
    """

    def __init__(self, tts_model_path: str, speaker_bank_dir: str, device: str = "cuda:0"):
        self.tts_model_path = tts_model_path
        self.speaker_bank_dir = speaker_bank_dir
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the VibeVoice TTS model/processor on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "MeetingSynthesizer requires torch and the VibeVoice community "
                "package. Install with: git clone "
                "https://github.com/vibevoice-community/VibeVoice && "
                "pip install -e VibeVoice"
            ) from e

        self._processor = VibeVoiceProcessor.from_pretrained(self.tts_model_path)
        # TODO-verify: kwarg names (torch_dtype vs dtype, attn_implementation)
        # against the installed vibevoice-community package version.
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.tts_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.tts_model_path, torch_dtype=torch.bfloat16
            )
        self._model = model.to(self.device).eval()
        if hasattr(self._model, "set_ddpm_inference_steps"):
            self._model.set_ddpm_inference_steps(num_steps=10)

    # ------------------------------------------------------------------
    # Voice bank
    # ------------------------------------------------------------------

    def _bank_voices(self) -> list[Path]:
        """Sorted reference wavs in the speaker bank (sorted => stable draws)."""
        root = Path(self.speaker_bank_dir)
        voices = sorted(p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_EXTS)
        if not voices:
            raise FileNotFoundError(f"no reference audio found under {root}")
        return voices

    def assign_voices(self, script: DialogueScript) -> dict[str, str]:
        """Deterministically map each script speaker to a bank reference wav.

        Uses the script content hash as the RNG seed, so the same script
        always gets the same voices; distinct speakers get distinct voices
        whenever the bank is large enough.
        """
        voices = self._bank_voices()
        rng = random.Random(int(_script_hash(script)[:16], 16))
        pool = list(voices)
        rng.shuffle(pool)
        mapping: dict[str, str] = {}
        for i, speaker in enumerate(script.speakers):
            mapping[speaker] = str(pool[i % len(pool)])
        return mapping

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synth_turn(self, text: str, voice_path: str) -> "Any":
        """Synthesize one single-speaker turn; returns float32 numpy at 24 kHz.

        TODO-verify with real weights: exact ``generate`` kwargs and the
        ``speech_outputs`` field of the returned object, per
        vibevoice-community demo/inference_from_file.py.
        """
        import numpy as np
        import torch

        formatted = f"Speaker 1: {text}"
        inputs = self._processor(
            text=[formatted],
            voice_samples=[[voice_path]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.3,
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )
        speech = outputs.speech_outputs[0]
        wav = speech.detach().to(torch.float32).cpu().numpy().reshape(-1)
        return np.asarray(wav, dtype=np.float32)

    def synth(self, script: DialogueScript, out_wav: str) -> MeetingRecord:
        """Synthesize a full meeting from ``script`` into ``out_wav``.

        Turns are generated sequentially with short deterministic pauses;
        cumulative sample counts yield exact per-turn Segment timestamps.
        """
        import numpy as np

        self._ensure_model()
        voice_map = self.assign_voices(script)
        sh = _script_hash(script)
        pause_rng = random.Random(int(sh[16:32], 16))

        sr = _TTS_SAMPLE_RATE
        pieces: list[np.ndarray] = []
        segments: list[Segment] = []
        cursor = 0  # samples
        for turn in script.turns:
            wav = self._synth_turn(turn.text, voice_map.get(turn.speaker, next(iter(voice_map.values()))))
            start = cursor
            end = cursor + wav.shape[0]
            segments.append(
                Segment(start=start / sr, end=end / sr, speaker=turn.speaker, text=turn.text)
            )
            pieces.append(wav)
            pause = int(round(pause_rng.uniform(0.25, 0.9) * sr))
            pieces.append(np.zeros(pause, dtype=np.float32))
            cursor = end + pause

        mix = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 1.0:
            mix = mix / peak * 0.95
        _write_wav(out_wav, mix, sr)

        return MeetingRecord(
            audio_path=str(out_wav),
            duration_s=mix.shape[0] / sr,
            sample_rate=sr,
            language=script.language,
            source="tts_synth",
            split="train",
            segments=segments,
            meta={
                "domain": script.domain,
                "script_hash": sh,
                "voice_map": voice_map,
                "tts_model": self.tts_model_path,
            },
        )
