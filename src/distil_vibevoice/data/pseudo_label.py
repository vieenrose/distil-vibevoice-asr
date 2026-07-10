"""Pseudo-labeling with the VibeVoice-ASR teacher model.

Wraps the original ``vibevoice`` package (path (A) of the research findings:
``git clone https://github.com/microsoft/VibeVoice && pip install -e .``).
Audio is loaded with soundfile and resampled to 24 kHz mono with
``scipy.signal.resample_poly`` before being handed to the processor.

Confidence filtering and two-pass agreement use a *local* mixed-token
(zh chars + en words) edit-distance implementation. This intentionally
duplicates ``distil_vibevoice.eval.mer.tokenize_mixed`` / cpWER logic so this
data module does not import the eval package (keeps the dependency graph
acyclic); the eval package remains the canonical implementation for reported
metrics.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

from distil_vibevoice.data.manifest import MeetingRecord, parse_teacher_output

logger = logging.getLogger(__name__)

__all__ = ["TeacherLabeler", "filter_by_confidence", "two_pass_agreement"]

TEACHER_SAMPLE_RATE = 24_000

_INSTALL_HINT = (
    "The VibeVoice package is required for teacher pseudo-labeling. Install with:\n"
    "  git clone https://github.com/microsoft/VibeVoice.git && pip install -e VibeVoice\n"
    "(flash-attn optional: pip install flash-attn --no-build-isolation)"
)


def _load_audio_24k_mono(audio_path: str) -> tuple["Any", int]:
    """Load audio, downmix to mono, resample to 24 kHz.

    Returns ``(wav_float32_24k, native_sample_rate)``.
    """
    import numpy as np

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("soundfile is required to load audio: pip install soundfile") from e

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # (frames, channels) -> mono
    if sr != TEACHER_SAMPLE_RATE:
        from fractions import Fraction

        from scipy.signal import resample_poly

        frac = Fraction(TEACHER_SAMPLE_RATE, sr).limit_denominator(10_000)
        wav = resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32)
    return wav, sr


class TeacherLabeler:
    """Runs VibeVoice-ASR inference and emits MeetingRecords with confidence."""

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16"):
        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_asr import (
                VibeVoiceASRForConditionalGeneration,
            )
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e

        self.model_path = model_path
        self.device = device
        self._torch = torch
        # The ASR repo ships no tokenizer; it is pulled from Qwen2.5-7B.
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model = self.model.to(device).eval()

    def _generate(self, wav: "Any", hotwords: list[str] | None) -> tuple[str, float | None]:
        """Run one greedy generation; return (decoded_text, mean_logprob|None)."""
        torch = self._torch
        kwargs: dict[str, Any] = dict(
            audio=[wav],
            sampling_rate=TEACHER_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        if hotwords:
            # Injected verbatim into the user turn ("with extra info: ...").
            kwargs["context_info"] = "\n".join(hotwords)
        inputs = self.processor(**kwargs)
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=32_768,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_ids = out.sequences[0, input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)

        mean_logprob: float | None = None
        scores = getattr(out, "scores", None)
        if scores:
            logps: list[float] = []
            n = min(len(scores), int(gen_ids.shape[0]))
            for step in range(n):
                tok = int(gen_ids[step])
                if tok == self.processor.pad_id:
                    continue
                step_logp = torch.log_softmax(scores[step][0].float(), dim=-1)[tok]
                logps.append(float(step_logp))
            if logps:
                mean_logprob = sum(logps) / len(logps)
        return text, mean_logprob

    def label_file(self, audio_path: str, hotwords: list[str] | None = None) -> MeetingRecord:
        """Transcribe one audio file into a pseudo-labeled MeetingRecord.

        ``rec.meta['mean_logprob']`` holds the mean generated-token logprob
        when generation scores are available.
        """
        wav, native_sr = _load_audio_24k_mono(audio_path)
        text, mean_logprob = self._generate(wav, hotwords)
        segments = parse_teacher_output(text)
        meta: dict[str, Any] = {
            "teacher_model": self.model_path,
            "raw_output": text,
        }
        if mean_logprob is not None:
            meta["mean_logprob"] = mean_logprob
        if hotwords:
            meta["hotwords"] = list(hotwords)
        return MeetingRecord(
            audio_path=audio_path,
            duration_s=len(wav) / TEACHER_SAMPLE_RATE,
            sample_rate=native_sr,
            language="zh-TW-en",
            source="teacher_pseudo_label",
            split="train",
            segments=segments,
            meta=meta,
        )

    def label_batch(
        self, paths: list[str], hotwords: list[str] | None = None
    ) -> list[MeetingRecord]:
        """Transcribe a list of audio files sequentially.

        Per-file failures are logged and skipped, so the output may be shorter
        than the input list; check ``rec.audio_path`` to align.
        """
        records: list[MeetingRecord] = []
        for path in paths:
            try:
                records.append(self.label_file(path, hotwords=hotwords))
            except Exception:
                logger.exception("TeacherLabeler failed on %s; skipping", path)
        return records


def filter_by_confidence(
    records: Iterable[MeetingRecord], min_mean_logprob: float = -0.5
) -> list[MeetingRecord]:
    """Keep records whose ``meta['mean_logprob']`` >= threshold.

    Records missing the key are dropped (their confidence cannot be
    established), with a logged warning.
    """
    kept: list[MeetingRecord] = []
    for rec in records:
        lp = rec.meta.get("mean_logprob")
        if lp is None:
            logger.warning(
                "filter_by_confidence: %s has no meta['mean_logprob']; dropping",
                rec.audio_path,
            )
            continue
        if float(lp) >= min_mean_logprob:
            kept.append(rec)
    return kept


# ---------------------------------------------------------------------------
# Local mixed tokenization + edit distance (documented duplication of eval.*)
# ---------------------------------------------------------------------------

_CJK_RANGES = "㐀-䶿一-鿿〇\uf900-\ufaff\U00020000-\U0003134f"
_MIXED_TOKEN_RE = re.compile(
    f"[{_CJK_RANGES}]"  # CJK: one char = one token
    "|[a-zà-öø-ɏ]+(?:['’][a-zà-öø-ɏ]+)*"  # latin word
    "|[0-9]+"  # digit run
)


def _tokenize_mixed(text: str) -> list[str]:
    """CJK chars as single tokens, latin words / digit runs as tokens.

    Mirrors ``eval.mer.tokenize_mixed``: NFKC-normalizes first (folding
    fullwidth latin/digits such as ＫＰＩ / １２３ — common in CJK teacher
    transcripts — to their ASCII forms) and lowercases, so the agreement
    filter sees the same tokens the canonical eval tokenizer would.
    """
    return _MIXED_TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over token lists (two-row DP)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + (ta != tb),  # substitution
            )
        prev = cur
    return prev[-1]


def _concat_tokens(rec: MeetingRecord) -> list[str]:
    segs = sorted(rec.segments, key=lambda s: (s.start, s.end))
    tokens: list[str] = []
    for seg in segs:
        tokens.extend(_tokenize_mixed(seg.text))
    return tokens


def two_pass_agreement(
    rec_a: MeetingRecord, rec_b: MeetingRecord, max_cpwer: float = 0.05
) -> bool:
    """True when two labeling passes agree within ``max_cpwer``.

    Computes a cpWER-like rate: mixed-token edit distance between the
    time-ordered concatenated transcripts, normalized by the reference
    (rec_a) token count. Speaker permutation matching is omitted because the
    comparison is speaker-agnostic (concatenation ignores speaker labels);
    the eval package's ``cpwer`` is the canonical metric.
    """
    tok_a = _concat_tokens(rec_a)
    tok_b = _concat_tokens(rec_b)
    if not tok_a and not tok_b:
        return True
    if not tok_a or not tok_b:
        return False
    rate = _edit_distance(tok_a, tok_b) / len(tok_a)
    return rate <= max_cpwer
