# distil-vibevoice-asr → MOSS-Transcribe-Diarize zh-TW/EN

On-device **zh-TW (Traditional Chinese, Taiwan) / English** meeting
transcription with **speaker diarization + timestamps**, built on
**[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)**
(0.9B, Whisper-medium encoder + Qwen3-0.6B decoder, Apache-2.0).

> ## Status (2026-07-21): purification-first rebuild, replacing the earlier fine-tuned lineage
>
> Two prior approaches are **superseded** and kept below for the record only:
> a VibeVoice-ASR distillation (§"Deprecated: VibeVoice-ASR distillation") and
> a fine-tuned MOSS-TD lineage (v1–v7, harness-laden windowed pipeline). Both
> were abandoned in favour of the plan below, after the fine-tuned lineage was
> found to be over-specialised and structurally fragile relative to the base
> model. **No fine-tuning happens anywhere in the current pipeline.** The base
> model is used as published; all engineering effort goes into a byte-verified
> C++ port, a minimal windowed pipeline, and quantization — each stage gated
> against measurement, not assumption.

## Live demo

**[Luigi/moss-transcribe-diarize-cpp](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-cpp)**
— native C++ (ggml, CPU) on the HF Space, windowed pipeline + cross-window
speaker linking, mixed-precision Q8_0 weights (1.55 GB). The former WASM demo
(`Luigi/moss-transcribe-diarize-wasm`) is now **private** — its `campplus.gguf`
dependency was rehosted to the model repo below, and it predates the current
engine; it is not being maintained as a second front end right now.

## The four-stage plan

Each stage is gated by measurement before the next begins. Full detail,
including every rejected candidate and why, lives in project memory
(`project-staged-delivery` — not in this repo, but summarised here).

### Stage 1 — byte-identical C++ port. **Done.**

The engine is [`vieenrose/RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp)
branch **`moss-pure`**, which vendors
[`localai-org/moss-transcribe.cpp`](https://github.com/localai-org/moss-transcribe.cpp)
(MIT) **unmodified**. `moss-pure` starts from the commit *before* the original
MOSS-TD work was ever added to that fork — it carries none of the harnesses
(audio-KV eviction, repetition penalty, EOS-coverage suppression, loop guards,
speaker-linking-by-fiat) that accumulated around the fine-tuned lineage. The
whole point of this stage was to separate "what the model actually does" from
"what a decade of patches made it look like it does."

Byte-identity gate, verified non-windowed, single pass, no post-processing:

| reference | f16 GGUF | f32 GGUF (pinned ISA) |
|---|---|---|
| PyTorch f32 | differs (near-tie flips vary by CPU SIMD width) | **byte-identical**, CPU and CUDA |
| PyTorch bf16 / f16 | not used as reference — see below | — |

Two findings that made the gate meaningful rather than accidental:
- **The reference must be f32.** bf16 (what the model authors' own README runs
  on CUDA) has an 8-bit mantissa vs f16's 10 — it is *less* precise than the
  port it would be judging, and diverges at near-ties. PyTorch-f16 diverges
  too (different accumulation order).
- **The build's ISA must be pinned**, not `-march=native`. The same source
  built on two machines can select different SIMD kernels and flip a
  near-tied token — measured directly (one timestamp in 92 differed between a
  local build and the HF Space's CPU). `GGML_NATIVE=OFF`, explicit
  `x86-64-v3` (AVX2+FMA+F16C), `GGML_LLAMAFILE=ON` (required — off, one more
  near-tie flips).

Quantization is **not** part of stage 1: f32 was the gate; smaller weights are
a stage-3 decision, re-gated on their own terms (below).

### Stage 2 — windowed pipeline + cross-window speaker linking. **Done, deployed.**

Long audio initially looked like it needed windowing to avoid truncation —
it didn't; the real cause was the GGUF's fixed 5120-token generation cap
(from `generation_config.json`), which a duration-scaled budget fixes with no
windowing at all (coverage 67%→100%, WER 0.552→0.161 on a 16-min meeting).
Windowing was kept anyway, once measured to be a genuine efficiency win: **3.3×
faster, ~half the peak memory** vs single-pass on real 16-minute meetings —
and it is required for meetings measured in hours, where a single continuous
decode's KV cache and attention span both grow unbounded.

- **Window length: fixed at 90s**, chosen after a sweep (60/90/180/300/450s)
  scored two ways — WER/speaker-accuracy against real AMI ground truth (EN),
  and a **turn-crossing check** (does a merged segment ever span a genuine
  speaker change, vs legitimately combine consecutive same-speaker utterances)
  against a validated windowless reference (zh, which has no independent
  ground truth). 90s is the best point found for zh and not distinguishable
  from other lengths for en.
- **Cross-window speaker identity** is not tracked by the model at all — each
  window's `[Sxx]` tags reset independently. Fixed with CAM++ (unmodified,
  from RapidSpeech.cpp's `rapidspeech-core`): pool each window-local speaker's
  audio into one embedding (per-utterance embedding fragmented badly — most
  utterances are under 2s, too little audio for a stable embedding), then
  **constrained agglomerative clustering** with a *cannot-link* prior (two
  speakers active in the same window are provably distinct — enforced as a
  soft penalty, not a hard rule, since the engine occasionally over-splits one
  real speaker within a window). An earlier greedy streaming version let one
  bad merge cascade through a running-mean centroid, corrupting two real
  speakers into one identity (68% accuracy); the constrained version fixed it
  to 99%+.
- **Streaming audio reader.** The pipeline never holds more than one window's
  audio in memory — verified peak RSS flat at ~0.76 GB from 16 minutes to
  2h3min of real audio (a naive whole-file-in-memory version grew ~3.8 MB per
  minute of audio). Verified byte-identical output vs the non-streaming path
  before shipping.

**Validated at scale, and the headline number needed correcting.** Tuned and
first validated on one AMI meeting (WER 0.150, speaker accuracy 99.4%). Tested
on 6 diverse real meetings (4 recording sites) plus two multi-hour real
sessions (87.5 min, 2h03min): **mean WER across the 6 meetings is 0.262 — the
single-meeting number was not representative.** Confirmed via windowless
control that this gap is base-model accuracy varying by meeting, not
something windowing/linking introduces (windowless scores the same on the
hard meetings). No crashes across ~4 hours of new audio tested.

### Stage 3 — quantization at equal accuracy. **Done, deployed.**

Deployed today: **`q4mix-v2`**, 759 MB — every linear at Q4_K, `token_embd`
at f16, with the decoder tensors taken from a **silence-robust QAT**
checkpoint (scripts 60–87; see `docs/integration-note-2026-07-23.md`). The
later q4 campaign superseded this section's "nothing below Q8_0 for the
encoder" conclusion. `q8mix` (1.55 GB) remains the higher-fidelity option.
The Q8_0 analysis below is the stage-3 work that mapped the
quantization-sensitive tensors and led there.

First deployment of this stage: **mixed-precision Q8_0**, 1.55 GB (2.3×
smaller than f32).

Uniform Q8_0 (0.99 GB) looked fine on short clips and on text-similarity
metrics (93% char agreement) — but on a 16-minute Chinese meeting it silently
stopped emitting utterance boundaries: 69 segments where the reference emits
312. Bisected by holding one tensor family at f16 and the rest at Q8_0 (and
the inverse): **`token_embd.weight`** alone reproduces the collapse, and the
**full Qwen3 decoder** (attention + FFN together) collapses too even though
each half is individually clean — a compounding effect, not one tensor. The
Whisper encoder and audio adaptor tolerate Q8_0 with zero measured damage.

Fix: hold `token_embd` + the full decoder at f16, quantize encoder + adaptor
to Q8_0 → 312/312 utterances (100.00% text agreement with f32), byte-identical
to f32 on the zh 5-minute golden clip, at 1.55 GB. Pushed further down
(encoder@q6_k/q5_k/q4_k) — all rejected: quality is **non-monotonic** with
bit-width (q5_k measured *worse* than q4_k), and q6_k costs real English
accuracy for only 10% additional size savings. No safe stopping point found
below Q8_0 for the encoder.

KV-cache optimization (eviction, KV quantization) was evaluated and found
**not worth doing**: windowing already bounds the engine's KV cache at 90s of
context, so there's little left to save.

### Stage 4 — audit the existing zh-TW/EN training data. **In progress.**

Reviewing `data/pseudo/ivod_ft_v4.jsonl` (131 real 立法院 sessions, ~97 audio
hours) on four axes: transcript correctness, timestamp accuracy, speaker-tag
accuracy, utterance-segmentation correctness. Review only — no training
implied by this stage. Findings so far:
- **Coverage confirmed as the headline defect, still current**: 57.8% mean /
  58.7% median session-level label coverage (p10: 43.7%) — unlabelled gaps
  are largely real speech, not silence.
- **The existing labels were not produced by MOSS-TD** (`"source":
  "ivod_wx_py"` in the manifest — a WhisperX-based pipeline predating this
  project's MOSS-TD work). Cross-checking against the now-validated pure
  engine's own output is in progress.

## Engine and demo repo map

| What | Where |
|---|---|
| **Pure C++ engine** (byte-identical, MIT-vendored) | [`vieenrose/RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp) branch `moss-pure` |
| **Live demo** (windowed + linked, deployed) | [`Luigi/moss-transcribe-diarize-cpp`](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-cpp) |
| **GGUF weights** (f32 gate reference, deployed q4mix-v2, q8mix, campplus) | [`Luigi/moss-transcribe-diarize-zhtw-gguf`](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-gguf) |
| Reference conversion tooling / parity suite (cited, not vendored) | [`localai-org/moss-transcribe.cpp`](https://github.com/localai-org/moss-transcribe.cpp) (MIT) |

2026-07-23 cleanup: the abandoned fine-tuned lineage was removed from the Hub
— its GGUF artifacts (`moss-td-zhtw-v5kl…v71-*`) were deleted from the model
repo's tip (recoverable via the repo's git history), and the safetensors
(`…-zhtw`) and ONNX (`…-zhtw-onnx`) fine-tune repos were deleted outright.
The GGUF repo now carries only `moss-transcribe-base-f32.gguf` (gate
reference), `moss-transcribe-base-q4mix-v2.gguf` (deployed),
`moss-transcribe-base-q8mix.gguf`, and `campplus.gguf`.

## Open items

- **RapidSpeech.cpp has no LICENSE file** (neither this fork nor
  `RapidAI/RapidSpeech.cpp` upstream). The ASR engine itself has zero
  unlicensed dependency (verified via `ldd`: only MIT ggml + the vendored MIT
  port). Cross-window speaker linking (`rapidspeech-core`, for CAM++) does
  pull in the surrounding unlicensed upstream tree as build dependencies —
  shipped as an accepted, explicit risk on that one feature, documented in
  `windowing.py`/`app.py` in the Space repo.
- Stage 3's CPU-side profiling (prefill/decode breakdown, thread-count sweep,
  NUMA pinning) has not been done — everything was profiled on CUDA first per
  standing instruction; the deployed demo runs CPU-only.
- Stage 4 audit is in progress; findings above are partial.

---

## Deprecated work (kept for the record)

- [`docs/DEPRECATED_VIBEVOICE_DISTILLATION.md`](docs/DEPRECATED_VIBEVOICE_DISTILLATION.md) —
  the original plan: prune-and-distill VibeVoice-ASR (8.7B → 1.5B). Superseded
  by adopting MOSS-Transcribe-Diarize directly (smaller, better, Apache-2.0).
- The fine-tuned MOSS-TD lineage (v1–v7) that followed is documented in git
  history (see commits before 2026-07-21) rather than a separate doc — it was
  superseded by the purification-first rebuild above before its own writeup
  was ever finalised as a standalone reference.
