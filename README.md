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

Deployed: **mixed-precision Q8_0**, 1.55 GB (2.3× smaller than f32).

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
| **GGUF weights** (f32 gate reference, deployed q8mix, campplus) | [`Luigi/moss-transcribe-diarize-zhtw-gguf`](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-gguf) |
| Reference conversion tooling / parity suite (cited, not vendored) | [`localai-org/moss-transcribe.cpp`](https://github.com/localai-org/moss-transcribe.cpp) (MIT) |

The model repo above also still carries GGUF artifacts from the abandoned
fine-tuned lineage (`moss-td-zhtw-v5kl…v71-*`) — kept for reference, **not**
part of the current pipeline; only `moss-transcribe-base-f32.gguf`,
`moss-transcribe-base-q8mix.gguf`, and `campplus*.gguf` are live.

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
# Deprecated: VibeVoice-ASR distillation

*(Original plan, kept for the record. Superseded first by a fine-tuned MOSS-TD lineage, then by the purification-first rebuild described above.)*

This README section is the canonical description of the (deprecated) distillation
plan. Module APIs are defined by the cross-module contract (see
`src/distil_vibevoice/`), and hyperparameters live in `configs/`.

---

## 1. Teacher architecture

VibeVoice-ASR = frozen audio tokenizers + connectors + a Qwen2.5-7B LLM that emits
a JSON array of `{Start, End, Speaker, Content}` segments.

```
                    24 kHz mono audio (<= 60 min / pass)
                               |
        +----------------------+----------------------+
        |                                             |
+-------v--------+                          +---------v-------+
| Acoustic       |                          | Semantic        |
| tokenizer      |                          | tokenizer       |
| sigma-VAE conv |                          | same topology   |
| 3200x down     |                          | latent dim 128  |
| 7.5 tok/s      |                          |                 |
| latent dim 64  |                          |                 |
+-------+--------+                          +---------+-------+
        |            (both FROZEN at all stages)      |
+-------v--------+                          +---------v-------+
| acoustic       |                          | semantic        |
| connector MLP  |                          | connector MLP   |
+-------+--------+                          +---------+-------+
        |                                             |
        +----------------------+----------------------+
                               |
              speech tokens spliced into Qwen2.5 chat template
                  <speech_start> ... <speech_end>
                               |
                 +-------------v--------------+
                 |  Qwen2.5-7B LLM backbone   |   <-- this is what we prune+distill
                 |  hidden 3584, 28 layers    |
                 |  28 Q / 4 KV heads, hd 128 |
                 |  ffn 18944, vocab 152064   |
                 |  RoPE 1e6, ctx 131072      |
                 +-------------+--------------+
                               |
   [{"Start":0,"End":15.4,"Speaker":0,"Content":"..."}, ...]

   (diffusion head / TTS branch: DROPPED for ASR distillation)
```

## 2. Distillation cascade

```
  8.7B teacher ──(width prune)──> 4B ──(distill vs 8B)──> 4B distilled
                                                             │
                        ┌────────────────(width prune)───────┘
                        v
                      1.5B ──(distill vs 4B, +10% direct-8B batches)──> 1.5B distilled
                        │
                        └──(QAT int4-w/int8-a + length & speaker curricula)──> mobile export
```

**Stage 1 — 8B → 4B** (`configs/prune_4b.yaml`, `configs/distill_stage1_4b.yaml`)

- Minitron-style **width** pruning of the teacher's own LLM: hidden 3584→2560,
  intermediate 18944→13312, Q heads 28→20, KV heads 4→4, **keep all 28 layers**.
  Activation-magnitude importance over 64 calibration batches.
- Distill: `L = 0.5·KL(teacher‖student, T=2.0) + 0.3·CE(labels) + 0.2·MSE(hidden, learned linear proj per mapped layer)`.
- **Speaker-tag and timestamp tokens upweighted 4×** in CE/KL.
- lr 1e-4 cosine, warmup 500, bf16, grad-accum, 8-bit optimizer,
  seq-len curriculum 4096 → 8192 → 16384.

**Stage 2 — 4B → 1.5B** (`configs/prune_1p5b.yaml`, `configs/distill_stage2_1p5b.yaml`)

- Prune again: hidden 2560→1536, intermediate→8960, Q heads 20→12, KV 4→2.
- Distill with the 4B as teacher, **plus 10% of batches distilled directly from the 8B**.
- Then **QAT int4-weight/int8-act** (`configs/qat_export.yaml`) with a
  **length curriculum** (30 s → 5 min → 15 min) and **speaker-count curriculum** (2 → 4 → 8).

Embeddings and connectors are width-pruned along the same channel index sets
(keep all vocab rows, cut columns). Acoustic + semantic encoders stay frozen
throughout.

## 3. Data plan

| Bucket | Share | Contents |
|---|---|---|
| TTS synthetic meetings | ~40% | VibeVoice-TTS, TW-accent voice cloning, LLM/template-scripted zh-TW/en code-switched dialogues, RIR/MUSAN/codec augmentation |
| Teacher-pseudo-labeled real audio | ~40% | Legislative Yuan IVOD, TW podcasts, YODAS zh/en, Common Voice zh-TW — labeled by the 8B teacher, confidence-filtered |
| Simulated mixtures | ~15% | `simulate_meetings` with **exact overlap labels** |
| Gold meetings | ~5% | Human-verified (AMI en, curated zh-TW meetings) |

Key sources (verified, ungated): `fsicoli/common_voice_22_0` (zh-TW, CC0),
`espnet/yodas2` (zh000 manual subs, en000+), `openfun/tw-ly-ivod` +
`openfun/ivod-fine-tune` (real zh-TW parliamentary meetings, CC-BY-4.0),
`CAiRE/ASCEND` (zh-en code-switch, CC-BY-SA-4.0), `edinburghcstr/ami` (CC-BY-4.0),
MUSAN (openslr 17) + RIRS_NOISES (openslr 28) for augmentation.

Rules (enforced by `configs/data.yaml` + `distil_vibevoice.data`):

- **ALL Chinese text targets normalized to Traditional/Taiwan** via OpenCC
  profile `s2twp`, with English spans protected from conversion.
- **Dedupe all training audio against eval sets** by spectral audio fingerprint.

## 4. Pipeline (numbered scripts)

Scripts live in `scripts/` and are thin CLIs over `distil_vibevoice` modules;
each takes `--config configs/<name>.yaml`.

| # | Script | Does |
|---|---|---|
| 00 | `00_download_data.py` | fetch source corpora + MUSAN/RIRs |
| 10 | `10_gen_dialogues.py` | template/LLM zh-TW-en meeting scripts |
| 11 | `11_tts_synth.py` | VibeVoice-TTS synthetic meetings + augmentation |
| 12 | `12_simulate_meetings.py` | overlap mixtures with exact labels |
| 20 | `20_pseudo_label.py` | 8B teacher labels real audio (hotword context) |
| 21 | `21_normalize_dedupe.py` | s2twp normalization + eval-set dedupe |
| 30 | `30_prune.py` | importance scoring + width pruning |
| 31 | `31_distill.py` | KL+CE+hidden distillation (2×GPU) |
| 40 | `40_qat_export.py` | QAT int4/int8 + mobile export |
| 50 | `50_eval_gates.py` | MER/cpWER/DER/timestamp gates |

Make targets wrap the common paths: `make setup | test | lint | label |
distill-4b | distill-1p5b | eval`.

## 5. Runtime (long meetings on device)

Chunked inference (`distil_vibevoice.runtime`): 10–15 min windows with 30–60 s
overlap; a **speaker roster is carried between chunks** via prompt-based context
injection; chunks are stitched with timestamp alignment and **Hungarian
speaker-permutation matching** on the overlap region.

### Multi-hour meetings (2–8 h) with stable global speaker identity

Pairwise overlap stitching alone is fragile over many hours: a speaker who is
silent during one window overlap has no text to match on, so the boundary hands
them a fresh label and every downstream window inherits the error. For long
recordings the transcriber can additionally anchor every window-local speaker to
a **persistent global identity** via a `SpeakerRegistry`, turning the chained
pairwise stitches into a **star topology** so one bad boundary never cascades.

- **Voice-embedding anchoring.** Each window's speakers are embedded
  (`runtime.embeddings`: the dependency-light `MfccStatsEmbedder`, or an
  ONNX ECAPA-TDNN via `load_embedder("onnx", model_path=…)`) and matched against
  EMA-updated per-speaker centroids plus short text snippets. Stitch text
  continuity stays *primary at the overlap* (it is the strongest signal when a
  speaker actually talks across the boundary); `registry.match` is the anchor
  that re-identifies a speaker who was silent through the overlap and would
  otherwise be split.
- **End-of-meeting consolidation.** A segment-level embedding store feeds a
  constrained average-linkage agglomerative pass (`runtime.consolidate`) that
  retroactively **merges global ids that turned out to share a voice** (fixing
  historical stitch/anchor errors) while leaving genuinely distinct speakers
  apart. Runs automatically on finish (`consolidate_on_finish=True`).
- **Constant per-window context.** The carried roster is capped (~12) by recent
  activity, so prompt size stays bounded no matter the meeting length.
- **Pause / resume.** Registry state (centroids + snippets + segment store +
  next-id counter) serializes to a `.json` + `.npz` sidecar pair
  (`registry.save` / `SpeakerRegistry.load`, round-trip exact). Pass
  `registry_state=<base_path>` to `ChunkedTranscriber` to persist on finish and
  transparently resume a paused meeting.

```python
from distil_vibevoice.runtime import ChunkedTranscriber, load_embedder, SpeakerRegistry

tr = ChunkedTranscriber(
    labeler,                          # any label_file(path, hotwords=...) backend
    embedder=load_embedder("mfcc"),   # or a SpeakerRegistry / registry_state alone
    consolidate_on_finish=True,
    registry_state="runs/meeting_A/registry",  # save + resume
)
rec = tr.transcribe("meeting_A.wav")  # rec.meta["registry_speakers"], ["consolidated"]
```

Passing **neither** `embedder` nor `registry` nor `registry_state` keeps the
legacy stitch-only path byte-for-byte. Global identity quality is measured by
`eval.consistency.speaker_consistency` (fraction of hyp speaker-time whose global
label matches the reference speaker); `scripts/11_evaluate.py --consolidate
[--registry-state PATH]` engages the registry and reports it alongside the gates.

### Mobile RAM budget — MEASURED (1.5B tied int4 ≈ 2.06 GB @ 8k ctx)

The table below is now **measured**, not projected, against a real bit-packed
int4 artifact of the 1.544 B **tied-embedding** student (SMOKE weights →
garbage outputs, but geometry / size / speed are the real target's). Full
methodology and reproduction in [`docs/MOBILE_BENCHMARK.md`](docs/MOBILE_BENCHMARK.md).

| Component | Assumption | ~GB | Source |
|---|---|---:|---|
| LLM weights (linears + lm_head, int4) + tied embedding/norms (bf16) | int4 g128, bit-packed to disk | **1.27** | **measured** (`packed_int4.pt` = 1,269,701,521 B) |
| KV cache | 28 layers × 2 KV heads × hd 128, fp16, 8192 ctx | **0.235** | **measured** (analytic + verified vs `generate()`; 28,672 B/token) |
| σ-VAE audio encoders | frozen, not in this artifact (needs `vibevoice` pkg) | ~0.40 | **estimate** |
| Activations + runtime overhead | mobile NPU runtime | ~0.15 | estimate (x86 torch baseline measured ~0.43) |
| **Total mobile stack @ 8k** | | **≈ 2.06** | 1.27 + 0.235 measured, +0.40 +0.15 est |

**Fits the 6 GB-RAM budget with ~4 GB to spare.** Clean deployment load of only
the packed int4 artifact adds +1.185 GB resident (process RSS 1.62 GB incl.
torch runtime). Quantizer = torchao `IntxWeightOnlyConfig(int4, PerGroup(128))`,
the ARM / on-device weight-only path.

**Throughput (decode):** GPU (RTX 5090) int4 = **60.5 tok/s** (≈ 8.1× real-time
vs the 7.5 tok/s-of-audio streaming target). CPU 8-thread x86 proxy = **1.05
tok/s** — a *directional, worst-case, unoptimized-reference-kernel, wrong-silicon*
floor (≈ 0.14× real-time), **never a phone number**. A real ARM/NPU packed-int4
XNNPACK/ExecuTorch/GGUF-q4 kernel is markedly faster; phone tok/s still needs a
real device. GGUF q4 path is staged & converter-verified (Q4_K_M ≈ 1.0–1.1 GB
expected) but the F16 write + quantize are documented manual steps.

(Authoritative *projected* numbers come from `runtime.ram_budget.estimate_ram`;
the export gate in `configs/qat_export.yaml` caps total at 2.4 GB — the measured
2.06 GB clears it.)

### Validated so far vs blocked / remaining

**Validated (measured, this machine):**
- **Backbone extraction** — `models/teacher_llm` = 7.616 B standalone
  Qwen2ForCausalLM, the real ASR-tuned VibeVoice-ASR LLM backbone (knows the
  structured JSON output format).
- **Prune + tie** — 1.544 B tied-embedding student with the exact target
  geometry (28 layers, hidden 1536, 2 KV heads, hd 128, vocab 152064).
- **int4 footprint & speed** — measured above; fits 6 GB.
- **Real-data simulation pipeline** — CV22 zh-TW real clips (438 decoded) →
  30 simulated multi-speaker code-switch meetings via `03b_build_simulated.py`;
  200 code-switch multi-speaker scripts; manifest + augment paths exercised
  end-to-end (no `vibevoice` pkg needed).
- **Real-tokenizer text distillation** — real Qwen2.5 152064-vocab BPE
  tokenizer; a correctness bug in `DistillCollator._collect_special_ids` found &
  fixed; 30-step run loss 9.66 → 2.83, finite, no OOM (peak 18.6 / 14.8 GiB on
  the two GPUs). Structured-target density ≈ 10.75 tok/s of audio.
- **252 CPU-only unit tests green.**

**Blocked / remaining (needs installs, gold data, or real hardware):**
- **Audio path** — splicing frozen σ-VAE acoustic/semantic latents into the
  student requires the `vibevoice` package (not installed here); TTS synthesis
  (`03`) and teacher pseudo-labeling (`04`) depend on it too.
- **Gold zh-TW meetings** — ~30–50 h of human-verified recordings for the 5%
  gold bucket and final eval (user-owned; not synthesizable).
- **Real distilled weights** — current student weights are SMOKE; a full
  distill run over real data is required before any WER/quality claim.
- **Real phone / ARM deployment** — packed-int4 NPU tok/s, encoder RAM &
  latency, sustained-clock thermals over an hours-long meeting.

## 6. Eval gates

`distil_vibevoice.eval.gates.run_gates` must pass before promoting a checkpoint
(`configs/eval_gates.yaml`):

| Metric | Overall | Code-switch slice |
|---|---|---|
| MER (zh chars + en words) | ≤ 0.12 | ≤ 0.15 |
| cpWER (speaker-attributed) | ≤ 0.16 | ≤ 0.19 |
| DER (collar 0.25 s) | ≤ 0.12 | ≤ 0.12 |
| Timestamp MAE (s) | ≤ 0.30 | ≤ 0.30 |

Additional slices: zh-only, en-only, long-form (≥15 min, chunked path),
overlap-heavy (see `configs/eval_gates.yaml`).

## 7. Hardware & environment

- Training box: **2× RTX 5090 (32 GB)** — student on `cuda:0`, frozen teacher on
  `cuda:1` (see distill configs). bf16 everywhere; 8-bit optimizer for the 4B stage.
- Python ≥ 3.11; venv already exists at `.venv` (python 3.12,
  **torch 2.10.0+cu128 preinstalled**).
- **torch is intentionally NOT a declared dependency** in `pyproject.toml` — the
  system-provided CUDA 12.8 build in `.venv` is used as-is; `pip install -e .`
  must not touch it.
- Teacher weights live in `models/teacher/` (git-ignored).

### Quickstart

```bash
cd /home/luigi/distil-vibevoice-asr
make setup          # .venv/bin/pip install -e ".[dev,train,audio]"
make test           # CPU-only unit tests, no network
make label          # pseudo-label real audio with the 8B teacher
make distill-4b     # stage 1: prune + distill
make distill-1p5b   # stage 2: prune + distill (+10% direct-8B)
make eval           # release gates
```

## 8. Repo layout

```
configs/                  stage hyperparameters (YAML)
scripts/                  numbered pipeline CLIs
src/distil_vibevoice/
  data/                   manifest, normalize_zhtw, pseudo_label, dedupe,
                          dialogue_scripts, tts_synth, augment, simulate_meetings
  pruning/                importance, prune
  distill/                losses, collator, trainer
  eval/                   mer, cpwer, der, timestamps, gates
  runtime/                chunked_inference, speaker_stitch, ram_budget
tests/                    CPU-only, tiny random Qwen2 models, no downloads
models/teacher/           8B teacher weights (git-ignored)
data/                     corpora + manifests (git-ignored)
```

`Segment` / `MeetingRecord` from `distil_vibevoice.data.manifest` are the shared
data types everywhere; manifests are JSONL of `MeetingRecord`.
