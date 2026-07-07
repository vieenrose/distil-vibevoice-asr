# distil-vibevoice-asr

Prune-and-distill **microsoft/VibeVoice-ASR** (8.7B) → **4B** → **1.5B** for
**on-device meeting transcription** (6 GB-RAM phone) with **speaker diarization +
timestamps**, targeting **zh-TW (Traditional Chinese, Taiwan) + English
code-switched meetings**.

This README is the canonical description of the plan. Module APIs are defined by
the cross-module contract (see `src/distil_vibevoice/`), and hyperparameters live
in `configs/`.

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
