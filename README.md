# distil-vibevoice-asr → MOSS-Transcribe-Diarize zh-TW

On-device **zh-TW (Traditional Chinese, Taiwan) / English** meeting
transcription with **speaker diarization + timestamps**.

> ## ⚠️ Status: the VibeVoice distillation path is DEPRECATED
>
> This repo started as a prune-and-distill of **microsoft/VibeVoice-ASR** (8.7B →
> 1.5B). That work is preserved below (§"Deprecated: VibeVoice-ASR distillation")
> for the record, but it is **superseded** by fine-tuning
> **[OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)**
> (0.9B, Apache-2.0). Why the pivot:
> - **Quality**: per its paper, MOSS beats VibeVoice-ASR-8.7B on every ASR
>   benchmark (e.g. AISHELL-4 CER 14.84 vs 21.40; podcast 5.97 vs 27.94).
> - **Size**: 0.9B natively vs a 1.5B distillation target — smaller *and*
>   better, no multi-stage prune/distill cascade needed.
> - **License**: Apache-2.0 (VibeVoice-ASR is research-only).
> - **Same output contract**: speaker-attributed, timestamped segments in one
>   pass (`[start][Sxx]text[end]`), which is the product requirement.
>
> **Everything published and live is the MOSS adaptation.** The VibeVoice code
> (`src/distil_vibevoice/{pruning,distill}`, `configs/`) remains importable and
> its 252 CPU tests pass, but it is not the maintained path.

## Published artifacts (MOSS adaptation)

| What | Where |
|---|---|
| Fine-tuned model (v6-stream, deployed) | [`Luigi/moss-transcribe-diarize-zhtw`](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw) |
| **GGUF quantized models** (q4_K_M, v5 + v6-stream) | [`Luigi/moss-transcribe-diarize-zhtw-gguf`](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-gguf) |
| **WASM browser demo** (ggml C++ engine, live) | [`Luigi/moss-transcribe-diarize-wasm`](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-wasm) (HF Space) |
| **RapidSpeech.cpp engine** (WASM CPU/WebGPU + Jetson Nano CUDA ports) | [`vieenrose/RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp) (`main` = WASM/CPU, `jetson-nano-gen1` = CUDA 10.2/sm_53) |
| Quantized ONNX graphs (web / mobile / sherpa) | [`Luigi/moss-transcribe-diarize-zhtw-onnx`](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-onnx) |
| ONNX browser demo | [`Luigi/zh-tw-transcriber-local`](https://huggingface.co/spaces/Luigi/zh-tw-transcriber-local) (HF Space) |
| Precomputed long-meeting viewer | [`Luigi/zh-tw-meeting-transcriber`](https://huggingface.co/spaces/Luigi/zh-tw-meeting-transcriber) |
| sherpa-onnx C++ runtime port | [`vieenrose/sherpa-onnx@feature/moss-transcribe-diarize`](https://github.com/vieenrose/sherpa-onnx/tree/feature/moss-transcribe-diarize) |

---

# MOSS-Transcribe-Diarize zh-TW adaptation (current work)

Base model: **MOSS-Transcribe-Diarize** — Whisper-Medium encoder (80-bin mel,
4× time-merge, VQ-adaptor) + Qwen3-0.6B decoder; audio token id `151671`,
12.5 audio-tokens/s; single-pass output stream `[start][Sxx]text[end]`.

## What's different vs. the original MOSS-Transcribe-Diarize?

Same architecture, very different deployment envelope. This project **fine-tunes**
(not prunes) [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
— Whisper-medium encoder + Qwen3-0.6B decoder, 0.9B params, single-pass
`[start][Sxx]text[end]` output, Apache-2.0 — and re-engineers everything around it:

| | Original MOSS-TD 0.9B | This project (v6-stream lineage) |
|---|---|---|
| **Architecture / license** | Whisper-medium enc + Qwen3-0.6B dec · Apache-2.0 | identical (fine-tuned, nothing pruned) |
| **Language & script** | general Mandarin, Simplified-leaning output | **Traditional Chinese (Taiwan)** enforced end-to-end + zh-TW/EN code-switch; conservative ITN |
| **Domain** | general speech | real meetings: ~3 000 synthetic zh-TW meetings (TTS) + 55.8 h 立法院 IVOD with fused whisperX×pyannote labels (speaker-purity filtered) |
| **Held-out zh-TW meeting MER** | 0.395* | **~0.18** (*script-normalized; part of the base gap is Simplified↔Traditional) |
| **Long-meeting diarization (123 min)** | DER 0.74 | **DER 0.195 · consistency 0.905** (cross-window ECAPA/CAM++ linking) |
| **Diarization under fine-tuning** | n/a (base) | defended: 8× speaker-tag CE + KL-anchor to base at `[Sxx]` positions — FT rounds no longer collapse speakers |
| **Code-switch regression (ASCEND)** | reference | no regression; v6 improves all buckets vs v5 (all-MER 0.417 → **0.285**) |
| **Long-audio decoding** | full attention, KV grows O(audio) | **streaming fine-tune: bounded 45 s audio-KV window** (monotonic eviction) — flat memory, ~20 % faster decode, DER parity via linking |
| **Decode robustness** | — | engine guards: tick-stall loop breaker, speech-aware premature-EOS suppression, per-window watchdog + recovery |
| **Quantization** | bf16 release | diarization-defended **q4 QAT** → q4_K_M GGUF (707 MB) · int8 ONNX (ternary/q3 measured & rejected; encoder-ternary in progress) |
| **Runtimes** | HF transformers (GPU) | + **in-browser WASM** (multithread/iOS/WebGPU), native CPU C++ ([RapidSpeech.cpp](https://github.com/vieenrose/RapidSpeech.cpp)), Jetson Nano (CUDA 10.2/sm_53), sherpa-onnx, ONNX web |
| **Live demos** | — | [WASM (fully local)](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-wasm) · [native C++ Space](https://huggingface.co/spaces/Luigi/moss-transcribe-diarize-cpp) |


## Reproducibility map (numbered scripts)

All scripts are thin CLIs under `scripts/`; run with `.venv/bin/python`. Models
live in `models/` and data in `data/` (both git-ignored — regenerate via the
steps below). Held-out eval meetings `15361/15362/15857` are **never** trained on.

| Stage | Script | Produces |
|---|---|---|
| **Real data** — collect IVOD | `01b_collect_ivod.py` | `data/raw/ivod_*/` + manifest (robust dead-air skip via `robust_speech_start`) |
| Fuse real labels | `38_build_ivod_targets.py` | `data/pseudo/ivod_ft_v4.jsonl` — whisperx×pyannote fused, speaker-coverage purity filter (131 mtgs / 55.8 h) |
| **Synthetic data** — scripts | `dialogue_scripts.py` (12 domains) | code-switched zh-TW/en meeting scripts |
| TTS meetings | `24_bulk_tts.py` (VibeVoice community fork, `.venv_tts`) | `data/pseudo/tts_all.jsonl` (~3000 mtgs, exact labels) |
| **Fine-tune** v1/v2 | `27_ft_moss.py` | `models/moss_ft_zhtw{,_v2}` (SFT, CE on assistant tokens) |
| Fine-tune v3/v4 | `39_ft_moss_v3.py` | `models/moss_ft_zhtw_v4` — 300 s windows, mix 40% clean / 30% RIR+MUSAN-aug / 30% real IVOD |
| **Eval** synthetic MER | `26_eval_moss.py` | held-out MER/DER |
| Long-clip gates | `36_dump_moss_outputs.py` → `40_eval_longclip.py` | leakage / tag-drop / DER / consistency at 90/180/300 s |
| Base-domain regression | `45_regression_base_domains.py` | ASCEND zh/en/mixed MER vs base (guards forgetting) |
| **Diarization** linking sweep | `34b`/`51_linking_sweep.py` | best cross-window linking config |
| **On-device** ONNX export | `30_export_moss_onnx.py` (`--last-logits`, `--fp16-kv`), `31_export_moss_qwen3style.py` | web / sherpa decoder graphs (parity-checked) |
| Web quantization | `41_quantize_web.py`, `44_export_ecapa_onnx.py` | int8 / q4 graphs, ECAPA ONNX (conv-DFT STFT, cosine 1.0000) |
| QAT + mixed-precision | `48_q4_sensitivity.py` (NAS) → `49_qat_q4.py` → `50_export_q4_mixed.py` | quantization-robust q4 |
| Fine-tune **v5** (diarization defense) | `40_ft_moss_v5.py` | `models/moss_ft_zhtw_v5*` — 8× speaker-tag CE + KL-anchor to base ([Sxx] distribution), optional diarization-defended QAT |
| **Quant ladder** q4→q3→ternary | `52_qat_ladder.py` | k-bit STE QAT (`--bits 4/3/2`, `--freeze-rest`, anneal, self-KL) — negative result, see below |
| Fine-tune **v6-stream** (streaming) | `53_streaming_ft.py` | `models/moss_ft_zhtw_v6_stream3` — bounded 45 s audio-KV window (4D eviction mask + frozen full-attention teacher KL + silence-tail aug) |
| **Demo build** | `37_build_space_example.py`, `43_dump_web_assets.py` | precomputed examples + browser assets |

## Fine-tuning recipe (v1 → v4)

SFT, cross-entropy on assistant tokens only, bf16, gradient checkpointing,
cosine LR, single GPU. Target format `[start][Sxx]text[end]`, Traditional text.

| Ver | Change | Held-out MER | Real 123-min DER / consistency |
|---|---|---|---|
| base | — | 0.395* | 0.74 / — |
| v1 | 400 steps @ lr 1e-5 | 0.201 | — |
| **v2** | +2000 steps @ 5e-6 | **0.183** | 0.180 / — |
| v3 | 300 s windows + real IVOD (30 %), 3000 steps | 0.18 | 0.244 / 0.814 *(diarization regressed)* |
| **v4** | v3 data with **speaker-coverage purity filter** (`38`), 2000 steps | ~0.18 | **0.195 / 0.905** *(recovered)* |
| **v5** | 8× speaker-tag CE + **KL-anchor to base at [Sxx] positions** (`40`) | ~0.18 | speaker count restored to base (3/3 on held-out 5-min) |
| **v6-stream** | streaming FT: bounded 45 s audio-KV window (`53`), silence-tail aug, speaker-position KL | ASCEND all buckets **better than v5** (0.285 vs 0.417 all) | linked DER 0.132 ≈ v5's 0.128 |

\* base-model gap is largely the Simplified↔Traditional script mismatch.

**Key v4→v5 fix**: transcription-focused FT rounds progressively *forgot* the
base model's speaker separation (base 3 → v4 1 speakers on a held-out 5-min
meeting) — [Sxx] tags are a tiny token fraction, so uniform CE happily trades
them away. Weighted CE alone couldn't restore voice discrimination (the model
games CE via turn-taking cues); the fix is KL-anchoring the *speaker-tag
distribution* to a frozen base teacher. The same defense is kept through QAT
("diarization-defended QAT") and the v6 streaming FT.

**v6-stream (deployed)**: fine-tuned so decode works with a **bounded 45 s
audio-KV window** (monotonic eviction in the engine): audio KV memory is O(45 s)
instead of O(meeting), per-token attention cost is constant, decode ~20 %
faster, sub-second timestamps preserved. Under the bounded window the model
cannot re-identify voices older than 45 s and correctly assigns fresh local
[Sxx] ids — global identity is restored by the ECAPA/CAM++ linking stage, so
streaming diarization is gated on **linked** DER (0.132 vs full-attention v5's
0.128 = parity). Recipe essentials (all needed): 4D eviction attention mask
built via `offset_mapping` (per-segment tokenizations do *not* concatenate —
BPE merges across `[end][start]`); window curriculum 120→75→45 s; frozen
full-attention teacher KL (all positions + extra weight on speaker positions);
**silence-tail augmentation** (without it the FT hallucinates speech after the
meeting ends — the base model didn't).

## Quantization ladder: q4 wins, q3 and ternary are dead ends (measured)

`52_qat_ladder.py` generalizes the q4 QAT to k-bit (bits=2 → block-scaled
ternary, BitNet b1.58 format). Findings on the 0.6B decoder (native CPU, 8T):

| rung | decode | CER / DER | verdict |
|---|---|---|---|
| q4_K_M (QAT) | 7.8 tok/s | 11.0 % / 0.12 | **shipped** |
| q3_K FFN | 7.0 tok/s | ~11 % / 0.076 | dead: file *bigger* than q4 (k-quant bumps) + slower kernel |
| ternary FFN (TQ2_0) | ~2.1× q4 (kernel bench) | ~21 % / 0.18+ | dead: capacity wall |

Three independent ternary recipes (freeze-rest; co-train + self-KL + 4→3→2
anneal + sensitivity-spared layers; CE-only + anneal) all converge to ~21 CER =
**the 0.6B decoder cannot absorb ternary-FFN damage at fine-tuning scale** —
a capacity wall, not a training bug. Two transferable QAT lessons:
(1) fake-quantizing a *subset* of layers while training *all* params lets the
unquantized params absorb the error — loss looks great, the deployed subset is
PTQ-garbage; freeze everything but the quantized latents (`--freeze-rest`),
which needs ~10× the LR. (2) self-distill KL against your *own* unquantized
weights diverges (moving target); use CE or a frozen teacher.

**Key v3→v4 fix**: v3's fused IVOD labels allowed segments spanning speaker
turns, teaching merged-turn output (consistency 0.912→0.814). A 15 s length cap
was not viable (median whisperx segment is 18 s → keeps only 15 % of speech);
instead the winning pyannote speaker must cover ≥70 % of the segment span
(`38_build_ivod_targets.py`). This dropped exactly the boundary-crossers and
recovered diarization.

## Evaluation principles

- **Script-normalize before scoring.** MER/CER are computed after converting
  both hyp and reference to one script (OpenCC), so Traditional-vs-Simplified
  output is not counted as a recognition error.
- **Separate repairable from irreparable metrics.** Simplified-script leakage
  and number formatting are fixed deterministically in post (see below) — they
  are reported but do **not** gate publication. Recognition (MER), diarization
  (DER / consistency), timestamps, and base-domain regression (ASCEND) do.
- **Always check base-domain regression.** Every fine-tune is scored on ASCEND
  (real zh/en conversational code-switch) vs the base model, so a zh-TW FT can't
  silently degrade the base model's general ASR.

## Long-meeting diarization

MOSS is single-pass; long audio is processed in windows and speakers are linked
across them (`src/distil_vibevoice/runtime/`):

1. Transcribe each window (`[start][Sxx]…` labels are only consistent *within* a
   window).
2. Embed every segment with **ECAPA-TDNN** (`runtime/embeddings.py`).
3. One global agglomerative-clustering pass (`runtime/linking.py`: core segments
   ≥3 s at cosine 0.45 average-linkage, short segments → nearest centroid) gives
   one consistent speaker set across the whole meeting.

Validated vs pyannote references on real meetings: **DER 0.056** (30-min,
7/8 speakers), **consistency 0.905** (123-min, 300 s windows). Window size is a
measured accuracy↔speed tradeoff: 90 s ≈ 180 s on diarization; only 300 s is
meaningfully better on very long meetings (consistency 0.905 vs 0.87), and the
180 s→300 s gap is **intrinsic to window context, not linker-fixable**
(`51_linking_sweep.py` recovers only ~0.003 of the 0.032 gap).

## Written-form post-processing (deterministic, recognition unchanged)

An acoustic model can't reliably learn script choice or number place-value, so
these are rule modules applied after decoding:

- **`runtime/lenient_parser.py`** — accepts `[start]text[end]` without an
  `[Sxx]` tag (real far-field audio drops it) and inherits the previous speaker;
  the strict parser otherwise discards whole windows.
- **OpenCC `s2tw`** — Simplified→Traditional (pure script conversion; never
  `s2twp`, which would corrupt proper nouns like 高端疫苗).
- **`data/itn.py`** — conservative inverse text normalization via `cn2an`
  (二十三→23, 百分之五十→50%) with idiom guards (千萬, 百分之百, …) and
  ordinal protection (第八/第一 stay words). A JS port (`space_local/itn.js`) is
  differential-tested byte-identical.

## On-device: ONNX, quantization, QAT

- **Export** (`30`, `31`): 3-graph web layout (encoder / embedding / decoder
  with dynamic KV) and sherpa fixed-KV layout; `--last-logits` (lm_head on the
  final position only — the full-prefill logits are a ~1.4 GB tensor in 32-bit
  wasm), optional `--fp16-kv`. All parity-checked vs PyTorch.
- **Quantization** (`41`, `44`): encoder int8 (MatMul-only — `ConvInteger`
  unsupported in ORT CPU/web), embedding int8, decoder int8 / q4 (MatMulNBits
  block-32 symmetric); ECAPA exported with a conv1d-DFT replacing `torch.stft`
  (parity cosine 1.0000).
- **Accuracy (script-normalized MER vs bf16 FT)**: int8 **0.03–0.04**; naive q4
  0.068; **QAT-q4 0.077→ near int8** after quantization-aware fine-tuning.
- **QAT + NAS** (`48`/`49`/`50`): STE int4 fake-quant matching MatMulNBits
  (`src/distil_vibevoice/quant/fakequant.py`); a layer sensitivity probe (the
  "NAS" — last layer + lm_head are most q4-sensitive) informs a mixed-precision
  export. **int8 is the shipped choice** (q4 saved only ~19 % of download for a
  real accuracy hit; accuracy is the priority).

## Browser demo (`space_local/`)

Fully-local: int8 ONNX + `onnxruntime-web` in a **Web Worker** (page never
freezes), any-length windowed audio + ECAPA linking, streaming transcript with
a liveness heartbeat, s2tw + ITN, SRT/JSON export. Weights and example media are
served from the ONNX model repo (HF Space storage is code-only). CPU-only —
WebGPU was dropped; cross-origin isolation for wasm threads via a COI service
worker. `space_local/pipeline.js` is the reference JS pipeline (env-agnostic;
node-validated against the Python sim `42_web_pipeline_sim.py`).

## sherpa-onnx C++ port

The native runtime lives in the fork branch above: MOSS-SATS offline recognizer
(adapted from the qwen3-asr impl), a character-state-machine SATS parser
(differential-tested byte-exact vs the Python reference), windowed decoding
(`--moss-sats-window-seconds`, default 300 — single-pass degenerates beyond
~4–5 min), int8 graphs.

## RapidSpeech.cpp port (ggml C++): WASM demo + Jetson Nano

The maintained native runtime is the
[`vieenrose/RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp)
fork — full MOSS-TD architecture (Whisper-medium encoder + Qwen3 decoder +
`[start][Sxx]text[end]` parsing) on ggml, from one q4_K_M GGUF.

- **WASM browser demo** (`main` branch, live Space above): multithreaded
  wasm + relaxed-SIMD variant, WebGPU option, iOS build (1.5 GB heap cap,
  iterative graph build for JSC's shallow worker stacks), CAM++ speaker
  linking, hotwords, thread auto-calibration (hybrid P/E-core laptops:
  `hardwareConcurrency` is misleading — a ~1.5 s microbenchmark picks the
  real optimum, cached per device), pause-snapped windows, silence gate,
  stall watchdog (a stalled window auto-recovers: worker killed, engine
  reloaded from cache, window skipped — one bad window no longer kills a
  2 h run).
- **KV policy**: engine default f16 KV (half the memory of f32, sub-second
  timestamps preserved; q8 KV snaps timestamps to whole seconds). Long
  browser windows use **45 s audio-KV eviction** (v6-stream model) instead of
  q8: flat heap at full f16 precision.
- **Jetson Nano gen1** (`jetson-nano-gen1` branch): CUDA 10.2 / sm_53 /
  C++14 / old-ggml adaptation; measured-best split is GPU encode + GPU
  prefill + CPU decode (GPU decode is host-KV-staging bound). Eviction
  ported for bounded-memory decode.
- **Speed levers measured**: CPU pinning (`taskset` P-cores) alone 5.1→7.8
  tok/s; eviction +20 %; next big lever is in-graph persistent KV +
  flash-decode (llama.cpp decodes the same arch ~14× faster — implementation
  headroom, not model cost).

---

# Deprecated: VibeVoice-ASR distillation

*(Original plan, kept for the record. Superseded by the MOSS adaptation above.)*

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
