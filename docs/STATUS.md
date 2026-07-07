# Project Status — distil-vibevoice-asr

End-to-end status of pruning-and-distilling **microsoft/VibeVoice-ASR** →
on-device (6 GB-RAM phone) hours-long **zh-TW + English** meeting transcription
with **diarization + timestamps + hotwords**.

Last consolidated: 2026-07-04. **252 CPU-only unit tests green.**
All 33 `distil_vibevoice` submodules import cleanly.

Honesty note: int4 **weight bytes** and **KV-cache bytes** are hardware-independent
(reported as measured). Inference **tok/s measured on x86 CPU** is only a
DIRECTIONAL proxy for a phone NPU, never a phone number. σ-VAE audio encoders are
NOT in `models/teacher_llm` (need the `vibevoice` package) — the ~0.40 GB encoder
contribution is a documented ESTIMATE.

---

## 1. Requirement-by-requirement

Legend: **Design** = specified in README/configs; **Implemented** = code exists +
unit-tested; **Validated** = exercised against real artifacts/data on this box.

| Requirement | Design | Implemented | Validated | Measured evidence |
|---|:--:|:--:|:--:|---|
| **Hours-long** (2–8 h, stable global speaker id) | ✅ | ✅ | ◑ | `runtime.chunked_inference` (10–15 min windows, 30–60 s overlap) + `speaker_registry` (star-topology anchoring) + `consolidate` (end-of-meeting merge) + pause/resume sidecar. Tested by `test_chunked_multihour`, `test_long_meetings`, `test_speaker_registry`, `test_consolidate`. Validated on synthetic long meetings; **not** yet on real hours-long audio. |
| **zh-TW + English code-switch** | ✅ | ✅ | ✅ | `data.normalize_zhtw` (OpenCC `s2twp`, English spans protected); 200 generated scripts all code-switch + multi-speaker; 30 simulated meetings from real CV22 zh-TW clips. Real Qwen2.5 152064 BPE tokenizer used in text-distill. Tests: `test_normalize_zhtw`, `test_dialogue_scripts`, `test_vocab_coverage`. |
| **6 GB-RAM phone** | ✅ | ✅ | ✅ | **MEASURED** total mobile stack ≈ **2.06 GB @ 8k ctx** (1.27 GB packed int4 weights + 0.235 GB KV, both measured; +0.40 encoder +0.15 overhead est). Fits 6 GB with ~4 GB spare. `runtime.ram_budget.estimate_ram`; export gate caps 2.4 GB. See `docs/MOBILE_BENCHMARK.md`. |
| **Diarization** (speaker attribution) | ✅ | ✅ | ◑ | `Speaker` field in structured output; `speaker_stitch` (Hungarian permutation match on overlap), `speaker_registry`, `runtime.embeddings` (`MfccStatsEmbedder` / ONNX ECAPA). Metric `eval.der`, `eval.cpwer`, `eval.consistency`. Validated on simulated mixtures with exact overlap labels; not on real gold diarization. |
| **Timestamps** | ✅ | ✅ | ◑ | `Start`/`End` fields; `eval.timestamps` (MAE, gate ≤ 0.30 s). Structured-target density ≈ 10.75 tok/s of audio (measured over 30 real records). Emitted by `data.manifest.format_target`; validated as a format, not against real audio-aligned refs. |
| **Transcript** (content) | ✅ | ✅ | ◑ | `Content` field; `eval.mer` (zh chars + en words, gate ≤ 0.12). Format validated; WER quality **blocked** on real distilled weights (current student is SMOKE). |
| **Hotwords** | ✅ | ✅ | ◑ | Hotword context injection in `data.pseudo_label.label_file(..., hotwords=...)` and the `ChunkedTranscriber` labeler contract. Tested `test_pseudo_label`. Not validated against real teacher inference (needs `vibevoice`). |

◑ = design + code + unit tests in place, but final validation blocked on real
audio path, gold data, or real distilled weights.

---

## 2. Pipeline script order (validated-against)

Scripts in `scripts/` are thin config-driven CLIs over `distil_vibevoice`.
Canonical order: `01 → 02 → 03(→03b) → 04 → 05 → 06(06b) → 07(07b/07c) → 08 → 09 → 10(10b) → 11`.

| # | Script | Purpose | Validated against (this box) |
|---|---|---|---|
| 00 | `00_extract_backbone.py` | Extract Qwen2.5-7B LLM backbone from VibeVoice-ASR checkpoint | ✅ produced `models/teacher_llm` (7.616 B real ASR-tuned weights) |
| 01 | `01_download_data.py` | Fetch source corpora + MUSAN/RIRs | ✅ fetched CV22 zh-TW TEST shard (136 MB tar, 5087 mp3, 866 speakers ≥3 clips) into `data/raw/common_voice_zhtw` |
| 02 | `02_generate_scripts.py` | Template/grammar zh-TW⇄en meeting scripts | ✅ 200 scripts → `data/scripts/scripts.jsonl`, all code-switch, 2–8 speakers, avg 40.2 turns |
| 03 | `03_synth_tts_meetings.py` | VibeVoice-TTS synthetic meetings + augmentation | ⛔ **blocked**: needs `vibevoice` pkg (TTS) |
| 03b | `03b_build_simulated.py` | Build simulated meetings from real decoded clips (no TTS) | ✅ decoded 438 real mp3→24k mono; 30 simulated meetings (0.54 h) with exact labels |
| 04 | `04_pseudo_label.py` | 8B teacher labels real audio (hotword ctx, confidence filter, s2twp) | ⛔ **blocked**: needs `vibevoice` pkg (teacher audio forward) |
| 05 | `05_build_manifests.py` | Merge buckets, dedupe vs eval, split, hours table | ✅ manifest + augment paths exercised end-to-end |
| 06 | `06_prune_4b.py` | Minitron importance + width-prune 8B→4B | ◑ recipe implemented + unit-tested (`test_pruning`); full 4B run not executed |
| 06b | `06b_prune_smoke.py` | Smoke prune to target geometry | ✅ produced `student_1p5b_smoke` |
| 07 | `07_distill_4b.py` | Stage-1 distill (KL+CE+hidden-MSE, 4× speaker/ts) | ◑ trainer unit-tested (`test_losses`, `test_collator`); full run not executed |
| 07b | `07b_distill_smoke.py` | Smoke distill (random tokens) | ✅ produced `distill_smoke_out` / tied smoke |
| 07c | `07c_textdistill_real.py` | Real-tokenizer text distillation | ✅ 30 steps, loss 9.66→2.83, finite, no OOM; **fixed** `DistillCollator._collect_special_ids` bug → `models/textdistill_real_out` |
| 08 | `08_prune_1p5b.py` | Prune 4B→1.5B (hidden 1536, 12Q/2KV) | ◑ recipe unit-tested; ran on smoke path to produce tied 1.544 B geometry |
| 09 | `09_distill_1p5b.py` | Stage-2 distill (4B TA + 10% direct-8B) | ◑ implemented + unit-tested; full run not executed |
| 10 | `10_qat_export.py` | torchao int4/int8 finalize + `estimate_ram` gate (>2.4 GB → exit 1) | ◑ implemented; export gate logic tested |
| 10b | `10b_export_benchmark.py` | Measure int4 footprint + tok/s + bit-pack to disk | ✅ **measured**: 1.27 GB packed int4, KV 0.235 GB @ 8k, 60.5 GPU / 1.05 CPU-proxy tok/s → `models/student_1p5b_int4_packed/packed_int4.pt` |
| 11 | `11_evaluate.py` | ChunkedTranscriber over eval manifest → `run_gates` (exit 1 on fail) | ◑ gate machinery unit-tested (`test_gates`); needs real weights + gold eval set |

---

## 3. Remaining critical path (with owners)

**Owner = me (this agent / on this box), given the right installs:**
1. Install the `vibevoice` package → unblock the audio path (σ-VAE latent
   splicing into the student input in `distill.collator`, TTS synthesis `03`,
   teacher pseudo-labeling `04`). This is the single biggest blocker.
2. Once audio path is live: run a real distill cascade (06→07→08→09) over real
   data to replace SMOKE weights with quality weights.
3. Run `10_qat_export.py` + `11_evaluate.py` for real MER/cpWER/DER/timestamp
   gate numbers.
4. Build the `llama-quantize` binary and execute the documented GGUF F16→Q4_K_M
   conversion (converter already verified against this Qwen2 arch; tokenizer in
   place) for a shippable ~1.0–1.1 GB q4 artifact.

**Owner = user (needs hardware / data we cannot synthesize):**
5. Record / source **~30–50 h of gold human-verified zh-TW meetings** for the 5%
   gold bucket and final eval — cannot be synthesized.
6. Provide a **real 6 GB-RAM phone / ARM device** to measure true packed-int4 NPU
   tok/s, encoder RAM & latency, and sustained-clock thermals over an hours-long
   meeting. (All on-device tok/s here are x86 CPU proxies only.)
7. Confirm licensing/attribution for any redistributed CC-BY / CC-BY-SA sources.

---

## 4. Key artifacts on disk

| Path | What | Status |
|---|---|---|
| `models/teacher_llm` | 7.616 B Qwen2ForCausalLM ASR backbone | REAL weights |
| `models/student_1p5b_tied_smoke` | 1.544 B tied-embedding student | SMOKE (real geometry/size/speed) |
| `models/student_1p5b_int4_packed/packed_int4.pt` | bit-packed int4 blob, 1.270 GB | REAL measured deployment size |
| `models/tokenizer` | Qwen2.5 152064-vocab BPE | REAL |
| `data/scripts/scripts.jsonl` | 200 code-switch multi-speaker scripts | REAL |
| `data/raw/common_voice_zhtw` | CV22 zh-TW test shard (5087 clips) | REAL |
| 30 simulated meetings (0.54 h) | from 438 real decoded clips, exact labels | REAL |

See `docs/MOBILE_BENCHMARK.md` (measured int4 footprint & speed) and
`docs/DATA_PLAN.md` (data buckets & sources).
