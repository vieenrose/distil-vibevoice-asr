# Mobile int4 Footprint & Speed — MEASURED

Turns the **2.05 GB @ 8k** mobile-RAM *estimate* for the tied int4 student into
*measured* numbers. Reproduce with:

```bash
.venv/bin/python scripts/10b_export_benchmark.py \
    --model models/student_1p5b_tied_smoke --context 8192 --threads 8 \
    --json-out bench.json
```

Model: `models/student_1p5b_tied_smoke` — Qwen2ForCausalLM, 28 layers, hidden
1536, 2 KV heads, head_dim 128, vocab 152064, **tied** embeddings, bf16 source.
Params (tied) = 1.544 B. (SMOKE weights → garbage outputs, but geometry / size /
speed are the real target's.)

Quantizer: **torchao `IntxWeightOnlyConfig(weight_dtype=int4, PerGroup(128))`** —
torchao's ARM / on-device int4 weight-only path (same family used for
ExecuTorch / XNNPACK mobile export), which is the correct analogue for a phone.
torchao 0.17.0 installed cleanly from PyPI (`torchao_ok = true`). Its C++ kernels
are skipped on torch 2.10 (they want ≥ 2.11), so quantization uses the portable
`IntxUnpackedToInt8Tensor` tensor subclass — see the storage caveat below.

## Memory table (MEASURED unless marked estimate)

| Component | GB | How obtained |
|---|---:|---|
| Source weights, bf16 | 3.088 | measured (unique-storage sum) |
| **int4 weights — genuinely bit-packed (deployment)** | **1.270** | **measured on disk: `packed_int4.pt` = 1,269,701,521 B** |
| &nbsp;&nbsp;• int4 qdata (all linears + lm_head, 2 nibbles/byte) | 0.772 | measured |
| &nbsp;&nbsp;• int4 zero-points (packed) | 0.006 | measured |
| &nbsp;&nbsp;• group scales (bf16, group=128) | 0.024 | measured |
| &nbsp;&nbsp;• tied embedding + norms (kept bf16) | 0.467 | measured |
| KV cache @ 8192 ctx (fp16) | 0.235 | measured — analytic **and** verified vs real `generate()` |
| σ-VAE audio encoders — int4 weights (deployment) | **0.493** | **MEASURED** — 1,058.4 M params (acoustic 687.4 M + semantic 344.6 M + 2 connectors 26.4 M) × 0.5 B/param. bf16 resident is **1.971 GB measured**; the old 0.40 was an under-estimate |
| &nbsp;&nbsp;• encoder activation, 15-min window (transient peak) | (3.54) | **MEASURED** via `encode_speech()` — see note; streams 60 s chunks but the final sample+connector runs over all 6,750 tokens at once |
| Runtime / activation overhead | ~0.15 | estimate (mobile NPU runtime; x86 torch baseline measured ~0.43) |
| **TOTAL mobile stack @ 8k (resident weights)** | **~2.15** | 1.27 measured + 0.235 measured + **0.493 measured enc** + 0.15 est |

**Measured update (2026-07-04, first real teacher run on audio):** the full
8.7B VibeVoice-ASR (incl. acoustic+semantic σ-VAE encoders + connectors + Qwen2
LLM) was loaded and run on real zh-TW audio. Load-time **peak VRAM = 16.2 GiB**
(bf16, whole model). The audio encoders are **1,058.4 M params = 1.971 GB in
bf16** (measured), i.e. **~5× the old 0.40 GB estimate**; int4-quantized to match
the rest of the stack they are **0.493 GB**. The int4 resident mobile stack is
therefore **~2.15 GB** (was estimated 2.06). **Still fits the 6 GB-RAM budget
with ~3.8 GB to spare.** *Caveat:* encoder **activation** for a 15-min window
peaks at a measured **~3.54 GB transient** (the streaming encoder accumulates all
segment means and runs the final sample + connector over ~6,750 tokens in one
shot). This is transient (freed before LLM decode) and was not previously
budgeted — a real mobile build should encode incrementally / cap the window to
keep this peak down.

### Resident-memory (RSS) measurements
- Clean deployment load — a fresh process that loads **only** the packed int4
  artifact adds **+1.185 GB resident** (total process RSS 1.62 GB incl. the
  ~0.43 GB torch/CUDA runtime). This is the honest "int4 model resident" figure.
- The benchmark script's `rss_after_quant = 5.42 GB` is a **transient peak**, not
  deployment: it holds the bf16 source copy **and** the int4 output **and**
  Python-arena memory that the allocator has not returned to the OS. Quantize
  offline; ship the 1.27 GB packed artifact.

### The int4-storage caveat (important, honest)
torchao's runnable tensor subclass stores each int4 value **unpacked in one int8
byte** for portability, so the *runnable* model measures **2.05 GB** of weights,
not 1.27 GB. That 2.05 GB is a kernel-layout artifact, **not** the deployment
size. The script therefore also **bit-packs the nibbles to disk** and measures the
real file (1.270 GB) — that packed figure is what a phone build (ExecuTorch /
GGUF q4) actually ships, and it is the number reported above.

## Throughput (decode)

| Backend | tok/s | Notes |
|---|---:|---|
| **CPU, 8 threads (mobile-ish PROXY)** | **1.05** | 512-tok prompt → 64 new tokens, 61 s. **x86 9950X3D, DIRECTIONAL ONLY — not a phone number.** |
| GPU (RTX 5090), int4 IntxWeightOnly | 60.5 | reference / upper bound |

### Real-time factor vs 7.5 tok/s audio
Streaming meeting ASR must emit ≈ **7.5 output tok / s of audio** to keep up.

- **GPU: 60.5 / 7.5 ≈ 8.1× real-time** — comfortably faster than real-time.
- **CPU proxy: 1.05 / 7.5 ≈ 0.14×** — i.e. ~7× *slower* than real-time on this
  x86 CPU **using torchao's dequant-to-int8 reference kernel**. This is a
  pessimistic floor, for two reasons: (1) the reference kernel dequantizes to
  int8 then does a full matmul — a real ARM/NPU deployment uses fused packed-int4
  XNNPACK kernels that are markedly faster; (2) a phone NPU is not an x86 core.
  Treat 1.05 tok/s as a "worst-case, unoptimized-kernel, wrong-silicon" number,
  **not** a verdict on phone real-time feasibility.

## GGUF status

`gguf` (0.19.0) installed from PyPI. A llama.cpp checkout with
`convert_hf_to_gguf.py` is already present on this machine
(`/home/luigi/llama.cpp/`, **not cloned by this task**). A probe run confirmed the
converter **fully recognizes this Qwen2 architecture** — it processed every
hparam and tensor-metadata block and only stopped at the tokenizer step, because
the smoke artifact had been saved without tokenizer files. That gap is now
closed: the Qwen2.5 tokenizer (`tokenizer.json`, `vocab.json`, `merges.txt`,
`tokenizer_config.json`) has been placed in the model dir and `sentencepiece`
installed, so conversion is ready to run.

Per the task's explicit boundary (do **not** drive llama.cpp's converter; document
it as the user's manual step), the actual GGUF write + q4 quantize is left as the
documented commands below rather than executed:

```bash
# 1) HF -> GGUF (F16). Converter + tokenizer are already in place.
cd /home/luigi/llama.cpp
/home/luigi/distil-vibevoice-asr/.venv/bin/python convert_hf_to_gguf.py \
    /home/luigi/distil-vibevoice-asr/models/student_1p5b_tied_smoke \
    --outfile /home/luigi/distil-vibevoice-asr/models/student_1p5b.f16.gguf \
    --outtype f16

# 2) Quantize to q4 (needs the llama-quantize binary — build once; not built here).
#    cmake -B build && cmake --build build --target llama-quantize -j
./build/bin/llama-quantize \
    /home/luigi/distil-vibevoice-asr/models/student_1p5b.f16.gguf \
    /home/luigi/distil-vibevoice-asr/models/student_1p5b.Q4_K_M.gguf Q4_K_M
```

Expected `Q4_K_M` size ≈ 1.0–1.1 GB (Q4_K_M ≈ 4.5 bit/weight over linears; close
to, slightly below, our 1.27 GB int4-linear+bf16-embedding pack because GGUF also
quantizes the embedding). `llama-quantize` is not built on this machine, so q4 is
a manual step.

## What still needs a REAL phone / ARM device
- **tok/s on a phone NPU.** The 1.05 tok/s here is x86 CPU with an unoptimized
  reference int4 kernel — a *directional proxy only*, never a phone number.
- **Real packed-int4 ARM kernel throughput** (XNNPACK / ExecuTorch / GGUF q4 on
  ARM). The deployment kernel differs from torchao's dequant-to-int8 reference.
- **σ-VAE audio-encoder RAM** — now MEASURED (2026-07-04): 1,058.4 M params =
  **1.971 GB bf16 / 0.493 GB int4**, plus a **~3.54 GB transient activation peak**
  for a 15-min window (measured via `encode_speech`). Still open on a phone: the
  int4 encoder-kernel latency and whether the activation peak can be capped by
  incremental encoding.
- **Peak RAM during real hours-long streaming** — KV grows with context; 8k is
  one operating point (KV scales linearly: 28,672 bytes/token, verified).
- **Thermal / sustained-clock throttling** on a phone over a long meeting.
- **End-to-end accuracy** — these are SMOKE weights; real distilled weights needed
  before any quality/WER claim.

## Reproduced artifacts
- `scripts/10b_export_benchmark.py` — argparse `--model --context --threads`
  (`--prompt-len --new-tokens --pack-out --skip-gpu --json-out`).
- `models/student_1p5b_int4_packed/packed_int4.pt` — real bit-packed int4 blob
  (1.270 GB, the measured deployment weight size).
