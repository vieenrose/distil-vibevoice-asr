# Integration note — MOSS-TD q4 update (2026-07-23)

For downstream applications embedding MOSS-Transcribe-Diarize via the
RapidSpeech.cpp (ggml) engine or the LiteRT port. Two changes to pick up.

## 1. New q4 weights: silence-robust `v2` (drop-in, recommended)

The v1 q4 decoder had a knife-edge failure on silent / unvoiced / near-silent
audio: instead of stopping, it free-ran garbage to the token budget (ggml:
`[0.00][S01][0.06]…` marker loops costing minutes of decode; LiteRT int4:
English meta-hallucinations like "the audio contains wind blowing…" inserted
into the transcript). `v2` fixes this in the weights (QAT), not in wrapper
logic — no client-side silence gating needed, and none should be added.

- ggml: `Luigi/moss-transcribe-diarize-zhtw-gguf` → `moss-transcribe-base-q4mix-v2.gguf`
- LiteRT: `Luigi/moss-transcribe-diarize-litert` → `moss_td_decoder_v2_q4b32_ekv2560.tflite`
  (encoder/embedder files unchanged — swap the decoder only)

Same file size, format, tensor layout, and API as v1 — a pure filename swap.
Behavior deltas vs v1, from the release validation: silent input now yields an
empty transcript in seconds; zh golden agreement 89.3→96.7; 5-meeting WER and
speaker accuracy statistically unchanged; en single-pass golden −1.5 pt
(97.2 vs 98.7) is the one known cost. If your app treats "empty transcript"
as an error, treat it as the valid no-speech result instead.

## 2. LiteRT integrators: set the CPU thread count EXPLICITLY

`CompiledModel.from_file(path)` without options lets the runtime pick the
thread count — and it picks very low (measured: decode 9.3 tok/s vs 20.1 with
threads set; 8.5× slower in one production host case). Always pass:

```python
from ai_edge_litert.cpu_options import CpuOptions
from ai_edge_litert.options import Options
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

opts = Options(hardware_accelerators=HardwareAccelerator.CPU,   # single enum, despite the plural name
               cpu_options=CpuOptions(num_threads=N))
cm = CompiledModel.from_file(path, options=opts)                # options XOR hardware_accel kwarg — not both
```

Choosing N: on x86 servers use the container's usable core count. On ARM
big.LITTLE use the BIG-core count only — measured on an Exynos 1280:
398 ms/token at 4 big-core threads vs 849 ms at 8 threads (little-core
contention doubles the cost). More threads is not faster on phones.
(The classic `Interpreter(num_threads=…)` path never had this problem.)

## Recommended engine flags (ggml path, validated)

`MT_KV_F16=1 MT_KV_EVICT_S=45` — near-lossless, large decode speedup on long
audio. Batched multi-window decode (`mtd_stream_transcribe_batch`, batch 2–8)
is byte-identical on x86 and near-lossless on ARM; it helps file/batch
processing only — do not use it to chase realtime latency.

Engine sources: `vieenrose/RapidSpeech.cpp` (branch `moss-pure`) and
`vieenrose/LiteRT` (branch `moss-td-port`, `litert/samples/asr/moss_td/`).
Reference deployments of both: HF Spaces `Luigi/moss-transcribe-diarize-cpp`
and `Luigi/moss-transcribe-diarize-litert`.
