---
title: zh-TW Transcriber (100% local)
emoji: 🔒
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
models:
  - Luigi/moss-transcribe-diarize-zhtw
short_description: Private on-device zh-TW/en speech-to-text in your browser
---

# zh-TW / English transcription — fully in your browser

[Luigi/moss-transcribe-diarize-zhtw](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw)
(0.9B) quantized to q4/int8 ONNX (~840 MB) and executed **entirely client-side**
with onnxruntime-web (WebGPU when available, wasm otherwise). Record from the
microphone or drop a short file (≤ 2 min): the timestamped, speaker-tagged,
Traditional-Chinese transcript streams out token by token. No audio or text
ever leaves the device.

Measured against the full bf16 fine-tune on held-out real meeting audio, the
quantized graphs score **MER 0.068** (same OpenCC s2tw output normalization on
both). The identical exported graphs power the
[sherpa-onnx C++ port](https://github.com/vieenrose/sherpa-onnx/tree/feature/moss-transcribe-diarize)
for phone deployment.

For hour-long meetings, use the
[ZeroGPU Space](https://huggingface.co/spaces/Luigi/zh-tw-meeting-transcriber-live)
(chunked windows + cross-window speaker linking) or the
[instant results viewer](https://huggingface.co/spaces/Luigi/zh-tw-meeting-transcriber).
