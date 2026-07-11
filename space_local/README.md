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
  - speechbrain/spkrec-ecapa-voxceleb
short_description: Meetings to transcripts with speakers — 100% in your browser
---

# zh-TW / English meeting transcription + speaker diarization — fully local

Speaker-attributed, timestamped transcription of Traditional-Chinese / English
meetings, running **entirely in your browser on CPU**. Your audio and the
transcript never leave the device — close the tab and they're gone.

Model: [Luigi/moss-transcribe-diarize-zhtw](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw)
(0.9B: Whisper-Medium encoder + Qwen3-0.6B decoder, fine-tuned for zh-TW
meetings), exported to int8 ONNX and executed with
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) inside a Web
Worker so the page never freezes. ~1.1 GB one-time download, cached by the
browser. Multi-threaded when the browser allows cross-origin isolation.

## What it does

- **Any-length audio.** Drop a meeting file (any length) or record from the mic
  (≤ 2 min). Audio is transcribed in windows; every segment is voice-
  fingerprinted with [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
  and one global clustering pass keeps speaker labels consistent across the
  whole meeting (S01 at minute 3 = S01 at minute 130).
- **Streaming.** The timestamped, speaker-tagged transcript appears as it
  decodes, with a live activity indicator so you can always tell it's working.
- **Written form.** Output is normalized deterministically — OpenCC `s2tw`
  (Traditional script) + conservative inverse text normalization
  (二十三→23, 百分之五十→50%, guarding idioms). Recognition is unchanged.
- **Export.** Download the result as SRT or JSON; click any line to seek the
  audio.

**Speaker-linking window** (selectable): 1.5 min (Fast) or 3 min (Best,
default). Longer windows give the model more context and slightly better
speaker consistency on very long meetings, at the cost of slower decode.

## Accuracy

Int8 vs the full bf16 fine-tune on held-out real meeting audio (script-
normalized MER): **0.03–0.04**. Diarization validated against pyannote
references on real meetings: DER **0.056** on a 30-min session, speaker
consistency **0.905** on a 123-min session.

Speed is CPU-bound: short clips are seconds; an hour of audio is many minutes
of browser compute, streamed throughout and stoppable at any time (partial
results kept). For fully-stable long-meeting diarization on device, the same
model runs natively via a
[sherpa-onnx C++ port](https://github.com/vieenrose/sherpa-onnx/tree/feature/moss-transcribe-diarize).

## Examples

Real Taiwan Legislative Yuan meetings (立法院 IVOD open data, **CC-BY-4.0**, via
[openfun/tw-ly-ivod](https://huggingface.co/datasets/openfun/tw-ly-ivod)),
including a full 2-hour committee session with precomputed results — instant
synchronized playback. Machine-generated transcripts may contain errors.
