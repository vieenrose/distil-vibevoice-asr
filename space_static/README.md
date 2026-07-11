---
title: zh-TW Meeting Transcriber
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
license: apache-2.0
models:
  - Luigi/moss-transcribe-diarize-zhtw
  - speechbrain/spkrec-ecapa-voxceleb
short_description: 2+ hour zh-TW/en meetings — transcript, speakers, timestamps
---

# zh-TW / English meeting transcription + speaker diarization

Interactive demo of
[Luigi/moss-transcribe-diarize-zhtw](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw)
(0.9B, fine-tuned from OpenMOSS-Team/MOSS-Transcribe-Diarize) on **real 2+ hour
Taiwan Legislative Yuan meetings**: synchronized transcript with speaker colors,
click-to-seek, per-speaker talk time, search, SRT/JSON export.

Long meetings are handled by transcribing **5-minute windows** and linking
speakers across windows with ECAPA-TDNN embeddings + one global agglomerative
clustering pass (validated on real meetings: DER 0.056 on a 30-min session,
speaker consistency 0.912 on a 123-min session).

The results shown were precomputed with exactly that pipeline. To run it live on your own
audio, use the [fully-local browser demo](https://huggingface.co/spaces/Luigi/zh-tw-transcriber-local)
or the Python snippet on the
[model card](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw#usage).

Example audio: 立法院 IVOD open data, **CC-BY-4.0**, via
[openfun/tw-ly-ivod](https://huggingface.co/datasets/openfun/tw-ly-ivod).
