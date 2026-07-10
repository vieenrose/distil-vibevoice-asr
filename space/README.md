---
title: zh-TW Meeting Transcriber
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.9.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
models:
  - Luigi/moss-transcribe-diarize-zhtw
  - speechbrain/spkrec-ecapa-voxceleb
short_description: 2+ hour zh-TW/en meetings — transcript, speakers, timestamps
---

# zh-TW / English meeting transcription + speaker diarization

Transcribes **hours-long Traditional-Chinese / English code-switched meetings**
with speaker labels and timestamps, using
[Luigi/moss-transcribe-diarize-zhtw](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw)
(0.9B, fine-tuned from OpenMOSS-Team/MOSS-Transcribe-Diarize).

**How long meetings are handled** (the same design used for on-device
deployment): the audio is split into 5-minute windows, each window is
transcribed in one pass (speaker tags are only consistent *within* a window),
then every segment is embedded with ECAPA-TDNN and a single global
agglomerative clustering pass links the speakers *across* windows into one
consistent set — `S01` in minute 3 is the same voice as `S01` in minute 130.

The bundled example is a real Taiwan Legislative Yuan meeting
(立法院 IVOD open data, **CC-BY-4.0**, via
[openfun/tw-ly-ivod](https://huggingface.co/datasets/openfun/tw-ly-ivod)) with
precomputed results so the full-meeting output shows instantly.
