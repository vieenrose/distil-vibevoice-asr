---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Transcribe-Diarize
language:
- zh
- en
tags:
- automatic-speech-recognition
- speaker-diarization
- timestamps
- zh-tw
- traditional-chinese
- code-switching
- meeting-transcription
pipeline_tag: automatic-speech-recognition
---

# MOSS-Transcribe-Diarize — zh-TW meeting fine-tune (v4)

A fine-tune of [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
(0.9B: Whisper-Medium encoder + Qwen3-0.6B decoder) for **Taiwanese-Mandarin /
English code-switched meeting transcription** with speaker diarization and
timestamps in a single pass.

What the fine-tune changes vs the base model:

- **Native Traditional Chinese output** (Taiwan conventions: 軟體/品質/影片…) —
  the base model outputs Simplified.
- **Taiwan business-meeting register**, including natural intra-sentential
  code-switching (「今天的 agenda 是主管職的 offer，跟大家 update 一下」).
- Improved recognition of common meeting vocabulary (deadline, KPI, sprint,
  quota, pipeline, review, …).

Output format is unchanged from the base model — a compact stream of
`[start][Sxx]text[end]` segments (speaker-attributed, timestamped).

**Written-form post-processing** (deterministic, recognition unchanged): apply
OpenCC `s2tw` for Traditional script + a conservative ITN pass for numbers
(二十三→23, 百分之五十→50%, guarding idioms like 千萬). Quantized ONNX graphs
(int8 / q4 / q4+fp16-KV mobile) for on-device use:
[Luigi/moss-transcribe-diarize-zhtw-onnx](https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-onnx).

## Results

Held-out evaluation on synthetic zh-TW/en multi-speaker business meetings
(exact reference labels; MER = mixed error rate, zh per-character / en per-word):

| Model | MER ↓ | DER ↓ (synthetic) | DER ↓ (real 123-min) |
|---|---|---|---|
| Base MOSS-Transcribe-Diarize | 0.395* | 0.53 | 0.74 |
| v2 fine-tune | 0.183 | 0.35 | 0.180 |
| **v4 (this)** | **0.18** | **0.34** | **0.195** |

v4 adds ~56 h of real far-field meeting audio (Taiwan Legislative Yuan IVOD,
whisperx×pyannote fused labels with a speaker-coverage purity filter) on top of
the synthetic corpus, trained in 300 s windows to match long-meeting inference.
It keeps v2's diarization quality on real 2-hour meetings (consistency 0.905)
while improving robustness to far-field audio.

\* much of the base-model gap is the Simplified↔Traditional script mismatch;
its underlying recognition is strong.

## Usage

Identical to the base model (custom code):

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
# helper package: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages, generate_transcription, resolve_device,
)

model_id = "Luigi/moss-transcribe-diarize-zhtw"
device = resolve_device("auto")
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype="auto"
).to(torch.bfloat16).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = build_transcription_messages("meeting.wav")
result = generate_transcription(model, processor, messages,
                                max_new_tokens=2048, do_sample=False,
                                device=device, dtype=torch.bfloat16)
for seg in parse_transcript(result["text"]):
    print(seg.start, seg.end, seg.speaker, seg.text)
```

## Training data

Fine-tuned (SFT, CE on assistant tokens) on ~3,000 synthetic multi-speaker
zh-TW/en business meetings (~270 h) generated with
[microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) TTS using
Taiwanese-accented voice prompts from Common Voice zh-TW, from template-generated
code-switched dialogue scripts (5 business domains). Labels (transcript, speaker,
timestamps) are exact by construction. Two rounds: 400 steps @ 1e-5, then
2,000 steps @ 5e-6, 120 s audio windows, bf16, single RTX 5090.

## Limitations — read before use

- **Evaluated on synthetic audio only.** No real-meeting benchmark yet; expect
  higher error on real far-field/noisy recordings.
- Trained register is business meetings with ≤4 speakers and modest overlap;
  heavy overlap or many speakers will degrade diarization.
- The base model's diarization weakens on far-field many-speaker audio
  (e.g. parliamentary recordings).
- Transcription only: the model does not summarize or title meetings (attempts
  to fine-tune those tasks onto the 0.6B decoder degraded output quality).
- Timestamps inherit base-model resolution; segment boundaries on rapid
  turn-taking may merge or split turns.

## Acknowledgements

- Base model: [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (Apache-2.0)
- TTS for training data: [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) (MIT)
- Voice prompts: Mozilla Common Voice zh-TW (CC0)
