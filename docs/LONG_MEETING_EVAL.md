# Real Long-Meeting Diarization Eval (IVOD)

**Meeting:** IVOD 16530 — 立法院第11屆第3會期內政委員會第4次全體委員會議 (2025-03-19),
first 60 min of a 156-min far-field parliamentary session, 24 kHz mono.
**Reference:** the catalog's pyannote diarization (a weak teacher-label reference, **not** gold):
309 turns / **19 distinct speakers** in the scored 60 min.

## Headline result

| Config | Global speakers | speaker_consistency ↑ | DER ↓ |
|---|---|---|---|
| FULL (registry + consolidation) | 2 (1 speech) | **0.295** | **0.740** |
| NO-CONSOLIDATE (registry only) | 2 | — (same output) | — |
| LEGACY (stitch only) | not scored (GPU-contention OOM; superseded by probe) | — | — |

The pipeline collapsed a 19-speaker meeting to ~1 speech speaker.

## Root cause — the TEACHER, not the post-processing

Per-window probe of the raw teacher (VibeVoice-ASR) on three 5-min speech windows:

| Window | Distinct speakers emitted by teacher |
|---|---|
| 29–34 min | 2 |
| 40–45 min | 2 |
| 50–55 min | 1 |

The teacher emits **1–2 speakers per window** on this audio, vs ~19–24 in the reference.
FULL and NO-CONSOLIDATE produce identical speaker counts, so **consolidation is not the
collapser** — the chunk-and-recombine machinery faithfully reflects teacher output that has
≤2 speakers/window. The synthetic multi-hour test (clean concatenated speech) passes; on the
clean `sim_meeting_0000` clip the teacher diarizes multiple speakers correctly. The failure is
specific to **far-field, many-speaker, rapid-turn parliamentary audio** — a condition outside
VibeVoice-ASR's tuned envelope (its model-card examples are clean 2-speaker podcasts).

## Implications

1. **IVOD parliamentary is HARDER than the product target.** Business meetings (2–10 speakers,
   closer mics, clearer turns) sit much nearer the teacher's comfort zone; validate diarization
   there, not on parliamentary far-field.
2. **Use IVOD's pyannote labels for diarization supervision, the teacher for ASR text.** On hard
   audio pyannote (19–24 spk) is far better than the teacher (1–2 spk). Distilling the teacher's
   diarization would inherit its ceiling; pairing teacher-text + pyannote-speaker labels is stronger.
3. If far-field many-speaker is a real target, the architecture likely needs a dedicated diarizer
   fused with the ASR, not distilled VibeVoice diarization alone.

## Operational notes found

- **~26 min of leading dead air** in the IVOD capture (`[Noise]`/`[Silence]`); needs VAD trimming
  before labeling to avoid wasting windows.
- **Runtime transcribe path does not apply OpenCC** — output here was Simplified (报告 not 報告).
  `normalize_zhtw` runs only in the pseudo-label path; the runtime `ChunkedTranscriber` should
  normalize too for the product.
- **RAM measurement was unreliable** (GPU contention from a leaked process; constant ~3.6 GiB at
  both 5- and 15-min windows, n_segments=1). The encode-activation figure needs a clean,
  single-tenant re-measurement before it's treated as a hard mobile constraint.

## Verdict (parliamentary)

The **post-processing does not invent or collapse speakers beyond what the teacher provides.** The
open risk here is **teacher diarization quality on hard acoustic conditions** — a data/domain-scoping
decision, not a pipeline bug.

---

# Synthetic clean-audio eval (scripts/11c) — the loop-closing test

Ran the same machinery on CLEAN synthetic meetings built from distinct Common Voice zh-TW speakers
(**exact ground-truth labels**).

## Part A — short meetings, single window (teacher raw diarization)

| Meeting | true spk | hyp spk | consistency ↑ | DER ↓ |
|---|---|---|---|---|
| sim_0006 (101s) | 6 | 7 | 0.977 | 0.147 |
| sim_0018 (91s)  | 6 | 6 | 0.994 | 0.117 |
| sim_0021 (99s)  | 6 | 7 | 0.929 | 0.180 |
| sim_0022 (101s) | 6 | 6 | 0.993 | 0.072 |
| sim_0027 (111s) | 6 | 6 | 0.992 | 0.199 |

**On clean audio the teacher diarizes excellently** (consistency 0.93–0.99). Confirms the parliamentary
collapse was an acoustic-envelope limit of the teacher, and that the eval metric is sound.

## Part B — one 18.3-min recurring-speaker meeting, forced 120 s windows (recombination test)

| Config | true spk | hyp spk | consistency ↑ | DER ↓ |
|---|---|---|---|---|
| FULL (registry + consolidation) | 5 | **1** | 0.197 | 0.829 |
| NO-CONSOLIDATE (registry only)  | 5 | **1** | 0.200 | 0.828 |
| LEGACY (stitch only)            | 5 | **15** | 0.293 | 0.769 |

**Cross-window recombination fails on real speech** even though per-window diarization is excellent:
registry-based configs COLLAPSE to 1 speaker; text-only stitching OVER-FRAGMENTS to 15.

## Root cause — the placeholder embedder cannot separate real voices

`MfccStatsEmbedder` on 6 real CV speakers: mean **same-speaker cosine 0.977**, **cross-speaker 0.896**
→ separation only **0.081**, both far above the registry `match_threshold=0.60`. So every segment
matches the first centroid → registry merges everyone. (The synthetic multi-hour unit test passes only
because its `_synth_voice` tones are artificially separable.)

## Resolution — ECAPA embedder + global reclustering (implemented)

Two changes, both validated, make multi-window diarization reliable:

**1. ECAPA-TDNN embedder** (`EcapaEmbedder`, speechbrain `spkrec-ecapa-voxceleb`, 192-dim, in
`runtime/embeddings.py` via `load_embedder('ecapa')`). Gate on exact-labeled real CV speakers:
same-speaker cosine **0.582** vs cross **0.144** → separation **0.438** (5.4× the MFCC 0.081); the
same/cross distributions barely overlap. **ONNX export is the remaining productionization TODO** for
on-device.

**2. Global per-segment reclustering** (`consolidate(..., mode="recluster")`, threshold ~0.7).
The incremental registry (first-match-wins + EMA) is fragile — once it merges two speakers it can't
split them, and mean-based `consolidate` can only merge further. Reclustering the raw per-segment
embeddings globally decouples the final diarization from the incremental assignment.

| Approach (ECAPA, 18-min 5-speaker clean meeting) | consistency ↑ | DER ↓ | speakers |
|---|---|---|---|
| Incremental registry (was shipping) | 0.552 | 0.488 | 3 |
| **`mode="recluster"` @0.7 (now default-available)** | **0.861** | **0.192** | 8→maps to 5 |
| ceiling: global-cluster on clean boundaries | 0.970 | — | — |

So consistency went **0.20 (MFCC) → 0.55 (ECAPA, incremental) → 0.86 (ECAPA + reclustering)**,
approaching the ~0.95 per-window ceiling. Enable in the runtime with
`ChunkedTranscriber(embedder=load_embedder('ecapa'), consolidate_mode='recluster', recluster_threshold=0.7)`.
`recluster_threshold` must match the embedder's separation (~0.7 for ECAPA). The residual 0.86→0.95
gap is teacher segment-boundary noise (mixed/short segments), improvable with per-window-speaker
embedding aggregation.

**Answer to "is there a reliable way?": yes** — ECAPA + global reclustering. Text-only stitching is
not a fallback (over-fragments); the MFCC placeholder is not usable on real speech.
