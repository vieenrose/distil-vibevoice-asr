# Data Acquisition Plan — zh-TW/en Meeting-ASR Distillation

Verified 2026-07-04 against the Hugging Face API + repo trees. **Every HF source
below is PUBLIC and UNGATED** — verification found **no dead or gated repos**, so
no repo-id replacements were needed. The edits to `configs/data.yaml` and
`scripts/01_download_data.py` reconcile *notes, scoping, and a latent bug* with
that reality (see "Config/script reconciliation" at the bottom).

Free disk on the target volume: **137 GB** (`/`, 83% used of 784 GB).

---

## 1. Source table

| Source | HF id (config/scope) | Mix role | Usable hours | License → obligation | Disk (scoped) | Gated? | Download command |
|---|---|---|---|---|---|---|---|
| common_voice_zhtw | `fsicoli/common_voice_22_0` (`zh-TW`) | pseudo, **tts_voice_bank**, simulated | ~30h validated read speech (~50k clips) | CC0-1.0 → **none** (public domain) | ~1.8 GB | No | `hf download fsicoli/common_voice_22_0 --repo-type dataset --include 'audio/zh-TW/*' 'transcript/zh-TW/*' --local-dir data/raw/common_voice_zhtw` |
| yodas_zh | `espnet/yodas2` (`zh000`) | pseudo | ~500h raw → **~200h TW** after Traditional filter | CC-BY-3.0 → **attribution** | ~60 GB | No | `hf download espnet/yodas2 --repo-type dataset --include 'data/zh000/*' --local-dir data/raw/yodas_zh` |
| yodas_en | `espnet/yodas2` (`en000`) | pseudo (English side) | ~400h | CC-BY-3.0 → **attribution** | ~69 GB | No | `hf download espnet/yodas2 --repo-type dataset --include 'data/en000/*' --local-dir data/raw/yodas_en` |
| ivod_meta | `openfun/tw-ly-ivod` | pseudo (index) | **0h bundled** (metadata/transcript index; ~1000h+ LY audio fetched separately) | CC-BY-4.0 → **attribution** | ~0.85 GB (jsonl) | No | `hf download openfun/tw-ly-ivod --repo-type dataset --local-dir data/raw/ivod_meta` |
| ivod_fine_tune | `openfun/ivod-fine-tune` | **tts_voice_bank**, simulated, gold(transcripts) | ~130h real zh-TW (speaker-labeled sentence segments) | CC-BY-4.0 → **attribution** | ~18.6 GB+ | No | `hf download openfun/ivod-fine-tune --repo-type dataset --local-dir data/raw/ivod_fine_tune` |
| ascend | `CAiRE/ASCEND` | pseudo, simulated | ~10.6h zh-en code-switch | **CC-BY-SA-4.0 → share-alike** | ~1.2 GB (all splits) | No | `hf download CAiRE/ASCEND --repo-type dataset --local-dir data/raw/ascend` |
| ami | `edinburghcstr/ami` (`ihm/*`) | gold (**English only**) | ~100h diarized English meetings | CC-BY-4.0 → **attribution** | ~24.5 GB (ihm); +~17 GB sdm | No | `hf download edinburghcstr/ami --repo-type dataset --include 'ihm/*' --local-dir data/raw/ami` |
| musan | openslr.org/17 | augmentation (noise) | n/a | CC-BY-4.0 → attribution | ~11 GB | No | via `01_download_data.py --only musan` |
| rirs_noises | openslr.org/28 | augmentation (reverb) | n/a | Apache-2.0 → notice | ~2 GB | No | via `01_download_data.py --only rirs_noises` |
| **gold_zhtw_selfrecord** | — (not on HF) | **gold (zh-TW)** | **0h — must self-record ~30-50h** | owned | ~5-10 GB (planned) | — | in-house recording + human diarization/transcription |

**Totals**

- Usable audio hours (all fetchable sources): **~870h** (~30 CV + ~200 YODAS-zh(TW) + ~400 YODAS-en + ~130 IVOD-ft + ~10 ASCEND + ~100 AMI). Plus ivod_meta's ~1000h+ LY audio if the videos are pulled from LY IVOD separately.
- If **everything is stored uncompressed at once**: ≈ 1.8 + 60 + 69 + 0.85 + 18.6 + 1.2 + 24.5 + 11 + 2 ≈ **~189 GB > 137 GB free — does NOT fit.**

### Store vs stream (fit into 137 GB)

**Store persistently (~59 GB, reused across the whole pipeline):**
`common_voice_zhtw` (1.8), `ivod_fine_tune` (18.6 — voice bank + gold transcripts),
`ivod_meta` (0.85), `ascend` (1.2), `musan` (11), `rirs_noises` (2), plus AMI-ihm
(24.5) for the English gold / diarization-pipeline validation.

**Stream or stage shard-by-shard, delete after pseudo-labeling (never all resident):**
`yodas_en` (69 GB) and `yodas_zh` (60 GB). Both ship a loading script (no HF viewer),
so pull one `.tar.gz` shard, run `04_pseudo_label.py`, write the manifest, delete the
shard, repeat. Peak transient footprint ≈ one shard (~1.5 GB) + features.

### Staged download order

1. **Small essentials first (~35 GB):** `common_voice_zhtw`, `ivod_fine_tune`,
   `ivod_meta`, `ascend`, `musan`, `rirs_noises`. Unblocks voice bank, TTS synthesis,
   simulation, and augmentation.
2. **AMI-ihm (~24.5 GB):** English gold + diarization-pipeline validation.
3. **YODAS en000 then zh000 — streamed shard-by-shard:** pseudo-label and discard;
   never leave the full 60/69 GB resident.

---

## 2. Mix-ratio mapping and the GAP

`data.yaml mix_ratios` are proportions of the total training pool. Anchoring on a
**1000h** target corpus:

| Bucket | Ratio | Target hrs | Fed by | Available now | Gap |
|---|---|---|---|---|---|
| tts_synthetic | 0.40 | **400h** | VibeVoice-TTS **generated** (voice bank = CV zh-TW client_ids + IVOD-ft speakers) | 0h generated (voice-bank material present) | **Generate 400h** via `03_synth_tts_meetings.py` — compute-bound, not data-bound |
| pseudo_labeled | 0.40 | **400h** | YODAS-zh(TW) ~200 + YODAS-en ~400 + CV ~30 + ASCEND ~10 + IVOD-meta LY audio | **~640h+ raw** (surplus) | **None — surplus.** Cap/curate down to 400h |
| simulated_mixtures | 0.15 | **150h** | `simulate_meetings` over single-speaker clips (CV zh-TW, IVOD-ft, ASCEND) | 0h built; abundant source clips | **Generate 150h** via `05`/simulate — compute-bound |
| gold | 0.05 | **50h** | real verified **zh-TW** diarized meetings | **0h real zh-TW** | **~30-50h short — must SELF-RECORD** |

**The one hard data gap: gold zh-TW meetings.**
No public corpus supplies real multi-speaker *diarized* zh-TW meeting audio:
- **AMI** is ~100h but **English** — usable for diarization-pipeline validation and
  English-side gold, it does **not** count toward the zh-TW gold bucket.
- **IVOD fine-tune** is genuine zh-TW LY speech but shipped as **single-speaker
  sentence segments** (no overlap/turn structure) — great as gold *transcripts* +
  TW-accent voice bank + simulate-mixture source, but **not** diarized meeting gold.

→ **Action: self-record + human-annotate ~30-50h of real zh-TW meetings**
(placeholder source `gold_zhtw_selfrecord` added to `data.yaml`). Until then, run the
gold bucket on AMI(en) for pipeline shakeout and treat zh-TW gold as the release blocker.

Secondary "gaps" are generation/compute, not acquisition: tts_synthetic (400h) and
simulated (150h) must be produced by the pipeline from material already on disk.

---

## 3. License obligations for RELEASE

**Model weights (distilled 4B → 1.5B):** trained on a mixed pool. Model *weights*
are generally an unencumbered artifact under these dataset licenses, but ship a
**DATA_CARD listing every source + license** and give **attribution** (CC-BY: YODAS,
IVOD ×2, AMI). CC0 (Common Voice) requires nothing.

**Any released DERIVED DATASET (manifests, pseudo-labels, TTS/simulated mixes that
embed source audio or transcripts):**
- **ASCEND is CC-BY-SA-4.0 (share-alike).** Any redistributed dataset that
  **contains ASCEND-derived audio/transcripts must itself be CC-BY-SA-4.0** and carry
  attribution. This is viral over the *dataset*. **Mitigation:** keep ASCEND-derived
  records in a *separately-licensed CC-BY-SA shard*, or exclude ASCEND from any
  permissively-licensed dataset release, so the share-alike term does not force the
  whole corpus to CC-BY-SA.
- **CC-BY sources (YODAS CC-BY-3.0; IVOD tw-ly-ivod + ivod-fine-tune CC-BY-4.0; AMI
  CC-BY-4.0):** redistribution requires **attribution + license notice + indication
  of changes** (s2twp normalization, teacher re-labeling, segmentation). No
  share-alike.
- **CC0 (Common Voice zh-TW):** no obligation.
- **YODAS caveat:** labels are YouTube captions and audio is user-uploaded; CC-BY-3.0
  is the *dataset* license — redistributing the raw audio carries the usual
  YouTube-provenance risk. Safer to release **pseudo-label manifests (ids + text +
  timestamps)** rather than the audio.
- **Self-recorded zh-TW gold:** you own it — license it deliberately (recommend CC-BY
  to match, and secure recorded-consent from meeting participants).

**Practical release recipe:** (1) permissive corpus/model = CC0 + CC-BY sources, with
a DATA_CARD + attributions, **excluding ASCEND audio**; (2) a *separate* CC-BY-SA
shard for anything ASCEND-derived; (3) prefer manifests-over-audio for YODAS/IVOD to
sidestep third-party media redistribution.

---

## Config/script reconciliation (done)

`configs/data.yaml`
- Added a verification banner (all sources public+ungated, 2026-07-04) and per-source
  reality notes: CV22 legacy `.py` loader breaks the viewer + `load_dataset()` on
  datasets>=3.0 (fetch by path); YODAS configs have no viewer + Traditional filter;
  **ivod_meta is a metadata/transcript index — audio not bundled**; ivod_fine_tune is
  single-speaker segments (not diarized); AMI is English-only gold.
- Added placeholder source **`gold_zhtw_selfrecord`** (hf_id null) marking the ~30-50h
  self-record gap for the zh-TW gold bucket.

`scripts/01_download_data.py`
- **Fixed the latent marker bug:** `.download_complete` is now written only if
  `_has_payload(dest)` finds ≥1 real file, so an include glob that matches nothing no
  longer marks a source "complete" and wrongly skips a later correct rerun.
- Scoped the **AMI** default to `include=['ihm/*']` (~24.5 GB instead of ~84 GB whole
  repo); documented adding `sdm/*` for far-field.
- Skip sources with no downloadable id (the self-record placeholder) cleanly instead
  of failing.

Tests: `pytest tests/ -q` → **235 passed** (CUDA disabled).
