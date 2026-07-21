# MOSS-TD regression checklist

Every item below exists because a real defect hid there. The date-stamped notes
say which bug earned the check — none of this is speculative.

The through-line: **three separate defects in one day were invisible to the eval
battery** because each lived in an axis nobody measured (deployment quant,
deployment engine, dense continuous speech). Passing ASCEND and a demo clip
proves very little.

---

## 0. Before trusting any number

| # | Rule | Earned by |
|---|---|---|
| 0.1 | **Evaluate the artifact that ships.** fp-HF numbers do not predict GGUF behaviour. | fp ASCEND rated v7 at 0.102 vs v6.1 0.267; at q4 the real gap was 0.3165 vs 0.3515 — a different conclusion. |
| 0.2 | **Validate the instrument against a reference implementation** before concluding from it. | An entire root-cause investigation (label coverage, v3→v4 filter history, lineage table) explained an artifact that existed only in the ggml path. HF had the contradicting data all along. |
| 0.3 | **Verify a control actually does what it claims** before trusting a null result. | Two confident false negatives: a duration sweep on audio whose speech ended at 216 s (so 240/300 s clips were identical by construction), and `RS_AUDIO_KV_WINDOW=0` misread twice. |
| 0.4 | **Validate a metric against known-GOOD and known-BAD models.** | The first margin metric sampled positions the model chose, so a marker-spamming model scored best of all (+7.15, above base). It nearly certified a model that emitted 49 characters. |
| 0.5 | **A metric and an optimisation target must not share a blind spot.** | The one-sided margin hinge and the margin metric shared one; the optimiser satisfied both while destroying the model. |
| 0.6 | **One measurement is not a result when margins are thin.** Repeat across quant/seed. | v7.2 at q4 outscored its own f16 on dense audio. Structural decisions are coin flips below ~1 logit of margin. |

---

## 1. Structure (markers, speakers) — the axis that broke most often

Run the **deployed engine** (`moss-td-test`), not HF, across this matrix:

- audio: **dense continuous speech**, **multi-speaker synthetic**, **the demo clip**
- quant: **f16 AND q4** (and any quant under consideration)
- eviction: **45 s AND off**

| # | Check | Pass | Earned by |
|---|---|---|---|
| 1.1 | Marker count per window vs base and ground truth | within ~2× of base; not 1 | v5–v7.1 emitted **1 segment per window** on continuous speech where base emits 14–26. Invisible on the pause-structured demo clip, which gave a plausible-looking 9. |
| 1.2 | Marker count not *inflated* | not >2× base | v7.3 emitted 114 markers on audio with 12 utterances, and 49 chars total. Under-emission and over-emission are both failures. |
| 1.3 | Speaker count vs reference, **at every quant** | matches at all quants | v7 kept 3 speakers at f16 and collapsed to 1 at q4. An f16-only check passes it. |
| 1.4 | **Cells agree with each other** | spread ≤ ~25% | Disagreement between f16/q4/eviction cells means margins are thin and the model is non-deterministic — a finding in itself. |
| 1.5 | Structural logit margin (diagnostic) | ≫1 logit; base ≈4.9 | The root cause of 1.1–1.4. At 0.18 logits, ggml vs PyTorch kernel noise flips the decision. |

## 2. Recognition quality

| # | Check | Pass | Earned by |
|---|---|---|---|
| 2.1 | In-domain MER (IVOD demo clip vs reference) | no worse than previous release | — |
| 2.2 | **ASCEND per bucket (zh / en / mixed)** at deployment quant | no bucket regresses | The FT halved general accuracy: en .353→.687, all .198→.392. Nothing in the in-domain gate saw it. |
| 2.3 | Character count vs reference | within ~10% | Catches silent truncation and marker-spam collapse (49 chars vs 1276). |
| 2.4 | Never accept a quant on structural counts alone | MER measured too | q5 matched f16 on speakers/markers but scored MER 0.157 vs 0.066. |

## 3. Long audio and memory

| # | Check | Pass | Earned by |
|---|---|---|---|
| 3.1 | Coverage: last marker ≈ audio length | no early stop | Eviction-era models drift and stop early. |
| 3.2 | Markers still emitted in the final third | no late drift | Degradation appears only after most audio is evicted. |
| 3.3 | KV buffer bounded under eviction | flat vs audio length | The product reason eviction exists. |
| 3.4 | No repetition loop on long dense windows at **q4** | none | The original 2 h `這個需求的` wall appeared only at q4, only ≥300 s, only on repetitive real audio. |
| 3.5 | Trailing silence produces no text | no hallucination | Tail estimator smeared text to 318 s when speech ended at 216 s; a model hallucinated 「請問您是誰？」 into silence. |

## 4. Pipeline (JS) — separate from the model

Bisect **HF vs engine** early: identical text with different structure means a pipeline bug, not a model bug.

| # | Check | Pass | Earned by |
|---|---|---|---|
| 4.1 | Speaker linking preserves the model's `[Sxx]` when embeddings are absent | no collapse to one speaker | `linkSpeakers` inherited the neighbour's cluster, discarding the engine's correct `[S02]`. |
| 4.2 | Window boundary emits no duplicate | none | Overlap (up to `SNAP_S`=12 s) was transcribed by both windows on healthy windows. |
| 4.3 | No segment with impossible speaking rate | <15 ch/s | A 0.9 s / 38-char fragment (42 ch/s) was a ragged tail kept and re-transcribed. |
| 4.4 | No segment implausibly slow | >1.5 ch/s | Marker-less tail spread text across trailing silence at 1.0 ch/s. |
| 4.5 | No overlapping segments | ends ≤ next start | A kept tail ran to 172.8 s while the next window began at 171.9 s. |
| 4.6 | Cosine threshold semantics | similarity vs distance not confused | `threshold=0.65` was applied as a distance ceiling, merging anything above cosine 0.35. |

## 5. Deployment — the fix must actually reach users

| # | Check | Pass | Earned by |
|---|---|---|---|
| 5.1 | Bump the `?v=` cache-bust with every asset change | new query | JS was replaced but `?v=` left stale; returning visitors kept the old script. A shipped fix appeared to do nothing, twice. |
| 5.2 | Verify in a **fresh/incognito** browser, not just curl | rendered value correct | Server served the fix while the user's cached tab did not. |
| 5.3 | User-visible strings match what ships | model/size/quant correct | UI advertised "Q4_K · ~700 MB" while downloading a 2.13 GB f16. The identifier lives in ≥4 places per demo. |
| 5.4 | Both demo copies patched | wasm **and** cpp | `static/app-native.js` lives only in the Space repo and carried duplicate copies of the same bugs. |
| 5.5 | Engine pin actually contains the fix | commit is an ancestor | The cpp Space ran engine `0facfb6`, which predates `MossLoopGuard`; the guard was believed deployed for hours. |
| 5.6 | Confirm the running process, not the repo | health endpoint | `app.py` said f16 while the live process still served q5 mid-restart. |

---

## Minimum gate before shipping a model

`scripts/68_stage_gate.py` runs 1.1–1.4 and 2.2 in one pass. Add manually:
long-audio coverage (3.1–3.3), the sanity checks (`exp/sanity.py` → 4.3–4.5),
and a fresh-browser deployment check (5.2).

**Do not ship on:** loss value (meaningless under fake-quant and hinge losses),
fp-only evals, a single quant, a single measurement, or a demo clip with pause
structure.
