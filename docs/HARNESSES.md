# Harnesses between the model and what a user sees

Every mechanism below alters output relative to a plain greedy decode of the
raw model. Each was added to fix a measured defect, and each therefore makes a
surprising result ambiguous: model, quantization, or harness? This document is
the map, and `RS_RAW=1` is the switch that removes the engine-side half.

Status as of 2026-07-21. Deployed artifact: `moss-td-base-q5_k_m.gguf` (base
weights, 0.80 GB), demos at build `a08edf4e-q5c`.

---

## A. Engine — output-altering (`rapidspeech/src/arch/moss_td.cpp`)

These five have no counterpart in the official PyTorch pipeline, which is plain
greedy. **All five are disabled by `RS_RAW=1`.**

| # | mechanism | default | disable | why it exists |
|---|---|---|---|---|
| A1 | **Audio-KV eviction** `RS_AUDIO_KV_WINDOW` | 45 s | `=0` | 3x faster decode (55.7 vs 169.3 ms/tok on 600 s dense) and lower KV (1008 vs 1260 MB). Cost: deviates from no-eviction output. **Amount never validated on real meetings — under review.** |
| A2 | **Repetition penalty** `RS_REP_PENALTY` | 1.10, window 64 | `=1.0` | q4/QAT models looped on dense speech. f16 is clean either way. |
| A3 | **EOS coverage suppression** `RS_EOS_MINCOVER` | 0.6 | `=0` | Model emitted EOS at 75 s of a 172 s window, silently losing the rest. Suppresses EOS while marked coverage < 0.6 x speech_end (only when audio > 30 s, capped at 64 suppressions, and only if > 15 s of audio remains). |
| A4 | **Loop guard** `MossLoopGuard` | on | (RS_RAW only) | Three degenerate-repetition signatures: advancing-clock cycle, ticking-but-stalled clock, tight no-timestamp loop. On fire, trims to the first occurrence and stops. Pure state machine, unit-tested in `tests/test_moss_loop_guard.cpp`. |
| A5 | **Decode cap** | `min(5120, max(MAX_DECODE_TOKENS, audio_T))` | — | Bounds runaway generation. Raised 3072 -> 5120 to match official. |

Measured effect of turning A1–A4 off together, 5-min example, q5_k_m:
`normal 18 segments / 1022 chars` vs `RS_RAW=1 40 segments / 1266 chars`.

### Engine — non-semantic switches
`RS_NO_FLASH`, `RS_KV_F16` / `RS_KV_F32` / `RS_KV_Q8`, `RS_KV_HEADROOM`,
`RS_DECODE_LEGACY`, `RS_NO_TIME_MARKERS`, `RS_MEL_FILE` (inject a mel to rule
out the front-end), `RS_ENC_MARGIN`, `RS_THREADS`, `RS_PROFILE`,
`RS_DEBUG_MOSS`, `RS_DEBUG_ASR_DECODE`, `RS_LOG_*`.

---

## B. Demo JS (`app-wasm.js`, `static/app-native.js` — kept in sync)

No master switch yet. Ordered as the data flows.

| # | mechanism | parameter | why it exists |
|---|---|---|---|
| B1 | **Pause-snapped windowing** | `WINDOW_S` 90/180/**300** (UI default 5 min), `SNAP_S` = 12 (>=90 s) | Windows must be disjoint and cut at a pause, or boundaries split utterances. Single-pass truncates on long audio in BOTH implementations, so windowing is mandatory. |
| B2 | **Per-window silence gate** | `rms < 0.002` | Skips all-silence windows, which the model hallucinates on. **Whole-window only — a window containing speech plus 100 s of trailing dead air still passes.** |
| B3 | **Ragged-tail retreat** | `MAX_CH_PER_S = 15` | If a window's last segments imply an impossible speaking rate, retreat the cursor rather than trust them. 15 is ~3x the fastest real rate observed (5.3). |
| B4 | **Marker-less window fallback** | triggers when coverage < min(MIN_ADV, cut/2) | A fully marker-less 180 s window is otherwise one 700-char mega-segment. Splits at sentence punctuation, then distributes timestamps proportionally to text length (`tsEstimated`). |
| B5 | **`collapseLoops`** | — | Cross-window duplicate removal. |
| B6 | **End inference + rate bound** | `MIN_CH_PER_S = 2`, `MAX_INFER_GAP_S = 10` | An end taken from the next segment's start is only valid when adjacent. Across a long gap it stretches over silence. Added 2026-07-21 after a 10-char utterance got a 106 s extent, which poisoned its CAM++ embedding and split it into a phantom speaker. |
| B7 | **Boundary overlap filter + end clamp** | — | Drops segments starting before the cursor; clamps ends to the cursor. |
| B8 | **`linkSpeakers`** | CAM++ AHC, threshold 0.65 | Cross-window speaker identity. Units with no embedding keep their OWN cluster (inheriting the neighbour's silently discarded the model's own tag). |
| B9 | **s2tw + number ITN** | OpenCC `cn`->`tw` | zh-TW orthography is a PIPELINE behaviour, not the weights. **Never s2twp** — it corrupts proper nouns (高端疫苗 -> 高階疫苗). |
| B10 | **iOS single 28 s chunk** | `IS_IOS` | Memory cap. |
| B11 | **mp3 chunk-failure anchoring** | — | A failed chunk leaves a silence-padded gap instead of shifting every later timestamp (once silently dropped ~2 min). |
| B12 | **Autoscroll follow** | — | Rewritten 2026-07-21 to use page scroll after the transcript pane stopped being its own scroll container. |

---

## C. Offline harnesses (not in the product)

| tool | purpose |
|---|---|
| `/tmp/claude-1001/exp/winharness.mjs` | Headless replica of the browser window loop driving the real engine. `--window`, `--out`, inherits `CUDA_VISIBLE_DEVICES` and `RS_*`. |
| `exp/cmp_pair.py` | Two runs vs official + GT. Restricts MER to the official reference's covered span; reports density. |
| `exp/adjudicate.py` | Per-diff adjudication against a reference — the only method that separated the quant candidates. |
| `exp/score.py` | Monotonic (order-preserving) DP alignment; greedy matching reports absurd drift on repeated passages. |
| `exp/score_sd.py`, `exp/score_meet.py` | DER + speaker consistency vs GT and official. |
| `exp/deviation.py`, `exp/sanity.py`, `exp/cmp_windowed.py` | Deviation accounting, reference-free checks, windowed-vs-single. |
| `scripts/70_quant_sweep.py` | Multi-clip quant sweep (single-clip rankings are coin flips). |
| `scripts/69_official_parity.py` | Engine configs vs official PyTorch; text MER and timestamp MAE scored separately. |
| `scripts/68_stage_gate.py`, `61_v8_gate.py` | Staged release gates (ASCEND + DER + long-window). |
| `scripts/66_margin_probe.py` | Structural-token logit margin (base 4.90 vs FT 0.98). |
| `scripts/72_selfdistil_labels.py` | Self-distillation corpus with coverage/density/loop gates. |
| `docs/REGRESSION_CHECKLIST.md` | The manual checklist; every item annotated with the defect that earned it. |

---

## Known gaps

1. **No master switch for section B.** `RS_RAW` covers the engine only; the JS
   harnesses still run. A `?raw=1` URL parameter is the obvious counterpart.
2. **B2 is whole-window only.** Trailing dead air inside a speechy window is
   what produced the hallucinated tail segment on the 5-min example.
3. **A1's amount is unvalidated on real audio.** The only curve (45/60/75 -> 5
   deviations, 90 -> 2, 120 -> 0) came from a synthetic 3-min clip on a
   different artifact. Deployed value is 45 — the worst end of that curve.
4. **B6 does not remove the hallucinated segment**, only bounds its neighbour.
   A trailing-artifact filter must run once at end-of-processing, not in
   `normalizeSegs`, which runs per window on still-growing data.
