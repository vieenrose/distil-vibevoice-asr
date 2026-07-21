#!/usr/bin/env python3
"""Score a MOSS-TD transcript against AMI ground truth.

Four axes, reported separately because they fail independently and a single
aggregate number hides which one broke:

  1. transcript correctness  -> WER against AMI word-level references
  2. timestamp accuracy      -> per-word onset error, after alignment
  3. speaker-tag accuracy    -> word-level speaker error under the best
                                permutation of our Sxx onto AMI's A-D
  4. coverage / segmentation -> how much reference speech we emitted at all,
                                and how our segment boundaries compare

Deliberately NOT a single score. On this project aggregate MER has repeatedly
saturated and ranked distinct transcripts identically, so each axis stays
visible.

Usage:
  .venv/bin/python scripts/81_score_vs_ami.py HYP.txt --meeting ES2004a \\
      --ami /tmp/claude-1001/ami_ann [--offset 0] [--max-time 1049]
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from itertools import permutations
from pathlib import Path
from xml.etree import ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"


def load_ami_words(ami_dir: Path, meeting: str):
    """[(start, end, token, speaker)] over all speakers, time-ordered.

    Punctuation-only <w punc="true"> tokens are dropped: MOSS emits its own
    punctuation and scoring it against the transcribers' would measure style,
    not recognition."""
    out = []
    for f in sorted((ami_dir / "words").glob(f"{meeting}.*.words.xml")):
        spk = f.name.split(".")[1]
        for w in ET.parse(f).getroot():
            if w.tag != "w" or w.get("punc") == "true":
                continue
            s, e = w.get("starttime"), w.get("endtime")
            if s is None or not (w.text or "").strip():
                continue
            out.append((float(s), float(e or s), w.text.strip(), spk))
    out.sort(key=lambda x: x[0])
    return out


SEG_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*(?:\[(S\d+)\])?\s*([^\[]*)")


def parse_hyp(text: str):
    """[(start, speaker, text)] from the model's raw [start][Sxx]text stream."""
    segs, spk = [], "S01"
    for m in SEG_RE.finditer(text):
        if m.group(2):
            spk = m.group(2)
        body = (m.group(3) or "").strip()
        if body:
            segs.append((float(m.group(1)), spk, body))
    return segs


def norm_tokens(s: str):
    """Lowercase, strip punctuation/marks. Keeps CJK chars as individual tokens
    so the same scorer works for the zh clip."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s一-鿿]", " ", s)
    toks = []
    for chunk in s.split():
        if re.search(r"[一-鿿]", chunk):
            toks.extend(list(chunk))
        else:
            toks.append(chunk)
    return toks


def levenshtein(a, b):
    """Edit distance with backpointers -> (distance, alignment pairs)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1, a[i - 1] == b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None, False)); i -= 1
        else:
            pairs.append((None, j - 1, False)); j -= 1
    return d[n][m], list(reversed(pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hyp")
    ap.add_argument("--meeting", default="ES2004a")
    ap.add_argument("--ami", type=Path, default=Path("/tmp/claude-1001/ami_ann"))
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds of the meeting the hypothesis starts at")
    ap.add_argument("--max-time", type=float, default=None,
                    help="ignore reference words past this time (clip length)")
    args = ap.parse_args()

    ref = load_ami_words(args.ami, args.meeting)
    if args.max_time is not None:
        ref = [w for w in ref if w[0] < args.offset + args.max_time]
    ref = [w for w in ref if w[0] >= args.offset]
    hyp_segs = parse_hyp(Path(args.hyp).read_text(encoding="utf-8"))

    # Flatten hypothesis to tokens, each carrying its segment's start + speaker.
    hyp_tok = []
    for start, spk, body in hyp_segs:
        for t in norm_tokens(body):
            hyp_tok.append((start + args.offset, spk, t))
    ref_tok = [(s, spk, t) for s, e, w, spk in ref for t in norm_tokens(w)]

    print(f"meeting {args.meeting} | ref {len(ref_tok)} words, "
          f"{len(set(w[1] for w in ref_tok))} speakers | "
          f"hyp {len(hyp_tok)} words, {len(hyp_segs)} segments, "
          f"{len(set(s[1] for s in hyp_segs))} speakers")

    # ---- 1. transcript correctness -----------------------------------------
    dist, pairs = levenshtein([t[2] for t in ref_tok], [t[2] for t in hyp_tok])
    wer = dist / max(1, len(ref_tok))
    matched = [(i, j) for i, j, ok in pairs if ok]
    print(f"\n1. WER {wer:.4f}  ({dist} errors / {len(ref_tok)} ref words); "
          f"{len(matched)} exact word matches")

    # ---- 2. timestamp accuracy ---------------------------------------------
    # Only over correctly-recognised words: a timestamp on a wrong word is
    # meaningless, and including it would mix axis 1 into axis 2.
    if matched:
        errs = sorted(abs(hyp_tok[j][0] - ref_tok[i][0]) for i, j in matched)
        med = errs[len(errs) // 2]
        p90 = errs[int(len(errs) * 0.9)]
        within = lambda th: 100 * sum(e <= th for e in errs) / len(errs)
        print(f"2. onset error: median {med:.2f}s  p90 {p90:.2f}s  "
              f"| within 1s {within(1.0):.0f}%  2s {within(2.0):.0f}%  "
              f"5s {within(5.0):.0f}%")
        print("   (segment-level starts vs word onsets, so sub-second error is "
              "expected even when perfect)")

    # ---- 3. speaker-tag accuracy -------------------------------------------
    ref_spks = sorted(set(w[1] for w in ref_tok))
    hyp_spks = sorted(set(w[1] for w in hyp_tok))
    # MANY-TO-ONE, not a 1:1 permutation. `zip(hyp_spks, perm)` truncates to the
    # shorter list, and hyp_spks is sorted alphabetically -- so with hyp>ref
    # (windowed output routinely over-segments: 7 hyp speakers vs 4 real ones)
    # every hyp label past the Nth alphabetically NEVER gets a mapping and is
    # scored wrong regardless of how consistently it tracks one real speaker.
    # Measured: this hid a cleanly-linked 45-segment/1900-char cluster (label
    # "S05") entirely, understating speaker accuracy by double digits.
    # Correct approach: credit each hyp label independently for whichever ref
    # speaker it overlaps with most among matched words -- this measures
    # cluster PURITY (does this label consistently mean one real speaker),
    # which is what "linking got the identity right" should mean. It does NOT
    # penalise over-segmentation (multiple hyp labels for one real speaker);
    # that is a separate, distinct defect from mislabelling and should not be
    # conflated into one number.
    votes: dict[str, dict] = {}
    for i, j in matched:
        h, r = hyp_tok[j][1], ref_tok[i][1]
        votes.setdefault(h, {}).setdefault(r, 0)
        votes[h][r] += 1
    best_map = {h: max(v, key=v.get) for h, v in votes.items()}
    best = sum(1 for i, j in matched if best_map.get(hyp_tok[j][1]) == ref_tok[i][1])
    if matched:
        print(f"3. speaker accuracy {100 * best / len(matched):.1f}% "
              f"on matched words (many-to-one, purity-style: {best_map})")
        if len(hyp_spks) != len(ref_spks):
            n_unattributed = len(hyp_spks) - len(set(best_map.values()))
            print(f"   NOTE speaker-count mismatch: hyp {len(hyp_spks)} vs "
                  f"ref {len(ref_spks)} — over-segmentation, not mislabelling; "
                  f"{len(hyp_spks)} hyp labels collapse onto "
                  f"{len(set(best_map.values()))} distinct real speakers")

    # ---- 4. coverage / segmentation ----------------------------------------
    ref_span = (ref_tok[0][0], ref_tok[-1][0]) if ref_tok else (0, 0)
    hyp_span = (hyp_tok[0][0], hyp_tok[-1][0]) if hyp_tok else (0, 0)
    print(f"4. ref speech spans {ref_span[0]:.0f}-{ref_span[1]:.0f}s | "
          f"hyp spans {hyp_span[0]:.0f}-{hyp_span[1]:.0f}s")
    print(f"   deletions {sum(1 for i, j, _ in pairs if j is None)}  "
          f"insertions {sum(1 for i, j, _ in pairs if i is None)}  "
          f"words/segment {len(hyp_tok) / max(1, len(hyp_segs)):.1f}")


if __name__ == "__main__":
    main()
