#!/usr/bin/env bash
# Overnight runner, chain A: stage-1 artifact evaluation.
#
# Waits for the in-flight English q8 run to finish first. That run predates the
# --out fix and still writes the shared out-new.json, so ANY concurrent engine
# run can clobber it -- which already happened once tonight and produced a
# W_zh_meet_base_q5.json that was byte-identical to the English f16 transcript.
# One engine run at a time until that job is gone.
#
# Nothing is deployed. Artifacts land on disk; a human decides.
# GPU1 only: GPU0 is reserved for another project.
set -uo pipefail

TMP=/tmp/claude-1001
LOG=$TMP/overnight_A.log
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  say "waiting for in-flight run pid $WAIT_PID (shared out-new.json)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "pid $WAIT_PID gone"
  # let its trailing `cp out-new.json W_en_meet_base_q8_0.json` land
  sleep 20
  if [ -f "$TMP/exp/W_en_meet_base_q8_0.json" ]; then
    say "en q8 result captured: $(stat -c%s "$TMP/exp/W_en_meet_base_q8_0.json") bytes"
  else
    say "WARN: W_en_meet_base_q8_0.json absent -- recover from out-new.json manually"
  fi
fi

say "=== PHASE A: stage-1 artifacts (q5_k_m / q5_k / iq4_xs) x (zh/en) ==="
bash $TMP/run_stage1_quants.sh 2>&1 | tee -a "$LOG"
say "=== PHASE A done — NOTHING DEPLOYED — review $LOG ==="
