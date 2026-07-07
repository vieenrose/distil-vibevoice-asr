#!/bin/bash
# Paced collect->label->discard loop. Bounded batches keep disk safe.
cd /home/luigi/distil-vibevoice-asr
YEARS="${1:-2025 2023 2022 2021 2020}"
BATCH=120           # per-year depth mining
MIN_DISK_GB=15     # pause collection if below this
for YR in $YEARS; do
  OFFSET=0
  while :; do
    FREE=$(df -B1 /home/luigi | tail -1 | awk '{print int($4/1e9)}')
    if [ "$FREE" -lt "$MIN_DISK_GB" ]; then echo "[grow] low disk ${FREE}GB, stopping"; exit 0; fi
    echo "[grow] === $YR collect batch (disk ${FREE}GB) ==="
    .venv/bin/python scripts/01b_collect_ivod.py --year $YR --limit $BATCH --max-minutes 10 \
      --kind full --require-transcript --done-manifest data/pseudo/ivod_v2.jsonl --sleep 1 --out data/raw/ivod_v2 >> data/grow_collect.log 2>&1
    NW=$(ls data/raw/ivod_v2/*.wav 2>/dev/null | wc -l)
    if [ "$NW" -eq 0 ]; then echo "[grow] $YR: no new wavs, next year"; break; fi
    echo "[grow] labeling $NW wavs..."
    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python \
      scripts/20_stream_label_cache.py --audio-glob "data/raw/ivod_v2/*.wav" \
      --latents-out data/latents/ivod_v2 --manifest data/pseudo/ivod_v2.jsonl \
      --delete-audio --max-audio-sec 300 --device cuda:0 >> data/grow_label.log 2>&1
    USABLE=$(.venv/bin/python -c "import json;print(sum(1 for l in open('data/pseudo/ivod_v2.jsonl') if any(not s['text'].startswith('[') for s in json.loads(l)['segments'])))" 2>/dev/null)
    echo "[grow] $YR batch done. total usable records: $USABLE"
  done
done
echo "[grow] all years done."
