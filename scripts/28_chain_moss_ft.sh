#!/bin/bash
cd /home/luigi/distil-vibevoice-asr
echo "[chain] waiting for augcache..."
while pgrep -f 25_augment_cache >/dev/null; do sleep 60; done
echo "[chain] augcache done ($(ls data/latents/tts/*.npz|wc -l) latents). starting MOSS FT..."
CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/27_ft_moss.py \
  --steps 400 --max-audio-s 120 > data/moss_ft.log 2>&1
echo "[chain] FT done. tail:"; tail -3 data/moss_ft.log
