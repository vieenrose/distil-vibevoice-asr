#!/bin/bash
# Wait for the two sharded labeling workers, merge manifests, then train (audio-free latents).
cd /home/luigi/distil-vibevoice-asr
echo "[chain] waiting for labeling shards $1 $2 ..."
while kill -0 "$1" 2>/dev/null || kill -0 "$2" 2>/dev/null; do sleep 60; done
echo "[chain] labeling done. merging manifests..."
cat data/pseudo/ivod_stream_manifest.jsonl data/pseudo/ivod_stream_shard0.jsonl data/pseudo/ivod_stream_shard1.jsonl 2>/dev/null \
  | grep -v '\[Noise\]' | grep -v '\[Silence\]' > data/pseudo/ivod_all_manifest.jsonl
N=$(wc -l < data/pseudo/ivod_all_manifest.jsonl)
H=$(.venv/bin/python -c "import json;print(round(sum(json.loads(l)['duration_s'] for l in open('data/pseudo/ivod_all_manifest.jsonl'))/3600,1))" 2>/dev/null)
echo "[chain] merged $N records (~${H}h speech). launching latent distill..."
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/07e_latent_distill.py \
  --manifest data/pseudo/ivod_all_manifest.jsonl --steps 30 --max-frames 4096 \
  > data/latent_distill.log 2>&1
echo "[chain] training done. tail:"; tail -5 data/latent_distill.log
