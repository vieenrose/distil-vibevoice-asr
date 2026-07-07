# Pipeline scripts

Thin, config-driven CLI wrappers around `distil_vibevoice` (src layout; every
script prepends `src/` to `sys.path`, so no install is needed beyond deps).
Each stage reads its YAML from `configs/` and is **idempotent/resumable** —
rerunning skips finished work. Run in order; 02+03 and 04 can run in parallel
(one GPU each).

```
01 → 02 → 03 ┐
      └→ 04 ┼→ 05 → 06 → 07 → 08 → 09 → 10 → 11
```

All wall-clock numbers below are **estimates for 2x RTX 5090 (32 GB)** and the
full-size data plan (~2,000 h per major bucket); scale linearly for smaller
runs. Disk figures assume 24 kHz 16-bit mono WAV (~173 MB/h).

| # | Script | What it does | Config | Est. wall-clock | Est. disk |
|---|--------|--------------|--------|-----------------|-----------|
| 01 | `01_download_data.py` | Download all sources into `data/raw/<name>` (HF CLI + direct URLs), print license reminders. `--only cv,musan` to subset. | `data.yaml` | 2–12 h (bandwidth-bound) | 0.5–1 TB (CV zh-TW ~2 GB, YODAS zh000 ~30 GB, YODAS en000 ~300–500 GB, IVOD meta ~1 GB + audio streamed later, ivod-fine-tune 50–300 GB selective, ASCEND ~4 GB, AMI ~100 GB, MUSAN 11 GB, RIRS 1.3 GB) |
| 02 | `02_generate_scripts.py` | 20 k template/grammar zh-TW⇄en code-switched meeting dialogues → `data/scripts/scripts.jsonl`. | `data.yaml` | < 10 min (CPU) | ~100 MB |
| 03 | `03_synth_tts_meetings.py` | VibeVoice-Large TTS renders each script (TW-accent voice bank) → `data/tts_raw/`, then RIR/MUSAN/codec augmentation → `data/tts_aug/`. | `data.yaml` | 4–7 days for ~2,000 h at ~0.5–1x realtime per GPU (run two shards, one per GPU, via `--limit`/split scripts.jsonl) | ~350 GB raw + ~350 GB aug |
| 04 | `04_pseudo_label.py` | 8B teacher labels real audio (IVOD/YODAS/CV/podcasts) in batches; confidence filter, optional `--two-pass` agreement, s2twp normalization → `data/pseudo/<source>_manifest.jsonl`. | `data.yaml` | 3–6 days for ~2,000 h at ~5–10x realtime per GPU (long-form batching; one source per GPU) | manifests only, < 5 GB (audio already on disk) |
| 05 | `05_build_manifests.py` | Merge buckets per `mix_ratios` (40/40/15/5), dedupe vs eval fingerprint index, stratified train/val split, hours table. | `data.yaml` | 10–60 min (fingerprinting is the cost) | < 1 GB |
| 06 | `06_prune_4b.py` | Extract Qwen2.5-7B backbone from the VibeVoice checkpoint (cached at `models/teacher_llm`, ~15 GB), Minitron importance on 64 calib batches, width-prune to 4B → `checkpoints/pruned_4b` + `hidden_keep_idx.pt`. | `prune_4b.yaml` | 30–60 min | ~8 GB (4B bf16) + 15 GB cache |
| 07 | `07_distill_4b.py` | Stage-1 distill (KL T=2 + CE + hidden-MSE, speaker/timestamp tokens 4x): student on cuda:0, frozen 8B teacher on cuda:1. 60 k steps, seq-len curriculum 4k→16k. **Text-only first milestone; audio-latent hookup is TODO-marked.** | `distill_stage1_4b.yaml` | ~1.5–2.5 days (60 k steps @ 2–3.5 s/step incl. teacher forward) | ~25 GB (3 kept ckpts × 8 GB) |
| 08 | `08_prune_1p5b.py` | Same recipe, 4B → 1.5B (hidden 1536, inter 8960, 12Q/2KV) → `checkpoints/pruned_1p5b`. | `prune_1p5b.yaml` | ~30 min | ~3 GB |
| 09 | `09_distill_1p5b.py` | Stage-2 distill from the 4B TA + 10% of batches direct from the 8B (`--no-direct-8b` to disable); length/speaker/seq-len curricula. 80 k steps. | `distill_stage2_1p5b.yaml` | ~1–2 days (80 k steps @ 1–2 s/step) | ~10 GB |
| 10 | `10_qat_export.py` | torchao int4-weight/int8-act finalize (QAT convert, PTQ fallback), export artifact + `estimate_ram` breakdown; **exits 1 if total > 2.4 GB ceiling**. Mobile container (GGUF vs ExecuTorch) is TODO-verify. | `qat_export.yaml` | 15–60 min (+ ~8 k QAT steps ≈ 4–8 h if QAT-prepared training is wired in) | ~1.2 GB export |
| 11 | `11_evaluate.py` | ChunkedTranscriber (15-min windows, 45 s overlap) over the eval manifest → `runs/eval/hyp.jsonl`, then `run_gates` overall + per slice (codeswitch / zh / en / longform / overlap). **Exit code 1 on any gate failure — CI-able.** | `eval_gates.yaml` | ~1–3 h for a 20 h eval set | < 1 GB |

## Typical invocations

```bash
.venv/bin/python scripts/01_download_data.py --only musan,rirs_noises,common_voice_zhtw
.venv/bin/python scripts/02_generate_scripts.py -n 20000
.venv/bin/python scripts/03_synth_tts_meetings.py --device cuda:0 --limit 10000
.venv/bin/python scripts/04_pseudo_label.py --audio-dir data/raw/ivod_audio --source ivod --two-pass
.venv/bin/python scripts/05_build_manifests.py
.venv/bin/python scripts/06_prune_4b.py
.venv/bin/python scripts/07_distill_4b.py
.venv/bin/python scripts/08_prune_1p5b.py
.venv/bin/python scripts/09_distill_1p5b.py
.venv/bin/python scripts/10_qat_export.py
.venv/bin/python scripts/11_evaluate.py --model checkpoints/qat_1p5b   # exit 1 = gates failed
```

## Notes

- **Teacher weights** must be fully downloaded at `models/teacher` before 04/06.
  The ASR repo ships no tokenizer — scripts pull it from `Qwen/Qwen2.5-7B`.
- **GPU placement**: distillation defaults to student on `cuda:0`, frozen
  teacher(s) on `cuda:1` (see the stage YAMLs).
- **Milestone order**: 07/09 run end-to-end **text-only** first; splicing the
  frozen acoustic/semantic encoder latents (`audio_latents`) into the student
  input is the TODO-marked follow-up in both scripts and in
  `distil_vibevoice.distill.collator`.
- **Licenses**: 01 prints per-source terms. CC-BY sources require attribution
  in anything redistributed; ASCEND (CC-BY-SA) constrains derived *dataset*
  redistribution, not model training.
