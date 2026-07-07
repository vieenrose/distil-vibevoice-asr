#!/usr/bin/env python
"""Produce a REAL int4 artifact of the tied student and MEASURE its footprint + speed.

This turns the 2.05 GB mobile-RAM *estimate* for the tied int4 geometry into
*measured* numbers: real bit-packed int4 weight bytes, process RSS, KV-cache
bytes at a chosen context length, and generation throughput on CPU (a
DIRECTIONAL proxy for a phone NPU -- x86 CPU is NOT a phone number) plus GPU
for reference.

Quantization backend: torchao `IntxWeightOnlyConfig(weight_dtype=int4,
granularity=PerGroup(128))`. This is torchao's ARM/on-device int4 weight-only
path (the same family used for ExecuTorch/XNNPACK mobile export), which is the
right analogue for the 6GB-RAM phone target. Note torchao's runnable tensor
subclass (`IntxUnpackedToInt8Tensor`) stores each int4 value UNPACKED in one
int8 byte for portability; the genuinely bit-packed 4-bit size (what a real
mobile/GGUF q4 deployment ships) is computed AND written to disk here so the
number is not hand-waved.

Usage:
    python scripts/10b_export_benchmark.py --model models/student_1p5b_tied_smoke \
        --context 8192 --threads 8
"""
import argparse
import gc
import json
import os
import time

import torch


def rss_gb() -> float:
    """Resident set size of this process in GB (Linux /proc, real measurement)."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)  # kB -> GB
    return float("nan")


def collect_unique_storage_bytes(model) -> int:
    """Sum real storage bytes of a module, de-duplicating shared/tied storages and
    descending into tensor subclasses (so int4 tensor-subclass internals count)."""
    seen: set = set()
    total = [0]

    def collect(obj):
        if isinstance(obj, torch.Tensor):
            try:
                names, _ = obj.__tensor_flatten__()
                for n in names:
                    collect(getattr(obj, n))
                return
            except Exception:
                pass
            try:
                st = obj.untyped_storage()
                key = st.data_ptr()
                if key not in seen:
                    seen.add(key)
                    total[0] += st.nbytes()
            except Exception:
                total[0] += obj.numel() * obj.element_size()

    for _, v in model.state_dict().items():
        collect(v)
    for _, v in model.named_buffers():
        collect(v)
    return total[0]


def int4_component_breakdown(model):
    """Walk quantized linear layers; return (qdata_numel, zero_point_numel,
    scale_bytes) for torchao IntxUnpackedToInt8Tensor weights, plus bf16 bytes
    for everything left unquantized (tied embedding, norms)."""
    qdata_numel = 0
    zp_numel = 0
    scale_bytes = 0
    for _, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and type(w).__name__ == "IntxUnpackedToInt8Tensor":
            names, _ = w.__tensor_flatten__()
            for n in names:
                t = getattr(w, n)
                if n == "qdata":
                    qdata_numel += t.numel()
                elif n == "zero_point":
                    zp_numel += t.numel()
                elif n == "scale":
                    scale_bytes += t.numel() * t.element_size()
    # ids of weights that are quantized subclasses -> exclude from bf16 accounting
    # (their .dtype reports the LOGICAL bf16, so a naive loop double-counts them).
    quant_ids = set()
    for _, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and type(w).__name__ == "IntxUnpackedToInt8Tensor":
            quant_ids.add(id(w))
    bf16_bytes = 0
    for _, p in model.named_parameters():
        if id(p) in quant_ids:
            continue
        if p.dtype in (torch.bfloat16, torch.float16, torch.float32):
            bf16_bytes += p.numel() * p.element_size()
    return qdata_numel, zp_numel, scale_bytes, bf16_bytes


def write_packed_int4(model, out_dir):
    """Genuinely bit-pack the int4 weights (2 nibbles/byte for qdata AND
    zero_point) and save to disk with bf16 scales + unquantized bf16 tensors.
    Returns real on-disk byte count. This is the honest "what ships" size."""
    os.makedirs(out_dir, exist_ok=True)
    blob = {}
    quant_ids = set()
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is not None and type(w).__name__ == "IntxUnpackedToInt8Tensor":
            quant_ids.add(id(w))
            names, _ = w.__tensor_flatten__()
            d = {n: getattr(w, n) for n in names}
            q = d["qdata"].to(torch.int8).flatten()
            if q.numel() % 2:  # pad to even for nibble packing
                q = torch.cat([q, q.new_zeros(1)])
            # map signed [-8,7] to unsigned [0,15] nibbles, pack 2/byte
            qu = (q & 0x0F).to(torch.uint8)
            packed_q = (qu[0::2] | (qu[1::2] << 4)).contiguous()
            blob[name + ".qdata_int4packed"] = packed_q
            blob[name + ".scale"] = d["scale"].to(torch.bfloat16).contiguous()
            zp = d["zero_point"].to(torch.int8).flatten()
            if zp.numel() % 2:
                zp = torch.cat([zp, zp.new_zeros(1)])
            zpu = (zp & 0x0F).to(torch.uint8)
            blob[name + ".zp_int4packed"] = (zpu[0::2] | (zpu[1::2] << 4)).contiguous()
    # unquantized bf16 tensors (tied embedding, norms) -- skip quantized weights
    for name, p in model.named_parameters():
        if id(p) in quant_ids:
            continue
        if p.dtype in (torch.bfloat16, torch.float16, torch.float32):
            blob[name] = p.detach().to(torch.bfloat16).contiguous()
    path = os.path.join(out_dir, "packed_int4.pt")
    torch.save(blob, path)
    return os.path.getsize(path)


def kv_cache_bytes(cfg, context_tokens, dtype_bytes=2):
    """Analytic KV-cache size for GQA: layers * 2(K,V) * kv_heads * head_dim *
    tokens * bytes_per_elem."""
    layers = cfg.num_hidden_layers
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_tok = layers * 2 * kv_heads * head_dim * dtype_bytes
    return per_tok * context_tokens, per_tok


def verify_kv_via_generate(model, device, context_tokens_probe=64):
    """Run a short generate() and sum the actual past_key_values tensor bytes,
    then scale to per-token to cross-check the analytic KV formula."""
    ids = torch.randint(0, 1000, (1, context_tokens_probe), device=device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    total = 0

    def add(t):
        nonlocal total
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()

    if pkv is None:
        return 0, 0.0
    # transformers 5.x DynamicCache: .layers -> objects with .keys/.values
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            add(getattr(layer, "keys", None))
            add(getattr(layer, "values", None))
    elif hasattr(pkv, "key_cache"):
        for t in list(pkv.key_cache) + list(pkv.value_cache):
            add(t)
    else:
        try:
            for layer in pkv:  # tuple-style
                for t in layer:
                    add(t)
        except TypeError:
            pass
    return total, (total / context_tokens_probe if total else 0.0)


def bench_generate(model, device, prompt_len, new_tokens, warmup=True):
    ids = torch.randint(0, 1000, (1, prompt_len), device=device)
    gen_kwargs = dict(max_new_tokens=new_tokens, do_sample=False, use_cache=True,
                      num_beams=1)
    if warmup:
        with torch.no_grad():
            model.generate(torch.randint(0, 1000, (1, min(16, prompt_len)), device=device),
                           max_new_tokens=4, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, **gen_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    produced = out.shape[1] - prompt_len
    return produced / dt, dt, produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/student_1p5b_tied_smoke")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--pack-out", default="models/student_1p5b_int4_packed")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from torchao.quantization import quantize_, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerGroup

    result = {}
    rss_baseline = rss_gb()

    # ---- load bf16 ----
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    cfg = m.config
    rss_after_load = rss_gb()
    bf16_bytes = collect_unique_storage_bytes(m)
    params_tied = sum(p.numel() for p in m.parameters())

    # ---- quantize int4 (torchao on-device path) ----
    torchao_ok = True
    try:
        quantize_(m, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(128)))
    except Exception as e:
        torchao_ok = False
        result["quant_error"] = repr(e)
    gc.collect()
    rss_after_quant = rss_gb()

    runnable_bytes = collect_unique_storage_bytes(m)
    qn, zpn, sb, bf16b = int4_component_breakdown(m)
    packed_qdata_gb = qn * 0.5 / 1e9
    packed_zp_gb = zpn * 0.5 / 1e9
    scale_gb = sb / 1e9
    unquant_bf16_gb = bf16b / 1e9
    true_packed_gb = packed_qdata_gb + packed_zp_gb + scale_gb + unquant_bf16_gb

    # ---- genuinely bit-pack to disk (real file bytes) ----
    disk_bytes = write_packed_int4(m, args.pack_out)

    # ---- KV cache ----
    kv_bytes, kv_per_tok = kv_cache_bytes(cfg, args.context, dtype_bytes=2)
    kv_probe_bytes, kv_probe_per_tok = verify_kv_via_generate(m, "cpu")

    # ---- CPU throughput (threads capped, mobile-ish DIRECTIONAL proxy) ----
    torch.set_num_threads(args.threads)
    cpu_tok_s, cpu_dt, cpu_prod = bench_generate(m, "cpu", args.prompt_len, args.new_tokens)

    # ---- GPU throughput for reference (re-quantize on cuda) ----
    gpu_tok_s = None
    gpu_note = "skipped"
    if not args.skip_gpu and torch.cuda.is_available():
        try:
            mg = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
            quantize_(mg, IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(128)))
            gpu_tok_s, gpu_dt, gpu_prod = bench_generate(mg, "cuda", args.prompt_len, args.new_tokens)
            gpu_note = "int4 IntxWeightOnly on cuda"
            del mg
            torch.cuda.empty_cache()
        except Exception as e:
            gpu_note = f"gpu int4 failed: {e!r}; trying bf16"
            try:
                mg = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
                gpu_tok_s, gpu_dt, gpu_prod = bench_generate(mg, "cuda", args.prompt_len, args.new_tokens)
                gpu_note = "bf16 on cuda (int4 cuda kernel unavailable)"
                del mg
                torch.cuda.empty_cache()
            except Exception as e2:
                gpu_note = f"gpu failed: {e2!r}"

    result.update({
        "model": args.model,
        "config": {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": getattr(cfg, "head_dim", None),
            "hidden_size": cfg.hidden_size,
            "vocab_size": cfg.vocab_size,
            "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
            "params_tied": params_tied,
            "params_after_quant_tie_broken": sum(p.numel() for p in m.parameters()),
        },
        "torchao_ok": torchao_ok,
        "bf16_weight_gb": round(bf16_bytes / 1e9, 4),
        "runnable_int4_in_int8_gb": round(runnable_bytes / 1e9, 4),
        "true_packed_int4_gb": round(true_packed_gb, 4),
        "true_packed_int4_ondisk_gb": round(disk_bytes / 1e9, 4),
        "int4_breakdown_gb": {
            "qdata_int4packed": round(packed_qdata_gb, 4),
            "zero_point_int4packed": round(packed_zp_gb, 4),
            "scales_bf16": round(scale_gb, 4),
            "unquantized_bf16_embedding_and_norms": round(unquant_bf16_gb, 4),
        },
        "rss_baseline_gb": round(rss_baseline, 4),
        "rss_after_bf16_load_gb": round(rss_after_load, 4),
        "rss_after_quant_gb": round(rss_after_quant, 4),
        "context_tokens": args.context,
        "kv_cache_gb": round(kv_bytes / 1e9, 4),
        "kv_bytes_per_token": kv_per_tok,
        "kv_probe_per_token_measured": round(kv_probe_per_tok, 1),
        "cpu_tok_s": round(cpu_tok_s, 3),
        "cpu_threads": args.threads,
        "cpu_gen_seconds": round(cpu_dt, 2),
        "cpu_tokens_produced": cpu_prod,
        "gpu_tok_s": round(gpu_tok_s, 3) if gpu_tok_s else None,
        "gpu_note": gpu_note,
    })
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
