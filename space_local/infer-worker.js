/* Inference worker: runs the whole ONNX pipeline off the main thread so the
 * page (and its liveness heartbeat) never freezes during wasm compute. Cross-
 * origin isolation is unreliable on static HF Spaces, so we do NOT rely on
 * SharedArrayBuffer threads or ORT's proxy — a dedicated worker is enough to
 * keep the UI responsive. */
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
import { MossPipeline } from "./pipeline.js";

const REPO =
  "https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-onnx/resolve/main/";
const WEIGHTS = REPO + "web/";
let pipe = null;

const post = (type, payload) => self.postMessage({ type, ...payload });

// HF's CDN sends these files with `Cache-Control: no-store`, so the browser's
// normal HTTP cache can never keep them — every page load would otherwise
// re-download the full ~1.1GB, every time. Cache them ourselves with the
// Cache Storage API instead (available directly in a worker's global scope,
// same-origin, no Service Worker involvement needed). Model URLs are
// immutable per deployed filename, so a plain cache-first strategy is safe;
// bump MODEL_CACHE's suffix if a future deploy ever reuses a filename for
// different content.
// v2: v5 reuses the same filenames (encoder.int8 / embedding.int8 / decoder.q4)
// with DIFFERENT weights than v4, so bump the suffix to force returning users
// to fetch the new v5 weights instead of serving stale v4 from cache.
const MODEL_CACHE = "moss-model-cache-v2";

// Without this, the ~1.1GB model cache lives in "best-effort" storage that
// the browser is free to silently evict under disk pressure (most
// aggressively on mobile / low-disk-space machines) — which looks exactly
// like "caching doesn't work, it redownloads every time" even though the
// Cache Storage writes above all succeeded. Requesting "persistent" mode
// makes eviction require explicit user action instead. Best-effort only:
// some browsers grant it silently, some prompt, Safari mostly ignores it —
// either way this never blocks or fails the actual caching path.
navigator.storage?.persist?.().catch(() => {});

// Stream into a SINGLE pre-sized buffer (Content-Length known ahead of time)
// instead of collecting chunks in a JS array and copying once at the end —
// the collect-then-copy pattern briefly doubles peak memory for large files
// (encoder/decoder are 300-600MB each) at the exact moment 2-3 other model
// files are also resident, which is the main contributor to slow, GC-heavy
// downloads and to tipping the tab over its WASM memory ceiling on long runs.
async function fetchProgress(url, onProgress) {
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength);
    return buf;
  }

  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split("/").pop()}: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || 0;
  const reader = r.body.getReader();
  const buf = total ? new Uint8Array(total) : null;
  const chunks = buf ? null : []; // fallback if server omits Content-Length
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    if (buf) buf.set(value, done);
    else chunks.push(value);
    done += value.length;
    onProgress(done, total || done);
  }
  const out = buf || (() => {
    const o = new Uint8Array(done);
    let off = 0;
    for (const c of chunks) { o.set(c, off); off += c.length; }
    return o;
  })();

  // Cache a copy for next time. AWAITED (not fire-and-forget): for the two
  // largest files (encoder/decoder, 300-600MB) an un-awaited write was
  // measured to frequently lose the race against the page navigating away
  // shortly after "ready" fires — a dedicated Worker is torn down the
  // moment its page navigates/closes, which cuts off any still-pending
  // write. Writing to Cache Storage is local disk I/O, much faster than the
  // network fetch that just happened, so this costs little and makes
  // "one-time download" an actual guarantee instead of a best effort.
  try {
    await cache.put(url, new Response(out, {
      headers: {
        "Content-Type": r.headers.get("Content-Type") || "application/octet-stream",
        "Content-Length": String(out.byteLength),
      },
    }));
  } catch { /* storage quota or similar — this file just re-downloads next time */ }

  return out.buffer;
}

async function ensureModel(quality) {
  if (pipe) return;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.proxy = false;  // we ARE the worker
  // int8 decoder only (highest accuracy; the q4 variant saved ~19% download
  // for a real accuracy hit, not worth it — diarization/accuracy first).
  // v5 (diarization-fixed) decoder is q4, NOT int8: naive int8 quantization
  // (per-tensor) rounds away the KL-restored voice->speaker discrimination and
  // collapses diarization back to 1 speaker, whereas q4 MatMulNBits (block-32,
  // per-block scales) preserves it — measured 3 speakers vs int8's 1 on the
  // 5-min held-out meeting. q4 also outputs Simplified more often, but the s2tw
  // post-process already fixes that; net win: the diarization fix survives to
  // the browser AND the decoder is smaller (374MB q4 vs 598MB int8).
  const models = {
    encoder: WEIGHTS + "encoder.int8.onnx",
    embedding: WEIGHTS + "embedding.int8.onnx",
    decoder: WEIGHTS + "decoder.q4.onnx",
    ecapa: WEIGHTS + "ecapa.onnx",
  };
  const [cfg, melBin, vocab] = await Promise.all([
    fetch("models/config.json").then((r) => r.json()),
    fetch("models/mel.bin").then((r) => r.arrayBuffer()),
    fetch("models/vocab.json").then((r) => r.json()),
  ]);
  const p = new MossPipeline(ort, cfg, melBin, vocab);
  // fp32 KV (onnxruntime-web lacks Float16Array support)
  p.eps = ["wasm"];
  // Fetch all 4 weight files CONCURRENTLY (was: strictly sequential, which
  // measured ~220s wall time — encoder 37s + embedding 6s + decoder 166s +
  // ecapa ~15s, one after another — vs a theoretical ~130-165s if the
  // fetches overlap and each streams close to its own measured rate).
  // Session CREATION (the actual wasm compile/init, CPU-bound) still happens
  // one at a time in a fixed order, right after each file's bytes land, so
  // peak concurrent memory stays bounded to "all 4 downloads in flight" not
  // "all 4 downloads + all 4 compiled sessions" at once.
  await p.load(models, fetchProgress,
    (name, done, total) => post("dl", { name, done, total }));
  pipe = p;
  post("ready", { threads: ort.env.wasm.numThreads });
}

let aborted = false;
let running = false; // defense-in-depth: the main thread's acquireBusy()
                      // lock should already prevent overlapping "run"
                      // requests, but a second one landing here anyway
                      // (e.g. a future code path that forgets the lock)
                      // must be rejected outright rather than interleaved
                      // with an in-flight transcribeMeeting() call, which
                      // would jumble two audios' segments/messages together.

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "abort") { aborted = true; return; }
  if (msg.type === "run") {
    if (running) {
      post("error", { message: "a transcription is already running" });
      return;
    }
    running = true;
    aborted = false;
    try {
      await ensureModel(msg.quality);
      const wav = new Float32Array(msg.wav);
      const signal = { get aborted() { return aborted; } };
      const segs = await pipe.transcribeMeeting(wav, {
        windowS: msg.windowS,
        signal,
        onStage: (wi, nw, stage, detail) => post("stage", { wi, nw, stage, detail }),
        onToken: (wi, nw, text, n, dt) => post("token", { wi, nw, text, n, dt }),
        onWindow: (linked, done, total) => post("window", { linked, done, total }),
      });
      post("done", { segs, aborted });
    } catch (err) {
      post("error", { message: err.message });
    } finally {
      running = false;
    }
  }
};
