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

async function fetchProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split("/").pop()}: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    chunks.push(value);
    done += value.length;
    onProgress(done, total);
  }
  const buf = new Uint8Array(done);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf.buffer;
}

async function ensureModel(quality) {
  if (pipe) return;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.proxy = false;  // we ARE the worker
  // int8 decoder only (highest accuracy; the q4 variant saved ~19% download
  // for a real accuracy hit, not worth it — diarization/accuracy first).
  const models = {
    encoder: WEIGHTS + "encoder.int8.onnx",
    embedding: WEIGHTS + "embedding.int8.onnx",
    decoder: WEIGHTS + "decoder.int8.onnx",
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
  await p.load(models, fetchProgress,
    (name, done, total) => post("dl", { name, done, total }));
  pipe = p;
  post("ready", { threads: ort.env.wasm.numThreads });
}

let aborted = false;

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "abort") { aborted = true; return; }
  if (msg.type === "run") {
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
    }
  }
};
