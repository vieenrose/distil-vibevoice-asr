/* UI + audio capture for the fully-local zh-TW transcriber. */
import { MossPipeline, parseLenient } from "./pipeline.js";

const ort = window.ort;
const $ = (id) => document.getElementById(id);
const MODELS = {
  encoder: "models/encoder.int8.onnx",
  embedding: "models/embedding.int8.onnx",
  decoder: "models/decoder.q4.onnx",
};
const MAX_S = 120;
const PALETTE = ["#4f7cff", "#e05563", "#2aa876", "#c78c2c", "#9761d8", "#3aa6b9"];

let pipe = null;
let s2tw = (t) => t;
try {
  if (window.OpenCC) s2tw = window.OpenCC.Converter({ from: "cn", to: "tw" });
} catch (e) { console.warn("opencc unavailable", e); }

const hasWebGPU = !!navigator.gpu;
$("ep-note").textContent = hasWebGPU
  ? "WebGPU detected — decoding will use your GPU."
  : "No WebGPU — falling back to CPU (wasm); expect a few tokens/second.";

async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get("Content-Length") || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let done = 0;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    chunks.push(value);
    done += value.length;
    onProgress && onProgress(done, total);
  }
  const buf = new Uint8Array(done);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf.buffer;
}

$("btn-load").onclick = async () => {
  $("btn-load").disabled = true;
  const bars = $("dl-bars");
  bars.innerHTML = Object.keys(MODELS).map((n) =>
    `<div>${n} <progress id="pg-${n}" max="100" value="0"></progress></div>`).join("");
  try {
    const [cfg, melBin, vocab] = await Promise.all([
      fetch("models/config.json").then((r) => r.json()),
      fetch("models/mel.bin").then((r) => r.arrayBuffer()),
      fetch("models/vocab.json").then((r) => r.json()),
    ]);
    pipe = new MossPipeline(ort, cfg, melBin, vocab);
    pipe.eps = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    await pipe.load(MODELS, fetchWithProgress, (name, done, total) => {
      const pg = $(`pg-${name}`);
      if (pg && total) pg.value = (100 * done) / total;
    });
    $("status").textContent = "Model ready. Record or pick a file.";
    $("btn-rec").disabled = false;
    $("file-in").disabled = false;
    $("loader").style.opacity = 0.6;
  } catch (e) {
    $("status").textContent = "Load failed: " + e.message;
    $("btn-load").disabled = false;
    console.error(e);
  }
};

/* ---- audio input ---------------------------------------------------------- */
async function blobTo16k(blob) {
  const arr = await blob.arrayBuffer();
  const ac = new AudioContext();
  const decoded = await ac.decodeAudioData(arr);
  ac.close();
  const secs = Math.min(decoded.duration, MAX_S);
  const oac = new OfflineAudioContext(1, Math.ceil(secs * 16000), 16000);
  const src = oac.createBufferSource();
  src.buffer = decoded;
  src.connect(oac.destination);
  src.start();
  const out = await oac.startRendering();
  return out.getChannelData(0);
}

let recorder = null, recChunks = [];
$("btn-rec").onclick = async () => {
  const btn = $("btn-rec");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Record";
      btn.classList.remove("rec");
      const wav = await blobTo16k(new Blob(recChunks));
      transcribe(wav);
    };
    recorder.start();
    btn.textContent = "■ Stop";
    btn.classList.add("rec");
    $("status").textContent = "Recording… press Stop when done (max 2 min used).";
    setTimeout(() => { if (recorder.state === "recording") recorder.stop(); }, MAX_S * 1000);
  } catch (e) {
    $("status").textContent = "Mic error: " + e.message;
  }
};

$("file-in").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("status").textContent = `Decoding ${f.name}…`;
  try {
    transcribe(await blobTo16k(f));
  } catch (err) {
    $("status").textContent = "Decode failed: " + err.message;
  }
};

/* ---- transcription -------------------------------------------------------- */
function fmt(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

function render(text) {
  const segs = parseLenient(text);
  const box = $("transcript");
  if (!segs.length) {
    box.innerHTML = `<div class="raw">${text.replace(/</g, "&lt;")}</div>`;
    return;
  }
  const speakers = [...new Set(segs.map((s) => s.speaker))];
  const color = Object.fromEntries(
    speakers.map((s, i) => [s, PALETTE[i % PALETTE.length]]));
  box.innerHTML = segs.map((s) =>
    `<div class="seg"><span class="ts">${fmt(s.start)}–${fmt(s.end)}</span>` +
    `<span class="spk" style="color:${color[s.speaker]}">${s.speaker}</span>` +
    `<span>${s.text.replace(/</g, "&lt;")}</span></div>`).join("");
  box.scrollTop = box.scrollHeight;
}

let busy = false;
async function transcribe(wav) {
  if (!pipe || busy) return;
  busy = true;
  $("btn-rec").disabled = true;
  $("file-in").disabled = true;
  const secs = wav.length / 16000;
  $("status").textContent = `Encoding ${secs.toFixed(0)} s of audio…`;
  $("stats").textContent = "";
  $("transcript").innerHTML = '<span class="sub">…</span>';
  try {
    const res = await pipe.generate(wav, {
      maxNew: 1500,
      onToken: (text, n, dt) => {
        render(s2tw(text));
        $("status").textContent = `Decoding… ${n} tokens`;
        $("stats").textContent =
          `${(n / dt).toFixed(1)} tok/s · ${dt.toFixed(0)} s elapsed`;
      },
    });
    render(s2tw(res.text));
    $("status").textContent = "Done.";
    $("stats").textContent =
      `${res.nTokens} tokens · ${res.seconds.toFixed(1)} s · ` +
      `${(res.nTokens / res.seconds).toFixed(1)} tok/s · ` +
      `RTF ${(res.seconds / secs).toFixed(2)} (all local)`;
  } catch (e) {
    $("status").textContent = "Inference failed: " + e.message;
    console.error(e);
  } finally {
    busy = false;
    $("btn-rec").disabled = false;
    $("file-in").disabled = false;
  }
}
