/* UI + audio capture for the fully-local zh-TW meeting transcriber. */
import { MossPipeline, parseLenientWithTail } from "./pipeline.js";

const ort = window.ort;
const $ = (id) => document.getElementById(id);
function modelSet(quality) {
  return {
    encoder: "models/encoder.int8.onnx",
    embedding: "models/embedding.int8.onnx",
    decoder: quality === "q4" ? "models/decoder.q4.onnx"
                              : "models/decoder.int8.onnx",
    ecapa: "models/ecapa.onnx",
  };
}
const MIC_MAX_S = 120;
const PALETTE = ["#4f7cff", "#e05563", "#2aa876", "#c78c2c", "#9761d8",
                 "#3aa6b9", "#d0679d", "#7a9a01", "#b05c45", "#5c7285"];

let pipe = null;
let abortCtl = null;
let s2tw = (t) => t;
try {
  if (window.OpenCC) s2tw = window.OpenCC.Converter({ from: "cn", to: "tw" });
} catch (e) { console.warn("opencc unavailable", e); }

const hasWebGPU = !!navigator.gpu;
const WINDOW_S = hasWebGPU ? 300 : 180; // wasm: smaller KV to stay in 32-bit memory
$("ep-note").textContent = hasWebGPU
  ? `WebGPU detected — ${WINDOW_S / 60}-minute windows on your GPU.`
  : `No WebGPU — CPU (wasm) fallback, ${WINDOW_S / 60}-minute windows; expect slow decoding.`;

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
  const MODELS = modelSet(
    document.querySelector('input[name="quality"]:checked')?.value || "int8");
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
    $("status").textContent = "Model ready. Record or pick a meeting file.";
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
async function blobTo16k(blob, maxS = 0) {
  const arr = await blob.arrayBuffer();
  const ac = new AudioContext();
  const decoded = await ac.decodeAudioData(arr);
  ac.close();
  const secs = maxS > 0 ? Math.min(decoded.duration, maxS) : decoded.duration;
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
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Record";
      btn.classList.remove("rec");
      transcribe(await blobTo16k(new Blob(recChunks), MIC_MAX_S));
    };
    recorder.start();
    btn.textContent = "■ Stop";
    btn.classList.add("rec");
    $("status").textContent = `Recording… press Stop (max ${MIC_MAX_S / 60} min).`;
    setTimeout(() => { if (recorder.state === "recording") recorder.stop(); },
               MIC_MAX_S * 1000);
  } catch (e) {
    $("status").textContent = "Mic error: " + e.message;
  }
};

$("file-in").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("status").textContent = `Decoding ${f.name}…`;
  try {
    transcribe(await blobTo16k(f)); // full file, no cap
  } catch (err) {
    $("status").textContent = "Decode failed: " + err.message;
  }
};

$("btn-abort").onclick = () => abortCtl && abortCtl.abort();

/* ---- rendering ------------------------------------------------------------ */
function fmt(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60),
        s = Math.floor(t % 60);
  return (h ? h + ":" : "") + String(m).padStart(h ? 2 : 1, "0") + ":" +
         String(s).padStart(2, "0");
}

function renderSegs(segs, tail = "") {
  const box = $("transcript");
  const speakers = [...new Set(segs.map((s) => s.speaker).filter((x) => x !== "…"))];
  const color = Object.fromEntries(
    speakers.map((s, i) => [s, PALETTE[i % PALETTE.length]]));
  color["…"] = "var(--muted, #888)";
  box.innerHTML = segs.map((s) =>
    `<div class="seg"><span class="ts">${fmt(s.start)}–${fmt(s.end)}</span>` +
    `<span class="spk" style="color:${color[s.speaker]}">${s.speaker}</span>` +
    `<span>${s2tw(s.text).replace(/</g, "&lt;")}</span></div>`).join("") +
    (tail ? `<div class="raw">${s2tw(tail).replace(/</g, "&lt;")}</div>` : "");
  box.scrollTop = box.scrollHeight;
  return speakers.length;
}

function offerSrt(segs) {
  const ts = (t) => {
    const h = String(Math.floor(t / 3600)).padStart(2, "0");
    const m = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
    const s = String(Math.floor(t % 60)).padStart(2, "0");
    const ms = String(Math.round((t % 1) * 1000)).padStart(3, "0");
    return `${h}:${m}:${s},${ms}`;
  };
  const srt = segs.map((s, i) =>
    `${i + 1}\n${ts(s.start)} --> ${ts(s.end)}\n[${s.speaker}] ${s2tw(s.text)}\n`
  ).join("\n");
  const url = URL.createObjectURL(new Blob([srt], { type: "text/plain" }));
  const a = $("dl-srt");
  a.href = url;
  a.download = "transcript.srt";
  a.style.display = "inline";
}

/* ---- transcription -------------------------------------------------------- */
let busy = false;
async function transcribe(wav) {
  if (!pipe || busy) return;
  busy = true;
  abortCtl = new AbortController();
  $("btn-rec").disabled = true;
  $("file-in").disabled = true;
  $("btn-abort").style.display = "inline";
  $("dl-srt").style.display = "none";
  const secs = wav.length / 16000;
  const nWin = Math.max(1, Math.ceil(secs / WINDOW_S));
  $("status").textContent =
    `${fmt(secs)} of audio → ${nWin} window(s) of ${WINDOW_S / 60} min…`;
  $("stats").textContent = "";
  $("transcript").innerHTML = '<span class="sub">…</span>';
  const t0 = Date.now();
  let lastLinked = [];
  try {
    const segs = await pipe.transcribeMeeting(wav, {
      windowS: WINDOW_S,
      signal: abortCtl.signal,
      onToken: (wi, nw, text, n, dt) => {
        const off = wi * WINDOW_S;
        const { segs: prov, tail } = parseLenientWithTail(text);
        const live = prov.map((s) => ({
          start: s.start + off, end: s.end + off,
          speaker: "…", text: s.text,
        }));
        renderSegs([...lastLinked, ...live], tail);
        $("status").textContent =
          `Window ${wi + 1}/${nw} · ${n} tokens · ${(n / dt).toFixed(1)} tok/s`;
      },
      onWindow: (linked, done, total) => {
        lastLinked = linked;
        const nSpk = renderSegs(linked);
        const el = (Date.now() - t0) / 1000;
        const eta = (el / done) * (total - done);
        $("stats").textContent =
          `${done}/${total} windows · ${linked.length} segments · ` +
          `${nSpk} speakers · ETA ${fmt(eta)}`;
      },
    });
    const el = (Date.now() - t0) / 1000;
    const nSpk = renderSegs(segs);
    const note = abortCtl.signal.aborted ? " (stopped early)" : "";
    $("status").textContent = "Done." + note;
    $("stats").textContent =
      `${fmt(secs)} audio · ${segs.length} segments · ${nSpk} speakers · ` +
      `${fmt(el)} compute · ${(secs / el).toFixed(2)}× realtime · all local`;
    if (segs.length) offerSrt(segs);
  } catch (e) {
    $("status").textContent = "Inference failed: " + e.message;
    console.error(e);
  } finally {
    busy = false;
    abortCtl = null;
    $("btn-rec").disabled = false;
    $("file-in").disabled = false;
    $("btn-abort").style.display = "none";
  }
}
