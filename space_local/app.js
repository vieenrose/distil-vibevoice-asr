/* UI for the fully-local zh-TW meeting transcriber.
 * Examples render instantly from precomputed JSON; user audio runs through
 * the in-browser pipeline (pipeline.js) with token-level streaming. */
import { MossPipeline, parseLenientWithTail } from "./pipeline.js";

const ort = window.ort;
const $ = (id) => document.getElementById(id);
const REPO =
  "https://huggingface.co/Luigi/moss-transcribe-diarize-zhtw-onnx/resolve/main/";
const WEIGHTS = REPO + "web/";
const EXAMPLES = REPO + "demo/";
const MIC_MAX_S = 120;
const PALETTE = ["#4f7cff", "#e05563", "#2aa876", "#c78c2c", "#9761d8",
                 "#3aa6b9", "#d0679d", "#7a9a01", "#b05c45", "#5c7285",
                 "#8884d8", "#c05780", "#4a9d5f", "#b8860b", "#7b68ee",
                 "#20a4b5", "#cc6699", "#899a20"];

const WINDOW_S = 180;  // CPU-only; 180 s windows validated (DER ~= 300 s)
let pipe = null, loading = null, abortCtl = null, busy = false;
let segs = [], rows = [], activeIdx = -1, hiddenSpk = new Set();
let s2tw = (t) => t;
try {
  if (window.OpenCC) s2tw = window.OpenCC.Converter({ from: "cn", to: "tw" });
} catch (e) { console.warn("opencc unavailable", e); }

const nThreads = window.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
$("input-note").textContent =
  `CPU-only · ${nThreads} thread${nThreads > 1 ? "s" : ""}` +
  (nThreads === 1 ? " (multithreading unavailable in this browser)" : "");

/* ============================ model loading ============================ */
function modelSet(quality) {
  return {
    encoder: WEIGHTS + "encoder.int8.onnx",
    embedding: WEIGHTS + "embedding.int8.onnx",
    decoder: WEIGHTS + (quality === "q4" ? "decoder.q4.onnx" : "decoder.int8.onnx"),
    ecapa: WEIGHTS + "ecapa.onnx",
  };
}

async function fetchWithProgress(url, onProgress) {
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
    onProgress && onProgress(done, total);
  }
  const buf = new Uint8Array(done);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf.buffer;
}

function setModelState(cls, msg) {
  $("model-state").className = cls;
  $("model-msg").textContent = msg;
}

function ensureModel() {
  if (pipe) return Promise.resolve();
  if (loading) return loading;
  const quality =
    document.querySelector('input[name="quality"]:checked')?.value || "int8";
  const MODELS = modelSet(quality);
  setModelState("loading", `Downloading model (${quality}) — one-time, browser-cached…`);
  $("btn-load").disabled = true;
  const bars = $("dl-bars");
  bars.innerHTML = Object.keys(MODELS).map((n) =>
    `<div>${n}<progress id="pg-${n}" max="100" value="0"></progress></div>`).join("");
  loading = (async () => {
    const [cfg, melBin, vocab] = await Promise.all([
      fetch("models/config.json").then((r) => r.json()),
      fetch("models/mel.bin").then((r) => r.arrayBuffer()),
      fetch("models/vocab.json").then((r) => r.json()),
    ]);
    const p = new MossPipeline(ort, cfg, melBin, vocab);
    p.eps = ["wasm"];
    ort.env.wasm.numThreads = nThreads;
    await p.load(MODELS, fetchWithProgress, (name, done, total) => {
      const pg = $(`pg-${name}`);
      if (pg && total) pg.value = (100 * done) / total;
    });
    pipe = p;
    bars.innerHTML = "";
    setModelState("ready",
      `Model ready · ${quality} · CPU ×${nThreads} · ${WINDOW_S / 60}-min windows`);
  })().catch((e) => {
    loading = null;
    setModelState("", "Load failed: " + e.message);
    $("btn-load").disabled = false;
    throw e;
  });
  return loading;
}
$("btn-load").onclick = () => ensureModel().catch(() => {});

/* ============================ transcript view ============================ */
function fmt(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60),
        s = Math.floor(t % 60);
  return (h ? h + ":" : "") + String(m).padStart(h ? 2 : 1, "0") + ":" +
         String(s).padStart(2, "0");
}

function colorsFor(list) {
  const speakers = [...new Set(list.map((s) => s.speaker).filter((x) => x !== "…"))];
  const c = Object.fromEntries(speakers.map((s, i) => [s, PALETTE[i % PALETTE.length]]));
  c["…"] = "var(--muted)";
  return c;
}

function renderLegend() {
  const talk = {};
  for (const s of segs) {
    if (s.speaker === "…") continue;
    talk[s.speaker] = (talk[s.speaker] || 0) + (s.end - s.start);
  }
  const color = colorsFor(segs);
  $("legend").innerHTML = Object.keys(talk)
    .sort((a, b) => talk[b] - talk[a])
    .map((s) => `<span class="lg${hiddenSpk.has(s) ? " off" : ""}" data-spk="${s}">` +
                `<i style="background:${color[s]}"></i>${s} · ${fmt(talk[s])}</span>`)
    .join("");
  document.querySelectorAll(".lg").forEach((el) => {
    el.onclick = () => {
      const s = el.dataset.spk;
      hiddenSpk.has(s) ? hiddenSpk.delete(s) : hiddenSpk.add(s);
      el.classList.toggle("off", hiddenSpk.has(s));
      applyFilter();
    };
  });
}

function renderTranscript(tail = "") {
  const color = colorsFor(segs);
  const box = $("transcript");
  box.innerHTML = (segs.map((s, i) =>
    `<div class="seg${s.speaker === "…" ? " live" : ""}" data-i="${i}">` +
    `<span class="ts">${fmt(s.start)}</span>` +
    `<span class="spk" style="color:${color[s.speaker]}">${s.speaker}</span>` +
    `<span>${s2tw(s.text).replace(/</g, "&lt;")}</span></div>`).join("") +
    (tail ? `<div class="tail">${s2tw(tail).replace(/</g, "&lt;")}</div>` : "")) ||
    '<div class="placeholder">…</div>';
  rows = [...box.querySelectorAll(".seg")];
  rows.forEach((r, i) => {
    r.onclick = () => {
      const a = $("audio");
      if ($("player-box").style.display !== "none" && isFinite(segs[i].start)) {
        a.currentTime = segs[i].start + 0.01;
        a.play().catch(() => {});
      }
    };
  });
  applyFilter();
  activeIdx = -1;
  if ($("autoscroll").checked) box.scrollTop = box.scrollHeight;
}

function applyFilter() {
  const q = $("search").value.trim().toLowerCase();
  rows.forEach((r, i) => {
    const s = segs[i];
    r.classList.toggle("hidden",
      hiddenSpk.has(s.speaker) || (q && !s.text.toLowerCase().includes(q)));
  });
}
$("search").oninput = applyFilter;

$("audio").addEventListener("timeupdate", () => {
  const t = $("audio").currentTime;
  let idx = -1;
  for (let i = 0; i < segs.length; i++) {
    if (segs[i].start <= t && t < segs[i].end) { idx = i; break; }
    if (segs[i].start > t) break;
  }
  if (idx === activeIdx) return;
  if (activeIdx >= 0 && rows[activeIdx]) rows[activeIdx].classList.remove("active");
  activeIdx = idx;
  if (idx >= 0 && rows[idx]) {
    rows[idx].classList.add("active");
    if ($("autoscroll").checked && !rows[idx].classList.contains("hidden")) {
      rows[idx].scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
});

function offerDownloads(finalSegs) {
  const ts = (t) => {
    const h = String(Math.floor(t / 3600)).padStart(2, "0");
    const m = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
    const s = String(Math.floor(t % 60)).padStart(2, "0");
    return `${h}:${m}:${s},${String(Math.round((t % 1) * 1000)).padStart(3, "0")}`;
  };
  const srt = finalSegs.map((s, i) =>
    `${i + 1}\n${ts(s.start)} --> ${ts(s.end)}\n[${s.speaker}] ${s2tw(s.text)}\n`).join("\n");
  $("dl-srt").href = URL.createObjectURL(new Blob([srt], { type: "text/plain" }));
  $("dl-srt").style.display = "inline";
  const js = JSON.stringify(finalSegs.map((s) =>
    ({ start: s.start, end: s.end, speaker: s.speaker, text: s2tw(s.text) })), null, 1);
  $("dl-json").href = URL.createObjectURL(new Blob([js], { type: "application/json" }));
  $("dl-json").style.display = "inline";
}

function resetView(placeholder) {
  segs = []; rows = []; hiddenSpk = new Set(); activeIdx = -1;
  $("legend").innerHTML = "";
  $("stats").textContent = "";
  $("dl-srt").style.display = "none";
  $("dl-json").style.display = "none";
  $("bar").style.display = "none";
  $("transcript").innerHTML = `<div class="placeholder">${placeholder}</div>`;
}

/* ============================ examples ============================ */
document.querySelectorAll("[data-ex]").forEach((btn) => {
  btn.onclick = async () => {
    if (busy) return;
    const stem = btn.dataset.ex;
    resetView("載入範例…");
    const data = await (await fetch(`${EXAMPLES}${stem}.json`)).json();
    segs = data.segments.map((s) => ({ ...s }));
    $("audio").src = `${EXAMPLES}${stem}.mp3`;
    $("player-box").style.display = "";
    renderLegend();
    renderTranscript();
    $("transcript").scrollTop = 0;
    const dur = Math.max(...segs.map((s) => s.end));
    const nSpk = new Set(segs.map((s) => s.speaker)).size;
    $("status").textContent =
      "真實會議 · 預先計算（本頁同一條 pipeline 產生）· 點任一句可跳播";
    $("stats").textContent =
      `${fmt(dur)} · ${segs.length} segments · ${nSpk} speakers`;
    offerDownloads(segs);
  };
});

/* ============================ audio input ============================ */
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
  return (await oac.startRendering()).getChannelData(0);
}

const drop = $("drop");
drop.onclick = () => $("file-in").click();
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("hover"); };
drop.ondragleave = () => drop.classList.remove("hover");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
};
$("file-in").onchange = (e) => {
  const f = e.target.files[0];
  if (f) handleFile(f);
  e.target.value = "";
};

async function handleFile(f) {
  if (busy) return;
  resetView(`解析 ${f.name}…`);
  $("status").textContent = `Decoding ${f.name}…`;
  try {
    $("audio").src = URL.createObjectURL(f);
    $("player-box").style.display = "";
    const wav = await blobTo16k(f);
    await transcribe(wav);
  } catch (err) {
    $("status").textContent = "Could not decode this file: " + err.message;
  }
}

let recorder = null, recChunks = [];
$("btn-rec").onclick = async () => {
  const btn = $("btn-rec");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  if (busy) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Record mic";
      btn.classList.remove("rec-on");
      const blob = new Blob(recChunks);
      $("audio").src = URL.createObjectURL(blob);
      $("player-box").style.display = "";
      resetView("處理錄音…");
      transcribe(await blobTo16k(blob, MIC_MAX_S));
    };
    recorder.start();
    btn.textContent = "■ Stop & transcribe";
    btn.classList.add("rec-on");
    $("status").textContent = `Recording… (max ${MIC_MAX_S / 60} min)`;
    setTimeout(() => { if (recorder.state === "recording") recorder.stop(); },
               MIC_MAX_S * 1000);
  } catch (e) {
    $("status").textContent = "Mic error: " + e.message;
  }
};

$("btn-abort").onclick = () => abortCtl && abortCtl.abort();

/* ============================ live transcription ============================ */
async function transcribe(wav) {
  if (busy) return;
  busy = true;
  abortCtl = new AbortController();
  $("btn-abort").style.display = "";
  const secs = wav.length / 16000;
  const nWin = Math.max(1, Math.ceil(secs / WINDOW_S));
  const t0 = Date.now();
  let lastLinked = [];
  try {
    await ensureModel();
    $("status").textContent =
      `${fmt(secs)} of audio → ${nWin} × ${WINDOW_S / 60}-min window(s)`;
    $("bar").style.display = "";
    const finalSegs = await pipe.transcribeMeeting(wav, {
      windowS: WINDOW_S,
      signal: abortCtl.signal,
      onStage: (wi, nw, stage, detail) => {
        const what = stage === "encode"
          ? `listening to the audio (chunk ${detail})`
          : `reading it into the model (${detail}) — first words follow`;
        $("status").textContent = `Window ${wi + 1}/${nw} · ${what}…`;
      },
      onToken: (wi, nw, text, n, dt) => {
        const off = wi * WINDOW_S;
        const { segs: prov, tail } = parseLenientWithTail(text);
        segs = [...lastLinked, ...prov.map((s) => ({
          start: s.start + off, end: s.end + off, speaker: "…", text: s.text,
        }))];
        renderTranscript(tail);
        $("status").textContent =
          `Window ${wi + 1}/${nw} · ${(n / dt).toFixed(1)} tok/s`;
      },
      onWindow: (linked, done, total) => {
        lastLinked = linked;
        segs = linked;
        renderLegend();
        renderTranscript();
        const el = (Date.now() - t0) / 1000;
        const eta = (el / done) * (total - done);
        $("bar").firstElementChild.style.width = `${(100 * done) / total}%`;
        $("stats").textContent =
          `${done}/${total} windows · ${linked.length} segments · ` +
          `${new Set(linked.map((s) => s.speaker)).size} speakers` +
          (done < total ? ` · ~${fmt(eta)} left` : "");
      },
    });
    segs = finalSegs;
    renderLegend();
    renderTranscript();
    const el = (Date.now() - t0) / 1000;
    $("bar").firstElementChild.style.width = "100%";
    $("status").textContent =
      abortCtl.signal.aborted ? "Stopped — partial result kept." : "Done · all local.";
    $("stats").textContent =
      `${fmt(secs)} audio · ${finalSegs.length} segments · ` +
      `${new Set(finalSegs.map((s) => s.speaker)).size} speakers · ` +
      `${fmt(el)} compute (${(secs / el).toFixed(2)}× realtime)`;
    if (finalSegs.length) offerDownloads(finalSegs);
  } catch (e) {
    if (pipe) $("status").textContent = "Inference failed: " + e.message;
    console.error(e);
  } finally {
    busy = false;
    abortCtl = null;
    $("btn-abort").style.display = "none";
  }
}
