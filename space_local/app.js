/* UI for the fully-local zh-TW meeting transcriber.
 * Examples render instantly from precomputed JSON; user audio runs through
 * the in-browser pipeline (pipeline.js) with token-level streaming. */
import { parseLenientWithTail } from "./pipeline.js";
import { itn } from "./itn.js";

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

// iOS Safari caps tab memory ~1.3-1.5GB; the mobile model set (q4 + fp16 KV)
// plus a shorter window fits. Detect and default accordingly.
// Window size trades diarization consistency (longer = better on very long
// meetings: 90s cons 0.874, 300s 0.905 on a 123-min meeting) against decode
// speed (longer = larger KV cache = slower tail). User-selectable; 90s ~= 180s
// on accuracy, 300s squeezes out ~0.03 consistency at the cost of speed.
function currentWindowS() {
  return +(document.querySelector('input[name="win"]:checked')?.value || 180);
}
let busy = false, aborted = false;
let segs = [], rows = [], activeIdx = -1, hiddenSpk = new Set();
let lastLinked = []; // segments finalized by prior windows in the current run

// A new file/mic/example request starting WHILE a previous one is still
// running (e.g. dropping a second file before the first finishes) used to be
// silently ignored — `busy` was only ever set true deep inside transcribe(),
// AFTER resetView() and an async audio-decode had already run, leaving a
// real window where two requests could race. From the outside this looked
// exactly like "I picked a new file but the transcript still shows the
// previous clip": the previous run just kept streaming into a view the user
// thought belonged to the new file. Fix: claim the lock SYNCHRONOUSLY,
// before any await, at every entry point, and visibly disable the controls
// that could start a second request while one is in flight.
function acquireBusy() {
  if (busy) return false;
  busy = true;
  setControlsEnabled(false);
  return true;
}
function releaseBusy() {
  busy = false;
  setControlsEnabled(true);
}
function setControlsEnabled(enabled) {
  $("file-in").disabled = !enabled;
  drop.classList.toggle("disabled", !enabled);
  document.querySelectorAll("[data-ex]").forEach((b) => { b.disabled = !enabled; });
}
let _s2tw = (t) => t;
try {
  if (window.OpenCC) _s2tw = window.OpenCC.Converter({ from: "cn", to: "tw" });
} catch (e) { console.warn("opencc unavailable", e); }
// written-form post-processing: Traditional script (s2tw) then number ITN.
const s2tw = (t) => itn(_s2tw(t));

// HTML-escaped display text, CACHED by source string: renderTranscript
// re-runs on every decoded token, and without this cache it re-converts
// (OpenCC + ITN + HTML-escape) EVERY prior segment's full text on EVERY
// token — for a long meeting with many already-closed segments, that's
// thousands of redundant conversions per window and is the main driver of
// the UI becoming unresponsive on longer audio. Segment text is immutable
// once a segment closes, so caching by the exact string is safe and bounded
// by the number of distinct segments ever seen in the session. (The
// in-progress "tail" text changes every token by definition, so it's
// converted directly, uncached, below — it's a single string, not O(n).)
const dispCache = new Map();
function dispText(text) {
  let v = dispCache.get(text);
  if (v === undefined) {
    v = s2tw(text).replace(/</g, "&lt;");
    dispCache.set(text, v);
  }
  return v;
}

$("input-note").textContent =
  "CPU-only · runs in a background worker";

/* ============================ inference worker ============================ */
let worker = null, workerReady = false, onWorkerMsg = null;
function getWorker() {
  if (worker) return worker;
  worker = new Worker("infer-worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "dl") {
      const pg = $(`pg-${m.name}`);
      if (pg && m.total) pg.value = (100 * m.done) / m.total;
      dlAgg[m.name] = { done: m.done, total: m.total };
      const done = Object.values(dlAgg).reduce((a, x) => a + x.done, 0);
      const total = Object.values(dlAgg).reduce((a, x) => a + (x.total || 0), 0);
      dlState = { done, total, files: Object.keys(dlAgg).length };
      beat("downloading model");
    } else if (m.type === "ready") {
      dlState = null;
      workerReady = true;
      $("dl-bars").innerHTML = "";
      setModelState("ready",
        `Model ready · CPU ×${m.threads}`);
    }
    if (onWorkerMsg) onWorkerMsg(m);
  };
  worker.onerror = (e) => {
    setModelState("", "Worker error: " + (e.message || "failed to start"));
    if (onWorkerMsg) onWorkerMsg({ type: "error", message: e.message || "worker failed" });
  };
  return worker;
}

function setModelState(cls, msg) {
  $("model-state").className = cls;
  $("model-msg").textContent = msg;
}

function primeDownloadBars(quality) {
  $("dl-bars").innerHTML = ["encoder", "embedding", "decoder", "ecapa"].map((n) =>
    `<div>${n}<progress id="pg-${n}" max="100" value="0"></progress></div>`).join("");
  setModelState("loading", `Downloading model (${quality}) — one-time, browser-cached…`);
  $("btn-load").disabled = true;
}

let loadKicked = false;
$("btn-load").onclick = () => {
  const quality = "int8";
  if (workerReady || loadKicked) return;
  loadKicked = true;
  primeDownloadBars(quality);
  // a bare load: run with an empty tail so the worker just initializes
  getWorker().postMessage({ type: "run", wav: new Float32Array(16000).buffer,
                            quality, windowS: currentWindowS() });
};

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
    `<span>${dispText(s.text)}</span></div>`).join("") +
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

// The worker posts a "token" message on every single decoded token (needed
// for the live streaming cursor). Two things scale with the total segment
// count accumulated so far (across ALL prior windows, so this only gets
// worse the longer the meeting runs) if done on every token: rebuilding the
// merged segment array, and — far more expensive — the transcript DOM
// render (OpenCC/ITN conversion + full innerHTML rebuild). On a long
// meeting this was the main driver of the tab becoming unresponsive and
// eventually crashing. Both are deferred into a single rAF callback, so
// during a burst of tokens only the LATEST state is kept and the array
// merge + DOM write happen at most once per animation frame instead of
// once per token.
let rafPending = false, rafProv = null, rafTail = "", rafOff = 0;
function scheduleTranscriptRender(prov, tail, off) {
  rafProv = prov; rafTail = tail; rafOff = off;
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    segs = [...lastLinked, ...rafProv.map((s) => ({
      start: s.start + rafOff, end: s.end + rafOff, speaker: "…", text: s.text,
    }))];
    renderTranscript(rafTail);
  });
}
function cancelScheduledRender() {
  rafPending = false;
  rafProv = null;
  rafTail = "";
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
  lastLinked = [];
  cancelScheduledRender();
  dispCache.clear();
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
    if (!acquireBusy()) return;
    try {
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
    } finally {
      releaseBusy();
    }
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
  if (!acquireBusy()) {
    $("status").textContent =
      "Still working on the previous audio — click ✕ Stop first, or wait for it to finish.";
    return;
  }
  resetView(`解析 ${f.name}…`);
  $("status").textContent = `Decoding ${f.name}…`;
  try {
    $("audio").src = URL.createObjectURL(f);
    $("player-box").style.display = "";
    const wav = await blobTo16k(f);
    await transcribe(wav); // transcribe() owns the lock from here; releases it when done
  } catch (err) {
    $("status").textContent = "Could not decode this file: " + err.message;
    releaseBusy();
  }
}

let recorder = null, recChunks = [];
$("btn-rec").onclick = async () => {
  const btn = $("btn-rec");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  // Held for the whole record+transcribe lifecycle, same lock a file upload
  // uses, so a file can't be dropped mid-recording (or vice versa).
  if (!acquireBusy()) {
    $("status").textContent =
      "Still working on the previous audio — click ✕ Stop first, or wait for it to finish.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      btn.textContent = "● Record mic";
      btn.classList.remove("rec-on");
      try {
        const blob = new Blob(recChunks);
        $("audio").src = URL.createObjectURL(blob);
        $("player-box").style.display = "";
        resetView("處理錄音…");
        await transcribe(await blobTo16k(blob, MIC_MAX_S)); // owns the lock from here
      } catch (err) {
        $("status").textContent = "Could not process the recording: " + err.message;
        releaseBusy();
      }
    };
    recorder.start();
    btn.textContent = "■ Stop & transcribe";
    btn.classList.add("rec-on");
    $("status").textContent = `Recording… (max ${MIC_MAX_S / 60} min)`;
    setTimeout(() => { if (recorder.state === "recording") recorder.stop(); },
               MIC_MAX_S * 1000);
  } catch (e) {
    $("status").textContent = "Mic error: " + e.message;
    releaseBusy();
  }
};

$("btn-abort").onclick = () => { aborted = true; if (worker) worker.postMessage({ type: "abort" }); };

/* ---- liveness heartbeat: time since the model last made progress ------- */
let lastBeat = 0, beatWhat = "", runT0 = 0, hbTimer = null;
let dlState = null;  // aggregate download state
let dlAgg = {};
function beat(what) { lastBeat = Date.now(); beatWhat = what; }
function hbStart() {
  runT0 = Date.now();
  beat("starting");
  $("heartbeat").style.display = "flex";
  hbTimer = setInterval(() => {
    const idle = (Date.now() - lastBeat) / 1000;
    const el = (Date.now() - runT0) / 1000;
    const dot = $("hb-dot");
    let note;
    if (dlState) {
      dot.className = "";
      const mb = (dlState.done / 1048576).toFixed(0);
      const tot = dlState.total ? ` / ${(dlState.total / 1048576).toFixed(0)} MB` : " MB";
      $("hb-text").textContent =
        `${fmt(el)} elapsed · downloading model ${mb}${tot} (one-time, cached)`;
      return;
    }
    if (idle < 8) { dot.className = ""; note = "model active"; }
    else if (idle < 90) {
      dot.className = "warn";
      note = `computing a large step — ${idle.toFixed(0)}s since last output (normal for prefill)`;
    } else {
      dot.className = "bad";
      note = `no progress for ${idle.toFixed(0)}s — likely stalled; use ✕ Stop and retry`;
    }
    $("hb-text").textContent =
      `${fmt(el)} elapsed · ${beatWhat} · ${note}`;
  }, 1000);
}
function hbStop() {
  clearInterval(hbTimer);
  $("heartbeat").style.display = "none";
}

/* ============================ live transcription ============================ */
async function transcribe(wav) {
  // Caller (handleFile / mic recorder.onstop) already called acquireBusy()
  // synchronously before doing any async work; this function owns releasing
  // it via finish() below.
  aborted = false;
  $("btn-abort").style.display = "";
  const secs = wav.length / 16000;
  const WINDOW_S = currentWindowS();
  const nWin = Math.max(1, Math.ceil(secs / WINDOW_S));
  const t0 = Date.now();
  lastLinked = [];
  hbStart();
  beat("loading model");
  const quality = "int8";
  if (!workerReady) primeDownloadBars(quality);
  $("status").textContent =
    `${fmt(secs)} of audio → ${nWin} × ${WINDOW_S / 60}-min window(s)`;
  $("bar").style.display = "";

  const finish = () => {
    hbStop(); releaseBusy(); onWorkerMsg = null;
    $("btn-abort").style.display = "none";
  };

  await new Promise((resolve) => {
    onWorkerMsg = (m) => {
      if (m.type === "stage") {
        dlState = null;
        beat(m.stage === "encode" ? `encoding ${m.detail}` : "prefill");
        const what = m.stage === "encode"
          ? `listening to the audio (chunk ${m.detail})`
          : `reading it into the model (${m.detail}) — first words follow`;
        $("status").textContent = `Window ${m.wi + 1}/${m.nw} · ${what}…`;
      } else if (m.type === "token") {
        beat(`decoding · ${(m.n / m.dt).toFixed(1)} tok/s`);
        const off = m.wi * WINDOW_S;
        const { segs: prov, tail } = parseLenientWithTail(m.text);
        scheduleTranscriptRender(prov, tail, off);
        $("status").textContent = `Window ${m.wi + 1}/${m.nw} · ${(m.n / m.dt).toFixed(1)} tok/s`;
      } else if (m.type === "window") {
        beat(`window ${m.done}/${m.total} linked`);
        lastLinked = m.linked;
        cancelScheduledRender();
        segs = m.linked;
        renderLegend(); renderTranscript();
        const el = (Date.now() - t0) / 1000;
        const eta = (el / m.done) * (m.total - m.done);
        $("bar").firstElementChild.style.width = `${(100 * m.done) / m.total}%`;
        $("stats").textContent =
          `${m.done}/${m.total} windows · ${m.linked.length} segments · ` +
          `${new Set(m.linked.map((s) => s.speaker)).size} speakers` +
          (m.done < m.total ? ` · ~${fmt(eta)} left` : "");
      } else if (m.type === "done") {
        cancelScheduledRender();
        segs = m.segs;
        renderLegend(); renderTranscript();
        const el = (Date.now() - t0) / 1000;
        $("bar").firstElementChild.style.width = "100%";
        $("status").textContent =
          m.aborted ? "Stopped — partial result kept." : "Done · all local.";
        $("stats").textContent =
          `${fmt(secs)} audio · ${m.segs.length} segments · ` +
          `${new Set(m.segs.map((s) => s.speaker)).size} speakers · ` +
          `${fmt(el)} compute (${(secs / el).toFixed(2)}× realtime)`;
        if (m.segs.length) offerDownloads(m.segs);
        finish(); resolve();
      } else if (m.type === "error") {
        $("status").textContent = "Inference failed: " + m.message;
        finish(); resolve();
      }
    };
    const buf = wav.buffer.slice(0);
    getWorker().postMessage({ type: "run", wav: buf, quality, windowS: WINDOW_S }, [buf]);
  });
}
