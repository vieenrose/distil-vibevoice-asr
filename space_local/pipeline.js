/* MOSS-Transcribe-Diarize zh-TW — in-browser inference pipeline.
 *
 * Environment-agnostic core: pass in an `ort` implementation
 * (onnxruntime-web in the browser, onnxruntime-node for offline tests).
 *
 * Stages (mirrors scripts/42_web_pipeline_sim.py exactly):
 *   melSpectrogram()  Whisper 80-bin log-mel, 30 s chunks padded to 3000 frames
 *   encodeAudio()     encoder.onnx, keep floor(floor(frames/2)/4) tokens/chunk
 *   generate()        embedding.onnx + splice + decoder.onnx greedy loop with
 *                     dynamic KV chaining (past_k_i/past_v_i <- present_*)
 *   decodeTokens()    GPT-2 byte-level BPE decode (vocab.json)
 *   parseLenient()    [start][Sxx]text[end] segments, tag-drop tolerant
 */

export class MossPipeline {
  constructor(ort, cfg, melBin, vocab) {
    this.ort = ort;
    this.cfg = cfg;
    const f32 = new Float32Array(melBin);
    this.window = f32.slice(0, cfg.n_fft);
    this.filters = f32.slice(cfg.n_fft); // [n_freq * n_mel], row-major [freq][mel]
    this.vocab = vocab;                  // id -> token string
    this.byteDecoder = buildByteDecoder();
    // precompute DFT basis for arbitrary n_fft (400): [n_freq][n_fft] cos/sin
    const { n_fft, n_freq } = cfg;
    this.dftCos = new Float32Array(n_freq * n_fft);
    this.dftSin = new Float32Array(n_freq * n_fft);
    for (let k = 0; k < n_freq; k++) {
      for (let n = 0; n < n_fft; n++) {
        const a = (2 * Math.PI * k * n) / n_fft;
        this.dftCos[k * n_fft + n] = Math.cos(a);
        this.dftSin[k * n_fft + n] = Math.sin(a);
      }
    }
    this.sessions = {};
  }

  async load(urls, fetchFn, onProgress) {
    for (const [name, url] of Object.entries(urls)) {
      const buf = await fetchFn(url, (done, total) =>
        onProgress && onProgress(name, done, total));
      this.sessions[name] = await this.ort.InferenceSession.create(buf, {
        executionProviders: this.eps || ["wasm"],
      });
    }
  }

  /* ---- mel: matches WhisperFeatureExtractor (center-pad reflect) -------- */
  melSpectrogram(wav /* Float32Array, <=30 s @16 kHz */) {
    const { n_fft, hop, n_mel, n_freq, chunk_frames } = this.cfg;
    const half = n_fft >> 1;
    const nSamples = 480000; // whisper pads/truncates to 30 s before STFT
    const padded = new Float32Array(nSamples + n_fft);
    // signal centered with reflect padding at both ends
    const sig = wav.length > nSamples ? wav.subarray(0, nSamples) : wav;
    padded.set(sig, half);
    for (let i = 0; i < half; i++) {
      padded[half - 1 - i] = sig[i + 1] || 0;
      // right reflect happens inside the zero region only if sig fills 30 s
    }
    if (sig.length === nSamples) {
      for (let i = 0; i < half; i++) {
        padded[half + nSamples + i] = sig[nSamples - 2 - i];
      }
    }
    const nFrames = chunk_frames; // 3000 (whisper drops the last frame)
    const frame = new Float32Array(n_fft);
    const power = new Float32Array(n_freq);
    const mel = new Float32Array(n_mel * nFrames); // [mel][frame]
    for (let t = 0; t < nFrames; t++) {
      const off = t * hop;
      for (let n = 0; n < n_fft; n++) frame[n] = padded[off + n] * this.window[n];
      for (let k = 0; k < n_freq; k++) {
        let re = 0.0, im = 0.0;
        const base = k * n_fft;
        for (let n = 0; n < n_fft; n++) {
          re += this.dftCos[base + n] * frame[n];
          im += this.dftSin[base + n] * frame[n];
        }
        power[k] = re * re + im * im;
      }
      for (let m = 0; m < n_mel; m++) {
        let acc = 0.0;
        for (let k = 0; k < n_freq; k++) acc += this.filters[k * n_mel + m] * power[k];
        mel[m * nFrames + t] = Math.log10(Math.max(acc, 1e-10));
      }
    }
    let mx = -Infinity;
    for (let i = 0; i < mel.length; i++) if (mel[i] > mx) mx = mel[i];
    const floor = mx - 8.0;
    for (let i = 0; i < mel.length; i++) {
      mel[i] = (Math.max(mel[i], floor) + 4.0) / 4.0;
    }
    return mel; // [80 * 3000]
  }

  async encodeAudio(wav /* Float32Array 16 kHz mono */) {
    const { sr, hop, n_mel, chunk_frames } = this.cfg;
    const chunkSamples = sr * 30;
    const parts = [];
    let total = 0;
    for (let off = 0; off < wav.length; off += chunkSamples) {
      const piece = wav.subarray(off, Math.min(off + chunkSamples, wav.length));
      const nFrames = Math.floor(piece.length / hop);
      const keep = Math.floor(Math.floor(nFrames / 2) / 4);
      if (keep === 0) continue;
      const mel = this.melSpectrogram(piece);
      const t = new this.ort.Tensor("float32", mel, [1, n_mel, chunk_frames]);
      const out = await this.sessions.encoder.run({ mel: t });
      const e = out.audio_embeds; // [1, 750, 1024]
      const dim = e.dims[2];
      parts.push(e.data.slice(0, keep * dim));
      total += keep;
    }
    const dim = 1024;
    const all = new Float32Array(total * dim);
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    return { data: all, count: total, dim };
  }

  async embed(ids) {
    const t = new this.ort.Tensor(
      "int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const out = await this.sessions.embedding.run({ input_ids: t });
    const k = Object.keys(out)[0];
    return out[k]; // [1, S, 1024]
  }

  /* ---- greedy generate with streaming callback -------------------------- */
  async generate(wav, { maxNew = 1024, onToken = null, signal = null } = {}) {
    const { prefix_ids, suffix_ids, audio_pad_id, eos_id,
            n_layers, kv_heads, head_dim } = this.cfg;
    const audio = await this.encodeAudio(wav);
    const ids = [...prefix_ids,
                 ...new Array(audio.count).fill(audio_pad_id),
                 ...suffix_ids];
    const embT = await this.embed(ids);
    const dim = embT.dims[2];
    const embs = embT.data instanceof Float32Array
      ? embT.data : new Float32Array(embT.data);
    embs.set(audio.data, prefix_ids.length * dim); // splice

    let feeds = {
      inputs_embeds: new this.ort.Tensor("float32", embs, [1, ids.length, dim]),
      attention_mask: onesMask(this.ort, ids.length),
    };
    for (let i = 0; i < n_layers; i++) {
      const empty = new this.ort.Tensor(
        "float32", new Float32Array(0), [1, kv_heads, 0, head_dim]);
      feeds[`past_k_${i}`] = empty;
      feeds[`past_v_${i}`] = empty;
    }

    const toks = [];
    let text = "";
    const t0 = Date.now();
    for (let step = 0; step < maxNew; step++) {
      if (signal && signal.aborted) break;
      const outs = await this.sessions.decoder.run(feeds);
      const logits = outs.logits;
      const vocab = logits.dims[2];
      const last = (logits.dims[1] - 1) * vocab;
      const data = logits.data;
      let best = 0, bestV = -Infinity;
      for (let v = 0; v < vocab; v++) {
        const x = data[last + v];
        if (x > bestV) { bestV = x; best = v; }
      }
      if (best === eos_id) break;
      toks.push(best);
      text = this.decodeTokens(toks);
      if (onToken) onToken(text, toks.length, (Date.now() - t0) / 1000);
      const curLen = outs.present_k_0.dims[2];
      const e = await this.embed([best]);
      feeds = {
        inputs_embeds: e,
        attention_mask: onesMask(this.ort, curLen + 1),
      };
      for (let i = 0; i < n_layers; i++) {
        feeds[`past_k_${i}`] = outs[`present_k_${i}`];
        feeds[`past_v_${i}`] = outs[`present_v_${i}`];
      }
    }
    return { text, nTokens: toks.length, seconds: (Date.now() - t0) / 1000 };
  }

  decodeTokens(ids) {
    const bytes = [];
    for (const id of ids) {
      const tok = this.vocab[id];
      if (tok === undefined) continue;
      for (const ch of tok) {
        const b = this.byteDecoder[ch];
        if (b !== undefined) bytes.push(b);
      }
    }
    return utf8Decode(Uint8Array.from(bytes));
  }
}

/* ---- lenient [start][Sxx]text[end] parser -------------------------------- */
const SEG_RE =
  /\[(\d{1,7}(?:\.\d{1,3})?)\](?:\[(S\d{1,3})\])?([^\[\]]+?)\[(\d{1,7}(?:\.\d{1,3})?)\]/g;

export function parseLenient(text) {
  const segs = [];
  let prev = "S01";
  SEG_RE.lastIndex = 0;
  let m;
  while ((m = SEG_RE.exec(text)) !== null) {
    const start = parseFloat(m[1]), end = parseFloat(m[4]);
    const spk = m[2] || prev;
    const body = m[3].trim();
    if (body && end >= start) {
      segs.push({ start, end, speaker: spk, text: body });
      prev = spk;
    }
  }
  return segs;
}

/* ---- GPT-2 byte-level decoder -------------------------------------------- */
function buildByteDecoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const dec = {};
  for (let i = 0; i < bs.length; i++) dec[String.fromCharCode(cs[i])] = bs[i];
  return dec;
}

function utf8Decode(bytes) {
  if (typeof TextDecoder !== "undefined") {
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
  return Buffer.from(bytes).toString("utf8");
}

function onesMask(ort, n) {
  const a = new BigInt64Array(n);
  a.fill(1n);
  return new ort.Tensor("int64", a, [1, n]);
}
