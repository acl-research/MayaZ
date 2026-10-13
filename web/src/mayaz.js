// MayaZ in the browser / Node: ONNX Runtime Web (WASM, single thread) + the ported renderer.
// Port of src/mayaz/runtime.py (softmax, buckets, temperatures), src/mayaz/model/tournament.py and
// server/app/inference.py (decide). Usage:
//   const m = await MayaZ.load({ model: <ArrayBuffer|URL>, tokenizer: <tokenizer.json object>, temperatures });
//   const r = await m.decide("choice", "The card was charged twice.", "Which intent?", [["refund"], ["greeting"]]);
import * as ort from "onnxruntime-web";
import { TextEncoder, collateOne, optionText, renderItem } from "./render.js";

export const DIRECT_MAX = 64;

export function softmax(z, t = 1) {
  const s = z.map((x) => x / t);
  const m = Math.max(...s);
  const e = s.map((x) => Math.exp(x - m));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}

export function bucket(k) {
  if (k < 2) throw new Error(`need at least 2 options, got ${k}`);
  return k === 2 ? "2" : k <= 5 ? "3-5" : k <= 20 ? "6-20" : "21+";
}

export function temperatureFor(temps, qtype, k) {
  const b = bucket(k);
  return temps.by_key[`${qtype}|${b}`] ?? temps.by_bucket[b] ?? temps.global;
}

// Same rounds and stable ordering as the Python tournament (strided chunks, top `keep` advance).
export async function tournament(keys, score, directMax = DIRECT_MAX, chunk = 32, keep = 4, eps = 1e-3) {
  const k = keys.length;
  if (k <= directMax) return score(keys);
  const n = Math.ceil(k / chunk);
  const first = new Float64Array(k);
  let finalists = [];
  for (let j = 0; j < n; j++) {
    const idx = [];
    for (let i = j; i < k; i += n) idx.push(i);
    const p = await score(idx.map((i) => keys[i]));
    idx.forEach((i, t) => { first[i] = p[t] / n; });
    const order = [...p.keys()].sort((a, b) => p[b] - p[a] || a - b);  // stable descending
    finalists.push(...order.slice(0, keep).map((t) => idx[t]));
  }
  finalists.sort((a, b) => a - b);
  const pf = await tournament(finalists.map((i) => keys[i]), score, directMax, chunk, keep, eps);
  const isFinal = new Set(finalists);
  let mass = 0;
  let nRest = 0;
  for (let i = 0; i < k; i++) if (!isFinal.has(i)) { mass += first[i]; nRest += 1; }
  const out = new Array(k);
  for (let i = 0; i < k; i++) if (!isFinal.has(i)) out[i] = mass > 0 ? (eps * first[i]) / mass : eps / nRest;
  finalists.forEach((i, t) => { out[i] = (1 - eps) * pf[t]; });
  return out;
}

export class MayaZ {
  static async load({ model, tokenizer, temperatures, wasmPaths }) {
    ort.env.wasm.numThreads = 1;
    if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
    let bytes = model;
    if (typeof model === "string") {
      const r = await fetch(model);
      if (!r.ok) throw new Error(`${model}: HTTP ${r.status}`);  // not "protobuf parsing failed" on a 404 page
      bytes = await r.arrayBuffer();
    }
    bytes = new Uint8Array(bytes);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"],
      graphOptimizationLevel: "all", enableCpuMemArena: false });
    return new MayaZ(session, new TextEncoder(tokenizer), temperatures);
  }

  constructor(session, tok, temps) {
    this.session = session;
    this.tok = tok;
    this.temps = temps;
  }

  // Raw logits for one rendered item (handy for parity checks against the Python pipeline).
  async logits(item) {
    const r = renderItem(item, this.tok);
    const feeds = Object.fromEntries(Object.entries(collateOne(r)).map(
      ([name, [data, dims, type]]) => [name, new ort.Tensor(type, data, dims)]));
    const out = await this.session.run(feeds);
    return { logits: Array.from(out.logits.data), tokens: r.inputIds.length, rendered: r };
  }

  async probs(item) {
    const { logits, tokens } = await this.logits(item);
    return { p: softmax(logits, temperatureFor(this.temps, item.type, logits.length)), tokens };
  }

  // options: [[key, description?], ...]  ->  {choice, probabilities, confidence, tokens}
  async decide(qtype, state, question, options) {
    const opts = options.map(([key, desc]) => ({ key, text: optionText(qtype, key, desc ?? null) }));
    const byKey = Object.fromEntries(opts.map((o) => [o.key, o]));
    const keys = opts.map((o) => o.key);
    let tokens = 0;
    const score = async (sub) => {
      const { p, tokens: n } = await this.probs({ type: qtype, state, question, options: sub.map((k) => byKey[k]) });
      tokens += n;
      return p;
    };
    const p = await tournament(keys, score);
    const i = p.indexOf(Math.max(...p));
    return { choice: keys[i], probabilities: Object.fromEntries(keys.map((k, j) => [k, p[j]])), confidence: p[i],
      tokens };
  }
}
