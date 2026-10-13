// Line-for-line port of src/mayaz/text.py and src/mayaz/model/render.py. Every token the model sees depends on
// this file: change it only together with the Python originals and keep the two in step.
import { Tokenizer } from "@huggingface/tokenizers";

export const NOUL_DEFAULTS = { true: "Yes, the statement holds.", false: "No, the statement does not hold." };
export const RESERVED = ["<OPT>", "</OPT>"];
export const DEFAULT_RENDER = {
  maxPositions: 512, maxTokens: 1536, optCap: 48, optMin: 8, qCap: 64, minStateTokens: 128,
};
const OPT_TOKEN = "[unused0]";
const CLOSE_TOKEN = "[unused1]";
// Added-token strings the tokenizer would otherwise match inside user text; "[" + " " breaks the match.
const SPECIAL = /\[(?=(?:CLS|SEP|PAD|MASK|UNK|unused\d+)\])/g;

// Python str.strip() (whitespace as str.isspace() defines it, which includes \x1c-\x1f).
const pyStrip = (s) => s.replace(/^[\s\x1c-\x1f\x85]+|[\s\x1c-\x1f\x85]+$/gu, "");

export function humanizeLabel(key) {
  let s = key.replace(/(?<=[a-z0-9])(?=[A-Z])/g, " ");
  s = s.replace(/[_\-]+/g, " ");
  return pyStrip(s.replace(/\s+/g, " ")).toLowerCase();
}

// Python json.dumps(obj, ensure_ascii=False): ", " and ": " separators. Non-integer floats are printed by JS
// rules; integral floats (1.0) cannot be told apart from ints in JS, so send such states as strings.
function pyJson(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return `[${v.map(pyJson).join(", ")}]`;
  if (typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(", ")}}`;
  return JSON.stringify(v);
}

export const stateText = (state) => (typeof state === "string" ? pyStrip(state) : pyJson(state));

export function optionText(qtype, key, desc) {
  const d = pyStrip(desc ?? "");
  if (qtype === "noul") return d || NOUL_DEFAULTS[key];
  if (qtype === "score") return d || key;
  const name = humanizeLabel(key);
  return d ? `${name}: ${d}` : name;
}

export class TextEncoder {
  constructor(tokenizerJson) {
    this.tok = new Tokenizer(tokenizerJson, {});
    const vocab = Object.fromEntries((tokenizerJson.added_tokens ?? []).map((t) => [t.content, t.id]));
    const id = (t) => vocab[t] ?? this.tok.model.tokens_to_ids.get(t);
    [this.clsId, this.sepId, this.padId] = [id("[CLS]"), id("[SEP]"), id("[PAD]")];
    [this.optId, this.closeId] = [id(OPT_TOKEN), id(CLOSE_TOKEN)];
    this.cache = new Map();
  }

  ids(text) {
    let r = this.cache.get(text);
    if (r === undefined) {
      if (this.cache.size > 50_000) this.cache.clear();
      r = this.tok.encode(text.replace(SPECIAL, "[ "), { add_special_tokens: false }).ids;
      this.cache.set(text, r);
    }
    return r;
  }
}

function fit(state, q, opts, cfg) {
  const lens = opts.map((o) => o.length);
  const optTokens = (c) => lens.reduce((a, n) => a + Math.min(n, c) + 2, 0);
  const fixed = 3 + q.length;
  const reserve = Math.min(state.length, cfg.minStateTokens);
  let c = cfg.optCap;
  while (c > cfg.optMin && fixed + optTokens(c) + reserve > cfg.maxTokens) c -= 1;
  const longest = opts.length ? Math.min(Math.max(...lens), c) + 2 : 0;
  const s = Math.min(state.length, cfg.maxTokens - fixed - optTokens(c), cfg.maxPositions - fixed - longest);
  if (s < 0) throw new Error(`item does not fit: ${opts.length} options, question ${q.length} tokens`);
  return [c, s];
}

// item: {type, state, question, options: [{key, text}]} (texts already built with optionText)
export function renderItem(item, tok, cfg = DEFAULT_RENDER) {
  const state = tok.ids(stateText(item.state));
  const q = tok.ids(pyStrip(item.question)).slice(0, cfg.qCap);
  const opts = item.options.map((o) => tok.ids(o.text).slice(0, cfg.optCap));
  const [c, s] = fit(state, q, opts, cfg);
  const ids = [tok.clsId, ...state.slice(0, s), tok.sepId, ...q, tok.sepId];
  const qStart = s + 2;
  const qEnd = s + 2 + q.length;
  const p0 = ids.length;
  const pos = [...Array(p0).keys()];
  const seg = Array(p0).fill(0);
  const starts = [];
  const ends = [];
  opts.forEach((o, k) => {
    const span = [tok.optId, ...o.slice(0, c), tok.closeId];
    starts.push(ids.length);
    ends.push(ids.length + span.length - 1);
    span.forEach((t, j) => { ids.push(t); pos.push(p0 + j); seg.push(k + 1); });
  });
  return { inputIds: ids, positionIds: pos, segmentIds: seg, optStart: starts, optEnd: ends, qStart, qEnd };
}

// Batch of one, no padding (what the server and the browser run). Returns plain typed arrays + shapes.
export function collateOne(r) {
  const L = r.inputIds.length;
  const K = r.optStart.length;
  const qMask = new Uint8Array(L);
  for (let i = r.qStart; i < r.qEnd; i++) qMask[i] = 1;
  const i64 = (a) => BigInt64Array.from(a, BigInt);
  return {
    input_ids: [i64(r.inputIds), [1, L], "int64"],
    position_ids: [i64(r.positionIds), [1, L], "int64"],
    segment_ids: [i64(r.segmentIds), [1, L], "int64"],
    q_mask: [qMask, [1, L], "bool"],
    opt_start: [i64(r.optStart), [1, K], "int64"],
    opt_end: [i64(r.optEnd), [1, K], "int64"],
    opt_mask: [new Uint8Array(K).fill(1), [1, K], "bool"],
  };
}
