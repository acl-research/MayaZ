"""Shared item renderer: training, ONNX export and the server all import this module (numpy + tokenizers only).

Row layout, one question per forward pass:

    [CLS] state [SEP] question [SEP] [OPT] opt_1 [/OPT] [OPT] opt_2 [/OPT] ... [OPT] opt_K [/OPT]

Parallel position ids: shared tokens count 0..P0-1 and every option restarts at P0, so no option is "first".
Segment ids: 0 for shared tokens, k+1 for option k, -1 for padding. The model builds its attention masks from
segment and position ids (option k sees the shared tokens and itself; shared tokens see everything).
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from mayaz.text import state_text

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizer.json"
OPT_TOKEN, CLOSE_TOKEN = "[unused0]", "[unused1]"
# Added-token strings the tokenizer would otherwise match inside user text; "[" + " " breaks the match.
_SPECIAL = re.compile(r"\[(?=(?:CLS|SEP|PAD|MASK|UNK|unused\d+)\])")


@dataclass(frozen=True)
class RenderConfig:
    max_positions: int = 512  # position span: shared tokens + the longest option must fit
    max_tokens: int = 1536  # total sequence length (all options side by side)
    opt_cap: int = 48  # content tokens per option, never divided by K
    opt_min: int = 8  # options are trimmed to a common cap no lower than this before the state is cut
    q_cap: int = 64
    min_state_tokens: int = 128


DEFAULT_RENDER = RenderConfig()


@dataclass(frozen=True)
class Rendered:
    input_ids: list[int]
    position_ids: list[int]
    segment_ids: list[int]
    opt_start: list[int]  # index of each option's [OPT]
    opt_end: list[int]  # index of each option's [/OPT]
    q_start: int  # question tokens are input_ids[q_start:q_end]
    q_end: int
    state_len: int


class TextEncoder:
    def __init__(self, path: str | Path = TOKENIZER_PATH):
        self.tok = Tokenizer.from_file(str(path))
        self.tok.no_padding()
        self.tok.no_truncation()
        tid = self.tok.token_to_id
        self.cls_id, self.sep_id, self.pad_id, self.mask_id = tid("[CLS]"), tid("[SEP]"), tid("[PAD]"), tid("[MASK]")
        self.opt_id, self.close_id = tid(OPT_TOKEN), tid(CLOSE_TOKEN)
        self._cache: dict[str, list[int]] = {}

    def ids(self, text: str) -> list[int]:
        if text not in self._cache:
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[text] = self.tok.encode(_SPECIAL.sub("[ ", text), add_special_tokens=False).ids
        return self._cache[text]


def _fit(state: list[int], q: list[int], opts: list[list[int]], cfg: RenderConfig) -> tuple[int, int]:
    """Return (common option cap, state length). Options shrink to a shared cap (down to opt_min) while the state
    would get fewer than min_state_tokens; only then is the state cut."""
    k = len(opts)
    lens = [len(o) for o in opts]

    def opt_tokens(c: int) -> int:
        return sum(min(n, c) + 2 for n in lens)

    fixed = 3 + len(q)
    reserve = min(len(state), cfg.min_state_tokens)
    c = cfg.opt_cap
    while c > cfg.opt_min and fixed + opt_tokens(c) + reserve > cfg.max_tokens:
        c -= 1
    longest = min(max(lens), c) + 2 if k else 0
    s = min(len(state), cfg.max_tokens - fixed - opt_tokens(c), cfg.max_positions - fixed - longest)
    if s < 0:
        raise ValueError(f"item does not fit: {k} options, question {len(q)} tokens, config {cfg}")
    return c, s


def render_item(item: dict, tok: TextEncoder, cfg: RenderConfig = DEFAULT_RENDER) -> Rendered:
    state = tok.ids(state_text(item["state"]))
    q = tok.ids(item["question"].strip())[: cfg.q_cap]
    opts = [tok.ids(o["text"])[: cfg.opt_cap] for o in item["options"]]
    c, s = _fit(state, q, opts, cfg)
    ids = [tok.cls_id, *state[:s], tok.sep_id, *q, tok.sep_id]
    q_start, q_end = s + 2, s + 2 + len(q)
    p0 = len(ids)
    pos, seg = list(range(p0)), [0] * p0
    starts, ends = [], []
    for k, o in enumerate(opts):
        span = [tok.opt_id, *o[:c], tok.close_id]
        starts.append(len(ids))
        ends.append(len(ids) + len(span) - 1)
        ids += span
        pos += range(p0, p0 + len(span))
        seg += [k + 1] * len(span)
    return Rendered(ids, pos, seg, starts, ends, q_start, q_end, s)


@functools.cache
def _default_pad_id() -> int:
    return Tokenizer.from_file(str(TOKENIZER_PATH)).token_to_id("[PAD]")


def collate(items: list[Rendered], pad_id: int | None = None) -> dict[str, np.ndarray]:
    """Pad a batch. Padded tokens: pad id, position 0, segment -1. Padded options: index 0, opt_mask False."""
    if pad_id is None:
        pad_id = _default_pad_id()
    b, L, K = len(items), max(len(r.input_ids) for r in items), max(len(r.opt_start) for r in items)
    out = {"input_ids": np.full((b, L), pad_id, np.int64), "position_ids": np.zeros((b, L), np.int64),
           "segment_ids": np.full((b, L), -1, np.int64), "q_mask": np.zeros((b, L), bool),
           "opt_start": np.zeros((b, K), np.int64), "opt_end": np.zeros((b, K), np.int64),
           "opt_mask": np.zeros((b, K), bool)}
    for i, r in enumerate(items):
        n, k = len(r.input_ids), len(r.opt_start)
        out["input_ids"][i, :n] = r.input_ids
        out["position_ids"][i, :n] = r.position_ids
        out["segment_ids"][i, :n] = r.segment_ids
        out["q_mask"][i, r.q_start:r.q_end] = True
        out["opt_start"][i, :k] = r.opt_start
        out["opt_end"][i, :k] = r.opt_end
        out["opt_mask"][i, :k] = True
    return out
