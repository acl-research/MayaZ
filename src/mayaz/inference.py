"""In-process inference: renderer, ONNX Runtime session, temperatures and tournament (numpy + tokenizers + onnxruntime).

Batch size 1, no padding. One lock around session.run: concurrent requests queue instead of multiplying memory.

Memory guard, three parts (measured on one long request: 21 options of ~50 tokens and a state filling the 512
positions, so a single pass would be 1,452 tokens; one thread, fresh process, glibc):
1. Tokenizer before session (Model.__init__) and a fixed mmap threshold with malloc_trim after every pass
   (mayaz.runtime.low_memory_malloc): freed activation buffers go back to the OS instead of staying resident.
2. A token budget per forward pass (`max_tokens`, or MAYAZ_MAX_TOKENS). A request whose pass would be longer is
   scored as a tournament of smaller option groups that each fit; a request within the budget is scored exactly
   as before.
3. A pass that still cannot fit raises TokenBudgetError (the server answers 422) rather than risking an OOM kill.

    peak memory   V1 settings, one pass   all three parts, default budget
    17M           321-326 MB              299 MB  (budget 1,536: one pass)
    32M           477 MB                  316 MB  (budget 1,024: 3 passes)
    68M           486-904 MB              309 MB  (budget 1,024: 3 passes)

Defaults (server/app/models.json): 17M 1,536 (the renderer's own cap), 32M and 68M 1,024.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from mayaz.model.render import DEFAULT_RENDER, TOKENIZER_PATH, TextEncoder, collate, render_item
from mayaz.model.tournament import tournament
from mayaz.runtime import Session, softmax, temperature_for
from mayaz.text import option_text

DIRECT_MAX = 64
MIN_BUDGET = 600  # a two-option pass at the full 512-position state renders to at most ~560 tokens


class TokenBudgetError(ValueError):
    """Even a two-option pass exceeds the token budget (only possible with a budget below MIN_BUDGET)."""


def execution_providers() -> list[str]:
    """MAYAZ_EP=cuda runs on a GPU (onnxruntime-gpu, evaluation only); the default and the deployment are CPU."""
    if os.environ.get("MAYAZ_EP", "cpu").lower() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def token_budget(default: int | None = None) -> int:
    """MAYAZ_MAX_TOKENS, else `default`, else the renderer's cap; clamped to [MIN_BUDGET, the renderer's cap]."""
    v = int(os.environ.get("MAYAZ_MAX_TOKENS") or default or DEFAULT_RENDER.max_tokens)
    return max(MIN_BUDGET, min(v, DEFAULT_RENDER.max_tokens))


class Model:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path = TOKENIZER_PATH,
                 temperatures_path: str | Path | None = None, max_tokens: int | None = None):
        # Tokenizer first, with one warm-up encode, then the ONNX session: in the opposite order the tokenizer's
        # one-off allocations land above the session's freed load buffers and pin them, and a 68M pass of ~600
        # tokens peaks at 680-980 MB instead of 315-360 MB (measured, fresh process, glibc).
        self.tok = TextEncoder(tokenizer_path)
        self.tok.ids("warm up the tokenizer before the session is created")
        self.session = Session(model_path, threads=1, providers=execution_providers())
        self.temps = (json.loads(Path(temperatures_path).read_text()) if temperatures_path
                      else {"global": 1.0, "by_bucket": {}, "by_key": {}})
        self.max_tokens = token_budget(max_tokens)
        self.lock = threading.Lock()

    def _probs(self, item: dict) -> tuple[np.ndarray, int]:
        r = render_item(item, self.tok)
        with self.lock:
            z = self.session.logits(collate([r], self.tok.pad_id))[0]
        return softmax(z, temperature_for(self.temps, item["type"], len(z))), len(r.input_ids)

    def decide(self, qtype: str, state, question: str, options: list[tuple[str, str | None]]) -> dict:
        item = {"type": qtype, "state": state, "question": question,
                "options": [{"key": k, "text": option_text(qtype, k, d)} for k, d in options]}
        by = {o["key"]: o for o in item["options"]}
        keys = list(by)
        tokens = 0

        def length(sub: list[str]) -> int:
            return len(render_item({**item, "options": [by[k] for k in sub]}, self.tok).input_ids)

        def group_size(sub: list[str]) -> int:
            """Largest group (< len(sub)) whose longest options still fit one pass; at least 2."""
            longest = sorted(sub, key=lambda k: -len(self.tok.ids(by[k]["text"])))
            for g in range(len(sub) - 1, 2, -1):
                if length(longest[:g]) <= self.max_tokens:
                    return g
            return 2

        def score(sub: list[str]) -> np.ndarray:
            nonlocal tokens
            if len(sub) == 1:  # a group of one (possible with small groups) needs no pass
                return np.ones(1)
            n = length(sub)
            if n > self.max_tokens:  # too long for one pass: a tournament of smaller groups
                if len(sub) <= 2:
                    raise TokenBudgetError(f"a {len(sub)}-option pass needs {n} tokens, budget {self.max_tokens}")
                m = group_size(sub)
                # keep at most half of each group, so every later round is smaller than this one
                return tournament(sub, score, direct_max=m, chunk=m, keep=max(1, min(4, m // 2)))
            p, n = self._probs({**item, "options": [by[k] for k in sub]})
            tokens += n
            return p

        p = tournament(keys, score, direct_max=DIRECT_MAX)
        i = int(np.argmax(p))
        return {"choice": keys[i], "probabilities": dict(zip(keys, p.tolist(), strict=True)),
                "confidence": float(p[i]), "tokens": tokens}
