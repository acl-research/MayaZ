"""Single-model tournament for option lists longer than one forward pass takes (numpy only; the server imports it).

K <= direct_max: one pass. Otherwise options are dealt into ceil(K / chunk) strided chunks (so neighbours in the
label order, often near-synonyms, are spread across chunks), each chunk is scored in its own pass, the top `keep`
of each chunk advance, and the finalists are scored again (recursively, if there are still too many).
Final distribution: finalists share 1 - eps by the final round; the rest share eps by their first-round probability.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ScoreFn = Callable[[list[str]], np.ndarray]


def tournament(keys: Sequence[str], score: ScoreFn, direct_max: int = 64, chunk: int = 32, keep: int = 4,
               eps: float = 1e-3) -> np.ndarray:
    keys = list(keys)
    k = len(keys)
    if k <= direct_max:
        return np.asarray(score(keys), dtype=np.float64)
    if not 0 < keep < chunk <= direct_max:
        raise ValueError(f"need 0 < keep < chunk <= direct_max, got {keep}, {chunk}, {direct_max}")
    n = -(-k // chunk)
    first = np.zeros(k)
    finalists: list[int] = []
    for j in range(n):
        idx = list(range(j, k, n))
        p = np.asarray(score([keys[i] for i in idx]), dtype=np.float64)
        first[idx] = p / n
        finalists += [idx[t] for t in np.argsort(-p, kind="stable")[:keep]]
    finalists.sort()
    pf = tournament([keys[i] for i in finalists], score, direct_max, chunk, keep, eps)
    out = np.zeros(k)
    rest = np.ones(k, bool)
    rest[finalists] = False
    mass = first[rest].sum()
    out[rest] = eps * first[rest] / mass if mass > 0 else eps / rest.sum()
    out[finalists] = (1 - eps) * pf
    return out
