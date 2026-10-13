"""M3.5: post-hoc temperature per (type x cardinality bucket), fitted on the served (INT8 ONNX) logits of calib-dev.

T minimises gold NLL over a log grid clamped to [0.5, 5]. Keys with fewer than min_n items fall back to their
bucket's temperature, then to the global one (mayaz.runtime.temperature_for does the lookup).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from mayaz.runtime import bucket

T_MIN, T_MAX = 0.5, 5.0
GRID = np.exp(np.linspace(np.log(T_MIN), np.log(T_MAX), 93))
GRID[0], GRID[-1] = T_MIN, T_MAX


def _nll(logits: list[np.ndarray], gold: list[int], T: float) -> float:
    total = 0.0
    for z, g in zip(logits, gold, strict=True):
        z = z / T
        m = z.max()
        total += m + np.log(np.exp(z - m).sum()) - z[g]
    return total


def fit_one(logits: list[np.ndarray], gold: list[int]) -> float:
    return float(GRID[int(np.argmin([_nll(logits, gold, T) for T in GRID]))])


def fit_temperatures(types: list[str], logits: list[np.ndarray], gold: list[int], min_n: int = 50) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, (t, z) in enumerate(zip(types, logits, strict=True)):
        b = bucket(len(z))
        groups[f"{t}|{b}"].append(i)
        buckets[b].append(i)

    def fit(idx):
        return fit_one([logits[i] for i in idx], [gold[i] for i in idx])

    return {"global": fit(range(len(logits))),
            "by_bucket": {b: fit(ix) for b, ix in sorted(buckets.items()) if len(ix) >= min_n},
            "by_key": {k: fit(ix) for k, ix in sorted(groups.items()) if len(ix) >= min_n},
            "n": {k: len(ix) for k, ix in sorted(groups.items())}, "clamp": [T_MIN, T_MAX], "min_n": min_n}
