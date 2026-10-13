"""Decision metrics.

Definitions (KL and Brier verified against the typed-decisions card's Uniform row):
- accuracy: argmax(pred) == gold_label; ties split credit (1/|argmax set| if gold is in it).
- kl_from_gold: mean KL(gold || pred), pred floored at 1e-12, zero-gold terms dropped.
- brier: mean over decisions of sum_k (pred_k - gold_k)^2 against the soft gold.
- ece: 10 equal-width bins on top-label confidence max(pred), correctness = tie-split accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

BUCKETS = ("2", "3-5", "6-20", "21+")


def bucket_of(k: int) -> str:
    if k < 2:
        raise ValueError(f"need at least 2 options, got {k}")
    if k == 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 20:
        return "6-20"
    return "21+"


@dataclass
class Decision:
    kind: str
    keys: list[str]
    pred: np.ndarray
    gold: np.ndarray
    gold_label: str
    group: str = ""

    @property
    def bucket(self) -> str:
        return bucket_of(len(self.keys))


def _correct(x: Decision) -> float:
    top = np.flatnonzero(np.isclose(x.pred, x.pred.max(), rtol=0, atol=1e-12))
    gi = x.keys.index(x.gold_label)
    return (gi in top) / len(top)


def accuracy(ds: list[Decision]) -> float:
    return float(np.mean([_correct(x) for x in ds]))


def kl_from_gold(ds: list[Decision], floor: float = 1e-12) -> float:
    vals = []
    for x in ds:
        g, p = x.gold, np.clip(x.pred, floor, None)
        m = g > 0
        vals.append(float(np.sum(g[m] * np.log(g[m] / p[m]))))
    return float(np.mean(vals))


def brier(ds: list[Decision]) -> float:
    return float(np.mean([np.sum((x.pred - x.gold) ** 2) for x in ds]))


def ece(ds: list[Decision], n_bins: int = 10) -> float:
    conf = np.array([x.pred.max() for x in ds])
    corr = np.array([_correct(x) for x in ds])
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(total)


def _block(ds: list[Decision]) -> dict:
    return {
        "n": len(ds), "accuracy": accuracy(ds), "kl": kl_from_gold(ds), "brier": brier(ds), "ece": ece(ds),
    }


def summarize(ds: list[Decision]) -> dict:
    by_bucket, by_type = defaultdict(list), defaultdict(list)
    for x in ds:
        by_bucket[x.bucket].append(x)
        by_type[x.kind].append(x)
    return {
        "overall": _block(ds),
        "by_bucket": {b: _block(by_bucket[b]) for b in BUCKETS if b in by_bucket},
        "by_type": {t: _block(v) for t, v in sorted(by_type.items())},
    }
