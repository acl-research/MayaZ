"""M3.5: temperatures per (type x cardinality bucket), fitted on served logits, clamped to [0.5, 5]."""
import numpy as np

from mayaz.eval.calibration import fit_temperatures
from mayaz.eval.metrics import BUCKETS, bucket_of
from mayaz.runtime import bucket, softmax, temperature_for


def test_runtime_bucket_matches_metrics():
    for k in range(2, 300):
        assert bucket(k) == bucket_of(k)
    assert set(BUCKETS) == {bucket(k) for k in (2, 3, 6, 21)}


def _data(rng, n, k, scale, acc):
    rows = []
    for _ in range(n):
        z = rng.normal(size=k)
        g = int(np.argmax(z)) if rng.random() < acc else int(rng.integers(k))
        rows.append(("choice", z * scale, g))
    return rows


def test_overconfident_logits_get_temperature_above_one():
    rng = np.random.default_rng(0)
    rows = _data(rng, 400, 4, 8.0, 0.6)
    t = fit_temperatures([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], min_n=30)
    assert t["by_key"]["choice|3-5"] > 1.5
    assert 0.5 <= t["global"] <= 5.0


def test_clamp_and_fallbacks():
    rng = np.random.default_rng(1)
    rows = _data(rng, 300, 2, 100.0, 0.5) + _data(rng, 5, 7, 1.0, 0.9)
    t = fit_temperatures([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], min_n=30)
    assert t["by_key"]["choice|2"] == 5.0  # clamped
    assert "choice|6-20" not in t["by_key"]  # too few items: falls back
    assert temperature_for(t, "choice", 7) == t["by_bucket"].get("6-20", t["global"])
    assert temperature_for(t, "unknown", 2) == t["by_bucket"]["2"]


def test_softmax_temperature_and_mask():
    p = softmax(np.array([[2.0, 0.0, -1e4]]), 2.0)
    assert np.isclose(p.sum(), 1) and p[0, 2] == 0.0
    assert np.isclose(p[0, 0] / p[0, 1], np.e)
