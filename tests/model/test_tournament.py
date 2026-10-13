"""Single-model tournament for option lists longer than one forward pass takes (locked decision; numpy only)."""
import numpy as np
import pytest

from mayaz.model.tournament import tournament


def scorer(scores):
    calls = []

    def fn(keys):
        calls.append(list(keys))
        s = np.array([scores[k] for k in keys])
        e = np.exp(s - s.max())
        return e / e.sum()
    return fn, calls


def test_direct_when_small():
    keys = [f"k{i}" for i in range(10)]
    fn, calls = scorer({k: float(i) for i, k in enumerate(keys)})
    p = tournament(keys, fn, direct_max=64)
    assert calls == [keys]
    assert p.shape == (10,) and np.isclose(p.sum(), 1.0) and p.argmax() == 9


@pytest.mark.parametrize("k", [65, 77, 151, 400])
def test_large_lists_find_the_best_option(k):
    rng = np.random.default_rng(k)
    keys = [f"k{i}" for i in range(k)]
    sc = dict(zip(keys, rng.normal(size=k).tolist(), strict=True))
    fn, calls = scorer(sc)
    p = tournament(keys, fn, direct_max=64, chunk=32, keep=4)
    assert np.isclose(p.sum(), 1.0) and (p >= 0).all()
    assert keys[int(p.argmax())] == max(sc, key=sc.get)
    assert all(len(c) <= 64 for c in calls)
    # every option is scored in some first-round chunk
    assert set().union(*map(set, calls)) == set(keys)


def test_finalists_carry_almost_all_mass_and_order_is_kept():
    keys = [f"k{i}" for i in range(100)]
    fn, _ = scorer({k: (10.0 if k in ("k3", "k70") else 0.0) for k in keys})
    p = tournament(keys, fn, direct_max=64, chunk=32, keep=4, eps=1e-3)
    assert p[3] + p[70] > 0.9
    assert np.isclose(p[3], p[70])


def test_chunks_interleave_so_neighbours_do_not_compete_alone():
    keys = [f"k{i}" for i in range(130)]
    fn, calls = scorer({k: 0.0 for k in keys})
    tournament(keys, fn, direct_max=64, chunk=32, keep=4)
    first = calls[0]
    assert first[:2] != ["k0", "k1"]  # strided, not contiguous
