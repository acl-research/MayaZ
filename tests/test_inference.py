"""mayaz.inference: low-memory session options and the per-pass token budget that keeps 512 MB hosts safe."""
import numpy as np
import pytest

from mayaz.inference import MIN_BUDGET, Model, TokenBudgetError, token_budget
from mayaz.model.render import DEFAULT_RENDER
from mayaz.runtime import session_options
from tests.server.toy_model import build_toy_model

LONG_STATE = " ".join(f"Entry {i}: the shipment was checked, signed for and moved to aisle {i % 9}." for i in range(80))
LONG_OPTS = [(f"choice_{i}", "a long description of this outcome with several qualifying details " * 3)
             for i in range(20)]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return build_toy_model(tmp_path_factory.mktemp("toy") / "model.onnx")


def passes(model):
    """Record the sequence length of every forward pass the model runs."""
    seen, run = [], model.session.logits

    def logits(batch):
        seen.append(batch["input_ids"].shape[1])
        return run(batch)
    model.session.logits = logits
    return seen


def test_session_options_disable_arena_and_memory_patterns_on_one_thread():
    o = session_options()
    assert not o.enable_cpu_mem_arena and not o.enable_mem_pattern
    assert o.intra_op_num_threads == 1 and o.inter_op_num_threads == 1


def test_budget_comes_from_env_then_default_and_is_clamped(monkeypatch):
    monkeypatch.delenv("MAYAZ_MAX_TOKENS", raising=False)
    assert token_budget() == DEFAULT_RENDER.max_tokens and token_budget(640) == 640
    assert token_budget(10) == MIN_BUDGET and token_budget(10_000) == DEFAULT_RENDER.max_tokens
    monkeypatch.setenv("MAYAZ_MAX_TOKENS", "700")
    assert token_budget(640) == 700


def test_requests_within_the_budget_are_scored_exactly_as_before(toy):
    opts = [("refund", None), ("card_lost", None), ("exchange_rate", None)]
    a = Model(toy, max_tokens=1536).decide("choice", "My card was charged twice.", "Which intent?", opts)
    small = Model(toy, max_tokens=MIN_BUDGET)
    seen = passes(small)
    b = small.decide("choice", "My card was charged twice.", "Which intent?", opts)
    assert a == b and len(seen) == 1


def test_long_requests_are_split_so_every_pass_fits_the_budget(toy):
    full = Model(toy, max_tokens=1536)
    seen_full = passes(full)
    r_full = full.decide("choice", LONG_STATE, "Which outcome?", LONG_OPTS)
    assert len(seen_full) == 1 and seen_full[0] > 640
    capped = Model(toy, max_tokens=640)
    seen = passes(capped)
    r = capped.decide("choice", LONG_STATE, "Which outcome?", LONG_OPTS)
    assert len(seen) > 1 and max(seen) <= 640
    assert set(r["probabilities"]) == {k for k, _ in LONG_OPTS}
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6 and r["choice"] in r["probabilities"]
    assert np.isfinite(list(r["probabilities"].values())).all() and r_full["tokens"] < r["tokens"]


def test_a_pass_that_cannot_fit_raises(toy):
    m = Model(toy)
    m.max_tokens = 100  # below MIN_BUDGET: only reachable by setting it directly
    with pytest.raises(TokenBudgetError):
        m.decide("noul", LONG_STATE, "Is this fine?", [("true", None), ("false", None)])


def test_many_long_options_at_the_minimum_budget_terminate_and_fit(toy):
    m = Model(toy, max_tokens=MIN_BUDGET)
    seen = passes(m)
    opts = [(f"k{i}", f"outcome {i} " + "with a long qualifying description " * (2 + i % 4)) for i in range(64)]
    r = m.decide("choice", LONG_STATE, "Which outcome?", opts)
    assert max(seen) <= MIN_BUDGET and len(r["probabilities"]) == 64
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_small_groups_with_a_single_leftover_option_are_scored(toy):
    m = Model(toy, max_tokens=MIN_BUDGET)
    seen = passes(m)
    opts = LONG_OPTS + [("extra", "one more long outcome description " * 3)]  # 21: groups of 2 leave one over
    r = m.decide("choice", LONG_STATE, "Which outcome?", opts)
    assert max(seen) <= MIN_BUDGET and abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_glibc_allocator_is_set_for_low_memory(monkeypatch, toy):
    import sys

    from mayaz import runtime
    if not sys.platform.startswith("linux"):
        pytest.skip("glibc only")
    assert runtime.Session(toy).low_memory is True
    monkeypatch.setenv("MAYAZ_MMAP_THRESHOLD", "0")
    assert runtime.Session(toy).low_memory is False
