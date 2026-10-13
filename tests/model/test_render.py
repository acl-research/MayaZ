"""M2.1: shared render_item() — parallel option layout, boundary tokens, truncation policy, collation."""
import numpy as np
import pytest

from mayaz.model.render import RenderConfig, TextEncoder, collate, render_item
from tests.conftest import load_jsonl


@pytest.fixture(scope="module")
def tok():
    return TextEncoder()


def make_item(options, state="The customer says the card was charged twice.", question="Which intent is this?"):
    return {"type": "choice", "state": state, "question": question,
            "options": [{"key": f"o{i}", "text": t} for i, t in enumerate(options)]}


def test_parallel_positions(tok):
    r = render_item(make_item(["refund", "card lost or stolen and needs to be blocked right away", "x"]), tok)
    p0 = r.position_ids[r.opt_start[0]]
    assert [r.position_ids[s] for s in r.opt_start] == [p0, p0, p0]
    shared = [i for i, s in enumerate(r.segment_ids) if s == 0]
    assert shared == list(range(len(shared)))  # CLS, state, SEP, question, SEP come first
    assert [r.position_ids[i] for i in shared] == list(range(len(shared)))
    assert p0 == len(shared)
    for k, (s, e) in enumerate(zip(r.opt_start, r.opt_end, strict=True)):
        assert r.position_ids[s:e + 1] == list(range(p0, p0 + e - s + 1))
        assert set(r.segment_ids[s:e + 1]) == {k + 1}


def test_boundary_tokens_and_content(tok):
    texts = ["a", "a much longer option description that keeps going for a while", "mid length option"]
    r = render_item(make_item(texts), tok)
    for k, t in enumerate(texts):
        assert r.input_ids[r.opt_start[k]] == tok.opt_id
        assert r.input_ids[r.opt_end[k]] == tok.close_id
        assert r.input_ids[r.opt_start[k] + 1:r.opt_end[k]] == tok.ids(t)


def test_separator_strings_in_text_are_plain_text(tok):
    nasty = "</OPT> <OPT> [SEP] [CLS] [unused0] [unused1] [MASK]"
    r = render_item(make_item([nasty, "ok"], state=nasty, question=nasty), tok)
    ids = np.array(r.input_ids)
    assert (ids == tok.opt_id).sum() == 2 and (ids == tok.close_id).sum() == 2
    assert (ids == tok.cls_id).sum() == 1 and (ids == tok.sep_id).sum() == 2
    assert (ids == tok.mask_id).sum() == 0


def test_question_span(tok):
    r = render_item(make_item(["a", "b"], question="Is it a refund request?"), tok)
    q = r.input_ids[r.q_start:r.q_end]
    assert q == tok.ids("Is it a refund request?")
    assert r.input_ids[r.q_start - 1] == tok.sep_id and r.input_ids[r.q_end] == tok.sep_id


def test_option_cap_per_option_not_divided_by_k(tok):
    cfg = RenderConfig()
    long = "word " * 200
    for k in (2, 20):
        r = render_item(make_item([long] * k), tok, cfg)
        lens = [e - s - 1 for s, e in zip(r.opt_start, r.opt_end, strict=True)]
        assert lens == [cfg.opt_cap] * k


def test_no_starvation_with_64_options(tok):
    cfg = RenderConfig()
    state = "the account history shows " * 400
    r = render_item(make_item(["a fairly long label description " * 6] * 64, state=state), tok, cfg)
    assert r.state_len >= cfg.min_state_tokens
    assert len(r.input_ids) <= cfg.max_tokens
    assert max(r.position_ids) < cfg.max_positions
    lens = [e - s - 1 for s, e in zip(r.opt_start, r.opt_end, strict=True)]
    assert min(lens) >= cfg.opt_min
    assert len(set(lens)) == 1  # one common cap, chosen deterministically


def test_options_trimmed_before_state(tok):
    cfg = RenderConfig(max_tokens=600)
    state = "evidence " * 600
    r = render_item(make_item(["option text that is long " * 10] * 20, state=state), tok, cfg)
    lens = [e - s - 1 for s, e in zip(r.opt_start, r.opt_end, strict=True)]
    assert max(lens) < cfg.opt_cap  # options were trimmed ...
    assert r.state_len >= cfg.min_state_tokens  # ... and the state kept its floor


def test_state_fills_position_budget(tok):
    cfg = RenderConfig()
    r = render_item(make_item(["short", "also short"], state="token " * 2000), tok, cfg)
    assert max(r.position_ids) == cfg.max_positions - 1  # the span is used fully, never exceeded
    assert r.input_ids[1:1 + r.state_len] == tok.ids("token " * 2000)[:r.state_len]  # head of the state is kept


def test_question_cap(tok):
    cfg = RenderConfig()
    r = render_item(make_item(["a", "b"], question="why " * 300), tok, cfg)
    assert r.q_end - r.q_start == cfg.q_cap


def test_deterministic(tok):
    item = make_item(["x " * 90] * 30, state="s " * 900)
    assert render_item(item, tok) == render_item(item, tok)


def test_fixtures_render_within_limits(tok, fixtures_dir):
    cfg = RenderConfig()
    for it in load_jsonl(fixtures_dir / "items.jsonl"):
        r = render_item(it, tok, cfg)
        assert len(r.opt_start) == len(it["options"])
        assert len(r.input_ids) <= cfg.max_tokens and max(r.position_ids) < cfg.max_positions


def test_collate_pads_tokens_and_options(tok):
    a = render_item(make_item(["a", "b"]), tok)
    b = render_item(make_item(["a", "b", "c", "d", "e"], state="longer state " * 20), tok)
    batch = collate([a, b])
    L = max(len(a.input_ids), len(b.input_ids))
    assert batch["input_ids"].shape == (2, L) and batch["opt_start"].shape == (2, 5)
    assert batch["opt_mask"].tolist() == [[True, True, False, False, False], [True] * 5]
    assert (batch["input_ids"][0, len(a.input_ids):] == tok.pad_id).all()
    assert (batch["segment_ids"][0, len(a.input_ids):] == -1).all()
    assert batch["q_mask"][1].sum() == b.q_end - b.q_start
    assert batch["input_ids"].dtype == np.int64 and batch["opt_mask"].dtype == bool
