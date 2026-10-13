import pytest

from mayaz.model.render import TextEncoder, collate, render_item


@pytest.fixture(scope="session")
def tok():
    return TextEncoder()


def make_item(options, state="The customer says the card was charged twice last week.", question="Which intent?"):
    return {"type": "choice", "state": state, "question": question,
            "options": [{"key": f"o{i}", "text": t} for i, t in enumerate(options)]}


def option_texts(k: int) -> list[str]:
    words = ["refund", "card lost", "transfer failed", "exchange rate", "top up", "pin blocked", "cash withdrawal",
             "direct debit", "virtual card", "verify identity"]
    return [f"{words[i % len(words)]} variant {i}: " + "detail " * (i % 7) for i in range(k)]


def batch_of(tok, items, **cfg):
    from mayaz.model.render import RenderConfig
    rc = RenderConfig(**cfg) if cfg else RenderConfig()
    return collate([render_item(it, tok, rc) for it in items])
