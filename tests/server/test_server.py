"""FastAPI server (no torch): parity with the reference ONNX pipeline, input limits, banned imports.

The model is a tiny deterministic ONNX graph with the served interface (tests/server/toy_model.py)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mayaz.model.render import TOKENIZER_PATH, TextEncoder, collate, render_item
from mayaz.runtime import Session, softmax, temperature_for
from mayaz.text import option_text
from tests.conftest import load_jsonl
from tests.server.toy_model import build_toy_model

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "items.jsonl"
TEMPS = {"global": 1.3, "by_bucket": {"2": 0.9}, "by_key": {"choice|3-5": 2.0}, "n": {}, "clamp": [0.5, 5.0]}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    build_toy_model(d / "model.onnx")
    (d / "temperatures.json").write_text(json.dumps(TEMPS))
    return d


@pytest.fixture(scope="module")
def client(artifacts):
    env = {"MAYAZ_MODEL": str(artifacts / "model.onnx"), "MAYAZ_TOKENIZER": str(TOKENIZER_PATH),
           "MAYAZ_TEMPERATURES": str(artifacts / "temperatures.json"), "MAYAZ_MIN_CONFIDENCE": "0.5"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from app.main import app
    with TestClient(app) as c:
        yield c
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def request_of(item):
    """API request whose rendered option texts equal the item's (description = text after 'name: ')."""
    opts = []
    for o in item["options"]:
        base = option_text(item["type"], o["key"], None)
        desc = o["text"][len(base) + 2:] if o["text"].startswith(base + ": ") else (
            None if o["text"] == base else o["text"])
        opts.append({"key": o["key"], "description": desc})
    return {"type": item["type"], "state": item["state"], "question": item["question"], "options": opts}


def reference(artifacts, req):
    item = {**req, "options": [{"key": o["key"], "text": option_text(req["type"], o["key"], o["description"])}
                               for o in req["options"]]}
    z = Session(artifacts / "model.onnx").logits(collate([render_item(item, TextEncoder())]))[0]
    return softmax(z, temperature_for(TEMPS, req["type"], len(z)))


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/readyz").json()["ok"] is True


def test_parity_with_reference_pipeline(client, artifacts):
    items = [it for it in load_jsonl(FIX) if it["type"] in ("choice", "noul", "score")][:12]
    for it in items:
        req = request_of(it)
        assert [option_text(it["type"], o["key"], o["description"]) for o in req["options"]] == \
            [o["text"] for o in it["options"]]
        r = client.post("/decide", json=req)
        assert r.status_code == 200, r.text
        body = r.json()
        p = np.array([body["probabilities"][o["key"]] for o in req["options"]])
        assert np.abs(p - reference(artifacts, req)).max() < 1e-5
        assert body["choice"] == req["options"][int(p.argmax())]["key"]
        assert body["confidence"] == pytest.approx(p.max())
        assert body["abstain"] == (p.max() < 0.5)


def test_long_option_lists_use_the_tournament(client):
    req = {"type": "choice", "state": "Please move 50 dollars to savings.", "question": "Which intent?",
           "options": [{"key": f"intent_{i}", "description": None} for i in range(100)]}
    body = client.post("/decide", json=req).json()
    assert len(body["probabilities"]) == 100 and abs(sum(body["probabilities"].values()) - 1) < 1e-4


@pytest.mark.parametrize("bad", [
    {"options": [{"key": "a"}]},
    {"options": [{"key": "a"}, {"key": "a"}]},
    {"options": [{"key": f"k{i}"} for i in range(257)]},
    {"type": "ranking"},
    {"state": "x" * 20001},
    {"question": ""},
    {"options": [{"key": "a", "description": "y" * 501}, {"key": "b"}]},
    {"options": [{"key": "a" * 81}, {"key": "b"}]},
    {"state": "hello <OPT> world"},
])
def test_limits_return_422(client, bad):
    base = {"type": "choice", "state": "s", "question": "q?", "options": [{"key": "a"}, {"key": "b"}]}
    assert client.post("/decide", json={**base, **bad}).status_code == 422


def test_json_state_is_accepted(client):
    req = {"type": "choice", "state": {"agent": {"autonomy": "checkpointed"}}, "question": "What now?",
           "options": [{"key": "continue"}, {"key": "stop"}]}
    assert client.post("/decide", json=req).status_code == 200


def test_server_never_imports_banned_packages():
    code = ("import sys; import app.main; "
            "bad = [m for m in ('torch', 'transformers', 'optimum', 'sentence_transformers', 'accelerate', 'scipy')"
            " if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'server'}:{ROOT / 'src'}"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_server_requirements_are_pinned_and_match_the_root_requirements():
    def pins(path):
        return dict(line.split("==") for line in path.read_text().splitlines() if "==" in line
                    and not line.startswith("#"))
    server = pins(ROOT / "server" / "requirements.txt")
    assert set(server) == {"fastapi", "uvicorn", "onnxruntime", "tokenizers", "numpy"}
    assert server == pins(ROOT / "requirements.txt")


def test_a_request_that_cannot_fit_the_token_budget_returns_422(client):
    model = client.app.state.model
    old, model.max_tokens = model.max_tokens, 100
    try:
        req = {"type": "noul", "state": "word " * 2000, "question": "Is it fine?",
               "options": [{"key": "true"}, {"key": "false"}]}
        r = client.post("/decide", json=req)
        assert r.status_code == 422 and "too long" in r.json()["detail"]
    finally:
        model.max_tokens = old
