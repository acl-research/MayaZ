"""Family step 5: one tier per process, chosen by MAYAZ_MODEL; registry consistency; RSS admission check."""
import json
import os
import re
from pathlib import Path

import pytest
from app.tiers import REGISTRY, default_tier, registry, resolve
from fastapi.testclient import TestClient

from mayaz.model.render import TOKENIZER_PATH
from tests.conftest import load_jsonl
from tests.server.test_server import artifacts, request_of  # noqa: F401  (module fixture reused)

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "items.jsonl"


def test_registry_lists_the_three_tiers_and_a_valid_default():
    tiers = registry()
    assert set(tiers) == {"mayaz-17m", "mayaz-32m", "mayaz-68m"}
    assert default_tier() == "mayaz-32m" in tiers
    for name, t in tiers.items():
        assert t["file"] == f"{name}.onnx" and t["temperatures"] == f"temperatures-{name}.json"
        if t["status"] == "ready":
            assert t["asset_url"].startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", t["sha256"])
            assert (ROOT / "server" / "artifacts" / t["temperatures"]).exists()


def test_resolve_tier_name_and_raw_path(tmp_path):
    r = resolve("mayaz-68m", tmp_path)
    assert r["tier"] == "mayaz-68m" and r["model"] == tmp_path / "mayaz-68m.onnx"
    assert r["temperatures"] == tmp_path / "temperatures-mayaz-68m.json"
    raw = resolve(str(tmp_path / "x.onnx"), tmp_path)
    assert raw["tier"] == "custom" and raw["model"] == tmp_path / "x.onnx"


def test_unknown_tier_fails_with_the_choices(tmp_path):
    with pytest.raises(ValueError, match="mayaz-17m"):
        resolve("mayaz-999m", tmp_path)


def _client(artifacts, **extra):  # noqa: F811
    env = {"MAYAZ_MODEL": str(artifacts / "model.onnx"), "MAYAZ_TEMPERATURES": str(artifacts / "temperatures.json"),
           "MAYAZ_TOKENIZER": str(TOKENIZER_PATH), **extra}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from app.main import app
    try:
        with TestClient(app) as c:
            yield c
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_response_and_model_endpoint_name_the_tier(artifacts):  # noqa: F811
    for c in _client(artifacts):
        item = next(i for i in load_jsonl(FIX) if len(i["options"]) >= 3)
        assert c.post("/decide", json=request_of(item)).json()["model"] == "custom"
        info = c.get("/model").json()
        assert info["model"] == "custom" and "sha256" not in info


def test_rss_over_budget_makes_readyz_report_not_ready(artifacts):  # noqa: F811
    for c in _client(artifacts, MAYAZ_RSS_BUDGET_MB="1"):
        r = c.get("/readyz").json()
        assert r["ok"] is False and "exceeds budget" in r["reason"]
        assert c.get("/healthz").json() == {"ok": True}


def test_registry_is_valid_json():
    assert json.loads(REGISTRY.read_text())["default"] in registry()


def test_every_tier_downloads_from_the_public_release_and_the_hub_with_a_token_budget():
    for name, t in registry().items():
        assert t["asset_url"] == f"https://github.com/acl-research/MayaZ/releases/download/v1.0.0/{name}.onnx"
        assert re.fullmatch(r"acl-research/MayaZ-(17|32|68)M", t["hf_repo"]) and t["hf_file"] in (
            "mayaz_fp32.onnx", "mayaz_int8_embed8.onnx")
        assert set(t["hf_variants"]) <= {"fp32", "int8", "int8_embed8"} and "fp32" in t["hf_variants"]
        assert 600 <= t["max_tokens"] <= 1536
