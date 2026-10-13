"""The MayaZ model family: one tier per process, chosen by MAYAZ_MODEL (a tier name, or a raw .onnx path).

models.json (next to this file) lists each tier's artifact file, temperatures file, precision and release asset
(url + sha256, used by scripts/fetch_model.sh at build time). A raw path is for tests and local runs.
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path(__file__).with_name("models.json")


def registry() -> dict[str, dict]:
    return json.loads(REGISTRY.read_text())["tiers"]


def default_tier() -> str:
    return json.loads(REGISTRY.read_text())["default"]


def resolve(name: str, artifacts: Path) -> dict:
    """{"tier", "model" (path), "temperatures" (path), "meta"} for a tier name or a raw .onnx path."""
    if name.endswith(".onnx"):
        return {"tier": "custom", "model": Path(name), "temperatures": artifacts / "temperatures.json",
                "meta": {"file": Path(name).name}}
    tiers = registry()
    if name not in tiers:
        raise ValueError(f"unknown MAYAZ_MODEL {name!r}; choose one of {sorted(tiers)} or a .onnx path")
    t = tiers[name]
    return {"tier": name, "model": artifacts / t["file"], "temperatures": artifacts / t["temperatures"], "meta": t}
