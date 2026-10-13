"""Model-facing text. Shared by data building, training, export and the server: change nothing here
without regenerating data, because every rendered token depends on it."""
from __future__ import annotations

import json
import re

RESERVED = ("<OPT>", "</OPT>")
NOUL_DEFAULTS = {"true": "Yes, the statement holds.", "false": "No, the statement does not hold."}


def humanize_label(key: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key)
    s = re.sub(r"[_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def state_text(state) -> str:
    if isinstance(state, str):
        return state.strip()
    return json.dumps(state, ensure_ascii=False)


def option_text(qtype: str, key: str, desc: str | None) -> str:
    desc = (desc or "").strip()
    if qtype == "noul":
        return desc or NOUL_DEFAULTS[key]
    if qtype == "score":
        return desc or key
    name = humanize_label(key)
    return f"{name}: {desc}" if desc else name
