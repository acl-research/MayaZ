"""TypeSafe-compatible `POST /v1/systemone`: the System One request/response shape over the same model as /decide.

Request:  {"model"?, "state", "questions": {id: {"type": "choice"|"noul"|"score", "instructions", "criteria"?}}}
  choice: criteria {label: description}          -> options [(label, description)]
  noul:   criteria {"true": d, "false": d} or none -> options [("true", d), ("false", d)] (P(yes) = P("true"))
  score:  criteria [level_0, ..., level_{n-1}]    -> options [("0", level_0), ...]
Response: {"model", "answers": {id: {...}}, "usage": {"input_tokens", "output_tokens"}, "runtime": {...}}
  choice: {"type", "choice", "probabilities", "confidence"}   confidence = (n * p_max - 1) / (n - 1)
  noul:   {"type", "noul", "probabilities"}                    noul = P(yes)
  score:  {"type", "score", "probabilities", "confidence"}     score = expected level; MAD-based confidence
The confidence formulas are TypeSafe's published summaries of the distribution, not extra model outputs.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas import MAX_DESC_CHARS, MAX_KEY_CHARS, MAX_QUESTION_CHARS, MAX_STATE_CHARS, _clean

MAX_QUESTIONS = 50
MAX_CHOICES = 255
MAX_LEVELS = 10


class Question(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    criteria: dict[str, str] | list[str] | None = None

    _i = field_validator("instructions")(_clean)

    @model_validator(mode="after")
    def _check(self) -> Question:
        c = self.criteria
        if self.type == "choice":
            if not isinstance(c, dict) or not 2 <= len(c) <= MAX_CHOICES:
                raise ValueError(f"choice needs criteria {{label: description}} with 2-{MAX_CHOICES} labels")
        elif self.type == "score":
            if not isinstance(c, list) or not 2 <= len(c) <= MAX_LEVELS:
                raise ValueError(f"score needs criteria [level, ...] with 2-{MAX_LEVELS} levels")
        elif c is not None and (not isinstance(c, dict) or set(c) - {"true", "false"}):
            raise ValueError('noul criteria may only describe "true" and "false"')
        keys, descs = (list(c), list(c.values())) if isinstance(c, dict) else ([], list(c or []))
        if any(len(k) > MAX_KEY_CHARS for k in keys) or any(len(d) > MAX_DESC_CHARS for d in descs):
            raise ValueError("label or description too long")
        for s in (*keys, *descs):
            _clean(s)
        return self

    def options(self) -> list[tuple[str, str | None]]:
        c = self.criteria
        if self.type == "choice":
            return list(c.items())
        if self.type == "score":
            return [(str(i), d) for i, d in enumerate(c)]
        c = c or {}
        return [("true", c.get("true")), ("false", c.get("false"))]


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def _check(self) -> SystemOneRequest:
        s = self.state if isinstance(self.state, str) else repr(self.state)
        if len(s) > MAX_STATE_CHARS:
            raise ValueError(f"state longer than {MAX_STATE_CHARS} characters")
        if isinstance(self.state, str):
            _clean(self.state)
        return self


def choice_confidence(p_max: float, n: int) -> float:
    return 1.0 if n == 1 else max(0.0, (n * p_max - 1) / (n - 1))


def score_confidence(probs: list[float]) -> float:
    n = len(probs)
    m = max(range(n), key=lambda i: probs[i])
    mad_unif = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - sum(p * abs(i - m) for i, p in enumerate(probs)) / mad_unif)


def answer(q: Question, r: dict) -> dict:
    """Shape one Model.decide result as a System One answer."""
    p = r["probabilities"]
    if q.type == "noul":
        return {"type": "noul", "noul": float(p["true"]), "probabilities": {"true": p["true"], "false": p["false"]}}
    if q.type == "choice":
        return {"type": "choice", "choice": r["choice"], "probabilities": p,
                "confidence": choice_confidence(r["confidence"], len(p))}
    levels = [p[str(i)] for i in range(len(p))]
    return {"type": "score", "score": float(sum(i * x for i, x in enumerate(levels))), "probabilities": p,
            "confidence": score_confidence(levels)}
