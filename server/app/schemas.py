"""Request / response models with hard limits (checked before any tokenisation)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RESERVED = ("<OPT>", "</OPT>")
MAX_STATE_CHARS = 20_000
MAX_QUESTION_CHARS = 1_000
MAX_KEY_CHARS = 80
MAX_DESC_CHARS = 500
MIN_OPTIONS, MAX_OPTIONS = 2, 256  # above 64 the single-model tournament runs (several passes)


def _clean(s: str) -> str:
    if any(t in s for t in RESERVED):
        raise ValueError("reserved token")
    return s


class Option(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)
    description: str | None = Field(default=None, max_length=MAX_DESC_CHARS)

    _k = field_validator("key")(_clean)

    @field_validator("description")
    @classmethod
    def _d(cls, v: str | None) -> str | None:
        return None if v is None else _clean(v)


class DecideRequest(BaseModel):
    type: Literal["choice", "score", "noul"]
    state: str | dict[str, Any]
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    options: list[Option] = Field(min_length=MIN_OPTIONS, max_length=MAX_OPTIONS)

    _q = field_validator("question")(_clean)

    @model_validator(mode="after")
    def _check(self) -> DecideRequest:
        keys = [o.key for o in self.options]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate option keys")
        s = self.state if isinstance(self.state, str) else repr(self.state)
        if len(s) > MAX_STATE_CHARS:
            raise ValueError(f"state longer than {MAX_STATE_CHARS} characters")
        if isinstance(self.state, str):
            _clean(self.state)
        if not self.question.strip():
            raise ValueError("empty question")
        return self


class DecideResponse(BaseModel):
    model: str
    choice: str
    probabilities: dict[str, float]
    confidence: float
    abstain: bool
