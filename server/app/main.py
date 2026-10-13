"""MayaZ decision API: FastAPI + ONNX Runtime, one worker, one inference thread, no torch.

Config (environment): MAYAZ_MODEL (a tier from app/models.json: mayaz-17m | mayaz-32m | mayaz-68m, default
mayaz-32m; or a raw .onnx path), MAYAZ_TOKENIZER, MAYAZ_TEMPERATURES (paths; default artifacts/ next to app/, the
tier's temperatures file), MAYAZ_MIN_CONFIDENCE (abstain below this max probability, default 0.5),
MAYAZ_RSS_BUDGET_MB (default 400: above it after loading, /readyz reports not ready with the reason),
MAYAZ_MAX_TOKENS (tokens per forward pass; default: the tier's max_tokens in models.json. Longer requests are
scored in smaller option groups so peak memory stays inside a 512 MB instance; see mayaz.inference).
One process serves exactly one tier.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before onnxruntime is imported

import logging  # noqa: E402
import time  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app.inference import Model, TokenBudgetError  # noqa: E402
from app.schemas import DecideRequest, DecideResponse  # noqa: E402
from app.systemone import SystemOneRequest, answer  # noqa: E402
from app.tiers import default_tier, resolve  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
log = logging.getLogger("mayaz")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _mem_mb(field: str = "VmRSS") -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith(field):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return -1.0


def _rss_mb() -> float:
    return _mem_mb("VmRSS")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    tier = resolve(os.environ.get("MAYAZ_MODEL") or default_tier(), ART)
    app.state.tier = tier
    app.state.model = Model(tier["model"], os.environ.get("MAYAZ_TOKENIZER", ART / "tokenizer.json"),
                            os.environ.get("MAYAZ_TEMPERATURES", tier["temperatures"]),
                            max_tokens=tier["meta"].get("max_tokens"))
    app.state.min_conf = float(os.environ.get("MAYAZ_MIN_CONFIDENCE", "0.5"))
    rss, budget = _rss_mb(), float(os.environ.get("MAYAZ_RSS_BUDGET_MB", "400"))
    app.state.not_ready = f"rss {rss:.0f} MB after loading exceeds budget {budget:.0f} MB" if rss > budget else None
    log.info("%s loaded in %.2f s, rss %.0f MB, %d tokens per pass%s", tier["tier"], time.time() - t0, rss,
             app.state.model.max_tokens,
             f" (NOT READY: {app.state.not_ready})" if app.state.not_ready else "")
    yield


app = FastAPI(title="MayaZ", version="1", lifespan=lifespan)


@app.exception_handler(TokenBudgetError)
def token_budget_exceeded(request: Request, exc: TokenBudgetError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"request too long for this server: {exc}"})


READY_PROBE = ("choice", "The card was charged twice.", "Which intent?", [("refund", None), ("greeting", None)])


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz(request: Request) -> dict:
    if request.app.state.not_ready:
        return {"ok": False, "reason": request.app.state.not_ready}
    r = request.app.state.model.decide(*READY_PROBE)
    return {"ok": abs(sum(r["probabilities"].values()) - 1) < 1e-4}


@app.get("/model")
def model_info(request: Request) -> dict:
    t = request.app.state.tier
    return {"model": t["tier"], **{k: v for k, v in t["meta"].items() if k not in ("asset_url", "sha256")}}


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest, request: Request) -> DecideResponse:
    t0 = time.time()
    r = request.app.state.model.decide(req.type, req.state, req.question,
                                       [(o.key, o.description) for o in req.options])
    log.info("decide k=%d tokens=%d ms=%.0f rss=%.0f", len(req.options), r["tokens"], 1000 * (time.time() - t0),
             _rss_mb())  # never the request text
    return DecideResponse(model=request.app.state.tier["tier"], choice=r["choice"],
                          probabilities=r["probabilities"], confidence=r["confidence"],
                          abstain=r["confidence"] < request.app.state.min_conf)


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest, request: Request) -> dict:
    """TypeSafe-compatible endpoint (see app/systemone.py): each question is one /decide call on the shared state."""
    t0 = time.time()
    model, answers, tokens = request.app.state.model, {}, 0
    for qid, q in req.questions.items():
        r = model.decide(q.type, req.state, q.instructions, q.options())
        tokens += r["tokens"]
        answers[qid] = answer(q, r)
    ms = 1000 * (time.time() - t0)
    log.info("systemone q=%d tokens=%d ms=%.0f rss=%.0f", len(answers), tokens, ms, _rss_mb())
    return {"model": request.app.state.tier["tier"], "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "runtime": {"server_ms": round(ms, 1), "rss_mb": round(_rss_mb(), 1), "hwm_mb": round(_mem_mb("VmHWM"), 1),
                        "ep": model.session.provider}}
