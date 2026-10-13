# MayaZ decision API

FastAPI + ONNX Runtime (no PyTorch). One process serves one model tier on one CPU thread; it fits Render's free
instance (0.1 vCPU, 512 MB) for every tier. Numbers: `DEPLOY_REPORT.md`.

| `MAYAZ_MODEL` | parameters | held-out test | p50 on 0.1 vCPU | peak memory |
|---|---|---|---|---|
| `mayaz-17m` | 17.7M | .781 | 0.6 s | 242 MB |
| `mayaz-32m` (default) | 33.9M | .782 | 1.5 s | 353 MB |
| `mayaz-68m` | 71.7M | .813 | 6.4 s | 367 MB |

## Deploy to Render (1-click)

1. Render dashboard → **New → Blueprint** → connect this repository. Render reads `server/render.yaml`.
2. Set `MAYAZ_MODEL` (default `mayaz-32m`). No token is needed: the model comes from the public `v1.0.0` release.
3. Deploy. The build downloads the chosen tier's model and checks its sha256; `GET /healthz` is the health check.

To switch tiers, change `MAYAZ_MODEL` and redeploy.

## Run locally

```bash
docker build -f server/Dockerfile --build-arg MAYAZ_MODEL=mayaz-17m -t mayaz-api .   # from the repo root
docker run --rm -p 8000:8000 --cpus=0.1 -m 512m mayaz-api
```

## API

`POST /decide`

```json
{"type": "choice", "state": "My card was charged twice.", "question": "What does the customer want?",
 "options": [{"key": "refund", "description": "money back"}, {"key": "cancel_order"}, {"key": "greeting"}]}
```

returns `{"model", "choice", "probabilities", "confidence", "abstain"}`. `type` is `choice`, `noul` (true/false)
or `score` (ordinal keys); 2–256 options (above 64 a tournament runs). `GET /model` describes the served tier;
`GET /readyz` runs one tiny inference and reports not-ready if memory after loading exceeded
`MAYAZ_RSS_BUDGET_MB` (default 400). `MAYAZ_MIN_CONFIDENCE` (default 0.5) sets `abstain`.

**Memory guard.** `MAYAZ_MAX_TOKENS` caps the tokens in one forward pass. The default is the tier's `max_tokens`
in `app/models.json`: 1,536 for 17M, 1,024 for 32M and 68M. Longer requests are scored in smaller option groups,
so peak memory stays inside a 512 MB instance. A pass that still cannot fit returns 422. `MAYAZ_MMAP_THRESHOLD`
(default 1 MiB) makes glibc return freed buffers after each pass. Measurements are in `DEPLOY_REPORT.md`.
