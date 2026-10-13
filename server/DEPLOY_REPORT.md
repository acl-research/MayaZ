# Deploy report: the MayaZ family on Render-free limits (0.1 vCPU, 512 MB)

One server process serves exactly one tier, chosen by `MAYAZ_MODEL` (`mayaz-17m` | `mayaz-32m` | `mayaz-68m`,
default `mayaz-32m`). Tier names resolve through `server/app/models.json` to a sha256-pinned model file, fetched at
build time from the public `v1.0.0` GitHub release by `server/scripts/fetch_model.sh`, and a temperatures file.

**Method.** The real server (`uvicorn app.main:app --workers 1`, one ONNX Runtime thread, CPU arena off) ran in a
cgroup with `MemoryMax=512M MemorySwapMax=0 CPUQuota=10%`, the equivalent of `docker run --cpus=0.1 -m 512m`.
Requests were held-out items, stratified by option count and sent one at a time: 200 per tier.

## Verdict per tier

| Tier | Served file | Size | p50 / p95 | Peak RSS | RSS growth, request 50 → 200 | Startup | Verdict |
|---|---|---|---|---|---|---|---|
| **mayaz-17m** | fp32 | 67.6 MB | **0.60 s / 2.20 s** | **242 MB** | +1.5 MB | 12.4 s | **fits** |
| **mayaz-32m** (default) | fp32 | 129.5 MB | 1.50 s / 5.31 s | 353 MB | −1 MB | 13.4 s | **fits** |
| **mayaz-68m** | INT8 + embed8 | 78.2 MB | 6.35 s / 17.7 s | 367 MB | +12 MB | 16.4 s | **fits-but-slow** |

- **fits:** peak RSS < 400 MB, no meaningful growth from request 50 to 200, and exact parity. Parity is the max
  |Δp| against the in-process reference over 100 items, and it is 0.0 for every row.
- **fits-but-slow:** fits, but with p95 > 10 s.

The 10% CPU quota dominates p95: a request waits for the next 100 ms period. 68M is CPU-bound. It is the tier for a
paid instance (1 vCPU cuts its latency roughly tenfold) or for the browser package.

Held-out accuracy:

| Tier | Internal held-out test | typed-decisions |
|---|---|---|
| 17M | .781 | .740 |
| 32M | .781 | .732 |
| 68M | .813 | .764 |

## Per-bucket latency (p50 / p95, seconds)

| Tier | 2 options | 3–5 | 6–20 | 21+ |
|---|---|---|---|---|
| 17M fp32 | 0.40 / 2.07 | 0.40 / 2.35 | 0.51 / 1.51 | 1.35 / 2.20 |
| 32M fp32 | 1.00 / 5.13 | 0.95 / 5.81 | 1.25 / 3.63 | 3.20 / 5.20 |
| 68M INT8 + embed8 | 4.95 / 16.98 | 4.50 / 19.17 | 5.65 / 12.73 | 11.40 / 16.88 |

Variants measured but not served:

| Variant | Measurement | Why not served |
|---|---|---|
| 32M INT8 | p50 2.25 s, peak 304 MB | slower than fp32 at one thread |
| 68M INT8, fp32 embeddings | 151.8 MB, peak 429 MB | over the memory budget |
| 17M INT8 + embed8 | 18.7 MB, p50 0.80 s, peak 176 MB | slower than fp32; kept for the browser |

## Memory guard for long inputs

All options of a decision sit side by side in one sequence, so a long state with many long options makes one long
forward pass, and peak memory grows with it. On V1 settings, the worst case was a 1,452-token pass (21 options of
about 50 tokens and a full 512-position state). It peaked at 477 MB on 32M and at 486–904 MB on 68M, depending on
heap state.

The serving code now has three safeguards:

1. **The allocator returns freed memory.**
   - The tokenizer is created, and encodes once, *before* the ONNX session. In the other order, the 68M peak at
     ~600 tokens is 680–980 MB instead of 315–360 MB.
   - glibc's mmap threshold is fixed at 1 MiB (`MAYAZ_MMAP_THRESHOLD`), and `malloc_trim` runs after every pass.
     Large activation buffers then go back to the OS instead of staying on the heap. Between passes RSS falls back
     to the post-load level (~150 MB for 68M).
2. **A token budget per forward pass.**
   - It is `max_tokens` in `app/models.json`, overridable with `MAYAZ_MAX_TOKENS`: 17M **1,536**, 32M **1,024**,
     68M **1,024**.
   - A request whose pass would be longer is scored as a tournament of smaller option groups that each fit.
     Requests within the budget are scored exactly as before.
3. **A hard stop.** The budget is clamped to at least 600 tokens, so a two-option pass always fits. If a pass is
   ever forced over the budget, the server returns 422 rather than risking an out-of-memory kill.

**Session settings** (`mayaz.runtime.session_options`):
- CPU memory arena off;
- memory-pattern pre-allocation off;
- one intra-op and one inter-op thread;
- no thread spinning.

**Peak memory on the worst-case request** (fresh process, one thread):

| Tier | V1 settings, one pass | All safeguards, default budget | Passes | Time on 1 core, before → after |
|---|---|---|---|---|
| 17M | 321–326 MB | **299 MB** | 1 | 2.1 s → 2.3 s |
| 32M | 477 MB | **316 MB** | 3 | 5.4 s → 7.5 s |
| 68M | 486–904 MB | **309 MB** | 3 | 13.2 s → 20.3 s |

Only requests above the budget get slower, because they are split. With a 640-token budget, 68M peaks at 207 MB
but makes 11 passes (45 s). Lower `MAYAZ_MAX_TOKENS` if you need more headroom, or raise it on bigger instances.

## Checks

- **Server tests (`tests/server`):**
  - parity with the reference pipeline within 1e-5;
  - 422 on every input limit: 2–256 options, unique keys, state ≤ 20,000 characters, question ≤ 1,000, key ≤ 80,
    description ≤ 500, reserved tokens, and an unknown type;
  - the tournament above 64 options;
  - the token-budget split and its 422;
  - JSON states;
  - tier resolution, including a clear error for an unknown tier;
  - `model` in every response and in `GET /model`;
  - RSS over `MAYAZ_RSS_BUDGET_MB` turns `/readyz` to not-ready;
  - no torch, transformers, optimum, sentence-transformers, accelerate or scipy is imported. The Dockerfile repeats
    this banned-package check after install.
- **Pinned requirements:** fastapi 0.141.1, uvicorn 0.52.4, onnxruntime 1.23.2 (the version the models were
  exported with), tokenizers 0.23.3, numpy 2.5.4.
