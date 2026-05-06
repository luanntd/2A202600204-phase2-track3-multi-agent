# Design Template

## Problem

Build an automated research assistant that accepts a natural-language query, retrieves relevant sources,
analyses them critically, and produces a well-structured, cited answer — without requiring manual
orchestration. The system must handle queries that span multiple information sources and require
synthesis, not just retrieval.

## Why multi-agent?

A single agent given the full task tends to skip the analysis step, hallucinate citations, or produce
shallow answers because it balances too many constraints at once (find sources + judge quality +
write well). Separating the pipeline into distinct roles forces each step to be thorough:

- The **Researcher** is constrained to only gather and report facts — no opinions.
- The **Analyst** is constrained to only critique — no writing.
- The **Writer** is constrained to synthesise from provided notes — no new facts.

This separation reduces hallucination risk and produces better-structured, more trustworthy output.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route to next worker or stop | Full `ResearchState` | Updated `route_history` + `iteration` | Infinite loop → guarded by `max_iterations` + timeout |
| Researcher | Search for sources; write factual notes | `request.query` | `sources`, `research_notes` | No matching sources → returns empty notes; Supervisor re-routes up to cap |
| Analyst | Extract claims, compare viewpoints, flag weak evidence | `research_notes` | `analysis_notes` | LLM API error → tenacity retries 3×; after that raises `AgentExecutionError` |
| Writer | Synthesise final answer with inline citations | `research_notes`, `analysis_notes` | `final_answer` | Missing notes → appends to `errors`, skips gracefully |

## Shared state

`ResearchState` (Pydantic model) is the single source of truth:

| Field | Type | Reason |
|---|---|---|
| `request` | `ResearchQuery` | Carries query + constraints (max_sources, audience) through the whole pipeline |
| `iteration` | `int` | Supervisor uses this to enforce `max_iterations` |
| `route_history` | `list[str]` | Audit trail: which agent ran in which order |
| `sources` | `list[SourceDocument]` | Researcher writes; Writer reads for citation numbering |
| `research_notes` | `str \| None` | Presence signals Researcher is done; absence triggers re-route |
| `analysis_notes` | `str \| None` | Presence signals Analyst is done |
| `final_answer` | `str \| None` | Presence signals Writer is done; CLI displays this |
| `agent_results` | `list[AgentResult]` | Per-agent output + metadata (tokens, cost, latency) for benchmark |
| `trace` | `list[dict]` | Event log for observability; exported to JSON |
| `errors` | `list[str]` | Non-fatal errors accumulate here for post-mortem |

## Routing policy

```
START
  │
  ▼
Supervisor ─── iteration ≥ max_iterations OR timeout ──► DONE (with error)
  │
  ├── research_notes is None ──────────────────────────► Researcher ─► Supervisor
  │
  ├── analysis_notes is None ─────────────────────────► Analyst ──► Supervisor
  │
  ├── final_answer is None ──────────────────────────► Writer ───► Supervisor
  │
  └── all outputs present ──────────────────────────────► DONE
```

## Guardrails

- **Max iterations**: `MAX_ITERATIONS=6` (env-configurable). Supervisor checks `state.iteration >= max_iterations` on every invocation.
- **Timeout**: `TIMEOUT_SECONDS=60` (env-configurable). Supervisor measures `time.monotonic()` from workflow start; stops and appends an error if exceeded.
- **Retry**: `tenacity` wraps `LLMClient.complete()` — 3 attempts, exponential backoff 2–10 s, retries on `APIError`, `APITimeoutError`, `RateLimitError`.
- **Fallback**: If all LLM retries fail, the exception propagates to the node; the workflow catches it and sets `state.errors`. The Supervisor then routes to `done` on the next tick.
- **Validation**: All inputs/outputs are Pydantic models. `ResearchQuery` enforces `min_length=5` on query; `BenchmarkMetrics` enforces `quality_score` in [0, 10].

## Benchmark plan

**Queries** (from `configs/lab_default.yaml`):
1. "Research GraphRAG state-of-the-art and write a 500-word summary"
2. "Compare single-agent and multi-agent workflows for customer support"
3. "Summarise production guardrails for LLM agents"

**Metrics**:

| Metric | Measurement method |
|---|---|
| Latency (s) | `perf_counter()` wall-clock per run |
| Cost (USD) | Sum of `cost_usd` from all `AgentResult.metadata` |
| Quality (0–10) | LLM-as-judge: GPT-4o-mini scores accuracy + completeness + clarity + citations |
| Citation coverage | Count of distinct `[N]` markers in `final_answer` |
| Failure rate | 1 if any `state.errors` containing "Max iterations" or "failed", else 0 |

**Expected outcome**: multi-agent achieves higher quality (better citation, more structured analysis)
at 3–5× latency and ~2× cost vs baseline.
