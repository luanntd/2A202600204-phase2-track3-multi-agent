"""Benchmark — measures latency, cost, quality, citation coverage, and failure rate."""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_JUDGE_SYSTEM = """You are an objective evaluator. Score the answer to the given research query
on a scale of 0-10.

Criteria:
- Accuracy & factual correctness (0-3)
- Completeness — does it address the query fully? (0-3)
- Clarity & structure (0-2)
- Use of evidence / citations (0-2)

Respond with ONLY a JSON object: {"score": <number>, "rationale": "<one sentence>"}
"""


def _count_citations(text: str) -> int:
    """Count distinct [N] citation markers."""
    return len(set(re.findall(r"\[(\d+)\]", text or "")))


def _estimate_cost(state: ResearchState) -> float:
    """Sum cost_usd across all agent results."""
    return sum(
        r.metadata.get("cost_usd") or 0.0
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    )


def _llm_quality_score(query: str, answer: str, llm: LLMClient) -> tuple[float, str]:
    """Ask an LLM judge for a quality score 0-10."""
    import json

    user_prompt = f"Query: {query}\n\nAnswer:\n{answer}\n\nScore this answer."
    try:
        resp = llm.complete(_JUDGE_SYSTEM, user_prompt)
        data = json.loads(resp.content.strip())
        score = float(data.get("score", 5))
        rationale = data.get("rationale", "")
        return min(max(score, 0.0), 10.0), rationale
    except Exception as exc:
        logger.warning("Quality scoring failed: %s", exc)
        return 5.0, "scoring failed"


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    judge_llm: LLMClient | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Time a runner, compute all metrics, return (state, metrics)."""
    started = perf_counter()
    failed = False
    state: ResearchState | None = None

    try:
        state = runner(query)
    except Exception as exc:
        logger.error("Runner %s failed: %s", run_name, exc)
        failed = True
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))

    latency = perf_counter() - started
    answer = (state.final_answer or "") if state else ""

    cost = _estimate_cost(state) if state else 0.0
    citations = _count_citations(answer)

    quality: float | None = None
    notes = ""
    if answer and not failed:
        llm = judge_llm or LLMClient(model="gpt-4o-mini", temperature=0.0)
        score, rationale = _llm_quality_score(query, answer, llm)
        quality = score
        notes = f"citations={citations}; {rationale}"
        if state and state.errors:
            notes += f"; errors={len(state.errors)}"
    elif failed:
        notes = "run failed"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost > 0 else None,
        quality_score=quality,
        notes=notes,
    )
    logger.info(
        "Benchmark %s: latency=%.2fs cost=$%.4f quality=%s citations=%d",
        run_name,
        latency,
        cost,
        quality,
        citations,
    )
    return state, metrics  # type: ignore[return-value]
