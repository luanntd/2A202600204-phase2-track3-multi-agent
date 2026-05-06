"""Analyst agent — extracts key claims, compares viewpoints, flags weak evidence."""

import logging
import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a critical analyst. Your ONLY job is to analyse \
the research notes and produce structured insights.

Output format (use these exact headers):
## Key Claims
- List 3-5 specific, evidence-backed claims from the research.

## Viewpoint Comparison
- Compare different perspectives or approaches mentioned in the research.

## Evidence Strength
- Rate each key claim: Strong / Moderate / Weak, with one-line justification.

## Gaps & Caveats
- Note missing information, conflicting data, or limitations.

Do NOT rewrite the research or produce a final answer — that is the Writer's job.
"""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        model = get_settings().openai_model
        self._llm = llm or LLMClient(model=model, temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.errors.append("Analyst called before research_notes available")
            return state

        with trace_span("analyst", {"notes_len": len(state.research_notes)}) as span:
            t0 = time.monotonic()

            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Produce structured analysis."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content

            elapsed = time.monotonic() - t0
            span["agent"] = self.name
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "latency_seconds": round(elapsed, 3),
                    },
                )
            )
            state.add_trace_event(
                "analyst_done",
                {
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "latency_seconds": round(elapsed, 3),
                },
            )
            logger.info("Analyst: done in %.2fs", elapsed)

        return state
