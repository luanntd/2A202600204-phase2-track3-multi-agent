"""Critic agent — fact-checks final answer and reports citation coverage."""

import logging
import re
import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a fact-checker reviewing a research answer.

For each factual claim in the answer:
1. Check whether it is supported by the provided research notes.
2. Flag any claims that appear unsupported or exaggerated.

Output format:
## Citation Coverage
<count> out of <total> claims are cited with [N] markers.

## Unsupported Claims
- List any claims not backed by the notes (or "None found").

## Overall Assessment
One sentence: pass / needs revision — and why.
"""


def count_citations(text: str) -> int:
    """Count distinct citation markers like [1], [2], [3]."""
    return len(set(re.findall(r"\[(\d+)\]", text)))


class CriticAgent(BaseAgent):
    """Optional fact-checking and citation-coverage agent."""

    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        model = get_settings().openai_model
        self._llm = llm or LLMClient(model=model, temperature=0.0)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer or not state.research_notes:
            state.errors.append("Critic skipped: missing final_answer or research_notes")
            return state

        with trace_span("critic", {}) as span:
            t0 = time.monotonic()

            citation_count = count_citations(state.final_answer)
            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Final answer:\n{state.final_answer}\n\n"
                "Fact-check the final answer."
            )
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

            elapsed = time.monotonic() - t0
            span["agent"] = self.name
            span["citation_count"] = citation_count

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=response.content,
                    metadata={
                        "citation_count": citation_count,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "latency_seconds": round(elapsed, 3),
                    },
                )
            )
            state.add_trace_event(
                "critic_done",
                {"citation_count": citation_count, "latency_seconds": round(elapsed, 3)},
            )
            logger.info("Critic: %d citations found in %.2fs", citation_count, elapsed)

        return state
