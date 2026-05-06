"""Writer agent — synthesises final answer with inline citations."""

import logging
import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a talented technical writer producing a final research answer
for {audience}.

Rules:
- Write approximately {word_count} words.
- Integrate insights from both research notes and analysis.
- Use inline citations [1], [2], … from the research notes whenever you state a fact.
- Structure: brief intro → main body → conclusion.
- Tone: clear, professional, and informative.
- Do NOT add new facts not present in the notes.
- If analysis notes are present, use them to add depth and interpretation,
  but do not contradict the research notes.
"""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        model = get_settings().openai_model
        self._llm = llm or LLMClient(model=model, temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.errors.append("Writer called before research_notes available")
            return state

        with trace_span("writer", {"has_analysis": state.analysis_notes is not None}) as span:
            t0 = time.monotonic()

            system = _SYSTEM_PROMPT.format(
                audience=state.request.audience,
                word_count=500,
            )
            analysis_section = (
                f"\nAnalysis notes:\n{state.analysis_notes}" if state.analysis_notes else ""
            )
            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}"
                f"{analysis_section}\n\n"
                "Write the final answer."
            )

            response = self._llm.complete(system, user_prompt)
            state.final_answer = response.content

            elapsed = time.monotonic() - t0
            span["agent"] = self.name
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
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
                "writer_done",
                {
                    "answer_len": len(response.content),
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "latency_seconds": round(elapsed, 3),
                },
            )
            logger.info("Writer: done in %.2fs answer_len=%d", elapsed, len(response.content))

        return state
