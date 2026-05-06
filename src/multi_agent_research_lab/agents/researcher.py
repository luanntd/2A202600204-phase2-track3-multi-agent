"""Researcher agent — search + LLM synthesis of research notes."""

import logging
import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a elite research assistant. Your ONLY job is to collect information
and write concise research notes.

Rules:
- Summarise the provided sources accurately.
- Capture key facts, numbers, and dates.
- Cite each source as [1], [2], … matching the order given.
- Do NOT analyse or draw conclusions — that is the Analyst's job.
- Output plain prose, 300-500 words.
- Do NOT include any commentary or opinions, just the facts from the sources.
- If a source doesn't contain relevant information, you can ignore it.
"""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        model = get_settings().openai_model
        self._llm = llm or LLMClient(model=model, temperature=0.2)
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher", {"query": state.request.query}) as span:
            t0 = time.monotonic()

            docs = self._search.search(state.request.query, max_results=state.request.max_sources)
            state.sources.extend(docs)
            logger.info("Researcher: retrieved %d sources", len(docs))

            sources_text = "\n\n".join(
                f"[{i + 1}] {doc.title}\n{doc.snippet}" for i, doc in enumerate(docs)
            )
            user_prompt = (
                f"Research query: {state.request.query}\n\n"
                f"Sources:\n{sources_text}\n\n"
                "Write research notes based on these sources."
            )

            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content

            elapsed = time.monotonic() - t0
            span["agent"] = self.name
            span["sources_count"] = len(docs)
            span["input_tokens"] = response.input_tokens
            span["output_tokens"] = response.output_tokens

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "sources_count": len(docs),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "latency_seconds": round(elapsed, 3),
                    },
                )
            )
            state.add_trace_event(
                "researcher_done",
                {
                    "sources": len(docs),
                    "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                    "latency_seconds": round(elapsed, 3),
                },
            )
            logger.info("Researcher: done in %.2fs tokens=%s", elapsed, response.input_tokens)

        return state
