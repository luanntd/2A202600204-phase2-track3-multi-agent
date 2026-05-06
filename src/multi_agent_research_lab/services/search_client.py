"""Search client — mock implementation with optional Tavily backend."""

import logging
import os
import re

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded mock corpus — deterministic, reproducible benchmarks
# ---------------------------------------------------------------------------
_MOCK_CORPUS: list[SourceDocument] = [
    SourceDocument(
        title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        url="https://arxiv.org/abs/2404.16130",
        snippet=(
            "GraphRAG extends RAG by first building a knowledge graph from the corpus, "
            "then leveraging graph-based community detection (Leiden algorithm) to generate "
            "multi-level summaries. At query time, community summaries are retrieved and "
            "synthesized, enabling answers that span the entire document collection — not "
            "just locally similar chunks."
        ),
        metadata={
            "year": 2024,
            "authors": "Edge et al.",
            "tags": ["graphrag", "rag", "knowledge-graph"],
        },
    ),
    SourceDocument(
        title="GraphRAG: Unlocking LLM Discovery on Narrative Private Data",
        url="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
        snippet=(
            "Microsoft Research's GraphRAG pipeline: extract entities and relationships via LLM, "
            "build a knowledge graph, cluster into communities, produce community reports, and use "
            "them for global summarization queries. Outperforms naive RAG on sensemaking tasks "
            "across large private datasets."
        ),
        metadata={
            "year": 2024,
            "source": "Microsoft Research",
            "tags": ["graphrag", "knowledge-graph", "enterprise"],
        },
    ),
    SourceDocument(
        title="Anthropic: Building Effective Agents",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        snippet=(
            "Key patterns for multi-agent systems: (1) prompt chaining — break tasks into "
            "sequential LLM calls; (2) routing — classifier directs inputs to specialist agents; "
            "(3) parallelisation — independent subtasks run in parallel; "
            "(4) orchestrator-subagents — a planner delegates to specialised workers; "
            "(5) evaluator-optimiser — one agent judges, another refines. "
            "Multi-agent is worth adding only when quality gains justify the latency and cost."
        ),
        metadata={
            "year": 2024,
            "source": "Anthropic",
            "tags": ["multi-agent", "agent-design", "patterns"],
        },
    ),
    SourceDocument(
        title="LangGraph: Stateful, Multi-Actor Applications with LLMs",
        url="https://langchain-ai.github.io/langgraph/concepts/",
        snippet=(
            "LangGraph models agent workflows as directed graphs where nodes are LLM calls or "
            "tools and edges encode conditional routing. The shared state object travels "
            "through nodes; "
            "conditional edges let a supervisor decide the next worker. Built-in support for "
            "persistence, human-in-the-loop, and streaming."
        ),
        metadata={
            "year": 2024,
            "source": "LangChain",
            "tags": ["langgraph", "multi-agent", "workflow"],
        },
    ),
    SourceDocument(
        title="Production Guardrails for LLM Agents",
        url="https://arxiv.org/abs/2402.01817",
        snippet=(
            "Critical guardrails for production LLM agents: (1) max iteration caps to prevent "
            "infinite loops; (2) per-step and total timeouts; (3) output schema validation with "
            "Pydantic; (4) retry with exponential backoff for transient API errors; (5) fallback "
            "responses when all retries fail; (6) input/output logging for post-mortem debugging. "
            "Rate-limiting and cost budgets are also recommended for multi-agent pipelines."
        ),
        metadata={"year": 2024, "tags": ["guardrails", "production", "reliability", "multi-agent"]},
    ),
    SourceDocument(
        title="Comparative Study: Single-Agent vs Multi-Agent for Customer Support",
        url="https://arxiv.org/abs/2401.12345",
        snippet=(
            "Empirical comparison on customer support tasks: single-agent (GPT-4o) achieves 71 % "
            "resolution rate at ~1.2 s latency. Multi-agent pipeline (classifier + specialist + "
            "escalation agent) achieves 84 % resolution rate but at ~4.1 s latency and 3× token "
            "cost. Multi-agent is justified for complex, multi-step queries; single-agent is "
            "preferred for routine, low-latency interactions."
        ),
        metadata={
            "year": 2024,
            "tags": ["customer-support", "multi-agent", "benchmark", "single-agent"],
        },
    ),
    SourceDocument(
        title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        url="https://arxiv.org/abs/2201.11903",
        snippet=(
            "Chain-of-thought (CoT) prompting — providing step-by-step reasoning examples — "
            "substantially improves LLM performance on multi-step reasoning tasks. This underpins "
            "many agent design patterns where breaking a complex task into sub-steps produces "
            "more reliable results than a single monolithic prompt."
        ),
        metadata={"year": 2022, "authors": "Wei et al.", "tags": ["prompting", "reasoning", "llm"]},
    ),
]

_TAG_STOPWORDS = {"and", "or", "the", "a", "an", "of", "for", "in", "is", "are", "with"}


def _score(doc: SourceDocument, query: str) -> int:
    """Simple keyword overlap scoring."""
    words = set(re.split(r"\W+", query.lower())) - _TAG_STOPWORDS
    text = (doc.title + " " + doc.snippet + " " + " ".join(doc.metadata.get("tags", []))).lower()
    return sum(1 for w in words if w and w in text)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily if TAVILY_API_KEY is set, otherwise falls back to the local mock corpus.
    """

    def __init__(self) -> None:
        self._tavily_key = os.getenv("TAVILY_API_KEY")

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self._tavily_key:
            return self._tavily_search(query, max_results)
        return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        scored = sorted(_MOCK_CORPUS, key=lambda d: _score(d, query), reverse=True)
        results = scored[:max_results]
        logger.debug("MockSearch query=%r returning %d docs", query, len(results))
        return results

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            client = TavilyClient(api_key=self._tavily_key)
            raw = client.search(query=query, max_results=max_results)
            docs = []
            for r in raw.get("results", []):
                docs.append(
                    SourceDocument(
                        title=r.get("title", "Untitled"),
                        url=r.get("url"),
                        snippet=r.get("content", ""),
                        metadata={"score": r.get("score", 0)},
                    )
                )
            logger.debug("TavilySearch query=%r returning %d docs", query, len(docs))
            return docs
        except Exception as exc:
            logger.warning("Tavily search failed (%s), falling back to mock", exc)
            return self._mock_search(query, max_results)
