"""LangGraph workflow — supervisor → researcher/analyst/writer loop."""

import logging
import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph internal state — Annotated lists use append-reducer
# ---------------------------------------------------------------------------
class _GS(TypedDict):
    request: dict[str, Any]
    iteration: int
    route_history: Annotated[list[str], operator.add]
    sources: Annotated[list[dict[str, Any]], operator.add]
    research_notes: str | None
    analysis_notes: str | None
    final_answer: str | None
    agent_results: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def _to_graph(state: ResearchState) -> _GS:
    return _GS(
        request=state.request.model_dump(),
        iteration=state.iteration,
        route_history=list(state.route_history),
        sources=[s.model_dump() for s in state.sources],
        research_notes=state.research_notes,
        analysis_notes=state.analysis_notes,
        final_answer=state.final_answer,
        agent_results=[r.model_dump() for r in state.agent_results],
        trace=list(state.trace),
        errors=list(state.errors),
    )


def _from_graph(gs: _GS) -> ResearchState:
    return ResearchState(
        request=ResearchQuery(**gs["request"]),
        iteration=gs["iteration"],
        route_history=gs["route_history"],
        sources=[SourceDocument(**s) for s in gs["sources"]],
        research_notes=gs.get("research_notes"),
        analysis_notes=gs.get("analysis_notes"),
        final_answer=gs.get("final_answer"),
        agent_results=[AgentResult(**r) for r in gs["agent_results"]],
        trace=gs["trace"],
        errors=gs["errors"],
    )


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph."""

    def __init__(self) -> None:
        self._start_time: float = 0.0

    def build(self) -> Any:
        """Create and compile the LangGraph StateGraph."""
        supervisor = SupervisorAgent(start_time=self._start_time)
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        def supervisor_node(gs: _GS) -> dict[str, Any]:
            state = _from_graph(gs)
            updated = supervisor.run(state)
            last_route = updated.route_history[-1] if updated.route_history else "done"
            return {
                "iteration": updated.iteration,
                "route_history": [last_route],
                "trace": [updated.trace[-1]] if updated.trace else [],
                "errors": updated.errors[len(gs["errors"]) :],
            }

        def researcher_node(gs: _GS) -> dict[str, Any]:
            state = _from_graph(gs)
            updated = researcher.run(state)
            new_sources = [s.model_dump() for s in updated.sources[len(gs["sources"]) :]]
            new_results = [
                r.model_dump() for r in updated.agent_results[len(gs["agent_results"]) :]
            ]
            new_trace = updated.trace[len(gs["trace"]) :]
            return {
                "sources": new_sources,
                "research_notes": updated.research_notes,
                "agent_results": new_results,
                "trace": new_trace,
                "errors": updated.errors[len(gs["errors"]) :],
            }

        def analyst_node(gs: _GS) -> dict[str, Any]:
            state = _from_graph(gs)
            updated = analyst.run(state)
            new_results = [
                r.model_dump() for r in updated.agent_results[len(gs["agent_results"]) :]
            ]
            new_trace = updated.trace[len(gs["trace"]) :]
            return {
                "analysis_notes": updated.analysis_notes,
                "agent_results": new_results,
                "trace": new_trace,
                "errors": updated.errors[len(gs["errors"]) :],
            }

        def writer_node(gs: _GS) -> dict[str, Any]:
            state = _from_graph(gs)
            updated = writer.run(state)
            new_results = [
                r.model_dump() for r in updated.agent_results[len(gs["agent_results"]) :]
            ]
            new_trace = updated.trace[len(gs["trace"]) :]
            return {
                "final_answer": updated.final_answer,
                "agent_results": new_results,
                "trace": new_trace,
                "errors": updated.errors[len(gs["errors"]) :],
            }

        def route_fn(gs: _GS) -> str:
            history = gs.get("route_history", [])
            if not history:
                return "researcher"
            last = history[-1]
            if last == "done":
                return END
            return last

        builder: StateGraph = StateGraph(_GS)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("researcher", researcher_node)
        builder.add_node("analyst", analyst_node)
        builder.add_node("writer", writer_node)

        builder.set_entry_point("supervisor")
        builder.add_conditional_edges(
            "supervisor",
            route_fn,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
        )
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Compile and invoke the graph, returning the final ResearchState."""
        self._start_time = time.monotonic()
        graph = self.build()
        initial = _to_graph(state)
        final_gs: _GS = graph.invoke(initial)
        result = _from_graph(final_gs)
        logger.info(
            "Workflow complete: iterations=%d routes=%s",
            result.iteration,
            result.route_history,
        )
        return result
