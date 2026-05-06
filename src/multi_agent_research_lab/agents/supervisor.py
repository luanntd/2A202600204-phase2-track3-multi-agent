"""Supervisor / router — rule-based routing with iteration and timeout guards."""

import logging
import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

_ROUTE_RESEARCHER = "researcher"
_ROUTE_ANALYST = "analyst"
_ROUTE_WRITER = "writer"
_ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (in order):
    1. iteration >= max_iterations → done (safety cap)
    2. research_notes missing → researcher
    3. analysis_notes missing → analyst
    4. final_answer missing → writer
    5. all outputs present → done
    """

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None, start_time: float | None = None) -> None:
        settings = get_settings()
        self._max_iterations = (
            max_iterations if max_iterations is not None else settings.max_iterations
        )
        self._timeout = settings.timeout_seconds
        self._start_time = start_time or time.monotonic()

    def run(self, state: ResearchState) -> ResearchState:
        elapsed = time.monotonic() - self._start_time

        all_done = (
            state.research_notes is not None
            and state.analysis_notes is not None
            and state.final_answer is not None
        )

        if all_done:
            next_route = _ROUTE_DONE
        elif elapsed >= self._timeout:
            logger.warning("Supervisor: timeout after %.1fs", elapsed)
            state.errors.append(f"Timeout after {elapsed:.1f}s — stopping early")
            next_route = _ROUTE_DONE
        elif state.iteration >= self._max_iterations:
            logger.warning("Supervisor: max iterations %d reached", self._max_iterations)
            state.errors.append(f"Max iterations ({self._max_iterations}) reached — stopping early")
            next_route = _ROUTE_DONE
        elif state.research_notes is None:
            next_route = _ROUTE_RESEARCHER
        elif state.analysis_notes is None:
            next_route = _ROUTE_ANALYST
        else:
            next_route = _ROUTE_WRITER

        logger.info(
            "Supervisor: iter=%d route=%s elapsed=%.1fs", state.iteration, next_route, elapsed
        )
        state.add_trace_event(
            "supervisor_route",
            {
                "iteration": state.iteration,
                "next_route": next_route,
                "elapsed_seconds": round(elapsed, 3),
                "has_research": state.research_notes is not None,
                "has_analysis": state.analysis_notes is not None,
                "has_answer": state.final_answer is not None,
            },
        )
        state.record_route(next_route)
        return state
