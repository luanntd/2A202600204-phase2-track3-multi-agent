"""Behavior tests for implemented agents (replaces StudentTodoError stubs)."""


from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _fresh_state(query: str = "Explain multi-agent systems in 500 words") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


class TestSupervisorRouting:
    def test_routes_to_researcher_when_no_notes(self) -> None:
        state = _fresh_state()
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "researcher"

    def test_routes_to_analyst_after_research(self) -> None:
        state = _fresh_state()
        state.research_notes = "Some research notes."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "analyst"

    def test_routes_to_writer_after_analysis(self) -> None:
        state = _fresh_state()
        state.research_notes = "Research."
        state.analysis_notes = "Analysis."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "writer"

    def test_routes_to_done_when_all_present(self) -> None:
        state = _fresh_state()
        state.research_notes = "Research."
        state.analysis_notes = "Analysis."
        state.final_answer = "Final answer."
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "done"

    def test_enforces_max_iterations(self) -> None:
        state = _fresh_state()
        # Force iteration past the cap
        state.iteration = 100
        result = SupervisorAgent(max_iterations=3).run(state)
        assert result.route_history[-1] == "done"
        assert any("Max iterations" in e for e in result.errors)

    def test_increments_iteration(self) -> None:
        state = _fresh_state()
        before = state.iteration
        SupervisorAgent().run(state)
        assert state.iteration == before + 1

    def test_records_trace_event(self) -> None:
        state = _fresh_state()
        SupervisorAgent().run(state)
        names = [e["name"] for e in state.trace]
        assert "supervisor_route" in names
