"""Command-line entrypoint for the lab."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace_json
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_BASELINE_SYSTEM = (
    "You are a research assistant. Answer the user's query accurately and completely.\n"
    "Structure your answer clearly. Cite sources when possible using [1], [2], … notation.\n"
    "Write approximately 500 words."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[bool, typer.Option("--save-trace", help="Save trace JSON")] = False,
) -> None:
    """Run a single-agent baseline (one LLM call)."""
    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    llm = LLMClient(model=get_settings().openai_model, temperature=0.3)
    response = llm.complete(_BASELINE_SYSTEM, f"Research query: {query}")
    state.final_answer = response.content
    state.add_trace_event(
        "baseline_done",
        {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))

    if save_trace:
        store = LocalArtifactStore()
        export_trace_json(state.trace, store.root / "trace_baseline.json")
        console.print("[dim]Trace saved to reports/trace_baseline.json[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[bool, typer.Option("--save-trace", help="Save trace JSON")] = False,
) -> None:
    """Run the multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()

    result = workflow.run(state)

    console.print(Panel.fit(result.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(
        f"[dim]Iterations: {result.iteration} | Route: {' → '.join(result.route_history)}[/dim]"
    )

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")

    if save_trace:
        store = LocalArtifactStore()
        path = export_trace_json(result.trace, store.root / "trace_multi_agent.json")
        console.print(f"[dim]Trace saved to {path}[/dim]")


@app.command()
def benchmark(
    queries: Annotated[
        list[str],
        typer.Option("--query", "-q", help="Query to benchmark (repeatable)"),
    ] = [],  # noqa: B006
    save_report: Annotated[bool, typer.Option("--save-report", help="Write reports/")] = True,
) -> None:
    """Benchmark single-agent vs multi-agent on one or more queries."""
    _init()
    settings = get_settings()

    if not queries:
        import yaml

        cfg_path = Path(__file__).parent.parent.parent / "configs" / "lab_default.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            queries = cfg.get("benchmark", {}).get("queries", [])
        if not queries:
            queries = ["Research GraphRAG state-of-the-art and write a 500-word summary"]

    console.print(f"[bold]Benchmarking {len(queries)} query/queries × 2 runners[/bold]")

    all_metrics = []
    all_answers: dict[str, str] = {}
    store = LocalArtifactStore()
    judge_llm = LLMClient(model="gpt-4o-mini", temperature=0.0)

    def _baseline_runner(q: str) -> ResearchState:
        import time as _time

        from multi_agent_research_lab.core.schemas import AgentName, AgentResult

        st = ResearchState(request=ResearchQuery(query=q))
        llm = LLMClient(model=settings.openai_model, temperature=0.3)
        t0 = _time.perf_counter()
        resp = llm.complete(_BASELINE_SYSTEM, f"Research query: {q}")
        latency = round(_time.perf_counter() - t0, 3)
        st.final_answer = resp.content
        st.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "latency_seconds": latency,
                },
            )
        )
        st.add_trace_event(
            "baseline_done",
            {
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cost_usd": resp.cost_usd,
                "latency_seconds": latency,
            },
        )
        return st

    def _multi_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(st)

    for i, query in enumerate(queries, 1):
        short = query[:50] + "…" if len(query) > 50 else query
        console.print(f"\n[bold]Query {i}:[/bold] {short}")

        run_name_b = f"baseline_q{i}"
        state_b, metrics_b = run_benchmark(run_name_b, query, _baseline_runner, judge_llm)
        all_metrics.append(metrics_b)
        all_answers[run_name_b] = state_b.final_answer or ""
        export_trace_json(state_b.trace, store.root / f"trace_{run_name_b}.json")

        run_name_m = f"multi_agent_q{i}"
        state_m, metrics_m = run_benchmark(run_name_m, query, _multi_runner, judge_llm)
        all_metrics.append(metrics_m)
        all_answers[run_name_m] = state_m.final_answer or ""
        export_trace_json(state_m.trace, store.root / f"trace_{run_name_m}.json")

    # Rich table
    table = Table(title="Benchmark Results")
    table.add_column("Run")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Notes")
    for m in all_metrics:
        cost = f"${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd else "—"
        quality = f"{m.quality_score:.1f}" if m.quality_score is not None else "—"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost, quality, m.notes[:60])
    console.print(table)

    if save_report:
        report_md = render_markdown_report(all_metrics, answers=all_answers)
        path = store.write_text("benchmark_report.md", report_md)
        console.print(f"\n[green]Report saved to {path}[/green]")

        # also save all answers as JSON for reference
        store.write_text(
            "benchmark_answers.json", json.dumps(all_answers, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    app()
