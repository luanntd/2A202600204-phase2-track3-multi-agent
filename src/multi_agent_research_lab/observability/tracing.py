"""Tracing hooks — minimal span context with optional JSON export."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal timing span used by all agents.

    Yields a mutable dict that agents can enrich with extra attributes.
    duration_seconds is set automatically on exit.
    """
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug(
            "span %s finished in %.4fs attrs=%s",
            name,
            span["duration_seconds"],
            span.get("attributes"),
        )


def export_trace_json(trace: list[dict[str, Any]], path: Path | str) -> Path:
    """Write the trace list to a JSON file and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    logger.info("Trace exported to %s (%d events)", out, len(trace))
    return out


def summarise_trace(trace: list[dict[str, Any]]) -> str:
    """Return a human-readable one-liner per trace event."""
    lines = []
    for event in trace:
        name = event.get("name", "?")
        payload = event.get("payload", event)
        latency = payload.get("latency_seconds", "")
        tokens = payload.get("tokens", "")
        parts = [f"[{name}]"]
        if latency:
            parts.append(f"latency={latency}s")
        if tokens:
            parts.append(f"tokens={tokens}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
