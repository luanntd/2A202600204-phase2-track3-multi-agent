"""Timing utility."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def elapsed_timer() -> Iterator[Callable[[], float]]:
    started = perf_counter()

    def elapsed() -> float:
        return perf_counter() - started

    yield elapsed
