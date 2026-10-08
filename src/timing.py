"""
High-resolution execution timing, warmup harness, and benchmarking utilities.
"""

import time
from typing import Any, Callable, Tuple


class PrecisionTimer:
    """Context manager and utility for nanosecond-precision execution timing."""
    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter_ns()
        self.elapsed_ms = (self.end_time - self.start_time) / 1e6


def benchmark_callable(
    fn: Callable[[], Any],
    warmup_runs: int = 2,
    num_runs: int = 1
) -> Tuple[Any, float]:
    """
    Executes a callable with warmup runs and returns (result, latency_ms).
    For multiple runs, returns the mean latency across runs.
    """
    for _ in range(warmup_runs):
        _ = fn()

    latencies = []
    result = None
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1e6)

    mean_latency = float(sum(latencies) / len(latencies))
    return result, mean_latency
