from contextlib import contextmanager
import time
from typing import Iterator, Dict


@contextmanager
def track_latency() -> Iterator[Dict[str, float]]:
    result = {}
    start = time.perf_counter()
    yield result
    result['latency_ms'] = (time.perf_counter() - start) * 1000
