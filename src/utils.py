from collections.abc import Callable
from typing import Generator, Any
from time import perf_counter
from sys import float_info
import numpy as np


def softmax(logits: list[float]) -> list[float]:
    np_logits = np.array(logits)
    exp_logits = np.exp(np_logits - np.max(np_logits))
    return list(exp_logits / np.sum(exp_logits))


def index_sort(iterable: list[float], count: int) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    working = list(iterable)
    limit = min(count, len(working))
    while len(res) < limit:
        max_value: float = -float_info.max
        max_index: int = -1
        for i, value in enumerate(working):
            if value > max_value:
                max_value = value
                max_index = i
        if max_index == -1:
            break
        res.append((max_index, max_value))
        working[max_index] = -float_info.max
    return res


def filter_proba(proba: list[float], allowed: set[int]) -> list[float]:
    res = list[float]()
    for i in range(len(proba)):
        val = proba[i]
        if i not in allowed:
            val = 0.
        res.append(val)
    return res


def batched(
        iterable: list[Any],
        batch_size: int
        ) -> Generator[list[Any], None, None]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:min(len(iterable), i + batch_size)]


def counter() -> dict[str, Callable[[], float]]:
    total_time = 0.
    start_time = 0.
    end_time = 0.

    def start() -> float:
        nonlocal start_time
        start_time = perf_counter()
        return start_time

    def end() -> float:
        nonlocal end_time
        end_time = perf_counter()
        return end_time

    def total() -> float:
        return total_time

    def duration() -> float:
        nonlocal total_time
        duration_time = end_time - start_time
        if duration_time < 0:
            raise RuntimeError("start - end - duration sequence")
        total_time += duration_time
        return duration_time

    return {
        "start": start,
        "end": end,
        "duration": duration,
        "total": total
    }


def clean(s: str) -> str:
    s = s.removesuffix(";")
    s = s.removeprefix('"')
    s = s.removesuffix('"')
    s = s.replace("\\\\", "\\")
    return s.strip()
