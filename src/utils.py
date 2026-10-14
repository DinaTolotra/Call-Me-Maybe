from typing import Generator, Any
from sys import float_info
import numpy as np


def softmax(logits: list[float]) -> list[float]:
    np_logits = np.array(logits)
    exp_logits = np.exp(np_logits - np.max(np_logits))
    return exp_logits / np.sum(exp_logits)


def index_sort(l: list[float], count: int) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    working = list(l)
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


def batched(l: list[Any], batch_size: int) -> Generator[list[Any], None, None]:
    for i in range(0, len(l), batch_size):
        yield l[i:min(len(l), i + batch_size)]
