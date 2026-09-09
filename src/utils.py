from sys import float_info
import numpy as np


def softmax(logits: list[float]) -> list[float]:
    np_logits = np.array(logits)
    exp_logits = np.exp(np_logits - np.max(np_logits))
    return exp_logits / np.sum(exp_logits)


def index_sort(l: list[float], count: int) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    while len(res) < len(l) and len(res) <= count:
        max_value: float = -float_info.max
        max_index: int = -1
        for i, value in enumerate(l):
            if value > max_value:
                max_value = value
                max_index = i
        if max_index != -1:
            res.append((max_index, max_value))
            l[max_index] = -float_info.max
    return res

def filter_proba(proba: list[float], allowed: set[int]) -> list[float]:
    res = list[float]()
    for i in range(len(proba)):
        val = proba[i]
        if i not in allowed:
            val = 0.
        res.append(val)
    return res
