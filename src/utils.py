from os.path import dirname, exists, isdir, isfile
from os import access, R_OK, W_OK, X_OK
from typing import Generator, Any
from sys import float_info
import numpy as np


def softmax(logits: list[float]) -> list[float]:
    if len(logits) == 0:
        return []
    np_logits = np.array(logits)
    exp_logits = np.exp(np_logits - np.max(np_logits))
    return list(exp_logits / np.sum(exp_logits))


def index_sort(iterable: list[float], count: int) -> list[tuple[int, float]]:
    res: list[tuple[int, float]] = []
    working = iterable.copy()
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
    res: list[float] = []
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


def check_input_file(fn_def_file: str, input_file: str) -> bool:
    return (
        exists(fn_def_file) and isfile(fn_def_file) and
        exists(input_file) and isfile(input_file) and
        access(fn_def_file, R_OK) and access(input_file, R_OK)
    )


def check_output_file(output_file: str) -> bool:
    return (
        (
            not exists(output_file) or
            access(output_file, W_OK)
        ) and
        exists(dirname(output_file)) and
        isdir(dirname(output_file)) and
        not isdir(output_file) and
        access(dirname(output_file), X_OK | W_OK)
    )


def clean(s: str) -> str:
    s = s.removesuffix(";")
    s = s.removeprefix('"')
    s = s.removesuffix('"')
    s = s.replace("\\\\", "\\")
    return s.strip()
