"""Small shared utilities for reproducibility, paths, and schedules."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_quantiles(values: Sequence[float], quantiles: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {str(q): math.nan for q in quantiles}
    return {str(q): float(np.quantile(arr, q)) for q in quantiles}


def chunked(iterable: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def pad_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    return ((length + multiple - 1) // multiple) * multiple


def pairwise_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0.0, right - left)


def cosine_ema(momentum_start: float, momentum_end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return momentum_end
    ratio = step / max(1, total_steps - 1)
    cosine = 0.5 * (1.0 - math.cos(math.pi * ratio))
    return momentum_start + (momentum_end - momentum_start) * cosine
