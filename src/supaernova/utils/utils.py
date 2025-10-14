from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import chainconsumer as cc

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt

    from supaernova.typing import T, Config


def pp(expression: object) -> None:
    __import__("pprint").pprint(expression)


def pf(expression: object) -> str:
    return __import__("pprint").pformat(expression)


def deepmerge(d1: "Config[T]", d2: "Config[T]") -> "Config[T]":
    out = d1.copy()
    for k, v in d2.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deepmerge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(
    input_path: str | Path | None = None,
    *,
    default_path: str | Path | None = None,
    relative_path: str | Path,
    mkdir: bool = False,
) -> Path:
    if input_path is None:
        if default_path is None:
            msg = "Cannot resolve `input_path=None` with `default_path=None"
            raise ValueError(msg)
        input_path = default_path
    input_path = Path(input_path)
    relative_path = Path(relative_path)
    if not input_path.is_absolute():
        input_path = relative_path / input_path
    final_path = input_path.resolve()
    if mkdir:
        final_path.mkdir(parents=True, exist_ok=True)
    return final_path


def jackknife_resample(arr: "npt.NDArray", func: "Callable") -> "npt.NDArray":
    n = len(arr)
    jackknife_values = np.zeros(n)
    for i in range(n):
        jackknife_values[i] = func(np.delete(arr, i))
    return jackknife_values


def max_central(
    data: "npt.NDArray",
) -> tuple[float, float, float]:
    c = cc.ChainConsumer()
    chain = cc.Chain(samples=pd.DataFrame({"data": data}), name="data")
    max_central = c.analysis.get_parameter_summary_max_central(chain, "data")
    center = max_central.center
    lower = max_central.center - max_central.lower
    upper = max_central.upper - max_central.center
    return (lower, center, upper)
