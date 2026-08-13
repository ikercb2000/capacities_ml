# imports
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike



# input validation
def _validate_positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_return_method(method: str) -> str:
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'.")
    return method


def _to_float_array(
    values: ArrayLike,
    *,
    ndim: tuple[int, ...] = (1, 2),
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim not in ndim:
        expected = " or ".join(str(value) for value in ndim)
        raise ValueError(f"Input must have {expected} dimensions.")
    return array


# container preservation
def _wrap_like(
    template: Any,
    values: np.ndarray,
    *,
    name: str | None = None,
):
    if isinstance(template, pd.Series):
        return pd.Series(
            values,
            index=template.index,
            name=template.name if name is None else name,
        )
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return values


# aligned pandas operations
def _binary_pandas_operation(
    left: ArrayLike,
    right: ArrayLike,
    operation: Callable[[Any, Any], Any],
):
    if isinstance(left, pd.DataFrame):
        if isinstance(right, pd.DataFrame):
            missing = set(left.columns) - set(right.columns)
            extra = set(right.columns) - set(left.columns)
            if missing or extra:
                raise ValueError("DataFrame columns must contain the same assets.")
            right = right.reindex(index=left.index, columns=left.columns)
        elif isinstance(right, pd.Series):
            if set(right.index) == set(left.columns):
                right = right.reindex(left.columns)
            elif right.index.equals(left.index):
                values = np.repeat(
                    right.to_numpy(dtype=float)[:, None],
                    left.shape[1],
                    axis=1,
                )
                right = pd.DataFrame(values, index=left.index, columns=left.columns)
            else:
                raise ValueError(
                    "A Series paired with a DataFrame must be indexed by dates "
                    "or by the DataFrame columns."
                )
        return operation(left, right)

    if isinstance(left, pd.Series):
        if isinstance(right, pd.Series):
            right = right.reindex(left.index)
        return operation(left, right)

    return operation(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
