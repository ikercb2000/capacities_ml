# imports
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# modules
from capacities_ml_fin.finance.utils import (
    _binary_pandas_operation,
    _to_float_array,
    _validate_positive_integer,
    _validate_return_method,
    _wrap_like,
)


# historical returns
def price_returns(
    prices: ArrayLike,
    *,
    periods: int = 1,
    method: str = "log",
):
    """Compute simple or logarithmic returns while preserving time alignment.

    Parameters
    ----------
    prices:
        One price series or a matrix whose rows are ordered observations and
        whose columns are assets.
    periods:
        Number of observations between consecutive prices.
    method:
        ``"log"`` for log returns or ``"simple"`` for arithmetic returns.

    Returns
    -------
    Same container type as ``prices`` when a pandas Series/DataFrame is used;
    otherwise a NumPy array. The first ``periods`` observations are ``NaN``.
    """
    periods = _validate_positive_integer(periods, name="periods")
    method = _validate_return_method(method)

    if isinstance(prices, (pd.Series, pd.DataFrame)):
        values = prices.astype(float)
        if method == "log":
            if (values <= 0.0).any().any() if isinstance(values, pd.DataFrame) else (values <= 0.0).any():
                raise ValueError("Log returns require strictly positive prices.")
            return np.log(values).diff(periods)
        return values.pct_change(periods=periods, fill_method=None)

    array = _to_float_array(prices)
    if method == "log" and np.any(array[np.isfinite(array)] <= 0.0):
        raise ValueError("Log returns require strictly positive prices.")

    result = np.full(array.shape, np.nan, dtype=float)
    current = array[periods:]
    previous = array[:-periods]
    if method == "log":
        result[periods:] = np.log(current) - np.log(previous)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result[periods:] = current / previous - 1.0
    return result


# loss convention
def to_losses(returns: ArrayLike):
    """Convert returns into losses using the convention ``loss = -return``."""
    if isinstance(returns, pd.Series):
        name = None if returns.name is None else f"{returns.name}_loss"
        return -returns.rename(name)
    if isinstance(returns, pd.DataFrame):
        return -returns
    return -_to_float_array(returns)


# historical aggregation
def aggregate_returns(
    returns: ArrayLike,
    *,
    horizon: int,
    method: str = "log",
):
    """Aggregate historical returns over a trailing horizon.

    Log returns are summed. Simple returns are compounded. The value at time
    ``t`` uses observations ``t-horizon+1, ..., t`` and therefore contains no
    future information.
    """
    horizon = _validate_positive_integer(horizon, name="horizon")
    method = _validate_return_method(method)

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        values = returns.astype(float)
        if method == "log":
            return values.rolling(horizon, min_periods=horizon).sum()
        return (1.0 + values).rolling(horizon, min_periods=horizon).apply(
            np.prod,
            raw=True,
        ) - 1.0

    array = _to_float_array(returns)
    result = np.full(array.shape, np.nan, dtype=float)
    if array.shape[0] < horizon:
        return result

    for end in range(horizon - 1, array.shape[0]):
        window = array[end - horizon + 1 : end + 1]
        if np.any(~np.isfinite(window), axis=0).any() if window.ndim == 2 else np.any(~np.isfinite(window)):
            if window.ndim == 1:
                continue
            valid = np.all(np.isfinite(window), axis=0)
            if method == "log":
                result[end, valid] = np.sum(window[:, valid], axis=0)
            else:
                result[end, valid] = np.prod(1.0 + window[:, valid], axis=0) - 1.0
            continue
        if method == "log":
            result[end] = np.sum(window, axis=0)
        else:
            result[end] = np.prod(1.0 + window, axis=0) - 1.0
    return result


# forward targets
def forward_returns(
    returns: ArrayLike,
    *,
    horizon: int,
    method: str = "log",
):
    """Construct future-return targets using only observations after time ``t``.

    The target stored at position ``t`` aggregates returns from ``t+1`` through
    ``t+horizon``. The final ``horizon`` observations are therefore ``NaN``.
    This convention is suitable for supervised financial forecasting datasets.
    """
    horizon = _validate_positive_integer(horizon, name="horizon")
    method = _validate_return_method(method)

    array = _to_float_array(returns)
    result = np.full(array.shape, np.nan, dtype=float)
    n_obs = array.shape[0]

    for position in range(max(0, n_obs - horizon)):
        window = array[position + 1 : position + horizon + 1]
        if window.ndim == 1:
            if not np.all(np.isfinite(window)):
                continue
            result[position] = (
                np.sum(window)
                if method == "log"
                else np.prod(1.0 + window) - 1.0
            )
            continue

        valid = np.all(np.isfinite(window), axis=0)
        if method == "log":
            result[position, valid] = np.sum(window[:, valid], axis=0)
        else:
            result[position, valid] = np.prod(1.0 + window[:, valid], axis=0) - 1.0

    return _wrap_like(returns, result)


def forward_losses(
    returns: ArrayLike,
    *,
    horizon: int,
    method: str = "log",
):
    """Construct future-loss targets as the negative future aggregate return."""
    future = forward_returns(returns, horizon=horizon, method=method)
    return to_losses(future)


# excess returns
def excess_returns(returns: ArrayLike, risk_free: ArrayLike | float):
    """Compute arithmetic excess returns ``return - risk_free``.

    ``risk_free`` may be a scalar or an aligned Series/DataFrame/array. When
    pandas objects are supplied, labels are aligned before subtraction.
    """
    return _binary_pandas_operation(returns, risk_free, lambda x, y: x - y)


# wealth index
def wealth_index(
    returns: ArrayLike,
    *,
    method: str = "simple",
    initial_value: float = 1.0,
):
    """Convert a return series into a cumulative wealth index."""
    method = _validate_return_method(method)
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be a positive finite number.")

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        values = returns.astype(float)
        if method == "simple":
            return initial_value * (1.0 + values).cumprod()
        return initial_value * np.exp(values.cumsum())

    array = _to_float_array(returns)
    if method == "simple":
        return initial_value * np.cumprod(1.0 + array, axis=0)
    return initial_value * np.exp(np.cumsum(array, axis=0))
