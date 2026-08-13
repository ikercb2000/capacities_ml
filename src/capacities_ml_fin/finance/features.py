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

from capacities_ml_fin.finance.returns import wealth_index


# volatility features
def realized_volatility(
    returns: ArrayLike,
    *,
    window: int = 20,
    demean: bool = False,
    annualize: bool = False,
    periods_per_year: int = 252,
):
    """Estimate trailing realized volatility from equally spaced returns.

    With ``demean=False``, the non-annualized estimate is the square root of
    the rolling sum of squared returns. With ``demean=True``, it is the rolling
    sample standard deviation multiplied by ``sqrt(window)``. If ``annualize``
    is true, the window estimate is scaled by ``sqrt(periods_per_year/window)``.
    """
    window = _validate_positive_integer(window, name="window")
    periods_per_year = _validate_positive_integer(
        periods_per_year,
        name="periods_per_year",
    )

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        values = returns.astype(float)
        if demean:
            result = values.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(window)
        else:
            result = np.sqrt((values**2).rolling(window, min_periods=window).sum())
        if annualize:
            result = result * np.sqrt(periods_per_year / window)
        return result

    array = _to_float_array(returns)
    result = np.full(array.shape, np.nan, dtype=float)
    for end in range(window - 1, array.shape[0]):
        sample = array[end - window + 1 : end + 1]
        if sample.ndim == 1:
            if not np.all(np.isfinite(sample)):
                continue
            value = np.std(sample, ddof=1) * np.sqrt(window) if demean else np.sqrt(np.sum(sample**2))
            result[end] = value
            continue
        valid = np.all(np.isfinite(sample), axis=0)
        if demean:
            result[end, valid] = np.std(sample[:, valid], axis=0, ddof=1) * np.sqrt(window)
        else:
            result[end, valid] = np.sqrt(np.sum(sample[:, valid] ** 2, axis=0))
    if annualize:
        result *= np.sqrt(periods_per_year / window)
    return result


# return features
def momentum(
    returns: ArrayLike,
    *,
    lookback: int = 252,
    skip: int = 21,
    method: str = "simple",
):
    """Compute trailing return momentum without using future observations.

    ``lookback`` is the total distance back from time ``t`` and ``skip`` is the
    most recent part of that interval to omit. Thus ``lookback=252, skip=21``
    aggregates the 231 observations ending 21 periods before ``t``.
    """
    lookback = _validate_positive_integer(lookback, name="lookback")
    if not isinstance(skip, (int, np.integer)) or int(skip) < 0:
        raise ValueError("skip must be a non-negative integer.")
    skip = int(skip)
    if skip >= lookback:
        raise ValueError("skip must be smaller than lookback.")
    method = _validate_return_method(method)
    window = lookback - skip

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        shifted = returns.astype(float).shift(skip)
        if method == "log":
            return shifted.rolling(window, min_periods=window).sum()
        return (1.0 + shifted).rolling(window, min_periods=window).apply(
            np.prod,
            raw=True,
        ) - 1.0

    array = _to_float_array(returns)
    shifted = np.full(array.shape, np.nan, dtype=float)
    if skip == 0:
        shifted[...] = array
    else:
        shifted[skip:] = array[:-skip]

    result = np.full(array.shape, np.nan, dtype=float)
    for end in range(window - 1, array.shape[0]):
        sample = shifted[end - window + 1 : end + 1]
        if sample.ndim == 1:
            if not np.all(np.isfinite(sample)):
                continue
            result[end] = np.sum(sample) if method == "log" else np.prod(1.0 + sample) - 1.0
            continue
        valid = np.all(np.isfinite(sample), axis=0)
        if method == "log":
            result[end, valid] = np.sum(sample[:, valid], axis=0)
        else:
            result[end, valid] = np.prod(1.0 + sample[:, valid], axis=0) - 1.0
    return result


# drawdown features
def drawdown(returns: ArrayLike, *, method: str = "simple"):
    """Compute drawdown from the running maximum of a cumulative wealth index."""
    wealth = wealth_index(returns, method=method)
    if isinstance(wealth, (pd.Series, pd.DataFrame)):
        return wealth / wealth.cummax() - 1.0
    array = np.asarray(wealth, dtype=float)
    return array / np.maximum.accumulate(array, axis=0) - 1.0


def max_drawdown(returns: ArrayLike, *, method: str = "simple"):
    """Return the most negative drawdown for each return series."""
    values = drawdown(returns, method=method)
    if isinstance(values, pd.DataFrame):
        return values.min(axis=0)
    if isinstance(values, pd.Series):
        return float(values.min())
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return float(np.nanmin(array))
    return np.nanmin(array, axis=0)


# liquidity features
def dollar_volume(price: ArrayLike, volume: ArrayLike):
    """Compute traded dollar volume as ``price * volume``."""
    return _binary_pandas_operation(price, volume, lambda x, y: x * y)


def turnover(volume: ArrayLike, shares_outstanding: ArrayLike):
    """Compute share turnover as ``volume / shares_outstanding``."""
    def operation(x, y):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            if (y <= 0.0).any().any() if isinstance(y, pd.DataFrame) else (y <= 0.0).any():
                raise ValueError("shares_outstanding must be strictly positive.")
        elif np.any(np.asarray(y, dtype=float) <= 0.0):
            raise ValueError("shares_outstanding must be strictly positive.")
        return x / y

    return _binary_pandas_operation(volume, shares_outstanding, operation)


def relative_bid_ask_spread(bid: ArrayLike, ask: ArrayLike):
    """Compute the quoted relative spread around the bid-ask midpoint."""
    def operation(bid_values, ask_values):
        if isinstance(bid_values, (pd.Series, pd.DataFrame)):
            invalid_order = ask_values < bid_values
            invalid_price = (ask_values <= 0.0) | (bid_values <= 0.0)
            order_error = (
                invalid_order.any().any()
                if isinstance(invalid_order, pd.DataFrame)
                else invalid_order.any()
            )
            price_error = (
                invalid_price.any().any()
                if isinstance(invalid_price, pd.DataFrame)
                else invalid_price.any()
            )
            if order_error:
                raise ValueError("ask must be greater than or equal to bid.")
            if price_error:
                raise ValueError("bid and ask must be strictly positive.")
        else:
            bid_array = np.asarray(bid_values, dtype=float)
            ask_array = np.asarray(ask_values, dtype=float)
            if np.any(ask_array < bid_array):
                raise ValueError("ask must be greater than or equal to bid.")
            if np.any(ask_array <= 0.0) or np.any(bid_array <= 0.0):
                raise ValueError("bid and ask must be strictly positive.")
        midpoint = (ask_values + bid_values) / 2.0
        return (ask_values - bid_values) / midpoint

    return _binary_pandas_operation(bid, ask, operation)


def amihud_illiquidity(
    returns: ArrayLike,
    traded_value: ArrayLike,
    *,
    window: int | None = None,
):
    """Compute the Amihud price-impact proxy ``abs(return) / traded_value``.

    If ``window`` is provided, the daily proxy is replaced by its trailing
    arithmetic mean over that many observations.
    """
    def operation(return_values, value_values):
        if isinstance(value_values, (pd.Series, pd.DataFrame)):
            invalid = value_values <= 0.0
            if invalid.any().any() if isinstance(invalid, pd.DataFrame) else invalid.any():
                raise ValueError("traded_value must be strictly positive.")
        elif np.any(np.asarray(value_values, dtype=float) <= 0.0):
            raise ValueError("traded_value must be strictly positive.")
        return np.abs(return_values) / value_values

    illiquidity = _binary_pandas_operation(returns, traded_value, operation)
    if window is None:
        return illiquidity

    window = _validate_positive_integer(window, name="window")
    if isinstance(illiquidity, (pd.Series, pd.DataFrame)):
        return illiquidity.rolling(window, min_periods=window).mean()

    array = _to_float_array(illiquidity)
    result = np.full(array.shape, np.nan, dtype=float)
    for end in range(window - 1, array.shape[0]):
        sample = array[end - window + 1 : end + 1]
        if sample.ndim == 1:
            if np.all(np.isfinite(sample)):
                result[end] = np.mean(sample)
            continue
        valid = np.all(np.isfinite(sample), axis=0)
        result[end, valid] = np.mean(sample[:, valid], axis=0)
    return result
