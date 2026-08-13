# imports
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# modules
from capacities_ml_fin.finance.utils import _to_float_array, _validate_positive_integer

from capacities_ml_fin.finance.returns import to_losses


# weight construction
def normalize_weights(weights: ArrayLike):
    """Normalize static or time-varying portfolio weights to sum to one.

    For a DataFrame or two-dimensional array, normalization is performed across
    assets for each row. For a Series or one-dimensional array, it is performed
    once across the vector.
    """
    if isinstance(weights, pd.Series):
        values = weights.astype(float)
        total = values.sum()
        if not np.isfinite(total) or np.isclose(total, 0.0):
            raise ValueError("weights must have a finite non-zero sum.")
        return values / total

    if isinstance(weights, pd.DataFrame):
        values = weights.astype(float)
        totals = values.sum(axis=1)
        if (~np.isfinite(totals)).any() or np.isclose(totals.to_numpy(), 0.0).any():
            raise ValueError("Each row of weights must have a finite non-zero sum.")
        return values.div(totals, axis=0)

    array = _to_float_array(weights)
    if array.ndim == 1:
        total = np.sum(array)
        if not np.isfinite(total) or np.isclose(total, 0.0):
            raise ValueError("weights must have a finite non-zero sum.")
        return array / total

    totals = np.sum(array, axis=1, keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(np.isclose(totals, 0.0)):
        raise ValueError("Each row of weights must have a finite non-zero sum.")
    return array / totals


def equal_weights(assets: int | Sequence[str]):
    """Return equal portfolio weights for a number or sequence of assets."""
    if isinstance(assets, (int, np.integer)):
        n_assets = _validate_positive_integer(int(assets), name="assets")
        return np.full(n_assets, 1.0 / n_assets)

    names = tuple(assets)
    if not names:
        raise ValueError("assets must contain at least one asset.")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Asset names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Asset names must be unique.")
    return pd.Series(1.0 / len(names), index=names, name="weight")


def market_cap_weights(market_caps: ArrayLike):
    """Construct cross-sectional portfolio weights from positive market caps."""
    if isinstance(market_caps, (pd.Series, pd.DataFrame)):
        values = market_caps.astype(float)
        if values.isna().any().any() if isinstance(values, pd.DataFrame) else values.isna().any():
            raise ValueError("market_caps must not contain missing values.")
        if (values < 0.0).any().any() if isinstance(values, pd.DataFrame) else (values < 0.0).any():
            raise ValueError("market_caps must be non-negative.")
        return normalize_weights(values)

    array = _to_float_array(market_caps)
    if np.any(~np.isfinite(array)):
        raise ValueError("market_caps must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError("market_caps must be non-negative.")
    return normalize_weights(array)


# weight timing
def lag_weights(weights: ArrayLike, *, periods: int = 1):
    """Lag portfolio weights explicitly to prevent accidental contemporaneous use."""
    periods = _validate_positive_integer(periods, name="periods")

    if isinstance(weights, pd.Series):
        raise ValueError(
            "Lagging a one-dimensional Series is ambiguous; provide time-varying "
            "weights as a DataFrame or two-dimensional array."
        )
    if isinstance(weights, pd.DataFrame):
        return weights.shift(periods)

    array = _to_float_array(weights)
    if array.ndim == 1:
        raise ValueError("Lagging requires time-varying two-dimensional weights.")
    result = np.full(array.shape, np.nan, dtype=float)
    result[periods:] = array[:-periods]
    return result


# portfolio aggregation
def portfolio_returns(asset_returns: ArrayLike, weights: ArrayLike):
    """Aggregate asset returns using static or time-varying portfolio weights.

    Weights are used exactly as supplied. They are not lagged or normalized
    implicitly; use :func:`lag_weights` and :func:`normalize_weights` when that
    behavior is desired.
    """
    if isinstance(asset_returns, pd.DataFrame):
        returns = asset_returns.astype(float)

        if isinstance(weights, pd.Series):
            if set(weights.index) != set(returns.columns):
                raise ValueError("Weight labels must match the return columns.")
            aligned = weights.reindex(returns.columns).astype(float)
            result = returns.mul(aligned, axis=1).sum(axis=1, min_count=len(returns.columns))
            result.name = "portfolio_return"
            return result

        if isinstance(weights, pd.DataFrame):
            if set(weights.columns) != set(returns.columns):
                raise ValueError("Weight columns must match the return columns.")
            aligned = weights.reindex(index=returns.index, columns=returns.columns).astype(float)
            result = (returns * aligned).sum(axis=1, min_count=len(returns.columns))
            result.name = "portfolio_return"
            return result

        weights_array = _to_float_array(weights)
        if weights_array.ndim == 1:
            if weights_array.shape[0] != returns.shape[1]:
                raise ValueError("Static weights must match the number of assets.")
            result = returns.mul(weights_array, axis=1).sum(
                axis=1,
                min_count=len(returns.columns),
            )
            result.name = "portfolio_return"
            return result
        if weights_array.shape != returns.shape:
            raise ValueError("Time-varying weights must have the same shape as returns.")
        result = pd.Series(
            np.sum(returns.to_numpy() * weights_array, axis=1),
            index=returns.index,
            name="portfolio_return",
        )
        return result

    returns_array = _to_float_array(asset_returns, ndim=(2,))
    weights_array = _to_float_array(weights)
    if weights_array.ndim == 1:
        if weights_array.shape[0] != returns_array.shape[1]:
            raise ValueError("Static weights must match the number of assets.")
        return returns_array @ weights_array
    if weights_array.shape != returns_array.shape:
        raise ValueError("Time-varying weights must have the same shape as returns.")
    return np.sum(returns_array * weights_array, axis=1)


def portfolio_losses(asset_returns: ArrayLike, weights: ArrayLike):
    """Compute portfolio losses using the convention ``loss = -return``."""
    return to_losses(portfolio_returns(asset_returns, weights))
