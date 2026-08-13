# Portfolio utilities

The portfolio module separates **weight construction**, **weight timing**, and **return aggregation**. It intentionally does not normalize or lag weights implicitly inside `portfolio_returns`.

## Equal weights

```python
from capacities_ml_fin.finance import equal_weights

w = equal_weights(["AAPL", "MSFT", "JPM"])
```

Named assets return a pandas Series. Passing an integer returns a NumPy vector.

## Normalize arbitrary weights

```python
from capacities_ml_fin.finance import normalize_weights

w = normalize_weights(raw_weights)
```

For a one-dimensional vector, normalization occurs once across assets. For a 2D array or DataFrame, each row is normalized independently.

The sum must be finite and non-zero. Negative weights are not rejected by `normalize_weights`; the function simply normalizes the supplied vector. If long-only constraints are required, enforce them before calling it.

## Market-cap weights

```python
from capacities_ml_fin.finance import market_cap_weights

w = market_cap_weights(market_caps)
```

Market capitalizations must be finite, non-missing and non-negative. They are then normalized cross-sectionally.

## Lag time-varying weights

A common portfolio backtest mistake is to apply weights estimated using information at date $t$ to the return realized during the same period when those weights would only have been available after formation.

The package makes lagging explicit:

```python
from capacities_ml_fin.finance import lag_weights

tradable_weights = lag_weights(signal_weights, periods=1)
```

`lag_weights` requires a two-dimensional time-varying weight object. A one-dimensional Series is interpreted as a static cross-sectional vector and is therefore rejected as ambiguous.

## Portfolio returns

```python
from capacities_ml_fin.finance import portfolio_returns

portfolio = portfolio_returns(asset_returns, tradable_weights)
```

For time-varying weights:

\[
r_{p,t}=\sum_i w_{t,i}r_{t,i}.
\]

For a static vector $w$, the same weights are applied to every row.

### Pandas alignment

If returns are a DataFrame and static weights are a Series, labels must match return columns. The function reorders the weights to the DataFrame column order before multiplication.

If both returns and weights are DataFrames, weight columns must match return columns and weights are reindexed to the return index and column order.

### No implicit behavior

`portfolio_returns` does **not**:

- normalize weights;
- lag weights;
- fill missing returns;
- renormalize after missing observations.

These choices can materially change empirical results, so they remain explicit.

## Portfolio losses

```python
from capacities_ml_fin.finance import portfolio_losses

loss = portfolio_losses(asset_returns, tradable_weights)
```

This applies the package convention

\[
L_t=-r_{p,t}.
\]

The resulting loss series can be passed directly to the risk module.

## Recommended backtest pattern

```python
weights = market_cap_weights(market_caps)
weights = lag_weights(weights, periods=1)
returns_p = portfolio_returns(asset_returns, weights)
losses_p = -returns_p
```

Whether a one-period lag is correct depends on the timing of the strategy. The utility exists to make that choice visible rather than to enforce a universal convention.

## API

See [Finance API — portfolio](../api/finance.md#portfolio).
