# Returns and losses

The returns module separates **historical aggregation** from **forward target construction**. This distinction is crucial for forecasting experiments.

## `price_returns`

```python
from capacities_ml_fin.finance import price_returns

log_returns = price_returns(prices, method="log")
simple_returns = price_returns(prices, method="simple")
```

For prices $P_t$:

- log return:

\[
r_t^{\log}=\log P_t-\log P_{t-h};
\]

- simple return:

\[
r_t^{\mathrm{simple}}=\frac{P_t}{P_{t-h}}-1.
\]

The first `periods` observations are `NaN` because there is no previous price at the requested distance.

Log returns require strictly positive finite prices.

## Historical aggregation

`aggregate_returns` forms a **trailing** return ending at time $t$.

```python
from capacities_ml_fin.finance import aggregate_returns

monthly_like = aggregate_returns(
    daily_returns,
    horizon=21,
    method="log",
)
```

For log returns:

\[
R_t^{(h)}
=
\sum_{j=0}^{h-1}r_{t-j}.
\]

For simple returns:

\[
R_t^{(h)}
=
\prod_{j=0}^{h-1}(1+r_{t-j})-1.
\]

This is a feature-like historical transformation: it contains no observations after $t$.

## Forward targets

`forward_returns` instead creates a target at position $t$ from the next `horizon` observations:

\[
Y_t
=
\operatorname{aggregate}(r_{t+1},\ldots,r_{t+h}).
\]

```python
from capacities_ml_fin.finance import forward_returns

target = forward_returns(
    daily_returns,
    horizon=5,
    method="log",
)
```

The final `horizon` rows are `NaN` because the required future observations do not exist.

This convention is useful for supervised forecasting datasets:

```python
features = feature_frame.loc[target.notna()]
y = target.dropna()
```

The future information belongs in `y`, not in predictors.

## Loss convention

The package uses

\[
L=-R.
\]

```python
from capacities_ml_fin.finance import to_losses, forward_losses

losses = to_losses(returns)
future_loss = forward_losses(returns, horizon=5)
```

This means larger values represent larger losses in the risk module.

## Excess returns

```python
from capacities_ml_fin.finance import excess_returns

excess = excess_returns(asset_returns, risk_free)
```

This computes arithmetic subtraction

\[
r_t^{e}=r_t-r_{f,t}.
\]

When pandas objects are supplied, labels are aligned before subtraction.

## Wealth index

```python
from capacities_ml_fin.finance import wealth_index

wealth = wealth_index(simple_returns, method="simple")
```

For simple returns:

\[
W_t=W_0\prod_{s\le t}(1+r_s).
\]

For log returns:

\[
W_t=W_0\exp\left(\sum_{s\le t}r_s\right).
\]

The initial level must be positive and finite.

## Example: feature and target timing

```python
from capacities_ml_fin.finance import (
    price_returns,
    aggregate_returns,
    forward_losses,
)

returns = price_returns(prices, method="log")
trailing_20d = aggregate_returns(returns, horizon=20, method="log")
y_5d_loss = forward_losses(returns, horizon=5, method="log")
```

At a given date $t$:

- `trailing_20d[t]` uses data up to and including $t$;
- `y_5d_loss[t]` intentionally uses $t+1$ through $t+5$ and should therefore only be used as a supervised target.

## API

See the [Finance API](../api/finance.md#returns) for generated signatures and parameter documentation.
