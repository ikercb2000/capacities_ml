# Example: build a point-in-time financial modeling dataset

This example shows how the finance utilities can be combined without hiding timing assumptions.

## 1. Convert prices to returns

```python
from capacities_ml_fin.finance import price_returns

returns = price_returns(prices, method="log")
```

## 2. Build trailing market features

```python
import pandas as pd

from capacities_ml_fin.finance import (
    amihud_illiquidity,
    dollar_volume,
    momentum,
    realized_volatility,
)

traded_value = dollar_volume(price, volume)

features = pd.DataFrame(
    {
        "volatility_20": realized_volatility(
            returns,
            window=20,
            annualize=True,
        ),
        "momentum_252_21": momentum(
            returns,
            lookback=252,
            skip=21,
            method="log",
        ),
        "amihud_20": amihud_illiquidity(
            returns,
            traded_value,
            window=20,
        ),
    }
)
```

All three features are historical at their row timestamp.

## 3. Create a forward target

```python
from capacities_ml_fin.finance import forward_losses

y = forward_losses(returns, horizon=5, method="log")
```

At date $t$, the target aggregates returns $t+1$ through $t+5$. It must not be used as a predictor.

## 4. Add fundamentals using availability dates

If exact availability dates are known:

```python
from capacities_ml_fin.finance import point_in_time_join

model_frame = point_in_time_join(
    market_frame,
    fundamental_frame,
    left_on="date",
    available_on="available_date",
    by="ticker",
)
```

If only period-end dates are known, a fixed lag can be used explicitly as an approximation:

```python
from capacities_ml_fin.finance import apply_publication_lag

fundamental_frame = apply_publication_lag(
    fundamental_frame,
    period_end="period_end",
    lag=45,
)
```

## 5. Validate the alignment

```python
from capacities_ml_fin.finance import validate_no_lookahead

validate_no_lookahead(model_frame)
```

## 6. Build portfolio losses

Assume a DataFrame of time-varying market caps and aligned asset returns:

```python
from capacities_ml_fin.finance import (
    lag_weights,
    market_cap_weights,
    portfolio_losses,
)

weights = market_cap_weights(market_caps)
weights = lag_weights(weights, periods=1)
portfolio_loss = portfolio_losses(asset_returns, weights)
```

The one-period lag is an empirical choice, not an automatic package assumption.

## 7. Hand the losses to the risk layer

```python
from capacities_ml_fin.risk import expected_shortfall

es_95 = expected_shortfall(
    portfolio_loss.dropna().to_numpy(),
    alpha=0.95,
)
```

This separation keeps dataset timing, portfolio construction and risk estimation auditable as independent steps.
