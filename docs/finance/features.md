# Financial features

The feature utilities are deliberately simple, transparent transformations of aligned market series. They do not download data and they do not perform implicit cross-sectional standardization.

## Realized volatility

```python
from capacities_ml_fin.finance import realized_volatility

vol = realized_volatility(
    returns,
    window=20,
    annualize=True,
    periods_per_year=252,
)
```

With `demean=False`, the window statistic before annualization is

\[
\sqrt{\sum_{j=0}^{w-1}r_{t-j}^2}.
\]

With `demean=True`, the implementation uses the rolling sample standard deviation multiplied by $\sqrt w$.

Annualization multiplies the window measure by

\[
\sqrt{\frac{\text{periods per year}}{w}}.
\]

## Momentum

```python
from capacities_ml_fin.finance import momentum

mom = momentum(
    returns,
    lookback=252,
    skip=21,
    method="simple",
)
```

`lookback` is the total distance back from $t$, while `skip` omits the most recent part of that interval. Thus `lookback=252, skip=21` aggregates the 231 observations ending 21 periods before the current row.

This construction is trailing and does not use future observations.

## Drawdown

```python
from capacities_ml_fin.finance import drawdown, max_drawdown

dd = drawdown(returns, method="simple")
worst = max_drawdown(returns, method="simple")
```

The function first constructs a wealth index, then compares it with its running maximum:

\[
D_t=\frac{W_t}{\max_{s\le t}W_s}-1.
\]

`max_drawdown` returns the most negative drawdown per series.

## Dollar volume

```python
from capacities_ml_fin.finance import dollar_volume

value = dollar_volume(price, volume)
```

The measure is simply

\[
\text{DollarVolume}_t=P_t\times V_t.
\]

Inputs are aligned when pandas objects are used.

## Turnover

```python
from capacities_ml_fin.finance import turnover

turn = turnover(volume, shares_outstanding)
```

\[
\text{Turnover}_t
=
\frac{\text{Volume}_t}{\text{SharesOutstanding}_t}.
\]

Shares outstanding must be strictly positive.

## Relative bid-ask spread

```python
from capacities_ml_fin.finance import relative_bid_ask_spread

spread = relative_bid_ask_spread(bid, ask)
```

The quoted spread is normalized by the midpoint:

\[
\text{Spread}
=
\frac{ask-bid}{(ask+bid)/2}.
\]

The implementation checks that prices are positive and `ask >= bid`.

## Amihud illiquidity

```python
from capacities_ml_fin.finance import amihud_illiquidity

illiq = amihud_illiquidity(
    returns,
    traded_value,
    window=20,
)
```

The daily proxy is

\[
\frac{|r_t|}{\text{TradedValue}_t}.
\]

If `window` is supplied, the function returns its trailing arithmetic mean.

## Building a feature frame

```python
features = pd.DataFrame(
    {
        "volatility_20": realized_volatility(returns, window=20),
        "momentum_252_21": momentum(returns, lookback=252, skip=21),
        "amihud_20": amihud_illiquidity(
            returns,
            dollar_volume(price, volume),
            window=20,
        ),
    }
)
```

When these features are used in a capacity model, decide which criteria are benefits and which are costs before fitting. For example, volatility and illiquidity may need to be listed in `CapacityNormalizer(cost_features=...)` depending on the predictive interpretation.

## API

See [Finance API — features](../api/finance.md#features).
