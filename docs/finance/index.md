# Finance

The finance module contains **data-construction utilities**, not a full market-data system. Its purpose is to make common empirical operations explicit and auditable before modeling.

The core design principle is to distinguish clearly between:

- **historical variables** available at or before time $t$;
- **forward targets** constructed from observations after time $t$;
- **portfolio weights** that may need to be lagged before they are applied;
- **fundamental information** that becomes available only after its reporting period ends.

## Typical workflow

```text
raw prices / volumes / fundamentals
          ↓
returns and point-in-time alignment
          ↓
trailing features
          ↓
forward return/loss targets
          ↓
model training
          ↓
portfolio weights
          ↓
lag weights if required
          ↓
portfolio returns / losses
          ↓
risk estimation and backtesting
```

## Main submodules

### Returns

`price_returns`, `aggregate_returns`, `forward_returns`, `forward_losses`, `excess_returns`, and `wealth_index` handle common return transformations while preserving pandas alignment when pandas objects are used.

[Returns and losses](returns.md)

### Features

The current feature set includes trailing realized volatility, momentum, drawdown, maximum drawdown, dollar volume, turnover, relative bid-ask spread and Amihud illiquidity.

[Financial features](features.md)

### Portfolios

Portfolio utilities construct equal or market-cap weights, normalize arbitrary weight vectors, lag time-varying weights explicitly, and aggregate asset returns.

[Portfolio utilities](portfolio.md)

### Point-in-time alignment

`point_in_time_join` implements backward as-of matching so a model row can only use information whose availability timestamp is no later than the model date.

[Point-in-time alignment](alignment.md)

## No silent timing assumptions

Several functions deliberately require the caller to make timing choices explicitly:

- `portfolio_returns` does **not** normalize or lag weights automatically;
- `apply_publication_lag` is explicitly documented as a fallback when exact publication dates are unavailable;
- `forward_returns` stores future targets at the current observation position, making the look-ahead intentional and visible;
- `validate_no_lookahead` can be used after joins to verify availability constraints.

This is preferable to hiding economically meaningful timing assumptions inside convenience functions.

## Container behavior

Most finance functions preserve pandas Series/DataFrame alignment and names where practical. NumPy arrays are also accepted for numerical workflows.

For panel-style operations such as point-in-time joins, pandas DataFrames are the intended interface because timestamps and group keys are part of the data semantics.
