# Conditional and rolling risk

The risk module provides two different ways to move from an unconditional sample to time-varying risk estimates:

- `ResidualBootstrapDistribution` builds a conditional empirical loss distribution around a fitted point predictor;
- `RollingRiskEstimator` recalculates a risk measure on a rolling or expanding historical window.

## Conditional residual-bootstrap distributions

```python
from sklearn.linear_model import LinearRegression
from capacities_ml_fin.risk import ResidualBootstrapDistribution

conditional = ResidualBootstrapDistribution(
    LinearRegression(),
    center_residuals=True,
    max_scenarios=256,
    random_state=0,
).fit(X_train, y_train)
```

The estimator is cloned and fitted. In-sample residuals are stored as

\[
\varepsilon_t=y_t-\widehat y_t.
\]

If `center_residuals=True`, their sample mean is removed.

For a new predictor row $x$ with point prediction $\widehat y(x)$, scenarios are formed as

\[
\widehat y(x)+\varepsilon_j.
\]

```python
scenarios = conditional.predict_scenarios(X_test)
```

Each row corresponds to one test observation and each column to a bootstrapped residual scenario.

To obtain `EmpiricalLossDistribution` objects directly:

```python
distributions = conditional.predict_distribution(X_test)
```

Then apply ordinary risk functions to each distribution's loss vector or use its capacity-aware quantile methods.

## Scenario subsampling

`max_scenarios` limits the residual sample used for prediction. If set, residuals are sampled according to the helper's bootstrap logic using `random_state` for reproducibility.

## Rolling capital estimation

```python
from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    RollingRiskEstimator,
)

estimator = RollingRiskEstimator(
    ExpectedShortfallDistortion(0.95),
    window=250,
    window_type="rolling",
    min_periods=250,
).fit(losses)
```

### Alignment and no contemporaneous loss

For each position $t$, the in-sample capital is computed from

```text
history = observations strictly before t
```

not including the realized loss at $t$ itself. This is the central timing guarantee of the rolling estimator.

```python
capital = estimator.predict_in_sample()
```

If the original losses are a pandas Series, the returned capital is aligned to the same index.

## Rolling versus expanding

- `window_type="rolling"`: only the most recent `window` finite historical aggregates are used;
- `window_type="expanding"`: the historical sample grows over time.

For an expanding estimator, `window` may be `None`.

## Multi-period horizon

`horizon > 1` first forms trailing aggregated losses of that length before estimating capital. For horizon $h$, an aggregate ending at historical position $s$ is the sum of the $h$ losses ending at $s$.

The resulting rolling forecast remains based only on historical aggregates available before the forecast position.

## Exponential decay

```python
estimator = RollingRiskEstimator(
    risk_measure,
    window=250,
    decay=0.98,
)
```

Weights are proportional to powers of `decay`, with newer observations receiving larger weights. They are normalized within each historical window.

If `risk_measure` is a `Distortion`, the estimator wraps it in `DistortionRiskMeasure`, which accepts the generated sample weights. A custom callable used with `decay` must also accept a `sample_weight=` keyword.

## Forecast from the latest window

After fitting:

```python
future_capital = estimator.predict(n_periods=5)
```

The current implementation estimates one capital value from the latest historical window and repeats it for the requested number of periods. It does not simulate future losses recursively.

## Updating

```python
estimator.update(new_losses)
```

appends observations and recomputes the historical aligned capital series.

## Several measures at once

```python
from capacities_ml_fin.risk import (
    ValueAtRiskDistortion,
    ExpectedShortfallDistortion,
    rolling_risk_estimates,
)

capital = rolling_risk_estimates(
    losses,
    {
        "VaR 95%": ValueAtRiskDistortion(0.95),
        "ES 95%": ExpectedShortfallDistortion(0.95),
    },
    window=250,
    min_periods=250,
)
```

The result is an aligned DataFrame with one column per named measure.

## API

See [Risk API — conditional and rolling](../api/risk.md#conditional-and-rolling-risk).
