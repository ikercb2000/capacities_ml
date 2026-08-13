# Choquet time-series forecasting

`ChoquetAutoRegressor` is a univariate autoregressive forecaster implementing the native sktime `BaseForecaster` interface.

## Model

With `lags=p`, the conditional mean is

\[
\widehat y_t
=
\beta_0
+
\phi\,C_\mu(y_{t-1},\ldots,y_{t-p})
+
x_t^\top\gamma,
\]

where:

- $C_\mu$ aggregates lagged observations through a learned capacity;
- $\phi$ scales the Choquet autoregressive component;
- $\beta_0$ is optional;
- exogenous regressors $x_t$ are optional.

The lag universe is automatically named

```text
lag_1, lag_2, ..., lag_p
```

so the fitted lag capacity can be interpreted using Shapley and interaction indices.

## Basic fit

```python
from capacities_ml_fin.ml.models import ChoquetAutoRegressor

forecaster = ChoquetAutoRegressor(lags=3)
forecaster.fit(y_train)
forecast = forecaster.predict(fh=[1, 2, 3])
```

The estimator expects a univariate pandas Series internally, following sktime's mtype conversion rules.

## Stationarity/contraction constraint

By default:

```python
enforce_stationarity=True
stability_bound=0.999
```

which constrains

\[
|\phi|\le 0.999.
\]

This is a contraction-style restriction on the outer autoregressive scale. It is not a generic proof of stationarity for every nonlinear specification, but it provides an explicit stability restriction for the model implemented here.

Because capacity parameters and $\phi$ multiply each other, the model is non-convex. The implementation therefore rejects the CVXPY backend; use SciPy or PYMOO.

## Reversible sktime pipelines

The repository notebook combines the model with log transformation and detrending:

```python
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender

pipeline = TransformedTargetForecaster(
    steps=[
        ("log", LogTransformer()),
        (
            "detrend",
            Detrender(
                forecaster=PolynomialTrendForecaster(degree=1)
            ),
        ),
        ("forecast", ChoquetAutoRegressor(lags=2)),
    ]
)
```

This lets sktime fit transformations only on training data and invert them automatically when producing forecasts.

## Temporal model selection

```python
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter

cv = ExpandingWindowSplitter(
    initial_window=60,
    step_length=12,
    fh=[1, 2, 3],
)

search = ForecastingGridSearchCV(
    forecaster=pipeline,
    cv=cv,
    param_grid={"forecast__lags": [1, 2, 3]},
    scoring=MeanAbsoluteError(),
)
search.fit(y_train)
```

This is preferable to ordinary shuffled cross-validation because future observations must not enter the training set of earlier folds.

## Absolute forecasting horizons

```python
from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon(y_test.index, is_relative=False)
forecast = search.predict(fh=fh)
```

The forecaster recursively generates every intermediate positive step required to reach the requested horizon, even if the requested `fh` skips some steps.

## Prediction intervals

The current interval implementation uses fitted one-step residual variance and Gaussian scaling:

```python
interval = search.predict_interval(
    fh=fh,
    coverage=0.95,
)
```

The scale increases with the square root of the relative horizon. These are model-based Gaussian intervals, not conformal intervals or bootstrap distributions.

## Exogenous variables

```python
forecaster.fit(y_train, X=X_train)
forecast = forecaster.predict(fh=fh, X=X_future)
```

If the model was fitted with exogenous data:

- future `X` is required;
- columns must match the fitted columns in the same order;
- `X` must include **every intermediate recursive forecasting date**, not just the explicitly requested horizon values.

These checks prevent silent misalignment during recursive forecasting.

## Fitted diagnostics

The estimator exposes:

```python
model.capacity_
model.phi_
model.intercept_
model.exogenous_coef_
model.fittedvalues_
model.resid_
model.sigma2_
model.n_params_
```

Convenience methods:

```python
model.resid()
model.aic()
model.bic()
```

AIC and BIC use a Gaussian residual likelihood with the fitted parameter count.

## Interpret lag importance

```python
from capacities_ml_fin.base.interpretation import (
    shapley_indices,
    pairwise_interactions,
)

shapley_indices(model.capacity_)
pairwise_interactions(model.capacity_)
```

A Shapley value attached to `lag_1` describes the learned global contribution of the most recent lag inside the Choquet aggregation. Pairwise interaction between `lag_1` and `lag_2` describes whether those lags act complementarily or redundantly in the capacity.

## Updating

Because the model implements the sktime update hook:

```python
model.update(y_new, X=X_new, update_params=True)
```

With `update_params=True`, the estimator refits on the updated internal history. With `update_params=False`, sktime appends data while the model retains current fitted parameters.

## Diagnostics before modeling

The package deliberately does not duplicate statsmodels' diagnostic ecosystem. A useful workflow is:

1. inspect trend/transformation needs;
2. use ADF/ACF/PACF or other diagnostics from statsmodels;
3. construct a reversible sktime preprocessing pipeline;
4. select lag order by temporal CV;
5. compare with classical forecasters under identical folds and metrics.

The repository `07_choquet_time_series.ipynb` follows exactly this pattern.

## API

See [`ChoquetAutoRegressor`](../api/ml_models.md#choquetautoregressor).
