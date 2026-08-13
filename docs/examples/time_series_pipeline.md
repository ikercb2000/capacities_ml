# Example: Choquet forecasting with statsmodels diagnostics and sktime

This example follows the intended division of labor:

- **statsmodels** for time-series diagnostics;
- **sktime** for transformations, temporal cross-validation and forecasting composition;
- `ChoquetAutoRegressor` as the nonlinear autoregressive model.

## 1. Chronological train/test split

```python
import numpy as np
import pandas as pd

from sktime.split import temporal_train_test_split

# y is a pandas Series with a supported time index
y_train, y_test = temporal_train_test_split(y, test_size=12)
```

## 2. Diagnostics on the training sample only

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

log_train = np.log(y_train)
# construct any trend-adjusted training diagnostic series here
adf_statistic, adf_pvalue, *_ = adfuller(detrended_log)
plot_acf(detrended_log, lags=18)
plot_pacf(detrended_log, lags=18, method="ywm")
```

Diagnostics inform the transformation/lag search; they should not be calculated using the future hold-out sample.

## 3. Reversible forecasting pipeline

```python
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.boxcox import LogTransformer
from sktime.transformations.series.detrend import Detrender

from capacities_ml_fin.ml.models import ChoquetAutoRegressor

pipeline = TransformedTargetForecaster(
    steps=[
        ("log", LogTransformer()),
        (
            "detrend",
            Detrender(
                forecaster=PolynomialTrendForecaster(degree=1)
            ),
        ),
        ("forecast", ChoquetAutoRegressor()),
    ]
)
```

## 4. Select lag order using expanding-window CV

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

## 5. Forecast the held-out dates

```python
from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon(y_test.index, is_relative=False)
forecast = search.predict(fh=fh)
intervals = search.predict_interval(fh=fh, coverage=0.95)
```

## 6. Inspect the selected lag capacity

```python
from capacities_ml_fin.base.interpretation import (
    shapley_indices,
    pairwise_interactions,
)

selected_pipeline = search.best_forecaster_
model = selected_pipeline.steps_[-1][1]

print("order", model.order_)
print("phi", model.phi_)
print("AIC", model.aic())
print("BIC", model.bic())
print("Shapley", shapley_indices(model.capacity_))
print("interactions", pairwise_interactions(model.capacity_))
```

The interpretation keys are `lag_1`, `lag_2`, etc.

## 7. Optional exogenous predictors

If exogenous variables are used, future `X` must contain all recursively required dates:

```python
model.fit(y_train, X=X_train)
forecast = model.predict(fh=fh, X=X_future)
```

A request for horizon steps 1 and 3 still requires exogenous data at step 2 because the model generates the missing intermediate forecast recursively.
