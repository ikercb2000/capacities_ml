# capacities_ml_fin

`capacities_ml_fin` is a Python package for working with capacities, Choquet integrals,
capacity-based machine-learning models, and non-additive risk measures.

[Documentation](https://ikercb2000.github.io/capacities_ml_fin/) · [Source code](https://github.com/ikercb2000/capacities_ml_fin)

The package provides the complete workflow around these objects: defining and
validating capacities, computing Choquet integrals, learning capacities from data,
interpreting interactions, selecting model complexity, and applying capacities to
financial loss distributions.

## Main features

- Explicit, k-additive, and Möbius capacity representations.
- Discrete Choquet integrals for individual observations and batches.
- Shapley values and interaction indices for capacity interpretation.
- Scikit-learn-compatible regression and classification estimators.
- Choquet autoregression implementing the native sktime forecasting interface.
- Choquet neural regressors and classifiers.
- Capacity preprocessing, regularization, and model-selection utilities.
- SciPy, CVXPY, and PYMOO optimization backends.
- Point-in-time financial data, return, feature, and portfolio utilities.
- Distortion, spectral, and generalized risk measures.
- Conditional and rolling risk forecasts with backtesting tools.

## Installation

The project requires Python 3.12 or later. Install the package and its development
environment from the repository root with Poetry:

```powershell
poetry install
```

Commands can then be run inside the project environment:

```powershell
poetry run python your_script.py
poetry run pytest
```

The local project can also be installed with pip:

```powershell
pip install .
```

Once published, the distribution can be installed from PyPI with
`pip install capacities-ml-fin`.

## Capacities and Choquet integrals

A capacity assigns a value to each coalition of variables. Unlike a probability
measure, it does not need to be additive, allowing the model to represent
complementarity and redundancy between variables.

```python
import numpy as np

from capacities_ml_fin.base.capacities import MobiusCapacity
from capacities_ml_fin.base.integrals.choquet import ordered_choquet

capacity = MobiusCapacity(
    coefficients={
        ("quality",): 0.35,
        ("cost",): 0.25,
        ("speed",): 0.20,
        ("quality", "speed"): 0.20,
    },
)

score = ordered_choquet(capacity, np.array([0.8, 0.4, 0.7]))
print(score)
```

`ExplicitCapacity` stores the complete capacity table, while `MobiusCapacity`
supports a sparse, directly evaluable Möbius representation. Both implement the
common `BaseCapacity` interface through `event_value()` and
`nested_event_values()`.

## Machine-learning models

The supervised estimators follow the scikit-learn API and can be used with
pipelines, cross-validation, metrics, and parameter search.

```python
from sklearn.pipeline import Pipeline

from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

model = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquetRegressor(
                sparsity=KAdditivity(order=2),
            ),
        ),
    ]
).set_output(transform="pandas")

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Capacity objects infer names from named coalition keys. Supervised estimators
infer them from DataFrame columns, or generate `x0`, `x1`, ... for array input.
For an intentionally sparse capacity whose keys omit some variables, pass
`n_elements=` or `var_names=` to state the otherwise unknowable elements.

The available estimators are:

- `ChoquetRegressor` for continuous targets.
- `ScaledChoquetRegressor` for continuous targets with a learned scale
  multiplying the Choquet integral.
- `ChoquetClassifier` for deterministic threshold classification. It can learn
  non-negative feature scales jointly with the capacity and exposes margins
  through `decision_function()`; it intentionally has no `predict_proba()`.
- `ChoquisticRegression` for probabilistic binary classification.
- `ChoquetAutoRegressor` for nonlinear autoregressive forecasting.
- `ChoquetNeuralRegressor` and `ChoquetNeuralClassifier` for models with a
  hidden layer of Choquet neurons.

Predictors should normally be oriented and normalized before fitting. The
`CapacityNormalizer` scales inputs and reverses cost criteria so that larger
transformed values always have the same interpretation.

Capacity complexity can be controlled with full, k-additive, or selected-pair
representations. L1 and L2 penalties are available, and the model-selection module
provides grids that can be passed directly to scikit-learn search objects.

## Time-series models

`ChoquetAutoRegressor` aggregates lagged observations through a learned capacity
and subclasses `sktime.forecasting.base.BaseForecaster`. It can therefore be used
with sktime forecasting horizons, temporal pipelines, splitters, metrics, tuning,
exogenous variables, prediction intervals, and update workflows.

```python
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster

from capacities_ml_fin.ml.models import ChoquetAutoRegressor

fh = ForecastingHorizon(y_test.index, is_relative=False)
model = TransformedTargetForecaster(
    [("forecast", ChoquetAutoRegressor(lags=5))]
)
model.fit(y_train, X=X_train, fh=fh)
forecast = model.predict(X=X_test)
intervals = model.predict_interval(X=X_test, coverage=0.95)
model.update(y_new, X=X_new)
```

The model exposes one-step training fitted values, residuals, AIC, BIC, and a
stationary contraction constraint. Input `y` should use a supported pandas time
index; when exogenous data are used, `X` must be indexed at the same dates and
must cover every recursive future step.

## Risk measures

The risk module applies capacities to events in a finite loss distribution. Larger
values must represent larger losses.

It includes empirical distributions, probability and distorted capacities,
probability envelopes, VaR, Expected Shortfall, generalized quantiles, spectral
measures, Kusuoka representations, conditional residual bootstrapping, rolling
capital estimation, and statistical backtesting.

```python
from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    RollingRiskEstimator,
    capital_backtest,
)

estimator = RollingRiskEstimator(
    ExpectedShortfallDistortion(alpha=0.95),
    window=256,
    min_periods=256,
).fit(losses)

capital = estimator.predict_in_sample()
diagnostics = capital_backtest(losses, capital)
```

Rolling forecasts use only information preceding each forecast date. Backtesting
utilities summarize capital breaches and provide coverage, independence, stress,
bootstrap, and HAC diagnostics.

## Package structure

| Folder | Contents |
|---|---|
| `base` | Capacity objects, Möbius transforms, Choquet integrals, validation, and interpretation |
| `ml` | Models, preprocessing, model selection, regularization, and optimization |
| `finance` | Point-in-time alignment, returns, market features, and portfolio utilities |
| `risk` | Loss distributions, distortions, risk measures, forecasting, and backtesting |

These domains are part of the same `capacities_ml_fin` distribution and therefore
share the same capacity abstractions and release version.

## Notebooks

Guided examples are available in `notebooks/`:

1. `01_capacities_and_choquet_integral.ipynb`
2. `02_choquet_regression.ipynb`
3. `03_choquet_classifier.ipynb`
4. `04_choquistic_regression.ipynb`
5. `05_optimization_backends.ipynb`
6. `06_risk_pipeline.ipynb`

The notebooks cover the main modeling workflows, preprocessing, model selection,
regularization, interpretation, evaluation metrics, optimization backends, and the
complete risk pipeline.

## Testing

Run the complete test suite with:

```powershell
poetry run pytest
```

The tests are organized by package component and include capacities, integrals,
estimators, optimization, preprocessing, interpretation, finance, and risk functionality.
