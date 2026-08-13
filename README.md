# capacities_ml_fin

`capacities_ml_fin` is a Python package for working with capacities, Choquet integrals,
capacity-based machine-learning models, and non-additive risk measures.

The package provides the complete workflow around these objects: defining and
validating capacities, computing Choquet integrals, learning capacities from data,
interpreting interactions, selecting model complexity, and applying capacities to
financial loss distributions.

## Main features

- Explicit, k-additive, and Möbius capacity representations.
- Discrete Choquet integrals for individual observations and batches.
- Shapley values and interaction indices for capacity interpretation.
- Scikit-learn-compatible regression and classification estimators.
- Choquet autoregression with a pmdarima-like forecasting interface.
- Choquet neural regressors and classifiers.
- Capacity preprocessing, regularization, and model-selection utilities.
- SciPy, CVXPY, and PYMOO optimization backends.
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
`pip install capacities_ml_fin`.

## Capacities and Choquet integrals

A capacity assigns a value to each coalition of variables. Unlike a probability
measure, it does not need to be additive, allowing the model to represent
complementarity and redundancy between variables.

```python
import numpy as np

from capacities_ml_fin.base.capacities import MobiusCapacity, VariableUniverse
from capacities_ml_fin.base.integrals.choquet import ordered_choquet

universe = VariableUniverse(("quality", "cost", "speed"))
capacity = MobiusCapacity(
    universe=universe,
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

from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

universe = VariableUniverse(("profitability", "liquidity", "volatility"))

model = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquetRegressor(
                universe=universe,
                sparsity=KAdditivity(order=2),
            ),
        ),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The available estimators are:

- `ChoquetRegressor` for continuous targets.
- `ChoquetClassifier` for direct threshold classification.
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

`ChoquetAutoRegressor` aggregates lagged observations through a learned capacity.
Its interface follows the usual time-series workflow:

```python
from capacities_ml_fin.ml.models import ChoquetAutoRegressor

model = ChoquetAutoRegressor(lags=5).fit(y_train)
forecast, intervals = model.predict(10, return_conf_int=True)
model.update(y_new)
```

It also supports aligned exogenous variables, in-sample predictions, residuals,
AIC, BIC, and stationary parameter constraints.

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
estimators, optimization, preprocessing, interpretation, and risk functionality.
