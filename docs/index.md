# capacities-ml-fin

`capacities-ml-fin` is a Python package for **finite capacities**, **Choquet integrals**, **capacity-based machine learning**, and **non-additive financial risk measurement**.

The package is designed around one idea: use a common capacity abstraction from the mathematical layer all the way to fitted machine-learning models and financial risk workflows. The same object that represents a non-additive set function can be integrated, interpreted through Shapley-style indices, learned from data, and reused as an event capacity in risk measurement.

<div class="grid cards" markdown>

-   :material-set-all: **Capacities and Choquet integration**

    ---

    Explicit, Möbius, and $k$-additive representations; exact validation; scalar and batch Choquet integrals.

    [Capacity theory](theory/capacities.md)

-   :material-brain: **Machine learning**

    ---

    Scikit-learn compatible regression, deterministic and probabilistic classification, neural Choquet models, and sktime forecasting.

    [Machine-learning guide](ml/index.md)

-   :material-chart-line: **Finance**

    ---

    Returns, forward targets, market features, portfolio aggregation, and point-in-time alignment utilities.

    [Finance guide](finance/index.md)

-   :material-shield-alert: **Risk**

    ---

    Distortions, generalized quantiles, spectral and Kusuoka risk measures, rolling capital estimation, and statistical backtesting.

    [Risk guide](risk/index.md)

</div>

## Design philosophy

The library separates three layers that are often mixed together in research code:

1. **Mathematical objects** — capacities, Möbius coefficients, Choquet integrals, Shapley values, and interactions.
2. **Learning infrastructure** — parameterizations, constraints, objectives, optimization backends, scikit-learn estimators, and sktime forecasters.
3. **Empirical finance and risk** — return construction, no-look-ahead alignment, portfolio losses, distorted event capacities, rolling risk, and backtesting.

This separation makes it possible to inspect a fitted model at the level of the learned capacity instead of treating the estimator as a black box.

## A first capacity

A finite capacity is a normalized monotone set function. Unlike an additive probability measure, it may assign interaction effects to coalitions of variables.

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

x = np.array([0.8, 0.4, 0.7])
score = ordered_choquet(capacity, x)
```

The Möbius representation is sparse: unspecified coefficients are treated as zero, while event values are reconstructed as needed.

## Learn a capacity from data

The supervised estimators follow the scikit-learn API:

```python
from sklearn.pipeline import Pipeline

from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", ChoquetRegressor(sparsity=KAdditivity(order=2))),
    ]
).set_output(transform="pandas")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

learned_capacity = pipeline.named_steps["model"].capacity_
```

The learned capacity can then be interpreted directly:

```python
from capacities_ml_fin.base.interpretation import (
    shapley_indices,
    pairwise_interactions,
)

importance = shapley_indices(learned_capacity)
interactions = pairwise_interactions(learned_capacity)
```

## Time-series forecasting

`ChoquetAutoRegressor` implements the native sktime forecaster interface. Its conditional mean is

\[
\widehat y_t
= \beta_0
+ \phi\, C_\mu(y_{t-1},\ldots,y_{t-p})
+ x_t^\top\gamma,
\]

where the exogenous term is optional.

```python
from capacities_ml_fin.ml.models import ChoquetAutoRegressor
from sktime.forecasting.compose import TransformedTargetForecaster

forecaster = TransformedTargetForecaster(
    [("forecast", ChoquetAutoRegressor(lags=3))]
)
forecaster.fit(y_train)
forecast = forecaster.predict(fh=[1, 2, 3])
```

See [Time series](ml/time_series.md) for recursive forecasting, exogenous variables, prediction intervals, and temporal model selection.

## Financial risk

The risk layer treats observations as finite loss scenarios and evaluates events using a capacity.

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
report = capital_backtest(losses, capital)
```

Rolling forecasts are aligned so that the capital assigned to date $t$ is calculated only from observations strictly before that date.

## Choose the right entry point

| Goal | Main object/function |
|---|---|
| Store a complete capacity table | `ExplicitCapacity` |
| Store sparse Möbius coefficients | `MobiusCapacity` |
| Construct a capacity known to be $k$-additive | `KAdditiveCapacity` |
| Learn a $k$-additive capacity | `KAdditivity` passed to an estimator |
| Compute a Choquet integral | `ordered_choquet`, `batch_choquet_integral` |
| Interpret variable importance | `shapley_indices` |
| Interpret pairwise complementarity/redundancy | `pairwise_interactions` |
| Continuous supervised prediction | `ChoquetRegressor` |
| Hard binary classification | `ChoquetClassifier` |
| Probabilistic binary classification | `ChoquisticRegression` |
| Fuzzy-input neural model | `FuzzyChoquetNeuralRegressor`, `FuzzyChoquetNeuralClassifier` |
| Aggregate model predictions | `aggregate_regression_predictions`, `aggregate_binary_probabilities` |
| Forecast a univariate series | `ChoquetAutoRegressor`, `FuzzyChoquetNeuralAutoRegressor` |
| Normalize/orient criteria | `CapacityNormalizer` |
| Build point-in-time financial datasets | `point_in_time_join` |
| Distortion risk | `DistortionRiskMeasure`, `distortion_risk_measure` |
| Generalized capacity quantiles | `generalized_value_at_risk` |
| Spectral/Kusuoka aggregation | `SpectralRiskMeasure`, `KusuokaRiskMeasure` |
| Rolling capital forecasts | `RollingRiskEstimator` |

## Documentation map

- [Getting started](getting-started/index.md) gives the shortest route to a working environment.
- [Theory](theory/index.md) explains the mathematical objects used by the code.
- [Machine learning](ml/index.md) documents estimators, pipelines, optimization and model selection.
- [Finance](finance/index.md) documents empirical data construction.
- [Risk](risk/index.md) documents non-additive risk measurement and backtesting.
- [API reference](api/index.md) gives object-by-object technical documentation generated from the source code and supplemented with manual guidance.
- [Examples](examples/index.md) contains complete workflows adapted from the repository notebooks.
