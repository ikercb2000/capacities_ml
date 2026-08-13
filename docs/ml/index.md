# Machine learning

The machine-learning layer learns capacities from data while preserving the scikit-learn estimator conventions wherever the modeling problem allows it. A separate native sktime estimator is provided for forecasting.

## Estimator map

| Task | Estimator | Output |
|---|---|---|
| Continuous regression | `ChoquetRegressor` | real-valued prediction |
| Deterministic binary classification | `ChoquetClassifier` | hard class label and signed margin |
| Probabilistic binary classification | `ChoquisticRegression` | class probabilities and log-odds |
| Nonlinear continuous prediction | `ChoquetNeuralRegressor` | one-hidden-layer neural output |
| Nonlinear binary classification | `ChoquetNeuralClassifier` | logits and probabilities |
| Univariate forecasting | `ChoquetAutoRegressor` | recursive sktime forecasts and intervals |

All supervised scikit-learn estimators infer feature names from DataFrame columns when available. Otherwise, the fitted variable universe uses generated names `x0`, `x1`, ... .

## Common workflow

A typical capacity-learning workflow is:

```text
raw criteria
    ↓
CapacityNormalizer
    ↓
capacity sparsity / shape specification
    ↓
Choquet estimator
    ↓
capacity_
    ↓
Shapley values + interaction indices
```

### Preprocess criterion direction

Capacity-based monotone aggregation is easiest to interpret when all input criteria share a common direction. `CapacityNormalizer` scales each feature to a common interval and optionally reverses cost criteria.

```python
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

normalizer = CapacityNormalizer(
    cost_features=["volatility", "drawdown"],
)
```

### Select capacity complexity

```python
from capacities_ml_fin.ml.optimization import KAdditivity

sparsity = KAdditivity(order=2)
```

`order=1` gives an additive baseline, while higher orders admit increasingly complex interactions.

### Fit an estimator

```python
from capacities_ml_fin.ml.models import ChoquetRegressor

model = ChoquetRegressor(sparsity=sparsity)
model.fit(X_train, y_train)
```

### Inspect the learned structure

```python
model.capacity_
model.problem_
model.result_
model.sparsity_
```

The capacity is the primary interpretable object. `problem_` and `result_` expose the optimization problem and backend-independent optimization result for diagnostics and research workflows.

## Scikit-learn composition

The estimators inherit from scikit-learn base classes and use the trailing-underscore convention for fitted attributes. They can be used inside `Pipeline`, `GridSearchCV`, scoring utilities, train/test splits and most standard meta-estimators compatible with their estimator tags.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", ChoquetRegressor(sparsity=KAdditivity(order=2))),
    ]
).set_output(transform="pandas")
```

For classification, note that `ChoquetClassifier` is intentionally a hard threshold model and does **not** expose `predict_proba()`. Use `ChoquisticRegression` when calibrated probability-like outputs are required by the model definition.

## Optimization layer

The estimators do not implement ad-hoc solver calls. They build a solver-independent `Problem` composed from:

- a capacity parameterization (`FullCapacity`, `KAdditivity`, `PairwiseInteractionSparsity`);
- an objective (`SquaredErrorObjective`, `LogisticNegativeLogLikelihood`, etc.);
- optional penalties;
- parameter blocks for capacity and non-capacity parameters;
- capacity normalization and monotonicity constraints;
- a selected backend (`SCIPY`, `PYMOO`, or `CVXPY` where supported).

See [Optimization](optimization.md) for details.

## Model selection

The model-selection helpers return candidate sparsity objects rather than manipulating estimator internals.

```python
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid

param_grid = capacity_parameter_grid(
    parameter_name="model__sparsity",
    orders=(1, 2, 3),
)
```

This can be passed directly to `GridSearchCV`.

## Time series is deliberately different

`ChoquetAutoRegressor` subclasses `sktime.forecasting.base.BaseForecaster`, not a scikit-learn regressor. It therefore follows sktime conventions for forecasting horizons, exogenous data, update operations and prediction intervals.

See [Time series](time_series.md).
