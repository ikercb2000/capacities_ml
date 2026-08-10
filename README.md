# capacities_ml

A Python package for learning capacities and capacity-based machine-learning models.

## Extended Choquet models

All supervised estimators expose scikit-learn's `fit`, `predict`, parameter and cloning
interfaces. Predictors should normally be scaled to `[0, 1]` before fitting.

```python
from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquisticRegression

model = ChoquisticRegression(
    universe=VariableUniverse(("market", "volatility", "liquidity")),
    class_weight="balanced",
).fit(X_train, y_train)

tail_probabilities = model.predict_proba(X_test)[:, 1]
capacity = model.capacity_
```

`ChoquisticRegression` implements the paper parametrization
`sigmoid(gamma * (C_mu(X) - beta))`. It learns a valid capacity, a positive
scale `gamma_` and a utility threshold `beta_` constrained to `[0, 1]` by
maximum likelihood. Its default capacity uses the full Möbius representation;
pass `KAdditivity(order=k)` and select `k` through cross-validation to reproduce
the paper's reduced models. Use `CapacityNormalizer`, including its
`cost_features` option, to normalize and orient criteria before fitting.

The autoregressive estimator follows the common pmdarima call pattern:

```python
from capacities_ml.models import ChoquetAutoRegressor

model = ChoquetAutoRegressor(lags=5).fit(y_train)
forecast, intervals = model.predict(10, return_conf_int=True)
model.update(y_new)
```

It supports aligned exogenous variables through `fit(y, X)` and future values through
`predict(n_periods, X)`, as well as `predict_in_sample`, `resid`, `aic` and `bic`.

## Model examples

The non-neural models have executable examples with several observations, fitted
parameters, predictions and diagnostics:

```powershell
poetry run python examples/choquet_regression_synthetic_n4.py
poetry run python examples/choquet_classifier_synthetic_n4.py
poetry run python examples/choquistic_regression_synthetic_n3.py
poetry run python examples/choquet_autoregression_synthetic.py
```

A trainable hidden layer of independently constrained Choquet neurons is available for
regression and binary classification:

```python
from capacities_ml.models import ChoquetNeuralClassifier

network = ChoquetNeuralClassifier(
    universe=VariableUniverse(("market", "volatility", "liquidity")),
    n_hidden=4,
    activation="tanh",
    random_state=7,
).fit(X_train, y_train)

learned_hidden_capacities = network.capacities_
```
