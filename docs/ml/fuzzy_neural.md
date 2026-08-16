# Fuzzy Choquet input layer

The paper-style estimators use a **staged** architecture:

\[
X \longrightarrow [0,1]\text{ normalization}
  \longrightarrow C_\mu(X)
  \longrightarrow \text{tanh hidden layer}
  \longrightarrow \text{output}.
\]

The normalized monotone capacity is estimated first and frozen. A conventional
scikit-learn neural network is then trained on the scalar Choquet input. The staged fit avoids optimizing several constrained capacities jointly inside
hidden neurons.

Unless `sparsity` is supplied, these tools fit a 2-additive capacity. For (p)
inputs this requires

\[
p + \binom{p}{2}
\]

Möbius terms rather than a complete exponential capacity.

## Reusable fuzzy input layer

`FuzzyChoquetInputLayer` is a supervised scikit-learn transformer. It can be
combined with a custom downstream estimator:

```python
from sklearn.neural_network import MLPRegressor
from capacities_ml_fin.ml.models import FuzzyChoquetInputLayer

layer = FuzzyChoquetInputLayer(task="regression")
Z_train = layer.fit_transform(X_train, y_train)
Z_test = layer.transform(X_test)

network = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="tanh",
    random_state=0,
).fit(Z_train, y_train)
```

Normalization is fitted only from `X_train`. Values outside the observed range
are clipped by default, preventing validation or future observations from
affecting the capacity scale.

## Ready-to-use regression and classification

```python
from capacities_ml_fin.ml.models import (
    FuzzyChoquetNeuralClassifier,
    FuzzyChoquetNeuralRegressor,
)

regressor = FuzzyChoquetNeuralRegressor(
    hidden_layer_sizes=(10,),
    random_state=0,
).fit(X_train, y_train)

classifier = FuzzyChoquetNeuralClassifier(
    hidden_layer_sizes=(10,),
    class_weight="balanced",
    random_state=0,
).fit(X_train, labels_train)
```

Both expose `capacity_`, `capacity_model_`, `normalizer_`, `network_`, and
`fuzzy_transform(X)`. The classification weights affect capacity estimation;
the downstream scikit-learn MLP is fitted on the resulting fuzzy input.

## Automatic selection from two to eight lags

`FuzzyChoquetNeuralAutoRegressor` accepts a univariate pandas Series. Its
default candidates are 2, 3, ..., 8 lags. The final part of the training series
is held out chronologically, every candidate capacity and network is fitted
only on the earlier part, and the lag order minimizing validation RMSE is
selected. The winning specification is then refitted using the complete
training series.

```python
from capacities_ml_fin.ml.models import FuzzyChoquetNeuralAutoRegressor

forecaster = FuzzyChoquetNeuralAutoRegressor(
    lag_candidates=tuple(range(2, 9)),
    validation_fraction=0.2,
    hidden_layer_sizes=(10,),
    random_state=0,
)
forecaster.fit(y_train)

print(forecaster.best_lag_)
print(forecaster.lag_scores_)
forecast = forecaster.predict(fh=[1, 2, 3])
```

Multi-step forecasts are recursive. Candidate validation is one-step and uses
only realized lag values available before each validation target.

## Interpretation

The selected capacity can be interpreted with the standard package tools:

```python
from capacities_ml_fin.base.interpretation import (
    interaction_indices,
    shapley_indices,
)

importance = shapley_indices(forecaster.capacity_)
interactions = interaction_indices(forecaster.capacity_)
```

For the autoregressor, the capacity variables are ordered from lag 1 through
`best_lag_`. Shapley values describe lag importance inside the fuzzy layer; they
are not a complete explanation of the downstream neural network.

## Model-selection warning

The trailing validation block used to select the lag order is part of model
development, not a final test set. Performance should still be reported on a
later untouched block or through an outer walk-forward evaluation.

