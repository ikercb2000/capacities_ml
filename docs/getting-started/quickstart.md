# Quickstart

This quickstart introduces the central workflow used throughout the package:

\[
\text{define or learn a capacity}
\longrightarrow
\text{evaluate a Choquet integral}
\longrightarrow
\text{interpret the capacity}.
\]

## 1. Define a complete capacity

For three variables, an explicit normalized capacity must provide the values of all non-empty coalitions.

```python
from capacities_ml_fin.base.capacities import ExplicitCapacity

capacity = ExplicitCapacity(
    values={
        "profitability": 0.25,
        "liquidity": 0.20,
        "solvency": 0.15,
        ("profitability", "liquidity"): 0.70,
        ("profitability", "solvency"): 0.60,
        ("liquidity", "solvency"): 0.30,
        ("profitability", "liquidity", "solvency"): 1.00,
    }
)
```

The constructor validates normalization and monotonicity. Coalition keys may use names or indices, but a single inferred universe cannot mix both styles.

## 2. Convert to the Möbius representation

```python
from capacities_ml_fin.base.capacities import (
    mobius_transform,
    inverse_mobius_transform,
)

mobius = mobius_transform(capacity)
recovered = inverse_mobius_transform(mobius)
```

The Möbius representation is useful because interaction order and sparsity are expressed directly in its coefficients.

## 3. Evaluate the Choquet integral

```python
import numpy as np

from capacities_ml_fin.base.integrals.choquet import (
    ordered_choquet,
    mobius_choquet,
)

x = np.array([0.70, 0.40, 0.80])

score_from_capacity = ordered_choquet(capacity, x)
score_from_mobius = mobius_choquet(mobius, x)
```

For a batch of observations:

```python
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral

X = np.array(
    [
        [0.70, 0.40, 0.80],
        [0.30, 0.90, 0.50],
        [0.80, 0.75, 0.60],
    ]
)

scores = batch_choquet_integral(X, capacity)
```

## 4. Interpret the capacity

```python
from capacities_ml_fin.base.interpretation import (
    shapley_indices,
    pairwise_interactions,
    interaction_signs,
)

importance = shapley_indices(capacity)
interactions = pairwise_interactions(capacity)
signs = interaction_signs(capacity)
```

Positive pairwise interaction is classified as complementary, negative interaction as redundant, and values numerically close to zero as neutral.

## 5. Learn a capacity with scikit-learn

Raw criteria often need a common scale and a common orientation. A cost criterion such as volatility can be reversed by `CapacityNormalizer`.

```python
from sklearn.pipeline import Pipeline

from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

model = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", ChoquetRegressor(sparsity=KAdditivity(order=2))),
    ]
).set_output(transform="pandas")

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Retrieve the fitted capacity:

```python
fitted = model.named_steps["model"]
fitted.capacity_
fitted.intercept_
```

## 6. Select interaction order

`capacity_parameter_grid` produces scikit-learn-ready candidates:

```python
from sklearn.model_selection import GridSearchCV
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid

search = GridSearchCV(
    model,
    capacity_parameter_grid(
        parameter_name="model__sparsity",
        orders=(1, 2, 3),
    ),
    scoring="neg_mean_squared_error",
    cv=5,
)
search.fit(X_train, y_train)
```

## 7. Move to the domain-specific guides

- [Regression](../ml/regression.md) for objectives, penalties and interpretation.
- [Classification](../ml/classification.md) for `ChoquetClassifier` and `ChoquisticRegression`.
- [Time series](../ml/time_series.md) for the native sktime forecaster.
- [Finance](../finance/index.md) for financial data construction.
- [Risk](../risk/index.md) for distorted and capacity-based risk measures.
