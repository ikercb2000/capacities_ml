# Example: Choquet regression pipeline

This example follows the repository regression notebook: generate criteria with a known pairwise-interaction structure, normalize their direction, select capacity order by cross-validation, optionally regularize interactions, and interpret the fitted capacity.

## 1. Synthetic data with interactions

```python
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from capacities_ml_fin.base.interpretation import (
    pairwise_interaction_matrix,
    pairwise_interactions,
    shapley_indices,
)
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid
from capacities_ml_fin.ml.models import ChoquetRegressor
from capacities_ml_fin.ml.optimization import L1Penalty, Solver
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

rng = np.random.default_rng(11)
n = 90

X = pd.DataFrame(
    {
        "profitability": rng.uniform(5.0, 25.0, n),
        "liquidity": rng.uniform(0.8, 2.5, n),
        "volatility": rng.uniform(0.10, 0.60, n),
    }
)

oriented = np.column_stack(
    (
        (X["profitability"] - 5.0) / 20.0,
        (X["liquidity"] - 0.8) / 1.7,
        (0.60 - X["volatility"]) / 0.50,
    )
)

y = (
    0.20
    + 0.25 * oriented[:, 0]
    + 0.20 * oriented[:, 1]
    + 0.15 * oriented[:, 2]
    + 0.25 * np.minimum(oriented[:, 0], oriented[:, 1])
    + 0.15 * np.minimum(oriented[:, 1], oriented[:, 2])
    + rng.normal(0.0, 0.025, n)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=7,
)
```

The response contains two minimum terms, which are exactly the kind of pairwise structure represented naturally by a 2-additive Möbius Choquet model.

## 2. Put normalization inside the pipeline

```python
pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquetRegressor(
                solver=Solver.SCIPY,
                solver_options={
                    "options": {"maxiter": 1500, "ftol": 1e-10}
                },
            ),
        ),
    ]
).set_output(transform="pandas")
```

The cost orientation of volatility is learned from the training range inside each CV fold.

## 3. Select capacity order

```python
search = GridSearchCV(
    pipeline,
    capacity_parameter_grid(
        parameter_name="model__sparsity",
        orders=(1, 2, 3),
    ),
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=4),
    n_jobs=1,
)
search.fit(X_train, y_train)

best_sparsity = search.best_params_["model__sparsity"]
```

The comparison asks whether additive, pairwise or third-order capacity structure improves held-out squared error.

## 4. Penalize only interaction parameters

```python
compilation = best_sparsity.compile(X.shape[1])
interaction_positions = np.array(
    [
        position
        for position, mask in enumerate(compilation.bundle.parameter_masks)
        if mask.bit_count() >= 2
    ],
    dtype=int,
)

penalty = L1Penalty(
    weight=5e-4,
    selection=interaction_positions,
)
```

Refit the selected structure:

```python
final_pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquetRegressor(
                sparsity=best_sparsity,
                penalty=penalty,
                solver=Solver.SCIPY,
                solver_options={
                    "options": {"maxiter": 2000, "ftol": 1e-12}
                },
            ),
        ),
    ]
).set_output(transform="pandas")

final_pipeline.fit(X_train, y_train)
```

## 5. Compare with a classical baseline

```python
linear_pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", LinearRegression()),
    ]
).set_output(transform="pandas").fit(X_train, y_train)

choquet_pred = final_pipeline.predict(X_test)
linear_pred = linear_pipeline.predict(X_test)

print("Choquet RMSE", mean_squared_error(y_test, choquet_pred) ** 0.5)
print("Linear RMSE", mean_squared_error(y_test, linear_pred) ** 0.5)
print("Choquet MAE", mean_absolute_error(y_test, choquet_pred))
print("Choquet R2", r2_score(y_test, choquet_pred))
```

## 6. Interpret the learned capacity

```python
fitted = final_pipeline.named_steps["model"]

print(pd.Series(shapley_indices(fitted.capacity_)))
print(pd.Series(pairwise_interactions(fitted.capacity_)))
print(
    pd.DataFrame(
        pairwise_interaction_matrix(fitted.capacity_),
        index=X.columns,
        columns=X.columns,
    )
)
```

The predictive evaluation and structural interpretation are separate outputs of the same fitted estimator.
