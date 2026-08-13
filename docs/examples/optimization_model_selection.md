# Example: capacity structure, constraints and model selection

This example shows how the optimization abstractions can be used without modifying the estimator implementation.

## 1. Compare interaction orders

```python
from capacities_ml_fin.ml.model_selection import capacity_sparsity_grid

candidates = capacity_sparsity_grid(
    orders=(1, 2, 3),
)
```

Each candidate is a `KAdditivity` object. When assigned to an estimator, it compiles its own parameter masks, bounds and monotonicity constraints.

## 2. Compare capacity shapes

```python
from capacities_ml_fin.ml.model_selection import capacity_shape_grid

shape_candidates = capacity_shape_grid(order=2)
```

The default list compares general, convex and concave 2-additive capacity specifications.

## 3. Fix a selected interaction to zero

```python
from capacities_ml_fin.ml.optimization import PairwiseInteractionSparsity

restricted = PairwiseInteractionSparsity(
    order=2,
    pairs=((0, 1),),
    target=0.0,
)
```

The compiler starts from the normal $k$-additive constraints and appends a linear equality constraint on the selected exact pairwise interaction index.

## 4. Use in a scikit-learn search

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    estimator,
    {
        "sparsity": [
            *candidates,
            restricted,
        ]
    },
    scoring="neg_mean_squared_error",
    cv=5,
)
search.fit(X, y)
```

For a pipeline, use a nested parameter name such as `model__sparsity`.

## 5. Add parameter-selective regularization

```python
import numpy as np
from capacities_ml_fin.ml.optimization import L1Penalty, KAdditivity

sparsity = KAdditivity(order=2)
compiled = sparsity.compile(X.shape[1])

interaction_positions = np.array(
    [
        i
        for i, mask in enumerate(compiled.bundle.parameter_masks)
        if mask.bit_count() == 2
    ]
)

penalty = L1Penalty(
    weight=1e-3,
    selection=interaction_positions,
)
```

Pass `penalty` to the estimator. Singleton capacity terms remain unpenalized.

## 6. Choose a solver deliberately

```python
from capacities_ml_fin.ml.optimization import Solver

Solver.SCIPY
Solver.PYMOO
Solver.CVXPY
```

The best backend depends on the objective:

- squared-error regression is naturally handled numerically with SciPy and may admit convex formulations;
- direct 0–1 classification is discontinuous, motivating the PYMOO default in `ChoquetClassifier`;
- Choquistic regression is explicitly restricted to SciPy/SQP by its implementation;
- Choquet autoregression is non-convex because $\phi$ multiplies the learned Choquet aggregate, so CVXPY is rejected.

The key architectural point is that capacity structure and solver choice are independent estimator parameters rather than hard-coded model variants.
