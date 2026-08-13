# Example: capacities, Choquet integral and interpretation

This example shows the complete mathematical workflow without fitting a machine-learning estimator.

## 1. Define an explicit capacity

```python
import numpy as np
import pandas as pd

from capacities_ml_fin.base.capacities import (
    ExplicitCapacity,
    KAdditiveCapacity,
    inverse_mobius_transform,
    mobius_transform,
)
from capacities_ml_fin.base.integrals.batch_integrals import (
    batch_choquet_integral,
    batch_choquet_integral_mobius,
)
from capacities_ml_fin.base.integrals.choquet import (
    mobius_choquet,
    ordered_choquet,
)
from capacities_ml_fin.base.interpretation import (
    interaction_signs,
    pairwise_interaction_matrix,
    pairwise_interactions,
    shapley_indices,
)

capacity = ExplicitCapacity(
    values={
        "profitability": 0.25,
        "liquidity": 0.20,
        "solvency": 0.15,
        ("profitability", "liquidity"): 0.70,
        ("profitability", "solvency"): 0.60,
        ("liquidity", "solvency"): 0.30,
        ("profitability", "liquidity", "solvency"): 1.00,
    },
)
```

The constructor checks the capacity table immediately. Because the coalition keys are named, the universe retains those names.

```python
pd.Series(capacity.to_named_dict(), name="capacity value")
```

## 2. Construct the equivalent 2-additive capacity from low-order values

```python
capacity_2add = KAdditiveCapacity(
    values={
        "profitability": 0.25,
        "liquidity": 0.20,
        "solvency": 0.15,
        ("profitability", "liquidity"): 0.70,
        ("profitability", "solvency"): 0.60,
        ("liquidity", "solvency"): 0.30,
    },
    k=2,
)
```

`KAdditiveCapacity` completes higher-order capacity values from the low-order Möbius structure instead of asking the caller to provide the full table.

## 3. Move to Möbius coefficients

```python
mobius = mobius_transform(capacity)
recovered = inverse_mobius_transform(mobius)

print(pd.Series(mobius.to_named_dict(), name="m(A)"))
```

Check that the explicit representation is recovered:

```python
original = capacity.to_named_dict()
round_trip = recovered.to_named_dict()

assert np.allclose(
    list(original.values()),
    [round_trip[key] for key in original],
)
```

## 4. Evaluate one observation

```python
x = np.array([0.70, 0.40, 0.80])

ordered_value = ordered_choquet(capacity, x)
mobius_value = mobius_choquet(mobius, x)

assert np.isclose(ordered_value, mobius_value)
```

The two formulas are different parameterizations of the same capacity, so they should agree numerically.

## 5. Evaluate a batch

```python
X = np.array(
    [
        [0.70, 0.40, 0.80],
        [0.30, 0.90, 0.50],
        [0.80, 0.75, 0.60],
        [0.25, 0.35, 0.20],
    ]
)

scores_explicit = batch_choquet_integral(X, capacity)
scores_mobius = batch_choquet_integral_mobius(X, mobius)

assert np.allclose(scores_explicit, scores_mobius)
```

## 6. Interpret importance and interaction

```python
importance = shapley_indices(capacity)
interactions = pairwise_interactions(capacity)
signs = interaction_signs(capacity)
matrix = pairwise_interaction_matrix(capacity)

print(pd.Series(importance, name="Shapley importance"))
print(pd.DataFrame({"interaction": interactions, "sign": signs}))
print(
    pd.DataFrame(
        matrix,
        index=capacity.var_names,
        columns=capacity.var_names,
    )
)
```

The exact interpretation functions operate through `event_value()`, so the same code can be run on the explicit, Möbius or learned capacity representation.

## What this example establishes

- Capacity values and Möbius coefficients are alternative representations.
- The ordered and Möbius Choquet formulas agree for equivalent capacities.
- Shapley importance is a property of the capacity, not of the storage representation.
- Pairwise interaction can be reported as values, a matrix, or simple sign classes.
