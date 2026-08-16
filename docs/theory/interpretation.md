# Shapley values and interaction indices

The interpretation module turns a learned or manually specified capacity into exact variable-importance and pairwise-interaction summaries.

## Shapley importance

For feature $i$, the Shapley importance index is the weighted average marginal contribution of $i$ over all coalitions not containing it:

\[
\phi_i(\mu)
=
\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(n-|S|-1)!}{n!}
\left[\mu(S\cup\{i\})-\mu(S)\right].
\]

The package computes this exactly through `event_value()` calls, so the same implementation works for explicit capacities, Möbius capacities, and any other `BaseCapacity`.

```python
from capacities_ml_fin.base.interpretation import shapley_indices

importance = shapley_indices(capacity)
```

If the capacity has named variables, the returned dictionary uses those names. Otherwise it uses integer indices.

For a single feature:

```python
from capacities_ml_fin.base.interpretation import shapley_index

value = shapley_index(capacity, "profitability")
```

## Pairwise Shapley interaction

For distinct $i$ and $j$, the package computes the exact second-order interaction

\[
I_{ij}(\mu)
=
\sum_{S\subseteq N\setminus\{i,j\}}
\frac{|S|!(n-|S|-2)!}{(n-1)!}
\Delta_{ij}\mu(S),
\]

where

\[
\Delta_{ij}\mu(S)
=
\mu(S\cup\{i,j\})
-\mu(S\cup\{i\})
-\mu(S\cup\{j\})
+\mu(S).
\]

```python
from capacities_ml_fin.base.interpretation import pairwise_interactions

interactions = pairwise_interactions(capacity)
```

Interpretation of the sign:

- $I_{ij}>0$: **complementarity** — the pair contributes more together than through their separate marginal effects.
- $I_{ij}<0$: **redundancy/substitutability**.
- $I_{ij}\approx0$: no detected pairwise interaction under the exact index.

`interaction_signs()` applies this classification numerically with a small zero tolerance.

## Interaction matrix

```python
from capacities_ml_fin.base.interpretation import pairwise_interaction_matrix

matrix = pairwise_interaction_matrix(capacity)
```

The result is symmetric with a zero diagonal. It can be wrapped in a DataFrame using `capacity.var_names`:

```python
import pandas as pd

frame = pd.DataFrame(
    matrix,
    index=capacity.var_names,
    columns=capacity.var_names,
)
```

## Interpreting a fitted estimator

Every main estimator exposing `capacity_` can use the same interpretation functions:

```python
model.fit(X, y)

importance = shapley_indices(model.capacity_)
interactions = pairwise_interactions(model.capacity_)
```

For `ChoquetAutoRegressor`, the feature names are automatically created as `lag_1`, `lag_2`, ..., making the Shapley indices interpretable as lag importance.


## Complexity

The current exact implementation enumerates subsets. This is desirable for correctness and representation independence but scales exponentially with the number of variables. For high-dimensional full capacities, interpretation can therefore become more expensive than prediction.

The implementation shares an event-value cache within multi-index calculations, reducing repeated capacity evaluations, but it remains an exact combinatorial method.

## Capacity interaction versus model effect

A capacity interaction index describes the **aggregation structure of the learned capacity**. It is not the same object as a local SHAP explanation of an arbitrary predictive model. In a Choquet model, this distinction is valuable: the capacity is itself a structured, interpretable parameter of the predictor.

For the regressor

\[
\widehat y = \beta_0 + C_\mu(x),
\]

`shapley_indices(model.capacity_)` explains the learned non-additive aggregation $C_\mu$, while the intercept is a separate additive term.

## See also

- [Regression](../ml/regression.md)
- [Classification](../ml/classification.md)
- [Time series](../ml/time_series.md)
- [Interpretation API](../api/interpretation.md)
