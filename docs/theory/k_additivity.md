# $k$-additivity

$k$-additivity limits the highest order of non-zero Möbius coefficients. It is one of the main tools in the package for controlling the complexity of a learned capacity.

## Definition

A capacity is at most $k$-additive if

\[
m(A)=0
\qquad\text{for every }|A|>k.
\]

A capacity is exactly $k$-additive if it is at most $k$-additive and at least one coalition of cardinality $k$ has a non-zero Möbius coefficient.

The practical parameter count is

\[
\sum_{j=1}^{k}\binom{n}{j}.
\]

For common cases:

| Order | Included terms | Parameter count before constraints |
|---|---|---|
| $k=1$ | singletons | $n$ |
| $k=2$ | singletons + pairs | $n+\binom n2$ |
| $k=3$ | singletons + pairs + triples | $n+\binom n2+\binom n3$ |
| $k=n$ | all non-empty coalitions | $2^n-1$ |

Normalization and monotonicity reduce the effective feasible parameter space.

## Two different APIs use $k$-additivity

The package has two similarly named concepts with different purposes.

### `KAdditiveCapacity`

`KAdditiveCapacity` is a **concrete capacity object**. You supply capacity values $\mu(A)$ for every coalition of size at most `k`. The constructor reconstructs the corresponding low-order Möbius coefficients, completes higher-order capacity values implied by those coefficients, and validates exact $k$-additivity.

```python
from capacities_ml_fin.base.capacities import KAdditiveCapacity

capacity = KAdditiveCapacity(
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

For three variables and `k=2`, the grand-coalition value is completed automatically from the low-order Möbius representation and must satisfy normalization after completion.

### `KAdditivity`

`KAdditivity` is an **optimization sparsity specification** used by estimators:

```python
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.models import ChoquetRegressor

model = ChoquetRegressor(
    sparsity=KAdditivity(order=2),
)
```

Here the optimization variables are Möbius terms whose coalition order is at most `order`. The constraint compiler adds normalization and monotonicity restrictions.

The fitted estimator decodes the optimized parameters into a capacity object and exposes it as `capacity_`.

## Capacity shape

`KAdditivity` accepts a `CapacityShape`:

```python
from capacities_ml_fin.ml.optimization import CapacityShape, KAdditivity

sparsity = KAdditivity(
    order=2,
    shape=CapacityShape.CONCAVE,
)
```

The current optimization layer supports `GENERAL`, `CONVEX`, and `CONCAVE` shapes for order 2. Higher-order convex/concave shape constraints are deliberately rejected by the model-selection helpers because they are not currently implemented in the same simplified form.

## Selecting $k$ with cross-validation

```python
from sklearn.model_selection import GridSearchCV
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid

search = GridSearchCV(
    estimator,
    capacity_parameter_grid(
        parameter_name="model__sparsity",
        orders=(1, 2, 3),
    ),
    scoring="neg_mean_squared_error",
    cv=5,
)
```

This treats interaction order as a model-complexity hyperparameter rather than choosing it solely from in-sample fit.

## Pairwise interaction restrictions

`PairwiseInteractionSparsity` starts from a $k$-additive Möbius parameterization and adds equality constraints on selected Shapley pairwise interaction indices.

```python
from capacities_ml_fin.ml.optimization import PairwiseInteractionSparsity

sparsity = PairwiseInteractionSparsity(
    order=2,
    pairs=((0, 1),),
    target=0.0,
)
```

This is useful when particular interactions should be suppressed or fixed during learning.

## When to use which order

- Use `order=1` as an additive baseline.
- Use `order=2` when pairwise complementarity/redundancy is the primary interpretability target.
- Use `order=3` or higher only when sample size and domain knowledge justify the additional parameters.
- Use the full capacity only for small feature sets or when exponential growth is acceptable.

## See also

- [Möbius representation](mobius.md)
- [Machine-learning optimization](../ml/optimization.md)
- [Preprocessing & model selection](../ml/preprocessing_model_selection.md)
