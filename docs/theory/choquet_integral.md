# Choquet integral

The discrete Choquet integral is the package's main non-additive aggregation operator. It combines ordered criterion values with capacity values of nested coalitions.

## Ordered form

Let

\[
x_{(1)}\le\cdots\le x_{(n)}
\]

be the sorted criterion values and define the nested upper sets

\[
A_{(i)}=\{(i),\ldots,(n)\}.
\]

The implementation computes

\[
C_\mu(x)
=
\sum_{i=1}^{n}
\bigl(x_{(i)}-x_{(i-1)}\bigr)\mu(A_{(i)}),
\qquad x_{(0)}=0.
\]

```python
import numpy as np
from capacities_ml_fin.base.integrals.choquet import ordered_choquet

x = np.array([0.70, 0.40, 0.80])
value = ordered_choquet(capacity, x)
```

The function validates that the number of input features equals `capacity.n_elements`.

## Why ordering matters

The weights applied to increments are not fixed feature coefficients. The relevant coalition depends on the ordering of the observation itself. This is the source of the Choquet integral's ability to model interactions while remaining monotone in its inputs when the capacity is monotone.

For normalized machine-learning criteria, the typical workflow is to transform all features to a common range before computing the integral. `CapacityNormalizer` is designed for this purpose.

## Möbius form

For a Möbius capacity, the package also uses

\[
C_\mu(x)
=
\sum_{\varnothing\ne A\subseteq N}
m(A)\min_{i\in A}x_i.
\]

```python
from capacities_ml_fin.base.integrals.choquet import mobius_choquet

value = mobius_choquet(mobius_capacity, x)
```

This form is particularly useful for sparse $k$-additive capacities because only stored non-zero/allowed coalitions need to be evaluated.

## Batch evaluation

For matrices with one observation per row:

```python
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral

scores = batch_choquet_integral(X, capacity)
```

The dispatcher uses specialized implementations:

- `ExplicitCapacity` → a direct capacity-value design matrix;
- `MobiusCapacity` → a Möbius minimum design matrix;
- any other `BaseCapacity` → row-by-row `ordered_choquet` evaluation.

This makes the same function suitable for manually constructed capacities, learned capacities, and risk-specific event capacities.

## Design matrices

Two lower-level functions are important for the learning layer.

### Capacity-value design

`capacity_design_matrix(X)` constructs a matrix with columns indexed by subset bitmasks. For each observation, the non-zero entries correspond to the nested coalitions induced by its feature ordering. A linear combination with the capacity-value vector reproduces the Choquet integral.

### Möbius design

`mobius_design_matrix(X, coalitions)` constructs

\[
Z_{t,A}=\min_{i\in A}X_{t,i}.
\]

Then

\[
C_\mu(X_t)=Z_t^\top m.
\]

This linearity in the capacity parameters is what makes least-squares and likelihood-based capacity learning possible under linear capacity constraints.

## Complete example

```python
import numpy as np

from capacities_ml_fin.base.capacities import ExplicitCapacity, mobius_transform
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.base.integrals.choquet import mobius_choquet, ordered_choquet

capacity = ExplicitCapacity(
    {
        "a": 0.2,
        "b": 0.3,
        ("a", "b"): 1.0,
    }
)
mobius = mobius_transform(capacity)

x = np.array([0.4, 0.8])
X = np.array([[0.4, 0.8], [0.9, 0.2]])

ordered = ordered_choquet(capacity, x)
from_mobius = mobius_choquet(mobius, x)
batch = batch_choquet_integral(X, capacity)
```

## See also

- [Capacities](capacities.md)
- [Möbius representation](mobius.md)
- [Regression](../ml/regression.md)
- [Integral API](../api/integrals.md)
