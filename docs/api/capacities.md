# Capacities API

This page documents the concrete finite-capacity objects and validation utilities.

## `BaseCapacity`

`BaseCapacity` defines the representation-independent interface used by integrals, interpretation functions and the risk package. The key semantic method is `event_value(event)`.

::: capacities_ml_fin.base.capacities.base.BaseCapacity

## `VariableUniverse`

A `VariableUniverse` attaches stable names and indices to a finite capacity domain. Estimators create one automatically from DataFrame column names or from the fitted feature count.

::: capacities_ml_fin.base.capacities.capacities.VariableUniverse

## `ExplicitCapacity`

Stores a complete table of capacity values $\mu(A)$. Construction validates monotonicity and normalization. Use this class when the full table is known or when an explicit representation is desirable for inspection.

```python
capacity = ExplicitCapacity(
    {
        "a": 0.3,
        "b": 0.4,
        ("a", "b"): 1.0,
    }
)
```

::: capacities_ml_fin.base.capacities.capacities.ExplicitCapacity

## `KAdditiveCapacity`

A concrete explicit capacity constructed from capacity values supplied for all coalitions up to order `k`. Higher-order values are completed from the implied low-order Möbius representation, and exact $k$-additivity is validated.

This object is different from `ml.optimization.KAdditivity`, which is a learning parameterization rather than a stored capacity.

::: capacities_ml_fin.base.capacities.k_additive.KAdditiveCapacity

## `MobiusCapacity`

Stores a sparse map of Möbius coefficients. `value(A)` returns the coefficient $m(A)$, while `event_value(event)` returns the capacity $\mu(A)$ reconstructed from all stored subset coefficients.

::: capacities_ml_fin.base.capacities.mobius.MobiusCapacity

## Möbius transforms

### `mobius_transform`

Converts a complete `ExplicitCapacity` into an equivalent `MobiusCapacity`.

::: capacities_ml_fin.base.capacities.mobius.mobius_transform

### `inverse_mobius_transform`

Expands a `MobiusCapacity` into a complete `ExplicitCapacity`.

::: capacities_ml_fin.base.capacities.mobius.inverse_mobius_transform

## Validation

### `validate_capacity`

Exhaustively checks normalization and monotonicity on a finite event domain, subject to a configurable maximum number of elements.

::: capacities_ml_fin.base.capacities.validation.validate_capacity

### `is_concave_capacity`

Checks submodularity/concavity by exhaustive enumeration.

::: capacities_ml_fin.base.capacities.validation.is_concave_capacity

### `is_convex_capacity`

Checks supermodularity/convexity by exhaustive enumeration.

::: capacities_ml_fin.base.capacities.validation.is_convex_capacity

## Lower-level container types

The following immutable containers support capacity implementations. Most end users will not need to construct them directly, but they are useful when extending the package.

### `CoalitionValue`

::: capacities_ml_fin.base.capacities.types.CoalitionValue

### `CapacityMap`

::: capacities_ml_fin.base.capacities.types.CapacityMap

### `MobiusMap`

::: capacities_ml_fin.base.capacities.types.MobiusMap

## Utility functions

### `powerset`

::: capacities_ml_fin.base.capacities.utils.powerset

### `subset_encoding`

::: capacities_ml_fin.base.capacities.utils.subset_encoding

### `subset_decoding`

::: capacities_ml_fin.base.capacities.utils.subset_decoding
