# Capacities

## Definition

For a finite universe $N$, a capacity is a normalized monotone set function

\[
\mu:2^N\to[0,1]
\]

satisfying

\[
\mu(\varnothing)=0,
\qquad
\mu(N)=1,
\]

and

\[
A\subseteq B\implies \mu(A)\le\mu(B).
\]

Unlike an additive probability measure, a capacity does **not** require

\[
\mu(A\cup B)=\mu(A)+\mu(B)
\]

for disjoint $A$ and $B$. This is what allows coalitions of variables to encode complementarity or redundancy.

## Variable universes

The package represents the finite universe through `VariableUniverse`.

```python
from capacities_ml_fin.base.capacities import VariableUniverse

universe = VariableUniverse(("profitability", "liquidity", "volatility"))
```

It stores:

- `var_names`: the ordered names;
- `n_elements`: the number of elements;
- `name_to_index`: a read-only name-to-index mapping.

A generated universe can also be created with:

```python
universe = VariableUniverse.from_size(3)
# names: x0, x1, x2
```

### Inferred universes

`ExplicitCapacity` and `MobiusCapacity` can infer the universe from coalition keys. If keys are named, names are retained in their first observed order. If keys are integer indices, the smallest universe containing the largest index is created.

```python
from capacities_ml_fin.base.capacities import ExplicitCapacity

capacity = ExplicitCapacity(
    {
        "quality": 0.4,
        "speed": 0.3,
        ("quality", "speed"): 1.0,
    }
)

capacity.var_names
```

For intentionally sparse inputs that do not mention every variable, pass `n_elements=`, `var_names=`, or a prebuilt `universe=` explicitly.

## `ExplicitCapacity`

`ExplicitCapacity` stores the table of capacity values directly.

For $n$ variables, a complete explicit capacity contains one value for each non-empty coalition. The empty coalition is not supplied: its value is fixed to zero.

```python
capacity = ExplicitCapacity(
    values={
        "a": 0.20,
        "b": 0.30,
        "c": 0.25,
        ("a", "b"): 0.60,
        ("a", "c"): 0.55,
        ("b", "c"): 0.65,
        ("a", "b", "c"): 1.00,
    }
)
```

The constructor calls `validate()`, so non-normalized or non-monotone tables are rejected at construction time.

### Access by coalition

```python
capacity.value(("a", "b"))
capacity.value({0, 1})
```

### Access by Boolean event

Generic algorithms should use an event mask:

```python
capacity.event_value([True, True, False])
```

The event above corresponds to the coalition $\{a,b\}$.

### Named table

```python
capacity.to_named_dict()
```

This is useful for display, serialization, and debugging.

## Common `BaseCapacity` interface

All capacity classes implement two key operations:

```python
capacity.event_value(event)
capacity.nested_event_values(order)
```

`event_value(event)` evaluates an arbitrary event. `nested_event_values(order)` evaluates the nested upper sets induced by an ordering of the variables. The latter operation is central to the ordered Choquet integral and lets sparse capacity implementations avoid materializing the complete $2^n$ table.

This interface is implemented not only by `ExplicitCapacity` and `MobiusCapacity`, but also by risk-specific classes such as `ProbabilityCapacity`, `DistortedCapacity`, `UpperEnvelopeCapacity`, and `LowerEnvelopeCapacity`.

## Validation

`validate_capacity` checks normalization and monotonicity by exhaustive enumeration for finite capacities:

```python
from capacities_ml_fin.base.capacities import validate_capacity

validate_capacity(capacity)
```

For large universes the number of events grows exponentially, so `max_elements` exists as an explicit safety limit.

The package also exposes:

```python
from capacities_ml_fin.base.capacities import (
    is_concave_capacity,
    is_convex_capacity,
)
```

A concave/submodular capacity satisfies

\[
\mu(A\cup B)+\mu(A\cap B)
\le
\mu(A)+\mu(B),
\]

while a convex/supermodular capacity satisfies the reverse inequality.

These checks also enumerate event pairs and therefore intentionally use a smaller default safety limit than basic monotonicity validation.

## Capacity values versus Möbius coefficients

This distinction is essential when using the API:

```python
explicit.value(("a", "b"))  # mu({a,b})
mobius.value(("a", "b"))    # m({a,b})
```

For representation-independent code:

```python
explicit.event_value([True, True, False])
mobius.event_value([True, True, False])
```

both return the corresponding capacity value $\mu(\{a,b\})$.

## Immutability

Capacity objects are frozen after successful validation. The low-level `CapacityMap` and `MobiusMap` are immutable as well. To change a capacity, construct a new object rather than mutating a coalition in place. This prevents a previously validated capacity from becoming mathematically invalid after construction.

## See also

- [Möbius representation](mobius.md)
- [$k$-additivity](k_additivity.md)
- [Choquet integral](choquet_integral.md)
- [Capacity API](../api/capacities.md)
