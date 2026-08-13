# Möbius representation

The Möbius transform provides an alternative parameterization of a finite capacity. It is especially useful for sparse capacities and for controlling interaction order.

## Transform

For a capacity $\mu$, the Möbius coefficient of coalition $A$ is

\[
m(A)
=
\sum_{B\subseteq A}
(-1)^{|A|-|B|}\mu(B).
\]

The inverse relation is

\[
\mu(A)
=
\sum_{B\subseteq A}m(B).
\]

The package implements both directions:

```python
from capacities_ml_fin.base.capacities import (
    mobius_transform,
    inverse_mobius_transform,
)

mobius = mobius_transform(explicit_capacity)
recovered = inverse_mobius_transform(mobius)
```

`mobius_transform` requires an `ExplicitCapacity`. The inverse produces a complete `ExplicitCapacity` by summing all Möbius coefficients over subsets of each coalition.

## `MobiusCapacity`

A Möbius capacity stores only the supplied coefficients. Unspecified coefficients are interpreted as zero.

```python
from capacities_ml_fin.base.capacities import MobiusCapacity

mobius = MobiusCapacity(
    coefficients={
        ("quality",): 0.35,
        ("cost",): 0.25,
        ("speed",): 0.20,
        ("quality", "speed"): 0.20,
    }
)
```

The normalization condition is encoded by

\[
\sum_{A\subseteq N}m(A)=1.
\]

Monotonicity is checked directly from the Möbius coefficients during construction.

## Coefficient access and event evaluation

For a `MobiusCapacity`, `value(A)` means **Möbius coefficient**:

```python
mobius.value(("quality", "speed"))
```

If that coefficient was not supplied, `value()` returns `0.0`.

To evaluate the capacity itself, use `event_value()`:

```python
mobius.event_value([True, False, True])
```

Internally this computes

\[
\mu(A)=\sum_{B\subseteq A}m(B).
\]

This distinction is why generic code in the package relies on the `BaseCapacity` event interface rather than calling `.value()`.

## Why sparse Möbius coefficients matter

A complete capacity on $n$ variables has up to

\[
2^n-2
\]

free nontrivial capacity values. If higher-order Möbius coefficients are zero, the representation can be dramatically smaller.

For example, a 2-additive capacity uses only singleton and pair terms:

\[
n+\binom{n}{2}
\]

Möbius coefficients instead of the full exponential table.

This is the representation used by the `KAdditivity` learning specification.

## Choquet integral in Möbius form

For nonnegative criteria, the discrete Choquet integral has the Möbius form

\[
C_\mu(x)
=
\sum_{\varnothing\ne A\subseteq N}
m(A)\min_{i\in A}x_i.
\]

The package implements it directly:

```python
from capacities_ml_fin.base.integrals.choquet import mobius_choquet

score = mobius_choquet(mobius, x)
```

For batches, `batch_choquet_integral_mobius` creates a matrix whose column for coalition $A$ is

\[
Z_{t,A}=\min_{i\in A}X_{t,i}
\]

and multiplies it by the aligned Möbius coefficient vector.

## Interpretation of low-order terms

For a 2-additive representation:

- singleton coefficients describe individual contributions;
- pair coefficients encode direct pair interactions in the Möbius basis;
- the exact Shapley importance of a feature redistributes interaction mass across participating variables;
- the exact Shapley pairwise interaction index is computed by the interpretation module from event values.

Do not equate a raw Möbius coefficient with a Shapley index unless the relevant special-case formula justifies it.

## Named export

```python
mobius.to_named_dict()
```

returns the stored coefficients keyed by variable names. This is useful when a learned `KAdditivity` model is decoded as a `MobiusCapacity`.

## See also

- [$k$-additivity](k_additivity.md)
- [Choquet integral](choquet_integral.md)
- [Interpretation](interpretation.md)
- [`MobiusCapacity` API](../api/capacities.md#mobiuscapacity)
