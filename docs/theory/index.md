# Theory overview

The package uses capacities as the common mathematical object behind Choquet aggregation, feature interaction, and non-additive risk measurement.

This section is intentionally practical: it introduces the definitions needed to understand the code, and then points to the concrete classes and functions implementing each concept.

## Conceptual map

A finite variable universe is

\[
N = \{1,\ldots,n\}.
\]

A normalized capacity is a set function

\[
\mu:2^N\to[0,1]
\]

such that

\[
\mu(\varnothing)=0,\qquad \mu(N)=1,
\]

and

\[
A\subseteq B\implies \mu(A)\le \mu(B).
\]

The package then uses four closely related views:

1. **Capacity values** $\mu(A)$ — stored by `ExplicitCapacity`.
2. **Möbius coefficients** $m(A)$ — stored by `MobiusCapacity`.
3. **The Choquet integral** $C_\mu(x)$ — computed by `ordered_choquet` or `mobius_choquet`.
4. **Interpretation indices** — Shapley importance and pairwise Shapley interaction.

Machine-learning estimators optimize a capacity subject to its normalization and monotonicity constraints. The risk package reuses the same interface for capacities on finite loss scenarios.

## Representation is not semantics

An important distinction in the API is that the same mathematical capacity can have different numerical representations.

- `ExplicitCapacity.value(A)` returns the stored capacity value $\mu(A)$.
- `MobiusCapacity.value(A)` returns the Möbius coefficient $m(A)$.
- `event_value(event)` has the **same semantic meaning for every `BaseCapacity`**: it evaluates the capacity $\mu(A)$ of the event represented by the Boolean mask.

For generic code that should work with any capacity representation, prefer `event_value()` and `nested_event_values()`.

## Where to continue

- [Capacities](capacities.md) — finite universes, normalization, monotonicity, convexity/concavity and the object model.
- [Möbius representation](mobius.md) — inversion formulas, sparsity and direct evaluation.
- [$k$-additivity](k_additivity.md) — interaction-order restrictions and the distinction between a stored $k$-additive capacity and a learned $k$-additive parameterization.
- [Choquet integral](choquet_integral.md) — ordered and Möbius formulas, batch evaluation and design matrices.
- [Shapley & interactions](interpretation.md) — exact variable importance and pairwise interaction indices.
