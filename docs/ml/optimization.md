# Optimization architecture

Capacity learning is implemented through a solver-independent optimization layer rather than embedding solver-specific code in every estimator.

## Overview

A fitted estimator conceptually builds:

```text
CapacitySparsity
      ↓ compile
capacity parameterization + constraints
      ↓
ParameterLayout + objective + initial values
      ↓
Problem
      ↓
Optimizer(solver=...)
      ↓
SciPy / PYMOO / CVXPY backend
      ↓
OptimizationResult
      ↓
decode learned capacity
```

This architecture is useful for research because capacity constraints, objectives and solver backends can be varied independently.

## Capacity sparsity specifications

### `FullCapacity`

Optimizes direct capacity values for all non-empty coalitions.

```python
from capacities_ml_fin.ml.optimization import FullCapacity

sparsity = FullCapacity()
```

The compiler builds direct normalization and monotonicity constraints.

### `KAdditivity`

Optimizes Möbius coefficients only up to the requested order.

```python
from capacities_ml_fin.ml.optimization import KAdditivity

sparsity = KAdditivity(order=2)
```

An equal-singleton normalized capacity is used as the feasible initial point.

### `PairwiseInteractionSparsity`

Adds equality constraints fixing selected pairwise Shapley interaction indices:

```python
from capacities_ml_fin.ml.optimization import PairwiseInteractionSparsity

sparsity = PairwiseInteractionSparsity(
    order=2,
    pairs=((0, 1),),
    target=0.0,
)
```

If `pairs=None`, the underlying interaction-constraint utility determines the complete selected set supported by the specification.

## Shape constraints

`CapacityShape` defines:

```python
CapacityShape.GENERAL
CapacityShape.CONVEX
CapacityShape.CONCAVE
```

The current concise Möbius shape restrictions are available for order 2. Model-selection helpers reject unsupported higher-order shape combinations early.

## Parameter layout

A single optimization vector may contain capacity and non-capacity parameters. `ParameterBlock` and `ParameterLayout` assign stable semantic slices.

For example, the regressor uses blocks equivalent to:

```text
capacity | intercept
```

Choquistic regression uses:

```text
capacity | gamma | beta
```

The autoregressor uses:

```text
capacity | phi | intercept? | exogenous?
```

This lets objectives and decoders refer to semantic blocks without hard-coding absolute indices.

## Objectives

The public objective classes include:

- `SquaredErrorObjective`
- `AbsoluteErrorObjective`
- `QuantileLossObjective`
- `LogisticNegativeLogLikelihood`
- `ZeroOneLossObjective`

Functional constructors with corresponding names ending in `_objective` are also exported.

An `ObjectiveSpec` can optionally provide a symbolic translation for CVXPY in addition to numerical evaluation.

## Penalties

`L1Penalty` and `L2Penalty` target a selected subset of the global parameter vector.

```python
from capacities_ml_fin.ml.optimization import L1Penalty

penalty = L1Penalty(
    weight=1e-3,
    selection=[3, 4, 5],
)
```

This is intentionally more flexible than a single global regularization coefficient: interaction terms, scale parameters, or any other block positions can be regularized selectively.

`L2Penalty` additionally exposes an analytical gradient.

## Constraints

The solver-independent primitives are:

- `VariableBounds`: componentwise lower/upper bounds;
- `LinearConstraintSystem`: interval constraints $l\le Ax\le u$;
- `NonlinearConstraintSpec`: callable nonlinear constraints;
- `ConstraintBundle`: common container.

Capacity-specific compilers build on these primitives to enforce normalization, monotonicity and selected interaction restrictions.

## `Problem`

`Problem` is the public capacity-aware optimization specification. It contains:

- `universe`;
- objective;
- sparsity specification;
- optional parameter layout;
- optional initial parameters;
- metadata and name.

The object compiles the requested capacity family and knows how to decode optimized capacity parameters back into a concrete capacity object.

This is what lets estimators expose both:

```python
model.problem_
model.result_
model.capacity_
```

without being coupled to one solver.

## Solvers

The public enum is:

```python
from capacities_ml_fin.ml.optimization import Solver

Solver.SCIPY
Solver.PYMOO
Solver.CVXPY
```

### SciPy

Useful for smooth or constrained numerical optimization and the default for several estimators. Choquistic regression specifically requires SciPy because the implemented paper formulation is fitted with sequential quadratic programming.

### PYMOO

Provides a genetic/evolutionary backend. `ChoquetClassifier` defaults to PYMOO because it directly minimizes a discontinuous 0–1 loss. Set a seed through solver options for reproducible evolutionary runs.

### CVXPY

Used when the objective has a symbolic CVXPY translation and constraints are supported. Generic nonlinear constraints currently have no automatic CVXPY translation. `ChoquetAutoRegressor` rejects CVXPY because the product of $\phi$ and capacity parameters makes that model non-convex.

## Backend-independent result

Every backend returns `OptimizationResult`, allowing estimators and tests to inspect success, parameters, objective value, iterations, diagnostics and messages in a common format.

## Extending the optimizer

A new capacity family should generally be implemented as a new `CapacitySparsity` whose `compile()` method returns a `SparsityCompilation`. A new solver backend should implement `OptimizerBackend.solve()` and return `OptimizationResult`.

This is preferable to adding special cases inside individual models.

## API

See the complete [Optimization API](../api/optimization.md).
