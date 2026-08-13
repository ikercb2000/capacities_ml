# Optimization API

The optimization layer is intentionally public so research code can build capacity-learning problems without going through a predefined estimator.

## Enums

### `Solver`

::: capacities_ml_fin.ml.optimization.enums.Solver

### `CapacityShape`

::: capacities_ml_fin.ml.optimization.enums.CapacityShape

### `CapacityRepresentation`

::: capacities_ml_fin.ml.optimization.enums.CapacityRepresentation

### `OptimizationSense`

::: capacities_ml_fin.ml.optimization.enums.OptimizationSense

## Sparsity / capacity families

### `CapacitySparsity`

Abstract interface for compiling a capacity family into parameters and constraints.

::: capacities_ml_fin.ml.optimization.sparsity.sparsity.CapacitySparsity

### `FullCapacity`

::: capacities_ml_fin.ml.optimization.sparsity.sparsity.FullCapacity

### `KAdditivity`

::: capacities_ml_fin.ml.optimization.sparsity.sparsity.KAdditivity

### `PairwiseInteractionSparsity`

::: capacities_ml_fin.ml.optimization.sparsity.sparsity.PairwiseInteractionSparsity

### `SparsityCompilation`

::: capacities_ml_fin.ml.optimization.sparsity.sparsity.SparsityCompilation

## Parameter layout

### `ParameterBlock`

::: capacities_ml_fin.ml.optimization.parametrization.parametrization.ParameterBlock

### `ParameterLayout`

::: capacities_ml_fin.ml.optimization.parametrization.parametrization.ParameterLayout

## Constraint primitives

### `VariableBounds`

::: capacities_ml_fin.ml.optimization.constraints.constraints.VariableBounds

### `LinearConstraintSystem`

::: capacities_ml_fin.ml.optimization.constraints.constraints.LinearConstraintSystem

### `NonlinearConstraintSpec`

::: capacities_ml_fin.ml.optimization.constraints.constraints.NonlinearConstraintSpec

### `ConstraintBundle`

::: capacities_ml_fin.ml.optimization.constraints.constraints.ConstraintBundle

## Objectives

### `ObjectiveSpec`

::: capacities_ml_fin.ml.optimization.objectives.objectives.ObjectiveSpec

### `SquaredErrorObjective`

::: capacities_ml_fin.ml.optimization.objectives.objectives.SquaredErrorObjective

### `AbsoluteErrorObjective`

::: capacities_ml_fin.ml.optimization.objectives.objectives.AbsoluteErrorObjective

### `QuantileLossObjective`

::: capacities_ml_fin.ml.optimization.objectives.objectives.QuantileLossObjective

### `LogisticNegativeLogLikelihood`

::: capacities_ml_fin.ml.optimization.objectives.objectives.LogisticNegativeLogLikelihood

### `ZeroOneLossObjective`

::: capacities_ml_fin.ml.optimization.objectives.objectives.ZeroOneLossObjective

## Penalties

### `L1Penalty`

::: capacities_ml_fin.ml.optimization.penalties.penalties.L1Penalty

### `L2Penalty`

::: capacities_ml_fin.ml.optimization.penalties.penalties.L2Penalty

## Problem and solver facade

### `Problem`

Capacity-aware solver-independent problem. It compiles the capacity parameterization and can decode a result back into a concrete capacity.

::: capacities_ml_fin.ml.optimization.problem.Problem

### `Optimizer`

Facade selecting SciPy, PYMOO or CVXPY backends.

::: capacities_ml_fin.ml.optimization.optimizer.Optimizer

### `OptimizationResult`

Common result returned by all backends.

::: capacities_ml_fin.ml.optimization.result.OptimizationResult

## Capacity constraint utilities

These are lower-level building blocks useful when adding a new capacity parameterization.

### `capacity_value_constraints`

::: capacities_ml_fin.ml.optimization.capacity_constraints.utils.capacity_value_constraints

### `mobius_capacity_constraints`

::: capacities_ml_fin.ml.optimization.capacity_constraints.utils.mobius_capacity_constraints

### `pairwise_interaction_constraints`

::: capacities_ml_fin.ml.optimization.capacity_constraints.utils.pairwise_interaction_constraints
