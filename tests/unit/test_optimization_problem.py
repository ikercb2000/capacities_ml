import numpy as np
import pytest

from capacities_ml.capacities import VariableUniverse
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.optimization import (
    CapacityRepresentation,
    ConstraintBundle,
    KAdditivity,
    Optimizer,
    OptimizationSense,
    Problem,
    Solver,
)
from capacities_ml.optimization.constraints import VariableBounds
from capacities_ml.optimization.objectives import SquaredErrorObjective


def test_capacity_problem_decodes_optimized_values():
    problem = Problem.from_capacity(
        universe=VariableUniverse(("x0", "x1")),
        objective=lambda parameters: float(np.sum(parameters**2)),
        initial_parameters=np.array([0.0, 0.2, 0.4, 1.0]),
    )

    capacity = problem.decode(np.array([0.0, 0.2, 0.4, 1.0]))

    assert capacity.var_names == ("x0", "x1")
    assert capacity.value({0, 1}) == 1.0


def test_mobius_problem_decodes_sparse_coefficients():
    problem = Problem.from_capacity(
        universe=VariableUniverse(("x0", "x1")),
        objective=lambda parameters: float(np.sum(parameters**2)),
        sparsity=KAdditivity(order=1),
        initial_parameters=np.array([0.3, 0.7]),
    )

    mobius_representation = problem.decode(np.array([0.3, 0.7]))

    assert isinstance(mobius_representation, MobiusRepresentation)
    assert mobius_representation.value({0}) == 0.3
    assert mobius_representation.value({0, 1}) == 0.0


def test_capacity_sparsity_builds_k_additive_problem_automatically():
    universe = VariableUniverse(("x0", "x1", "x2"))
    problem = Problem.from_capacity(
        universe=universe,
        objective=lambda parameters: float(np.sum(parameters**2)),
        sparsity=KAdditivity(order=2),
    )

    assert problem.n_parameters == 6
    assert problem.initial_parameters.tolist() == [
        1 / 3,
        1 / 3,
        1 / 3,
        0.0,
        0.0,
        0.0,
    ]
    assert problem.max_order == 2


def test_optimizer_selects_requested_backend():
    scipy_optimizer = Optimizer(solver=Solver.SCIPY)
    pymoo_optimizer = Optimizer(solver=Solver.PYMOO)

    assert scipy_optimizer.solver is Solver.SCIPY
    assert scipy_optimizer._create_backend().__class__.__name__ == "ScipyOptimizer"
    assert pymoo_optimizer._create_backend().__class__.__name__ == "PymooGeneticOptimizer"

    with pytest.raises(TypeError, match="solver must be a Solver"):
        Optimizer(solver="unknown")._create_backend()


def test_problem_normalizes_representation_to_enum():
    problem = Problem.from_capacity(
        universe=VariableUniverse(("x0",)),
        objective=lambda parameters: float(np.sum(parameters**2)),
        initial_parameters=np.array([0.0, 1.0]),
    )

    assert problem.representation is CapacityRepresentation.VALUES


def test_public_options_require_enums():
    with pytest.raises(TypeError, match="shape must be a CapacityShape"):
        KAdditivity(order=2, shape="convex")


def test_generic_constraint_bundle_is_independent_of_capacity_models():
    bundle = ConstraintBundle(bounds=VariableBounds.box(2, 0.0, 1.0))

    assert bundle.n_parameters == 2
    assert bundle.linear_constraints == ()


def test_optimizer_translates_supported_objective_for_cvxpy():
    objective = SquaredErrorObjective(
        target=np.zeros(2),
        predictor=lambda parameters: parameters,
        symbolic_predictor=lambda variable: variable,
    )
    problem = Problem.from_capacity(
        universe=VariableUniverse(("x0",)),
        objective=objective,
    )

    compiled = Optimizer(solver=Solver.CVXPY)._compile_problem(problem)

    assert compiled.sense is OptimizationSense.MINIMIZE
    assert compiled.n_parameters == 2
