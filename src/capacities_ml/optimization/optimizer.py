# imports
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# modules
from capacities_ml.optimization.problem import Problem, CvxpyOptimizationProblem, OptimizationProblem
from capacities_ml.optimization.result import OptimizationResult
from capacities_ml.optimization.enums import Solver
from capacities_ml.optimization.objectives import to_cvxpy

# optimization problem variable
ProblemT = TypeVar("ProblemT") # for any kind of ooptimisation problem in different packages

# optimizer backend base class
class OptimizerBackend(ABC, Generic[ProblemT]):
    """Abstract base class implemented by all optimization backends."""

    @abstractmethod
    def solve(self, problem: ProblemT) -> OptimizationResult:
        """Solve a problem and return a backend-independent result."""


# public optimizer facade
class Optimizer:
    """Select and run an optimization backend for a public ``Problem``."""
    
    def __init__(self, solver: Solver = Solver.SCIPY, **solver_options: Any) -> None:
        if not isinstance(solver, Solver):
            raise TypeError("solver must be a Solver enum member.")
        self.solver = solver
        self.solver_options = solver_options

    def solve(self, problem: Any) -> OptimizationResult:
        if not isinstance(problem, Problem):
            raise TypeError("problem must be a Problem instance.")
        compiled_problem = self._compile_problem(problem)
        backend = self._create_backend()
        return backend.solve(compiled_problem)

    def _compile_problem(self, problem: Any) -> Any:
        if self.solver is Solver.CVXPY:
            return self._compile_cvxpy(problem)
        return self._compile_numerical(problem)

    def _compile_numerical(self, problem: Any) -> Any:
        constraints = problem.parameterization.constraints
        return OptimizationProblem(
            objective=problem.objective,
            initial_parameters=problem.initial_parameters,
            bounds=constraints.bounds,
            linear_constraints=constraints.linear_constraints,
            nonlinear_constraints=constraints.nonlinear_constraints,
            name=problem.name,
            metadata=problem.metadata,
        )

    def _compile_cvxpy(self, problem: Any) -> Any:

        constraints = problem.parameterization.constraints
        if constraints.nonlinear_constraints:
            raise NotImplementedError(
                "Nonlinear constraints do not have a generic CVXPY translation yet."
            )
        return CvxpyOptimizationProblem(
            n_parameters=problem.n_parameters,
            objective_builder=to_cvxpy(problem.objective),
            bounds=constraints.bounds,
            linear_constraints=constraints.linear_constraints,
            initial_parameters=problem.initial_parameters,
            name=problem.name,
            metadata=problem.metadata,
        )

    def _create_backend(self) -> OptimizerBackend[Any]:
        if self.solver is Solver.SCIPY:
            from capacities_ml.optimization.backends.to_scipy import ScipyOptimizer

            return ScipyOptimizer(**self.solver_options)
        if self.solver is Solver.PYMOO:
            from capacities_ml.optimization.backends.to_pymoo import PymooGeneticOptimizer

            return PymooGeneticOptimizer(**self.solver_options)
        if self.solver is Solver.CVXPY:
            from capacities_ml.optimization.backends.to_cvxpy import CvxpyOptimizer

            return CvxpyOptimizer(**self.solver_options)
        raise ValueError(
            f"Unknown solver {self.solver!r}. Expected 'scipy', 'pymoo' or 'cvxpy'."
        )
