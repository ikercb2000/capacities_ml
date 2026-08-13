# imports
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import numpy as np
import cvxpy as cp

# modules
from capacities_ml_fin.ml.optimization.optimizer import OptimizerBackend
from capacities_ml_fin.ml.optimization.enums import OptimizationSense
from capacities_ml_fin.ml.optimization.problem import CvxpyOptimizationProblem
from capacities_ml_fin.ml.optimization.result import OptimizationResult

# cvxpy optimizer dataclass

@dataclass(slots=True)
class CvxpyOptimizer(OptimizerBackend[CvxpyOptimizationProblem]):
    """Convex backend for regression and Choquistic likelihood problems."""

    solver: str | None = None
    warm_start: bool = True
    verbose: bool = False
    solver_options: dict[str, Any] = field(default_factory=dict)

    def solve(self, problem: CvxpyOptimizationProblem) -> OptimizationResult:
        variable = cp.Variable(problem.n_parameters, name="parameters")
        if problem.initial_parameters is not None:
            variable.value = problem.initial_parameters

        expression = problem.objective_builder(variable)
        objective = (
            cp.Minimize(expression)
            if problem.sense is OptimizationSense.MINIMIZE
            else cp.Maximize(expression)
        )
        constraints: list[Any] = []

        if problem.bounds is not None:
            finite_lower = np.isfinite(problem.bounds.lower)
            finite_upper = np.isfinite(problem.bounds.upper)
            if np.any(finite_lower):
                constraints.append(variable[finite_lower] >= problem.bounds.lower[finite_lower])
            if np.any(finite_upper):
                constraints.append(variable[finite_upper] <= problem.bounds.upper[finite_upper])

        for system in problem.linear_constraints:
            values = system.matrix @ variable
            finite_lower = np.isfinite(system.lower)
            finite_upper = np.isfinite(system.upper)
            equality = finite_lower & finite_upper & np.isclose(system.lower, system.upper)
            lower_only = finite_lower & ~equality
            upper_only = finite_upper & ~equality
            if np.any(equality):
                constraints.append(values[equality] == system.lower[equality])
            if np.any(lower_only):
                constraints.append(values[lower_only] >= system.lower[lower_only])
            if np.any(upper_only):
                constraints.append(values[upper_only] <= system.upper[upper_only])

        if problem.additional_constraints_builder is not None:
            constraints.extend(problem.additional_constraints_builder(variable))

        cvx_problem = cp.Problem(objective, constraints)
        start = perf_counter()
        value = cvx_problem.solve(
            solver=self.solver,
            warm_start=self.warm_start,
            verbose=self.verbose,
            **self.solver_options,
        )
        runtime = perf_counter() - start

        parameters = (
            np.asarray(variable.value, dtype=float).reshape(-1)
            if variable.value is not None
            else np.full(problem.n_parameters, np.nan)
        )
        success_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        stats = cvx_problem.solver_stats

        return OptimizationResult(
            parameters=parameters,
            objective_value=float(value) if value is not None else np.inf,
            success=cvx_problem.status in success_statuses,
            status=str(cvx_problem.status),
            message=f"CVXPY status: {cvx_problem.status}",
            n_iterations=getattr(stats, "num_iters", None),
            runtime_seconds=runtime,
            solver_name=getattr(stats, "solver_name", self.solver),
            diagnostics={
                "solve_time": getattr(stats, "solve_time", None),
                "setup_time": getattr(stats, "setup_time", None),
                "extra_stats": getattr(stats, "extra_stats", None),
                "problem": cvx_problem,
            },
        )
