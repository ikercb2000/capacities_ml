# imports
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

# modules
from capacities_ml_fin.ml.optimization.optimizer import OptimizerBackend
from capacities_ml_fin.ml.optimization.problem import OptimizationProblem
from capacities_ml_fin.ml.optimization.result import OptimizationResult

# scipy optimizer dataclass
@dataclass(slots=True)
class ScipyOptimizer(OptimizerBackend[OptimizationProblem]):
    """Local constrained optimizer based on :func:`scipy.optimize.minimize`."""

    method: str = "SLSQP"
    tolerance: float | None = None
    options: dict[str, Any] = field(default_factory=lambda: {"maxiter": 2_000})

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        scipy_bounds = None
        if problem.bounds is not None:
            scipy_bounds = Bounds(problem.bounds.lower, problem.bounds.upper)

        constraints: list[Any] = []
        constraints.extend(
            LinearConstraint(system.matrix, system.lower, system.upper)
            for system in problem.linear_constraints
        )
        constraints.extend(
            NonlinearConstraint(
                spec.function,
                spec.lower,
                spec.upper,
                jac=spec.jacobian if spec.jacobian is not None else "2-point",
            )
            for spec in problem.nonlinear_constraints
        )

        start = perf_counter()
        raw_result = minimize(
            fun=problem.objective,
            x0=problem.initial_parameters,
            method=self.method,
            jac=problem.gradient,
            hess=problem.hessian,
            bounds=scipy_bounds,
            constraints=constraints,
            tol=self.tolerance,
            options=dict(self.options),
        )
        runtime = perf_counter() - start
        parameters = np.asarray(raw_result.x, dtype=float)
        violation = problem.maximum_constraint_violation(parameters)

        return OptimizationResult(
            parameters=parameters,
            objective_value=float(raw_result.fun),
            success=bool(getattr(raw_result, "success", False)),
            status=str(getattr(raw_result, "status", "unknown")),
            message=str(getattr(raw_result, "message", "")),
            n_iterations=getattr(raw_result, "nit", None),
            n_function_evaluations=getattr(raw_result, "nfev", None),
            runtime_seconds=runtime,
            solver_name=f"scipy:{self.method}",
            diagnostics={
                "maximum_constraint_violation": violation,
                "raw_result": raw_result,
            },
        )
