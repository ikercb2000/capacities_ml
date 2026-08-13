# imports
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# modules
from capacities_ml_fin.ml.optimization.optimizer import OptimizerBackend
from capacities_ml_fin.ml.optimization.problem import OptimizationProblem
from capacities_ml_fin.ml.optimization.result import OptimizationResult

# pymoo optimizer dataclass
@dataclass(slots=True)
class PymooGeneticOptimizer(OptimizerBackend[OptimizationProblem]):
    """Genetic optimizer for discontinuous 0-1 Choquet classifier losses."""

    population_size: int = 100
    n_generations: int = 300
    seed: int | None = None
    verbose: bool = False
    equality_tolerance: float = 1e-6

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        if problem.bounds is None:
            raise ValueError("A genetic search requires finite parameter bounds.")
        if not np.all(np.isfinite(problem.bounds.lower)) or not np.all(
            np.isfinite(problem.bounds.upper)
        ):
            raise ValueError("PymooGeneticOptimizer requires finite bounds for every parameter.")

        linear_ieq_count = 0
        linear_eq_count = 0
        for system in problem.linear_constraints:
            equality = np.isfinite(system.lower) & np.isfinite(system.upper) & np.isclose(
                system.lower, system.upper, atol=self.equality_tolerance
            )
            linear_eq_count += int(np.sum(equality))
            linear_ieq_count += int(np.sum(np.isfinite(system.upper) & ~equality))
            linear_ieq_count += int(np.sum(np.isfinite(system.lower) & ~equality))

        nonlinear_ieq_count = 0
        nonlinear_eq_count = 0
        for spec in problem.nonlinear_constraints:
            equality = np.isfinite(spec.lower) & np.isfinite(spec.upper) & np.isclose(
                spec.lower, spec.upper, atol=self.equality_tolerance
            )
            nonlinear_eq_count += int(np.sum(equality))
            nonlinear_ieq_count += int(np.sum(np.isfinite(spec.upper) & ~equality))
            nonlinear_ieq_count += int(np.sum(np.isfinite(spec.lower) & ~equality))

        n_ieq = linear_ieq_count + nonlinear_ieq_count
        n_eq = linear_eq_count + nonlinear_eq_count
        outer = self

        class WrappedProblem(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(
                    n_var=problem.n_parameters,
                    n_obj=1,
                    n_ieq_constr=n_ieq,
                    n_eq_constr=n_eq,
                    xl=problem.bounds.lower,
                    xu=problem.bounds.upper,
                )

            def _evaluate(self, x: np.ndarray, out: dict[str, object], *args: object, **kwargs: object) -> None:
                out["F"] = problem.objective(x)
                inequalities: list[float] = []
                equalities: list[float] = []

                def append_interval(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
                    equality = np.isfinite(lower) & np.isfinite(upper) & np.isclose(
                        lower, upper, atol=outer.equality_tolerance
                    )
                    equalities.extend((values[equality] - lower[equality]).tolist())
                    upper_rows = np.isfinite(upper) & ~equality
                    lower_rows = np.isfinite(lower) & ~equality
                    inequalities.extend((values[upper_rows] - upper[upper_rows]).tolist())
                    inequalities.extend((lower[lower_rows] - values[lower_rows]).tolist())

                for system in problem.linear_constraints:
                    append_interval(system.matrix @ x, system.lower, system.upper)
                for spec in problem.nonlinear_constraints:
                    append_interval(spec.values(x), spec.lower, spec.upper)

                if n_ieq:
                    out["G"] = np.asarray(inequalities, dtype=float)
                if n_eq:
                    out["H"] = np.asarray(equalities, dtype=float)

        algorithm = GA(pop_size=self.population_size, eliminate_duplicates=True)
        start = perf_counter()
        raw_result = minimize(
            WrappedProblem(),
            algorithm,
            termination=("n_gen", self.n_generations),
            seed=self.seed,
            verbose=self.verbose,
            save_history=False,
        )
        runtime = perf_counter() - start

        parameters = np.asarray(raw_result.X, dtype=float).reshape(-1)
        if raw_result.F is None:
            objective_value = float(problem.objective(parameters))
        else:
            objective_value = float(np.asarray(raw_result.F).reshape(-1)[0])
        violation = problem.maximum_constraint_violation(parameters)

        return OptimizationResult(
            parameters=parameters,
            objective_value=objective_value,
            success=bool(np.isfinite(objective_value) and violation <= max(self.equality_tolerance, 1e-6)),
            status="completed",
            message="pymoo genetic search completed.",
            n_iterations=self.n_generations,
            runtime_seconds=runtime,
            solver_name="pymoo:GA",
            diagnostics={
                "maximum_constraint_violation": violation,
                "raw_result": raw_result,
            },
        )
