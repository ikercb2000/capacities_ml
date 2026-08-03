# imports
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

# modules
from capacities_ml.capacities.capacities import Capacity, VariableUniverse
from capacities_ml.capacities.utils import subset_decoding
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.optimization.constraints import (
    LinearConstraintSystem,
    NonlinearConstraintSpec,
    VariableBounds,
)
from capacities_ml.optimization.enums import CapacityRepresentation, OptimizationSense
from capacities_ml.optimization.objectives import ObjectiveSpec
from capacities_ml.optimization.result import OptimizationResult
from capacities_ml.optimization.sparsity import (
    CapacitySparsity,
    FullCapacity,
    SparsityCompilation,
)

# optimization aliases
FloatArray = NDArray[np.float64]
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], ArrayLike]
HessianFunction = Callable[[FloatArray], ArrayLike]
CvxpyExpressionBuilder = Callable[[Any], Any]
CvxpyConstraintBuilder = Callable[[Any], Sequence[Any]]


# numerical optimization problem
@dataclass(slots=True)
class OptimizationProblem:
    """Numerical optimization problem used by SciPy and Pymoo."""

    objective: ObjectiveFunction
    initial_parameters: FloatArray
    bounds: VariableBounds | None = None
    linear_constraints: tuple[LinearConstraintSystem, ...] = ()
    nonlinear_constraints: tuple[NonlinearConstraintSpec, ...] = ()
    gradient: GradientFunction | None = None
    hessian: HessianFunction | None = None
    name: str = "optimization_problem"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_parameters, dtype=float)
        if initial.ndim != 1:
            raise ValueError("initial_parameters must be one-dimensional.")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_parameters must be finite.")
        self.initial_parameters = initial.copy()
        n_parameters = initial.size
        if n_parameters < 1:
            raise ValueError("At least one parameter is required.")
        if self.bounds is not None and self.bounds.n_parameters != n_parameters:
            raise ValueError("bounds have an incompatible size.")
        for constraint in self.linear_constraints:
            if constraint.n_parameters != n_parameters:
                raise ValueError(
                    f"Linear constraint {constraint.name!r} has an incompatible size."
                )

    @property
    def n_parameters(self) -> int:
        return self.initial_parameters.size

    def evaluate(self, parameters: ArrayLike) -> float:
        vector = np.asarray(parameters, dtype=float)
        if vector.shape != (self.n_parameters,):
            raise ValueError("parameters have an incompatible shape.")
        value = float(self.objective(vector))
        if not np.isfinite(value):
            raise FloatingPointError("The objective returned a non-finite value.")
        return value

    def maximum_constraint_violation(self, parameters: ArrayLike) -> float:
        violations = [0.0]
        if self.bounds is not None:
            violations.append(self.bounds.maximum_violation(parameters))
        violations.extend(
            constraint.maximum_violation(parameters)
            for constraint in self.linear_constraints
        )
        violations.extend(
            constraint.maximum_violation(parameters)
            for constraint in self.nonlinear_constraints
        )
        return float(max(violations))


# public capacity optimization problem
@dataclass(slots=True)
class Problem:
    """Solver-independent capacity problem specification."""

    universe: VariableUniverse
    objective: ObjectiveFunction | ObjectiveSpec
    sparsity: CapacitySparsity | None = None
    initial_parameters: ArrayLike | None = None
    name: str = "capacity_problem"
    metadata: dict[str, Any] = field(default_factory=dict)
    _compilation: SparsityCompilation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")
        if self.universe.n_vars < 1:
            raise ValueError("The problem universe must contain variables.")
        if self.sparsity is not None and not isinstance(self.sparsity, CapacitySparsity):
            raise TypeError("sparsity must be a CapacitySparsity instance or None.")
        if self.sparsity is None:
            self.sparsity = FullCapacity()
        self._compilation = self.sparsity.compile(self.universe.n_vars)
        initial = (
            self._compilation.initial_parameters
            if self.initial_parameters is None
            else np.asarray(self.initial_parameters, dtype=float)
        )
        if initial.shape != (self._compilation.bundle.n_parameters,):
            raise ValueError("initial_parameters have an incompatible size.")
        self.initial_parameters = initial.copy()

    @classmethod
    def from_capacity(
        cls,
        *,
        universe: VariableUniverse,
        objective: ObjectiveFunction | ObjectiveSpec,
        sparsity: CapacitySparsity | None = None,
        initial_parameters: ArrayLike | None = None,
        name: str = "capacity_problem",
        metadata: dict[str, Any] | None = None,
    ) -> "Problem":
        """Create a capacity problem with optional parameter sparsity."""
        return cls(
            universe=universe,
            objective=objective,
            sparsity=sparsity,
            initial_parameters=initial_parameters,
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @property
    def n_parameters(self) -> int:
        return self._compilation.bundle.n_parameters

    @property
    def parameterization(self):
        return self._compilation.bundle

    @property
    def representation(self) -> CapacityRepresentation:
        return self.parameterization.representation

    @property
    def parameter_masks(self) -> tuple[int, ...]:
        return self.parameterization.parameter_masks

    @property
    def max_order(self) -> int:
        return self.parameterization.max_order

    def decode(self, parameters: ArrayLike) -> Capacity | MobiusRepresentation:
        """Convert optimized parameters into a public capacity representation."""
        vector = np.asarray(parameters, dtype=float)
        if vector.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected {self.n_parameters} parameters; got {vector.shape}."
            )

        if self.representation is CapacityRepresentation.VALUES:
            values = {
                subset_decoding(mask, self.universe.n_vars): value
                for mask, value in zip(self.parameter_masks, vector)
                if mask
            }
            return Capacity(universe=self.universe, values=values)

        coefficients = {
            subset_decoding(mask, self.universe.n_vars): value
            for mask, value in zip(self.parameter_masks, vector)
        }
        return MobiusRepresentation(
            universe=self.universe,
            coefficients=coefficients,
        )

    def decode_result(self, result: OptimizationResult) -> Capacity | MobiusRepresentation:
        """Convert a solver result into a public capacity representation."""
        return self.decode(result.parameters)


# cvxpy optimization problem
@dataclass(slots=True)
class CvxpyOptimizationProblem:
    """Internal convex problem built by the optimizer for CVXPY."""

    n_parameters: int
    objective_builder: CvxpyExpressionBuilder
    sense: OptimizationSense = OptimizationSense.MINIMIZE
    bounds: VariableBounds | None = None
    linear_constraints: tuple[LinearConstraintSystem, ...] = ()
    additional_constraints_builder: CvxpyConstraintBuilder | None = None
    initial_parameters: FloatArray | None = None
    name: str = "cvxpy_optimization_problem"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sense, OptimizationSense):
            raise TypeError("sense must be an OptimizationSense enum member.")
        if self.n_parameters < 1:
            raise ValueError("n_parameters must be positive.")
        if self.bounds is not None and self.bounds.n_parameters != self.n_parameters:
            raise ValueError("bounds have an incompatible size.")
        for constraint in self.linear_constraints:
            if constraint.n_parameters != self.n_parameters:
                raise ValueError(
                    f"Linear constraint {constraint.name!r} has an incompatible size."
                )
        if self.initial_parameters is not None:
            initial = np.asarray(self.initial_parameters, dtype=float)
            if initial.shape != (self.n_parameters,):
                raise ValueError("initial_parameters have an incompatible size.")
            self.initial_parameters = initial.copy()
