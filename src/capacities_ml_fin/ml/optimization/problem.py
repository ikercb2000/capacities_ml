# imports
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray

# modules
from capacities_ml_fin.base.capacities import (
    ExplicitCapacity,
    MobiusCapacity,
    VariableUniverse,
)
from capacities_ml_fin.base.capacities.utils import subset_decoding
from capacities_ml_fin.ml.optimization.constraints import (
    ConstraintBundle,
    LinearConstraintSystem,
    NonlinearConstraintSpec,
    VariableBounds,
)
from capacities_ml_fin.ml.optimization.enums import CapacityRepresentation, OptimizationSense
from capacities_ml_fin.ml.optimization.objectives import ObjectiveSpec
from capacities_ml_fin.ml.optimization.parametrization import ParameterLayout, ParameterBlock
from capacities_ml_fin.ml.optimization.result import OptimizationResult
from capacities_ml_fin.ml.optimization.sparsity import (
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
    layout: ParameterLayout | None = None
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
        if self.layout is not None and self.layout.n_parameters != n_parameters:
            raise ValueError("layout has an incompatible size.")

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
    parameter_layout: ParameterLayout | None = None
    initial_parameters: ArrayLike | None = None
    name: str = "capacity_problem"
    metadata: dict[str, Any] = field(default_factory=dict)
    _compilation: SparsityCompilation = field(init=False, repr=False)
    _constraints: ConstraintBundle = field(init=False, repr=False)
    _capacity_slice: slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.universe, VariableUniverse):
            raise TypeError("universe must be a VariableUniverse instance.")
        if self.universe.n_elements < 1:
            raise ValueError("The problem universe must contain variables.")
        if self.sparsity is not None and not isinstance(self.sparsity, CapacitySparsity):
            raise TypeError("sparsity must be a CapacitySparsity instance or None.")
        if self.sparsity is None:
            self.sparsity = FullCapacity()
        self._compilation = self.sparsity.compile(self.universe.n_elements)
        capacity_parameterization = self._compilation.bundle
        n_capacity_parameters = capacity_parameterization.n_parameters

        if self.parameter_layout is None:
            self.parameter_layout = ParameterLayout(
                ParameterBlock("capacity", n_capacity_parameters)
            )
        else:
            try:
                capacity_slice = self.parameter_layout.slice("capacity")
            except KeyError as error:
                raise ValueError(
                    "parameter_layout must contain a 'capacity' block."
                ) from error
            if capacity_slice.stop - capacity_slice.start != n_capacity_parameters:
                raise ValueError(
                    "The 'capacity' block has an incompatible size."
                )

        self._capacity_slice = self.parameter_layout.slice("capacity")
        self._constraints = self._embed_capacity_constraints(
            capacity_parameterization.constraints
        )

        if self.initial_parameters is None:
            if self.parameter_layout.n_parameters != n_capacity_parameters:
                raise ValueError(
                    "initial_parameters are required when parameter_layout "
                    "contains additional blocks."
                )
            initial = self._compilation.initial_parameters
        else:
            initial = np.asarray(self.initial_parameters, dtype=float)
        if initial.shape != (self.parameter_layout.n_parameters,):
            raise ValueError("initial_parameters have an incompatible size.")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_parameters must be finite.")
        self.initial_parameters = initial.copy()

    def _embed_capacity_constraints(
        self,
        capacity_constraints: ConstraintBundle,
    ) -> ConstraintBundle:
        """Embed capacity constraints into the complete parameter vector."""
        total_parameters = self.parameter_layout.n_parameters
        start = self._capacity_slice.start
        bounds = self.parameter_layout.bounds().intersect(
            capacity_constraints.bounds.embed(
                start=start,
                total_parameters=total_parameters,
            )
        )
        linear_constraints = capacity_constraints.embedded(
            start=start,
            total_parameters=total_parameters,
        )
        nonlinear_constraints: list[NonlinearConstraintSpec] = []
        for constraint in capacity_constraints.nonlinear_constraints:
            nonlinear_constraints.append(
                NonlinearConstraintSpec(
                    function=lambda parameters, constraint=constraint: constraint.values(
                        np.asarray(parameters)[self._capacity_slice]
                    ),
                    lower=constraint.lower,
                    upper=constraint.upper,
                    jacobian=(
                        None
                        if constraint.jacobian is None
                        else lambda parameters, constraint=constraint: np.asarray(
                            constraint.jacobian(
                                np.asarray(parameters)[self._capacity_slice]
                            )
                        )
                    ),
                    name=constraint.name,
                )
            )
        return ConstraintBundle(
            bounds=bounds,
            linear_constraints=linear_constraints,
            nonlinear_constraints=tuple(nonlinear_constraints),
        )

    @classmethod
    def from_capacity(
        cls,
        *,
        universe: VariableUniverse,
        objective: ObjectiveFunction | ObjectiveSpec,
        sparsity: CapacitySparsity | None = None,
        parameter_layout: ParameterLayout | None = None,
        initial_parameters: ArrayLike | None = None,
        name: str = "capacity_problem",
        metadata: dict[str, Any] | None = None,
    ) -> "Problem":
        """Create a capacity problem with optional parameter sparsity."""
        return cls(
            universe=universe,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=parameter_layout,
            initial_parameters=initial_parameters,
            name=name,
            metadata={} if metadata is None else metadata,
        )

    @property
    def n_parameters(self) -> int:
        return self.parameter_layout.n_parameters

    @property
    def parameterization(self):
        return self._compilation.bundle

    @property
    def constraints(self) -> ConstraintBundle:
        """Return constraints embedded in the complete parameter vector."""
        return self._constraints

    @property
    def representation(self) -> CapacityRepresentation:
        return self.parameterization.representation

    @property
    def parameter_masks(self) -> tuple[int, ...]:
        return self.parameterization.parameter_masks

    @property
    def max_order(self) -> int:
        return self.parameterization.max_order

    def decode(self, parameters: ArrayLike) -> ExplicitCapacity | MobiusCapacity:
        """Convert optimized parameters into a public capacity representation."""
        vector = np.asarray(parameters, dtype=float)
        if vector.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected {self.n_parameters} parameters; got {vector.shape}."
            )
        capacity_vector = vector[self._capacity_slice]

        if self.representation is CapacityRepresentation.VALUES:
            values = {
                subset_decoding(mask, self.universe.n_elements): value
                for mask, value in zip(self.parameter_masks, capacity_vector)
                if mask
            }
            return ExplicitCapacity(universe=self.universe, values=values)

        coefficients = {
            subset_decoding(mask, self.universe.n_elements): value
            for mask, value in zip(self.parameter_masks, capacity_vector)
        }
        return MobiusCapacity(
            universe=self.universe,
            coefficients=coefficients,
        )

    def decode_result(self, result: OptimizationResult) -> ExplicitCapacity | MobiusCapacity:
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
    layout: ParameterLayout | None = None
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
        if self.layout is not None and self.layout.n_parameters != self.n_parameters:
            raise ValueError("layout has an incompatible size.")
