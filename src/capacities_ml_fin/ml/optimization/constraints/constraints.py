# imports
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

# modules
from capacities_ml_fin.ml.optimization.constraints.utils import (
    ConstraintFunction,
    FloatArray,
    JacobianFunction,
    _as_vector,
)

# variable bounds
@dataclass(frozen=True, slots=True)
class VariableBounds:
    """Componentwise parameter bounds ``lower <= x <= upper``."""

    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        lower = _as_vector(self.lower, name="lower")
        upper = _as_vector(self.upper, name="upper")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if np.any(lower > upper):
            raise ValueError("Every lower bound must be <= its upper bound.")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @classmethod
    def unbounded(cls, n_parameters: int) -> "VariableBounds":
        if n_parameters < 1:
            raise ValueError("n_parameters must be positive.")
        return cls(
            lower=np.full(n_parameters, -np.inf),
            upper=np.full(n_parameters, np.inf),
        )

    @classmethod
    def box(
        cls,
        n_parameters: int,
        lower: float,
        upper: float,
    ) -> "VariableBounds":
        return cls(
            lower=np.full(n_parameters, lower, dtype=float),
            upper=np.full(n_parameters, upper, dtype=float),
        )

    @property
    def n_parameters(self) -> int:
        return self.lower.size

    def maximum_violation(self, x: ArrayLike) -> float:
        vector = _as_vector(x, name="x")
        if vector.size != self.n_parameters:
            raise ValueError("x has an incompatible size.")
        lower_violation = np.max(np.maximum(self.lower - vector, 0.0))
        upper_violation = np.max(np.maximum(vector - self.upper, 0.0))
        return float(max(lower_violation, upper_violation))

    def embed(self, *, start: int, total_parameters: int) -> "VariableBounds":
        """Embed these bounds into an otherwise unbounded global vector."""
        if start < 0:
            raise ValueError("start cannot be negative.")
        stop = start + self.n_parameters
        if stop > total_parameters:
            raise ValueError("The embedded block exceeds total_parameters.")
        lower = np.full(total_parameters, -np.inf, dtype=float)
        upper = np.full(total_parameters, np.inf, dtype=float)
        lower[start:stop] = self.lower
        upper[start:stop] = self.upper
        return VariableBounds(lower, upper)

    def intersect(self, other: "VariableBounds") -> "VariableBounds":
        """Return the intersection of two bound systems."""
        if self.n_parameters != other.n_parameters:
            raise ValueError("Both bound systems must have the same size.")
        return VariableBounds(
            lower=np.maximum(self.lower, other.lower),
            upper=np.minimum(self.upper, other.upper),
        )


# linear constraint system
@dataclass(frozen=True, slots=True)
class LinearConstraintSystem:
    """Interval-form linear constraints ``lower <= matrix @ x <= upper``."""

    matrix: FloatArray
    lower: FloatArray
    upper: FloatArray
    name: str = "linear_constraints"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("matrix must contain only finite values.")
        lower = _as_vector(self.lower, name="lower")
        upper = _as_vector(self.upper, name="upper")
        if lower.size != matrix.shape[0] or upper.size != matrix.shape[0]:
            raise ValueError(
                "lower and upper must contain one value per constraint row."
            )
        if np.any(lower > upper):
            raise ValueError("Every lower constraint bound must be <= upper.")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def n_constraints(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_parameters(self) -> int:
        return self.matrix.shape[1]

    @classmethod
    def equality(
        cls,
        matrix: ArrayLike,
        rhs: ArrayLike,
        *,
        name: str = "equality_constraints",
    ) -> "LinearConstraintSystem":
        rhs_array = _as_vector(rhs, name="rhs")
        return cls(matrix=np.asarray(matrix, dtype=float), lower=rhs_array, upper=rhs_array, name=name)

    @classmethod
    def upper_bounded(
        cls,
        matrix: ArrayLike,
        upper: ArrayLike,
        *,
        name: str = "upper_bounded_constraints",
    ) -> "LinearConstraintSystem":
        upper_array = _as_vector(upper, name="upper")
        return cls(
            matrix=np.asarray(matrix, dtype=float),
            lower=np.full(upper_array.size, -np.inf),
            upper=upper_array,
            name=name,
        )

    @classmethod
    def lower_bounded(
        cls,
        matrix: ArrayLike,
        lower: ArrayLike,
        *,
        name: str = "lower_bounded_constraints",
    ) -> "LinearConstraintSystem":
        lower_array = _as_vector(lower, name="lower")
        return cls(
            matrix=np.asarray(matrix, dtype=float),
            lower=lower_array,
            upper=np.full(lower_array.size, np.inf),
            name=name,
        )

    def values(self, x: ArrayLike) -> FloatArray:
        vector = _as_vector(x, name="x")
        if vector.size != self.n_parameters:
            raise ValueError("x has an incompatible size.")
        return self.matrix @ vector

    def maximum_violation(self, x: ArrayLike) -> float:
        values = self.values(x)
        lower_violation = np.max(np.maximum(self.lower - values, 0.0), initial=0.0)
        upper_violation = np.max(np.maximum(values - self.upper, 0.0), initial=0.0)
        return float(max(lower_violation, upper_violation))

    def embed(self, *, start: int, total_parameters: int) -> "LinearConstraintSystem":
        """Embed this system into a larger global parameter vector."""
        if start < 0:
            raise ValueError("start cannot be negative.")
        stop = start + self.n_parameters
        if stop > total_parameters:
            raise ValueError("The embedded block exceeds total_parameters.")
        matrix = np.zeros((self.n_constraints, total_parameters), dtype=float)
        matrix[:, start:stop] = self.matrix
        return LinearConstraintSystem(matrix, self.lower, self.upper, name=self.name)

    @staticmethod
    def stack(
        systems: Iterable["LinearConstraintSystem"],
        *,
        name: str = "stacked_constraints",
    ) -> "LinearConstraintSystem":
        systems_tuple = tuple(systems)
        if not systems_tuple:
            raise ValueError("At least one constraint system is required.")
        n_parameters = systems_tuple[0].n_parameters
        if any(system.n_parameters != n_parameters for system in systems_tuple):
            raise ValueError("All systems must have the same number of parameters.")
        return LinearConstraintSystem(
            matrix=np.vstack([system.matrix for system in systems_tuple]),
            lower=np.concatenate([system.lower for system in systems_tuple]),
            upper=np.concatenate([system.upper for system in systems_tuple]),
            name=name,
        )


# nonlinear constraint specification
@dataclass(frozen=True, slots=True)
class NonlinearConstraintSpec:
    """Solver-independent nonlinear constraint specification."""

    function: ConstraintFunction
    lower: FloatArray
    upper: FloatArray
    jacobian: JacobianFunction | None = None
    name: str = "nonlinear_constraint"

    def __post_init__(self) -> None:
        lower = _as_vector(self.lower, name="lower")
        upper = _as_vector(self.upper, name="upper")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape.")
        if np.any(lower > upper):
            raise ValueError("Every lower constraint bound must be <= upper.")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def n_constraints(self) -> int:
        return self.lower.size

    def values(self, x: ArrayLike) -> FloatArray:
        values = _as_vector(self.function(np.asarray(x, dtype=float)), name="constraint values")
        if values.size != self.n_constraints:
            raise ValueError(
                f"Constraint {self.name!r} returned {values.size} values; "
                f"expected {self.n_constraints}."
            )
        return values

    def maximum_violation(self, x: ArrayLike) -> float:
        values = self.values(x)
        lower_violation = np.max(np.maximum(self.lower - values, 0.0), initial=0.0)
        upper_violation = np.max(np.maximum(values - self.upper, 0.0), initial=0.0)
        return float(max(lower_violation, upper_violation))


# generic constraint bundle
@dataclass(frozen=True, slots=True)
class ConstraintBundle:
    """Common container for bounds and solver-independent constraints."""

    bounds: VariableBounds
    linear_constraints: tuple[LinearConstraintSystem, ...] = ()
    nonlinear_constraints: tuple[NonlinearConstraintSpec, ...] = ()

    def __post_init__(self) -> None:
        n_parameters = self.bounds.n_parameters
        if any(
            constraint.n_parameters != n_parameters
            for constraint in self.linear_constraints
        ):
            raise ValueError("Linear constraints have an incompatible size.")

    @property
    def n_parameters(self) -> int:
        return self.bounds.n_parameters

    def embedded(
        self,
        *,
        start: int,
        total_parameters: int,
    ) -> tuple[LinearConstraintSystem, ...]:
        """Embed linear constraints into a larger global parameter vector."""
        return tuple(
            constraint.embed(start=start, total_parameters=total_parameters)
            for constraint in self.linear_constraints
        )

    def embedded_bounds(
        self,
        *,
        start: int,
        total_parameters: int,
    ) -> VariableBounds:
        """Embed bounds into a larger global parameter vector."""
        return self.bounds.embed(start=start, total_parameters=total_parameters)
