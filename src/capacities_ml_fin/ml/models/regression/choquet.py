# imports
from __future__ import annotations
from functools import partial
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

# modules
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.ml.models.regression.utils import (
    regression_predictor,
    scaled_regression_predictor,
)
from capacities_ml_fin.ml.models.utils import (
    capacity_design,
    fitted_universe,
    resolve_solver,
)
from capacities_ml_fin.ml.optimization import (
    FullCapacity,
    Optimizer,
    ParameterBlock,
    ParameterLayout,
    Problem,
    Solver,
)
from capacities_ml_fin.ml.optimization.objectives import SquaredErrorObjective
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# choquet regressor
class ChoquetRegressor(RegressorMixin, BaseEstimator):
    """Least-squares regression with a learned Choquet integral.

    The estimator learns a normalized monotone capacity together with an
    unconstrained intercept. It follows the scikit-learn estimator contract and
    can be used in pipelines, cross-validation, grid search, pickle, and joblib.

    Parameters
    ----------
    sparsity : CapacitySparsity, optional
        Capacity parameterization. ``None`` fits a full explicit capacity.
    solver : {"scipy", "cvxpy", "pymoo"} or Solver, default="scipy"
        Optimization backend.
    solver_options : dict, optional
        Keyword arguments forwarded to the selected backend.
    penalty : callable or penalty object, optional
        Regularization term added to the squared-error objective.

    Attributes
    ----------
    capacity_ : BaseCapacity
        Fitted immutable capacity.
    intercept_ : float
        Fitted additive intercept.
    universe_ : VariableUniverse
        Feature universe inferred from ``X``.
    problem_ : Problem
        Solver-independent optimization problem used during fitting.
    result_ : OptimizationResult
        Numerical optimizer result and diagnostics.
    n_features_in_ : int
        Number of input features seen during :meth:`fit`.
    feature_names_in_ : ndarray of str
        Input feature names when ``X`` provides string column names.

    Notes
    -----
    Inputs should be made commensurable before fitting, typically with
    :class:`~capacities_ml_fin.ml.preprocessing.CapacityNormalizer`.
    """

    def __init__(
        self,
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        penalty: Any = None,
    ) -> None:
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.penalty = penalty

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ChoquetRegressor":
        """Fit the capacity and the regression intercept."""
        matrix, target = validate_data(
            self,
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        self.universe_ = fitted_universe(self)
        target = np.asarray(target, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values.")
        self.solver_ = resolve_solver(self.solver)

        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe_.n_elements)
        design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        intercept_initial = float(np.mean(target - design @ capacity_initial))
        layout = ParameterLayout(
            ParameterBlock("capacity", compilation.bundle.n_parameters),
            ParameterBlock("intercept", 1),
        )
        capacity_slice = layout.slice("capacity")
        intercept_slice = layout.slice("intercept")

        predictor = partial(
            regression_predictor,
            design=design,
            capacity_slice=capacity_slice,
            intercept_slice=intercept_slice,
        )

        objective = SquaredErrorObjective(
            target=target,
            predictor=predictor,
            penalty=self.penalty,
            symbolic_predictor=predictor,
        )
        problem = Problem.from_capacity(
            universe=self.universe_,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(
                [capacity_initial, np.array([intercept_initial])]
            ),
            name="choquet_regression",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver_, **options).solve(problem)
        if not result.success:
            raise RuntimeError(
                f"Choquet regression optimization failed: {result.message}"
            )

        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.intercept_ = float(result.parameters[intercept_slice][0])
        self.sparsity_ = sparsity
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous responses."""
        check_is_fitted(self, ["capacity_", "intercept_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        return batch_choquet_integral(matrix, self.capacity_) + self.intercept_

    def fit_predict(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Fit the model and return predictions for ``X``."""
        return self.fit(X, y).predict(X)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return np.asarray(self.universe_.var_names, dtype=object)


# scaled Choquet regressor
class ScaledChoquetRegressor(RegressorMixin, BaseEstimator):
    """Least-squares regression of the form ``c + q * C_mu(X)``.

    This separate estimator implements the scaled Choquet submodel discussed by
    Grabisch after Wang et al. The capacity remains normalized and monotone;
    ``q`` controls the response scale without changing that normalization.

    Parameters
    ----------
    sparsity : CapacitySparsity, optional
        Capacity parameterization. ``None`` fits a full explicit capacity.
    solver : {"scipy", "pymoo"} or Solver, default="scipy"
        Numerical backend. CVXPY is unavailable because ``q * C_mu`` is
        bilinear in the jointly learned parameters.
    solver_options : dict, optional
        Keyword arguments forwarded to the selected backend.
    penalty : callable or penalty object, optional
        Regularization added to the squared-error objective.
    q_bounds : pair of float, default=(0.0, inf)
        Bounds for ``q``. The nonnegative default preserves monotonicity of the
        complete regression function. Negative values can be enabled explicitly.

    Attributes
    ----------
    capacity_ : BaseCapacity
        Fitted normalized monotone capacity.
    q_ : float
        Fitted scale multiplying the Choquet integral.
    intercept_ : float
        Fitted error-center/intercept ``c``.
    problem_ : Problem
        Joint constrained optimization problem.
    result_ : OptimizationResult
        Numerical optimizer result and diagnostics.

    Notes
    -----
    The initial ``q`` and ``c`` are obtained by ordinary least squares for the
    initial feasible capacity, as in the paper's proposed submodels. Joint
    numerical minimization then retains all capacity and ``q`` constraints.
    Inputs must be commensurable before aggregation.
    """

    def __init__(
        self,
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        penalty: Any = None,
        q_bounds: tuple[float, float] = (0.0, np.inf),
    ) -> None:
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.penalty = penalty
        self.q_bounds = q_bounds

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ScaledChoquetRegressor":
        """Fit the capacity, scale ``q``, and intercept ``c``."""
        matrix, target = validate_data(
            self, X, y, dtype=float, ensure_2d=True, ensure_min_samples=1
        )
        self.universe_ = fitted_universe(self)
        target = np.asarray(target, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values.")
        self.solver_ = resolve_solver(self.solver)
        if self.solver_ is Solver.CVXPY:
            raise ValueError(
                "The q-times-capacity model is non-convex; use SCIPY or PYMOO."
            )
        q_lower, q_upper = self._validated_q_bounds()

        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe_.n_elements)
        design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters

        # For a fixed feasible capacity, q and c are linear-regression terms.
        aggregate = design @ capacity_initial
        linear_initial, *_ = np.linalg.lstsq(
            np.column_stack([aggregate, np.ones(target.size)]), target, rcond=None
        )
        q_initial = float(np.clip(linear_initial[0], q_lower, q_upper))
        intercept_initial = float(np.mean(target - q_initial * aggregate))

        layout = ParameterLayout(
            ParameterBlock("capacity", compilation.bundle.n_parameters),
            ParameterBlock("q", 1, lower=q_lower, upper=q_upper),
            ParameterBlock("intercept", 1),
        )
        predictor = partial(
            scaled_regression_predictor,
            design=design,
            capacity_slice=layout.slice("capacity"),
            q_slice=layout.slice("q"),
            intercept_slice=layout.slice("intercept"),
        )
        objective = SquaredErrorObjective(
            target=target, predictor=predictor, penalty=self.penalty
        )
        problem = Problem.from_capacity(
            universe=self.universe_,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(
                [capacity_initial, np.array([q_initial, intercept_initial])]
            ),
            name="scaled_choquet_regression",
            metadata={"formulation": "c + q * C_mu(X)"},
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver_, **options).solve(problem)
        if not result.success:
            raise RuntimeError(
                f"Scaled Choquet regression optimization failed: {result.message}"
            )

        blocks = layout.unpack(result.parameters)
        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.q_ = float(blocks["q"][0])
        self.intercept_ = float(blocks["intercept"][0])
        self.sparsity_ = sparsity
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous responses."""
        check_is_fitted(self, ["capacity_", "q_", "intercept_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        return self.intercept_ + self.q_ * batch_choquet_integral(
            matrix, self.capacity_
        )

    def fit_predict(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Fit the model and return predictions for ``X``."""
        return self.fit(X, y).predict(X)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return np.asarray(self.universe_.var_names, dtype=object)

    def _validated_q_bounds(self) -> tuple[float, float]:
        try:
            bounds = np.asarray(self.q_bounds, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("q_bounds must be a pair of real numbers.") from error
        if bounds.shape != (2,):
            raise ValueError("q_bounds must contain exactly (lower, upper).")
        lower, upper = map(float, bounds)
        if np.isnan(lower) or np.isnan(upper):
            raise ValueError("q_bounds cannot contain NaN.")
        if lower > upper:
            raise ValueError("q_bounds lower bound must be <= upper bound.")
        return lower, upper
