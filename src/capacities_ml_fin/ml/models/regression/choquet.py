# imports
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted

# modules
from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.ml.models.utils import capacity_design, validate_features
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
    """Scikit-learn compatible least-squares Choquet regressor."""

    def __init__(
        self,
        universe: VariableUniverse,
        sparsity: CapacitySparsity | None = None,
        solver: Solver = Solver.SCIPY,
        solver_options: dict[str, Any] | None = None,
        penalty: Any = None,
    ) -> None:
        self.universe = universe
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.penalty = penalty

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ChoquetRegressor":
        """Fit the capacity and the regression intercept."""
        matrix, target = check_X_y(
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        matrix = validate_features(matrix, self.universe, fitting=True)
        target = np.asarray(target, dtype=float).reshape(-1)
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values.")
        if not isinstance(self.solver, Solver):
            raise TypeError("solver must be a Solver enum member.")

        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe.n_elements)
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

        def predictor(parameters: np.ndarray) -> np.ndarray:
            return design @ parameters[capacity_slice] + parameters[intercept_slice][0]

        def symbolic_predictor(variable: Any) -> Any:
            return design @ variable[capacity_slice] + variable[intercept_slice][0]

        objective = SquaredErrorObjective(
            target=target,
            predictor=predictor,
            penalty=self.penalty,
            symbolic_predictor=symbolic_predictor,
        )
        problem = Problem.from_capacity(
            universe=self.universe,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(
                [capacity_initial, np.array([intercept_initial])]
            ),
            name="choquet_regression",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver, **options).solve(problem)

        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.intercept_ = float(result.parameters[intercept_slice][0])
        self.n_features_in_ = matrix.shape[1]
        self.feature_names_in_ = np.asarray(self.universe.var_names, dtype=object)
        self.sparsity_ = sparsity
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous responses."""
        check_is_fitted(self, ["result_", "problem_", "intercept_"])
        matrix = validate_features(X, self.universe)
        design = capacity_design(
            matrix,
            self.problem_.parameter_masks,
            self.problem_.representation,
        )
        blocks = self.problem_.parameter_layout.unpack(self.result_.parameters)
        return design @ blocks["capacity"] + blocks["intercept"][0]

    def fit_predict(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Fit the model and return predictions for ``X``."""
        return self.fit(X, y).predict(X)

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return self.feature_names_in_.copy()
