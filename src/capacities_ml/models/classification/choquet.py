# imports
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

# modules
from capacities_ml.capacities import Capacity, VariableUniverse
from capacities_ml.models._choquet import capacity_design, validate_features
from capacities_ml.optimization import (
    FullCapacity,
    Optimizer,
    OptimizationResult,
    ParameterBlock,
    ParameterLayout,
    Problem,
    Solver,
)
from capacities_ml.optimization.objectives import ZeroOneLossObjective
from capacities_ml.optimization.sparsity import CapacitySparsity


# choquet linear classifier
class ChoquetClassifier(ClassifierMixin, BaseEstimator):
    """Scikit-learn compatible threshold Choquet classifier."""

    def __init__(
        self,
        universe: VariableUniverse,
        sparsity: CapacitySparsity | None = None,
        solver: Solver = Solver.PYMOO,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        self.universe = universe
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ChoquetClassifier":
        """Fit the capacity and the classification threshold."""
        matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=1)
        matrix = validate_features(matrix, self.universe, fitting=True)
        raw_target = column_or_1d(y, warn=True)
        if raw_target.shape != (matrix.shape[0],):
            raise ValueError("y must contain one label per observation.")
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(raw_target)
        if label_encoder.classes_.size != 2:
            raise ValueError("y must contain exactly two distinct labels.")
        if not isinstance(self.solver, Solver):
            raise TypeError("solver must be a Solver enum member.")

        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe.n_vars)
        design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        threshold_initial = float(
            np.clip(np.median(design @ capacity_initial), 0.0, 1.0)
        )
        layout = ParameterLayout(
            ParameterBlock("capacity", compilation.bundle.n_parameters),
            ParameterBlock("threshold", 1, lower=0.0, upper=1.0),
        )
        capacity_slice = layout.slice("capacity")
        threshold_slice = layout.slice("threshold")

        def classifier(parameters: np.ndarray) -> np.ndarray:
            scores = design @ parameters[capacity_slice]
            return (scores >= parameters[threshold_slice][0]).astype(int)

        objective = ZeroOneLossObjective(target=target, predictor=classifier)
        problem = Problem.from_capacity(
            universe=self.universe,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(
                [capacity_initial, np.array([threshold_initial])]
            ),
            name="choquet_linear_classifier",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver, **options).solve(problem)

        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.threshold_ = float(result.parameters[threshold_slice][0])
        self.classes_ = label_encoder.classes_
        self.n_classes_ = self.classes_.size
        self.n_features_in_ = matrix.shape[1]
        self.feature_names_in_ = np.asarray(self.universe.var_names, dtype=object)
        self.sparsity_ = sparsity
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return Choquet scores before thresholding."""
        check_is_fitted(self, ["result_", "problem_", "threshold_"])
        matrix = validate_features(X, self.universe)
        design = capacity_design(
            matrix,
            self.problem_.parameter_masks,
            self.problem_.representation,
        )
        blocks = self.problem_.parameter_layout.unpack(self.result_.parameters)
        return design @ blocks["capacity"]

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary labels using the fitted threshold."""
        scores = self.decision_function(X)
        encoded_predictions = (scores >= self.threshold_).astype(int)
        return self.classes_[encoded_predictions]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return deterministic class probabilities induced by the threshold."""
        predictions = self.predict(X)
        encoded_predictions = np.searchsorted(self.classes_, predictions)
        probabilities = np.zeros((predictions.size, 2), dtype=float)
        probabilities[:, 1] = encoded_predictions
        probabilities[:, 0] = 1.0 - encoded_predictions
        return probabilities

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return self.feature_names_in_.copy()
