# imports
from __future__ import annotations
from functools import partial
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

# modules
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.ml.models.classification.utils import (
    linear_classifier,
    normalized_feature_scales,
    scaled_linear_classifier,
    validate_unit_interval,
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
from capacities_ml_fin.ml.optimization.objectives import ZeroOneLossObjective
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# Choquet linear classifier
class ChoquetClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic threshold classifier with monotone Choquet projection.

    When ``learn_feature_scales=True``, non-negative feature scales are fitted
    jointly with the capacity and normalized so their maximum equals one.
    Inputs must be commensurable criteria in ``[0, 1]`` whose direction has
    already been oriented so that larger values favor the positive class.
    """

    def __init__(
        self,
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "pymoo",
        solver_options: dict[str, Any] | None = None,
        penalty: Any = None,
        learn_feature_scales: bool = True,
    ) -> None:
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.penalty = penalty
        self.learn_feature_scales = learn_feature_scales

    def __sklearn_tags__(self):
        """Declare the binary, normalized-input estimator contract."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.positive_only = True
        options = {} if self.solver_options is None else self.solver_options
        tags.non_deterministic = (
            self.solver in ("pymoo", Solver.PYMOO)
            and options.get("seed") is None
        )
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ChoquetClassifier":
        """Fit the capacity, optional feature scales and decision threshold."""
        # input and binary-label validation
        matrix, raw_target = validate_data(
            self,
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        self.universe_ = fitted_universe(self)
        matrix = validate_unit_interval(matrix)
        raw_target = column_or_1d(raw_target, warn=True)
        if raw_target.shape != (matrix.shape[0],):
            raise ValueError("y must contain one label per observation.")
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(raw_target)
        if label_encoder.classes_.size != 2:
            raise ValueError("y must contain exactly two distinct labels.")
        self.solver_ = resolve_solver(self.solver)
        if not isinstance(self.learn_feature_scales, (bool, np.bool_)):
            raise TypeError("learn_feature_scales must be boolean.")

        # capacity design and threshold initialization
        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(self.universe_.n_elements)
        design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        threshold_initial = float(
            np.clip(np.median(design @ capacity_initial), 0.0, 1.0)
        )
        blocks = [
            ParameterBlock("capacity", compilation.bundle.n_parameters)
        ]
        if self.learn_feature_scales:
            blocks.append(
                ParameterBlock(
                    "feature_scales",
                    self.universe_.n_elements,
                    lower=1e-8,
                    upper=1.0,
                )
            )
        blocks.append(ParameterBlock("threshold", 1, lower=0.0, upper=1.0))
        layout = ParameterLayout(*blocks)
        capacity_slice = layout.slice("capacity")
        threshold_slice = layout.slice("threshold")

        initial_parts = [capacity_initial]
        if self.learn_feature_scales:
            feature_scales_slice = layout.slice("feature_scales")
            initial_parts.append(np.ones(self.universe_.n_elements, dtype=float))
            classifier = partial(
                scaled_linear_classifier,
                matrix=matrix,
                parameter_masks=compilation.bundle.parameter_masks,
                representation=compilation.bundle.representation,
                capacity_slice=capacity_slice,
                feature_scales_slice=feature_scales_slice,
                threshold_slice=threshold_slice,
            )
        else:
            feature_scales_slice = None
            classifier = partial(
                linear_classifier,
                design=design,
                capacity_slice=capacity_slice,
                threshold_slice=threshold_slice,
            )
        initial_parts.append(np.array([threshold_initial]))

        # direct 0-1 loss optimization
        objective = ZeroOneLossObjective(
            target=target,
            predictor=classifier,
            penalty=self.penalty,
        )
        problem = Problem.from_capacity(
            universe=self.universe_,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(initial_parts),
            name="choquet_linear_classifier",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver_, **options).solve(problem)
        if not result.success:
            violation = result.diagnostics.get("maximum_constraint_violation")
            raise RuntimeError(
                "Choquet classification optimization failed: "
                f"{result.message} Maximum constraint violation: {violation}."
            )

        # fitted scikit-learn state
        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.threshold_ = float(result.parameters[threshold_slice][0])
        self.feature_scales_ = (
            normalized_feature_scales(result.parameters[feature_scales_slice])
            if feature_scales_slice is not None
            else np.ones(self.universe_.n_elements, dtype=float)
        )
        self.classes_ = label_encoder.classes_
        self.n_classes_ = self.classes_.size
        self.sparsity_ = sparsity
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return signed margins relative to the learned decision threshold."""
        return self.choquet_score(X) - self.threshold_

    def choquet_score(self, X: ArrayLike) -> np.ndarray:
        """Return the Choquet projection before thresholding."""
        check_is_fitted(self, ["capacity_", "threshold_", "feature_scales_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        matrix = validate_unit_interval(matrix)
        return batch_choquet_integral(
            matrix * self.feature_scales_,
            self.capacity_,
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary labels using the fitted threshold."""
        encoded_predictions = (self.decision_function(X) >= 0.0).astype(int)
        return self.classes_[encoded_predictions]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return np.asarray(self.universe_.var_names, dtype=object)
