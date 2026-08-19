# imports
from __future__ import annotations
from functools import partial
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data

# modules
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.ml.models.classification.utils import validate_unit_interval
from capacities_ml_fin.ml.models.utils import (
    capacity_design,
    fitted_universe,
    resolve_solver,
)
from capacities_ml_fin.ml.optimization import KAdditivity, Optimizer, Problem, Solver
from capacities_ml_fin.ml.optimization.objectives import LogisticNegativeLogLikelihood
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# shared-capacity logits
def _shared_capacity_logits(
    parameters: np.ndarray,
    *,
    positive_design: np.ndarray,
    negative_design: np.ndarray,
    softmax_scale: float,
) -> np.ndarray:
    """Return binary Softmax logits from two shared-capacity designs."""
    return softmax_scale * ((positive_design - negative_design) @ parameters)


# shared-capacity binary aggregator
class SharedCapacityBinaryAggregator(ClassifierMixin, BaseEstimator):
    """Class-wise binary Choquet aggregation using one shared capacity.

    Given source-model positive-class probabilities ``p``, the estimator uses
    the same normalized monotone capacity ``nu`` to compute
    ``a1 = C_nu(p)`` and ``a0 = C_nu(1 - p)``. The positive probability is the
    two-class Softmax, equivalently
    ``sigmoid(softmax_scale * (a1 - a0))``. Only capacity parameters are
    learned by constrained maximum likelihood.

    This aggregation is inspired by Uriz et al. (2023), while fitting the
    capacity with this package's solver-independent constrained optimization
    framework.
    """

    def __init__(
        self,
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        class_weight: dict[Any, float] | str | None = None,
        penalty: Any = None,
        softmax_scale: float = 3.0,
    ) -> None:
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.class_weight = class_weight
        self.penalty = penalty
        self.softmax_scale = softmax_scale

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "SharedCapacityBinaryAggregator":
        """Fit the single capacity by binary negative log-likelihood."""
        matrix, raw_target = validate_data(
            self,
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        matrix = validate_unit_interval(matrix)
        self.universe_ = fitted_universe(self)

        encoder = LabelEncoder()
        target = encoder.fit_transform(raw_target)
        if encoder.classes_.size != 2:
            raise ValueError("y must contain exactly two distinct labels.")

        if not isinstance(self.softmax_scale, (int, float, np.integer, np.floating)):
            raise TypeError("softmax_scale must be a positive finite number.")
        scale = float(self.softmax_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("softmax_scale must be a positive finite number.")
        self.softmax_scale_ = scale
        self.solver_ = resolve_solver(self.solver)

        weights = compute_sample_weight(self.class_weight, raw_target).astype(float)
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float).reshape(-1)
            if supplied.shape != target.shape:
                raise ValueError(
                    "sample_weight must contain one value per observation."
                )
            if np.any(~np.isfinite(supplied)) or np.any(supplied < 0.0):
                raise ValueError("sample_weight must be finite and non-negative.")
            weights *= supplied
        if not np.any(weights > 0.0):
            raise ValueError("At least one sample must have positive weight.")

        sparsity = (
            self.sparsity
            if self.sparsity is not None
            else KAdditivity(order=self.universe_.n_elements)
        )
        compilation = sparsity.compile(self.universe_.n_elements)
        positive_design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        negative_design = capacity_design(
            1.0 - matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        predictor = partial(
            _shared_capacity_logits,
            positive_design=positive_design,
            negative_design=negative_design,
            softmax_scale=scale,
        )
        objective = LogisticNegativeLogLikelihood(
            target=target,
            predictor=predictor,
            sample_weight=weights,
            penalty=self.penalty,
            symbolic_predictor=predictor,
        )
        problem = Problem.from_capacity(
            universe=self.universe_,
            objective=objective,
            sparsity=sparsity,
            name="shared_capacity_binary_aggregation",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver_, **options).solve(problem)
        if not result.success:
            raise RuntimeError(
                "Shared-capacity binary aggregation failed: " + result.message
            )

        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.classes_ = encoder.classes_
        self.sparsity_ = sparsity
        return self

    def class_scores(self, X: ArrayLike) -> np.ndarray:
        """Return ``[C_nu(1 - p), C_nu(p)]`` using the fitted capacity."""
        check_is_fitted(self, ["capacity_", "softmax_scale_"])
        matrix = validate_unit_interval(
            validate_data(self, X, reset=False, dtype=float)
        )
        negative = batch_choquet_integral(1.0 - matrix, self.capacity_)
        positive = batch_choquet_integral(matrix, self.capacity_)
        return np.column_stack((negative, positive))

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return ``softmax_scale * (a1 - a0)``."""
        scores = self.class_scores(X)
        return self.softmax_scale_ * (scores[:, 1] - scores[:, 0])

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the two-class Softmax probabilities."""
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return numerically stable log-probabilities."""
        logits = self.decision_function(X)
        return np.column_stack(
            (-np.logaddexp(0.0, logits), -np.logaddexp(0.0, -logits))
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels using the larger class-wise Choquet score."""
        encoded = (self.decision_function(X) >= 0.0).astype(int)
        return self.classes_[encoded]

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> np.ndarray:
        """Return the source-model names seen during fitting."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return np.asarray(self.universe_.var_names, dtype=object)
