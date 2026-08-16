# imports
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.validation import check_is_fitted, validate_data

# modules
from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.ml.models.classification import ChoquisticRegression
from capacities_ml_fin.ml.models.regression import ScaledChoquetRegressor
from capacities_ml_fin.ml.optimization import KAdditivity, Solver
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer


# fuzzy Choquet input layer
class FuzzyChoquetInputLayer(TransformerMixin, BaseEstimator):
    """Supervised scalar Choquet input layer.

    The layer first makes the input criteria commensurable on ``[0, 1]`` and
    then learns a normalized monotone capacity against the supplied target.
    ``transform`` returns the fitted Choquet integral as one fuzzy input.

    This is a staged implementation of the architecture described by Gomes
    et al. (2016): the capacity is fitted first and is then frozen while the
    downstream neural network is trained.  It deliberately differs from a
    network containing one independently constrained capacity per hidden unit.

    Parameters
    ----------
    task : {"regression", "classification"}, default="regression"
        Supervised objective used to estimate the capacity.
    sparsity : CapacitySparsity, optional
        Capacity family. By default a 2-additive capacity is used (1-additive
        for a single input feature).
    solver : {"scipy", "pymoo"} or Solver, default="scipy"
        Optimization backend. Classification currently requires SciPy.
    solver_options : dict, optional
        Options forwarded to the capacity estimator.
    class_weight : dict, "balanced", or None, default=None
        Classification weights used while fitting the fuzzy layer.
    clip : bool, default=True
        Whether normalized values outside the fitted training range are clipped.

    Attributes
    ----------
    capacity_ : BaseCapacity
        Learned normalized monotone capacity.
    capacity_model_ : BaseEstimator
        Complete fitted capacity estimator.
    normalizer_ : CapacityNormalizer
        Training-only min-max normalizer.
    n_features_in_ : int
        Number of input criteria.
    """

    def __init__(
        self,
        task: str = "regression",
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        class_weight: dict[Any, float] | str | None = None,
        clip: bool = True,
    ) -> None:
        self.task = task
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.class_weight = class_weight
        self.clip = clip

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "FuzzyChoquetInputLayer":
        """Fit normalization and the supervised capacity without data leakage."""
        matrix, target = validate_data(
            self, X, y, dtype=float, ensure_2d=True, ensure_min_samples=2
        )
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be either 'regression' or 'classification'.")

        self.normalizer_ = CapacityNormalizer(clip=self.clip).fit(matrix)
        normalized = self.normalizer_.transform(matrix)
        sparsity = self.sparsity
        if sparsity is None:
            sparsity = KAdditivity(order=min(2, matrix.shape[1]))

        if self.task == "regression":
            if sample_weight is not None:
                raise ValueError(
                    "sample_weight is currently available only for classification."
                )
            model = ScaledChoquetRegressor(
                sparsity=sparsity,
                solver=self.solver,
                solver_options=self.solver_options,
                q_bounds=(-np.inf, np.inf),
            ).fit(normalized, target)
        else:
            model = ChoquisticRegression(
                sparsity=sparsity,
                solver=self.solver,
                solver_options=self.solver_options,
                class_weight=self.class_weight,
            ).fit(normalized, target, sample_weight=sample_weight)

        self.capacity_model_ = model
        self.capacity_ = model.capacity_
        self.sparsity_ = sparsity
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Return one Choquet-fuzzified input per observation."""
        check_is_fitted(self, ["capacity_", "normalizer_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        normalized = self.normalizer_.transform(matrix)
        if self.task == "classification":
            values = self.capacity_model_.utility_function(normalized)
        else:
            # The neural network receives the normalized Choquet integral, not
            # the regression scale and intercept used to estimate the capacity.
            values = batch_choquet_integral(normalized, self.capacity_)
        return np.asarray(values, dtype=float).reshape(-1, 1)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> np.ndarray:
        """Return the name of the scalar fuzzy input."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
        return np.asarray(["choquet_fuzzy_input"], dtype=object)


# fuzzy Choquet neural regressor
class FuzzyChoquetNeuralRegressor(RegressorMixin, BaseEstimator):
    """Tanh neural regressor fed by a learned 2-additive Choquet input.

    Capacity fitting and neural fitting are sequential. This makes the fuzzy
    input explicit and avoids a large joint constrained optimization over
    several capacity-valued hidden units.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (10,),
        alpha: float = 0.0001,
        max_iter: int = 500,
        random_state: int | None = None,
        mlp_solver: str = "lbfgs",
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        clip: bool = True,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.mlp_solver = mlp_solver
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.clip = clip

    def fit(self, X: ArrayLike, y: ArrayLike) -> "FuzzyChoquetNeuralRegressor":
        """Fit the fuzzy layer and then the conventional neural network."""
        matrix, target = validate_data(
            self, X, y, dtype=float, ensure_2d=True, ensure_min_samples=2
        )
        target = np.asarray(target, dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError("y must contain only finite values.")
        self.fuzzy_layer_ = FuzzyChoquetInputLayer(
            task="regression",
            sparsity=self.sparsity,
            solver=self.solver,
            solver_options=self.solver_options,
            clip=self.clip,
        ).fit(matrix, target)
        fuzzy = self.fuzzy_layer_.transform(matrix)
        self.network_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="tanh",
            solver=self.mlp_solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(fuzzy, target)
        self.capacity_ = self.fuzzy_layer_.capacity_
        self.capacity_model_ = self.fuzzy_layer_.capacity_model_
        self.normalizer_ = self.fuzzy_layer_.normalizer_
        self.n_iter_ = self.network_.n_iter_
        self.loss_ = self.network_.loss_
        return self

    def fuzzy_transform(self, X: ArrayLike) -> np.ndarray:
        """Expose the scalar input presented to the fitted neural network."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        return self.fuzzy_layer_.transform(matrix)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous responses."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        return self.network_.predict(self.fuzzy_transform(X))


# fuzzy Choquet neural classifier
class FuzzyChoquetNeuralClassifier(ClassifierMixin, BaseEstimator):
    """Binary tanh MLP fed by a supervised 2-additive Choquet input layer."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (10,),
        alpha: float = 0.0001,
        max_iter: int = 500,
        random_state: int | None = None,
        mlp_solver: str = "lbfgs",
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
        solver_options: dict[str, Any] | None = None,
        class_weight: dict[Any, float] | str | None = None,
        clip: bool = True,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.mlp_solver = mlp_solver
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.class_weight = class_weight
        self.clip = clip

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "FuzzyChoquetNeuralClassifier":
        """Fit the classification capacity and then the neural classifier."""
        matrix, target = validate_data(
            self, X, y, dtype=float, ensure_2d=True, ensure_min_samples=2
        )
        self.fuzzy_layer_ = FuzzyChoquetInputLayer(
            task="classification",
            sparsity=self.sparsity,
            solver=self.solver,
            solver_options=self.solver_options,
            class_weight=self.class_weight,
            clip=self.clip,
        ).fit(matrix, target, sample_weight=sample_weight)
        fuzzy = self.fuzzy_layer_.transform(matrix)
        self.network_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="tanh",
            solver=self.mlp_solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(fuzzy, target)
        self.capacity_ = self.fuzzy_layer_.capacity_
        self.capacity_model_ = self.fuzzy_layer_.capacity_model_
        self.normalizer_ = self.fuzzy_layer_.normalizer_
        self.classes_ = self.network_.classes_
        self.n_iter_ = self.network_.n_iter_
        self.loss_ = self.network_.loss_
        return self

    def fuzzy_transform(self, X: ArrayLike) -> np.ndarray:
        """Expose the scalar input presented to the fitted neural network."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        matrix = validate_data(self, X, reset=False, dtype=float)
        return self.fuzzy_layer_.transform(matrix)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict binary labels."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        return self.network_.predict(self.fuzzy_transform(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probabilities for both classes."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        return self.network_.predict_proba(self.fuzzy_transform(X))

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return log-probabilities for both classes."""
        check_is_fitted(self, ["fuzzy_layer_", "network_"])
        return self.network_.predict_log_proba(self.fuzzy_transform(X))
