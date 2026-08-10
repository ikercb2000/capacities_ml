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
from sklearn.utils.validation import check_X_y, check_is_fitted

# modules
from capacities_ml.capacities import VariableUniverse
from capacities_ml.models.classification.utils import (
    apply_choquistic_link,
    choquet_scores,
    choquistic_logits,
    validate_unit_interval,
)
from capacities_ml.models.utils import capacity_design, validate_features
from capacities_ml.optimization import (
    KAdditivity,
    Optimizer,
    ParameterBlock,
    ParameterLayout,
    Problem,
    Solver,
)
from capacities_ml.optimization.objectives import LogisticNegativeLogLikelihood
from capacities_ml.optimization.sparsity import CapacitySparsity


# numerical lower bound for the strict gamma > 0 constraint
_MIN_GAMMA = 1e-8


# Choquistic regression
class ChoquisticRegression(ClassifierMixin, BaseEstimator):
    """
    Choquistic regression as defined by Tehrani, Cheng and Hüllermeier.
    """

    def __init__(
        self,
        universe: VariableUniverse,
        sparsity: CapacitySparsity | None = None,
        solver: Solver = Solver.SCIPY,
        solver_options: dict[str, Any] | None = None,
        class_weight: dict[Any, float] | str | None = None,
    ) -> None:
        self.universe = universe
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.class_weight = class_weight

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "ChoquisticRegression":
        """Fit the capacity, utility threshold and scale by maximum likelihood."""
        # input and binary-label validation
        matrix, raw_target = check_X_y(
            X,
            y,
            dtype=float,
            ensure_2d=True,
            ensure_min_samples=1,
        )
        matrix = validate_unit_interval(
            validate_features(matrix, self.universe, fitting=True)
        )
        encoder = LabelEncoder()
        target = encoder.fit_transform(raw_target)
        if encoder.classes_.size != 2:
            raise ValueError("y must contain exactly two distinct labels.")
        if not isinstance(self.solver, Solver):
            raise TypeError("solver must be a Solver enum member.")
        if self.solver is not Solver.SCIPY:
            raise ValueError(
                "The paper formulation is fitted with sequential quadratic "
                "programming; use Solver.SCIPY."
            )

        # optional class and observation weights
        weights = compute_sample_weight(self.class_weight, raw_target).astype(float)
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float).reshape(-1)
            if supplied.shape != target.shape:
                raise ValueError("sample_weight must contain one value per observation.")
            if np.any(~np.isfinite(supplied)) or np.any(supplied < 0.0):
                raise ValueError("sample_weight must be finite and non-negative.")
            weights *= supplied
        if not np.any(weights > 0.0):
            raise ValueError("At least one sample must have positive weight.")

        # full Mobius capacity by default, or the requested sparse capacity
        sparsity = (
            self.sparsity
            if self.sparsity is not None
            else KAdditivity(order=self.universe.n_vars)
        )
        compilation = sparsity.compile(self.universe.n_vars)
        design = capacity_design(
            matrix,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        initial_scores = design @ capacity_initial

        # paper parameters gamma and beta with feasible initial values
        negative_mean = float(
            np.average(initial_scores[target == 0], weights=weights[target == 0])
        )
        positive_mean = float(
            np.average(initial_scores[target == 1], weights=weights[target == 1])
        )
        beta_initial = float(
            np.clip((negative_mean + positive_mean) / 2.0, 0.0, 1.0)
        )
        gamma_initial = 1.0

        layout = ParameterLayout(
            ParameterBlock("capacity", compilation.bundle.n_parameters),
            ParameterBlock("gamma", 1, lower=_MIN_GAMMA),
            ParameterBlock("beta", 1, lower=0.0, upper=1.0),
        )
        capacity_slice = layout.slice("capacity")
        gamma_slice = layout.slice("gamma")
        beta_slice = layout.slice("beta")
        linear_predictor = partial(
            choquistic_logits,
            design=design,
            capacity_slice=capacity_slice,
            gamma_slice=gamma_slice,
            beta_slice=beta_slice,
        )

        # constrained maximum-likelihood problem
        objective = LogisticNegativeLogLikelihood(
            target=target,
            predictor=linear_predictor,
            sample_weight=weights,
        )
        problem = Problem.from_capacity(
            universe=self.universe,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(
                [capacity_initial, np.array([gamma_initial, beta_initial])]
            ),
            name="choquistic_regression",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver, **options).solve(problem)
        if not result.success:
            raise RuntimeError(f"Choquistic optimization failed: {result.message}")

        # fitted scikit-learn state
        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.gamma_ = float(result.parameters[gamma_slice][0])
        self.beta_ = float(result.parameters[beta_slice][0])
        self.classes_ = encoder.classes_
        self.n_features_in_ = matrix.shape[1]
        self.feature_names_in_ = np.asarray(self.universe.var_names, dtype=object)
        self.sparsity_ = sparsity
        return self

    # latent Choquet utility
    def utility_function(self, X: ArrayLike) -> np.ndarray:
        """Return the latent utility ``C_mu(X)`` from the paper's first stage."""
        check_is_fitted(self, ["result_", "problem_", "gamma_", "beta_"])
        matrix = validate_unit_interval(validate_features(X, self.universe))
        design = capacity_design(
            matrix,
            self.problem_.parameter_masks,
            self.problem_.representation,
        )
        blocks = self.problem_.parameter_layout.unpack(self.result_.parameters)
        return choquet_scores(design, blocks["capacity"])

    # logistic decision score
    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the fitted log-odds ``gamma * (C_mu(X) - beta)``."""
        scores = self.utility_function(X)
        return apply_choquistic_link(
            scores,
            gamma=self.gamma_,
            beta=self.beta_,
        )

    # class probabilities
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return probabilities for the two encoded classes."""
        positive = expit(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    # log class probabilities
    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Return numerically stable log-probabilities."""
        logits = self.decision_function(X)
        return np.column_stack((-np.logaddexp(0.0, logits), -np.logaddexp(0.0, -logits)))

    # binary prediction
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels using a probability threshold of 0.5."""
        encoded = (self.decision_function(X) >= 0.0).astype(int)
        return self.classes_[encoded]

    # feature names
    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Return the feature names used by the fitted estimator."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        return self.feature_names_in_.copy()
