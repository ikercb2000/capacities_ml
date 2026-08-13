# imports
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# modules
from capacities_ml_fin.risk.conditional.utils import _bootstrap_residuals
from capacities_ml_fin.risk.distributions import EmpiricalLossDistribution


# residual-bootstrap distribution
class ResidualBootstrapDistribution(BaseEstimator):
    """Conditional empirical distributions from point-model residuals.

    Parameters
    ----------
    estimator : estimator object
        Cloneable regressor implementing ``fit`` and ``predict``.
    center_residuals : bool, default=True
        Subtract the residual mean before resampling.
    max_scenarios : int, optional
        Maximum residual scenarios retained per prediction.
    random_state : int, optional
        Seed used when residual subsampling is required.

    Attributes
    ----------
    estimator_ : estimator object
        Fitted clone of the supplied estimator.
    residuals_ : ndarray
        In-sample residual scenarios.
    n_features_in_ : int
        Number of predictor columns seen during fitting.

    Notes
    -----
    Each conditional distribution shifts the empirical residual sample by the
    point prediction for its input row.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        center_residuals: bool = True,
        max_scenarios: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.center_residuals = center_residuals
        self.max_scenarios = max_scenarios
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ResidualBootstrapDistribution":
        """Fit the point estimator and retain its in-sample residual distribution."""
        matrix, target = check_X_y(X, y, dtype=float, ensure_min_samples=2)
        if self.max_scenarios is not None and self.max_scenarios < 1:
            raise ValueError("max_scenarios must be positive or None.")
        if not isinstance(self.center_residuals, (bool, np.bool_)):
            raise TypeError("center_residuals must be boolean.")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(matrix, target)
        fitted = np.asarray(self.estimator_.predict(matrix), dtype=float)
        if fitted.shape != target.shape:
            raise ValueError("The estimator must return one prediction per observation.")
        residuals = target - fitted
        if self.center_residuals:
            residuals = residuals - np.mean(residuals)
        self.residuals_ = residuals
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict_scenarios(self, X: ArrayLike) -> np.ndarray:
        """Return one row of residual-bootstrap scenarios per prediction."""
        check_is_fitted(self, ["estimator_", "residuals_", "n_features_in_"])
        matrix = check_array(X, dtype=float, ensure_2d=True)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; expected {self.n_features_in_}."
            )
        locations = np.asarray(self.estimator_.predict(matrix), dtype=float)
        residuals = _bootstrap_residuals(
            self.residuals_,
            self.max_scenarios,
            self.random_state,
        )
        return locations[:, None] + residuals[None, :]

    def predict_distribution(self, X: ArrayLike) -> list[EmpiricalLossDistribution]:
        """Return empirical conditional distributions for each input row."""
        return [
            EmpiricalLossDistribution(row)
            for row in self.predict_scenarios(X)
        ]
