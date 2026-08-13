# imports
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


# capacity normalizer
class CapacityNormalizer(TransformerMixin, BaseEstimator):
    """Scale capacity inputs to a common range with optional cost features.

    Features listed in ``cost_features`` are reversed after scaling so that
    larger transformed values always represent more desirable outcomes.
    """

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        cost_features: Sequence[int | str] | None = None,
        clip: bool = True,
    ) -> None:
        self.feature_range = feature_range
        self.cost_features = cost_features
        self.clip = clip

    def fit(self, X: ArrayLike, y: Any = None) -> "CapacityNormalizer":
        """Learn feature ranges and resolve cost-feature names."""
        matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=1)
        lower, upper = self._validate_feature_range()
        if not isinstance(self.clip, (bool, np.bool_)):
            raise TypeError("clip must be a boolean.")

        self.n_features_in_ = matrix.shape[1]
        columns = getattr(X, "columns", None)
        if columns is not None and all(isinstance(column, str) for column in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.cost_features_ = self._resolve_cost_features(columns)
        self.data_min_ = np.min(matrix, axis=0)
        self.data_max_ = np.max(matrix, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        safe_range = np.where(self.data_range_ == 0.0, 1.0, self.data_range_)
        self.scale_ = (upper - lower) / safe_range
        self.min_ = lower - self.data_min_ * self.scale_
        self.feature_range_ = (lower, upper)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform features into the configured capacity range."""
        check_is_fitted(self, ["data_min_", "scale_", "cost_features_"])
        matrix = self._validate_input(X)
        lower, upper = self.feature_range_
        transformed = matrix * self.scale_ + self.min_
        if self.cost_features_.size:
            transformed[:, self.cost_features_] = (
                lower + upper - transformed[:, self.cost_features_]
            )
        if self.clip:
            np.clip(transformed, lower, upper, out=transformed)
        return transformed

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Undo the normalization and cost-feature reversal."""
        check_is_fitted(self, ["data_min_", "scale_", "cost_features_"])
        transformed = self._validate_input(X)
        lower, upper = self.feature_range_
        restored = transformed.copy()
        if self.clip:
            np.clip(restored, lower, upper, out=restored)
        if self.cost_features_.size:
            restored[:, self.cost_features_] = (
                lower + upper - restored[:, self.cost_features_]
            )
        safe_scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return (restored - self.min_) / safe_scale

    def get_feature_names_out(
        self,
        input_features: ArrayLike | None = None,
    ) -> np.ndarray:
        """Return feature names unchanged."""
        check_is_fitted(self, ["n_features_in_"])
        if input_features is not None:
            features = np.asarray(input_features, dtype=object)
            if features.shape != (self.n_features_in_,):
                raise ValueError("input_features has an incompatible size.")
            return features
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_.copy()
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    def _validate_feature_range(self) -> tuple[float, float]:
        values = np.asarray(self.feature_range, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("feature_range must contain two finite values.")
        lower, upper = float(values[0]), float(values[1])
        if lower >= upper:
            raise ValueError("feature_range lower bound must be less than upper bound.")
        return lower, upper

    def _resolve_cost_features(self, columns: Any) -> np.ndarray:
        if self.cost_features is None:
            return np.empty(0, dtype=int)
        features = tuple(self.cost_features)
        if all(isinstance(feature, (int, np.integer)) for feature in features):
            indices = np.asarray(features, dtype=int)
        elif all(isinstance(feature, str) for feature in features):
            if columns is None:
                raise ValueError(
                    "String cost_features require named input columns."
                )
            names = tuple(columns)
            missing = [feature for feature in features if feature not in names]
            if missing:
                raise ValueError(f"Unknown cost features: {missing}.")
            indices = np.asarray([names.index(feature) for feature in features], dtype=int)
        else:
            raise TypeError("cost_features must contain only indices or only names.")
        if np.any(indices < 0) or np.any(indices >= self.n_features_in_):
            raise ValueError("A cost feature index is outside the input matrix.")
        return np.unique(indices)

    def _validate_input(self, X: ArrayLike) -> np.ndarray:
        matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=1)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; expected {self.n_features_in_}."
            )
        return matrix
