# imports
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# modules
from capacities_ml_fin.risk.distortions import Distortion
from capacities_ml_fin.risk.measures import DistortionRiskMeasure
from capacities_ml_fin.risk.rolling.utils import _series_values


# rolling risk estimator
class RollingRiskEstimator(BaseEstimator):
    """Estimate historical and future capital without look-ahead.

    Parameters
    ----------
    risk_measure : Distortion or callable
        Maps a historical loss sample to required capital.
    window : int or None, default=250
        Maximum history for rolling estimation.
    window_type : {"rolling", "expanding"}, default="rolling"
        Fixed-length or expanding estimation history.
    min_periods : int, optional
        Minimum history required before producing capital.
    horizon : int, default=1
        Number of consecutive losses aggregated before estimation.
    decay : float, optional
        Exponential decay in ``(0, 1]`` for older observations.

    Attributes
    ----------
    losses_ : ndarray
        Validated observed losses.
    aggregated_losses_ : ndarray
        Horizon-aggregated losses.
    capital_ : ndarray
        Capital forecast aligned with each realized loss.
    nobs_ : int
        Number of original observations.

    Notes
    -----
    Capital at time ``t`` uses only observations preceding the loss forecast.
    """

    def __init__(
        self,
        risk_measure: Any,
        *,
        window: int | None = 250,
        window_type: str = "rolling",
        min_periods: int | None = None,
        horizon: int = 1,
        decay: float | None = None,
    ) -> None:
        self.risk_measure = risk_measure
        self.window = window
        self.window_type = window_type
        self.min_periods = min_periods
        self.horizon = horizon
        self.decay = decay

    def fit(self, losses: ArrayLike, y: Any = None) -> "RollingRiskEstimator":
        """Compute historical capital forecasts from past loss windows."""
        values = _series_values(losses)
        self._validate_parameters()
        self.losses_ = values.copy()
        self.index_ = losses.index.copy() if isinstance(losses, pd.Series) else None
        self.name_ = losses.name if isinstance(losses, pd.Series) else None
        self.aggregated_losses_ = self._aggregate(values)
        self.capital_ = self._historical_capital(self.aggregated_losses_)
        self.nobs_ = values.size
        return self

    def predict_in_sample(self) -> np.ndarray | pd.Series:
        """Return capital aligned with the loss that it forecasts."""
        check_is_fitted(self, ["capital_", "losses_"])
        if self.index_ is not None:
            name = "capital" if self.name_ is None else f"{self.name_}_capital"
            return pd.Series(self.capital_.copy(), index=self.index_, name=name)
        return self.capital_.copy()

    def predict(self, n_periods: int = 1) -> np.ndarray:
        """Estimate capital from the latest available historical window."""
        check_is_fitted(self, ["aggregated_losses_"])
        if not isinstance(n_periods, int) or n_periods < 1:
            raise ValueError("n_periods must be a positive integer.")
        history = self.aggregated_losses_[np.isfinite(self.aggregated_losses_)]
        capital = self._window_measure(history)
        return np.repeat(capital, n_periods)

    def update(self, losses: ArrayLike) -> "RollingRiskEstimator":
        """Append observed losses and recompute the aligned capital series."""
        check_is_fitted(self, ["losses_"])
        additional = _series_values(losses)
        combined = np.concatenate([self.losses_, additional])
        return self.fit(combined)

    def _validate_parameters(self) -> None:
        if self.window_type not in {"rolling", "expanding"}:
            raise ValueError("window_type must be 'rolling' or 'expanding'.")
        if self.window_type == "rolling":
            if not isinstance(self.window, int) or self.window < 1:
                raise ValueError("A rolling estimator requires a positive integer window.")
        elif self.window is not None and (not isinstance(self.window, int) or self.window < 1):
            raise ValueError("window must be positive or None.")
        if self.min_periods is not None and (
            not isinstance(self.min_periods, int) or self.min_periods < 1
        ):
            raise ValueError("min_periods must be a positive integer or None.")
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if self.decay is not None and (
            not np.isfinite(self.decay) or not 0.0 < self.decay <= 1.0
        ):
            raise ValueError("decay must lie in (0, 1] or be None.")
        if not isinstance(self.risk_measure, Distortion) and not callable(self.risk_measure):
            raise TypeError("risk_measure must be a Distortion or callable.")

    def _aggregate(self, losses: np.ndarray) -> np.ndarray:
        if self.horizon == 1:
            return losses.copy()
        cumulative = np.concatenate([[0.0], np.cumsum(losses)])
        aggregated = np.full(losses.size, np.nan)
        aggregated[self.horizon - 1 :] = (
            cumulative[self.horizon :] - cumulative[: -self.horizon]
        )
        return aggregated

    def _historical_capital(self, aggregated: np.ndarray) -> np.ndarray:
        capital = np.full(aggregated.size, np.nan)
        minimum = self.min_periods
        if minimum is None:
            minimum = self.window if self.window_type == "rolling" else 1
        for position in range(aggregated.size):
            history = aggregated[:position]
            history = history[np.isfinite(history)]
            if history.size < minimum:
                continue
            capital[position] = self._window_measure(history)
        return capital

    def _window_measure(self, history: np.ndarray) -> float:
        if history.size == 0:
            raise ValueError("At least one historical aggregate is required.")
        if self.window_type == "rolling" and self.window is not None:
            history = history[-self.window :]
        weights = None
        if self.decay is not None:
            powers = np.arange(history.size - 1, -1, -1, dtype=float)
            weights = np.power(self.decay, powers)
            weights /= np.sum(weights)
        measure = (
            DistortionRiskMeasure(self.risk_measure)
            if isinstance(self.risk_measure, Distortion)
            else self.risk_measure
        )
        if weights is None:
            return float(measure(history))
        try:
            return float(measure(history, sample_weight=weights))
        except TypeError as error:
            raise TypeError(
                "A callable risk_measure must accept sample_weight when decay is used."
            ) from error
