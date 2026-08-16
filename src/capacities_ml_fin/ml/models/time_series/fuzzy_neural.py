# imports
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# modules
from capacities_ml_fin.ml.models.neural.fuzzy import FuzzyChoquetNeuralRegressor
from capacities_ml_fin.ml.models.time_series.utils import lag_matrix
from capacities_ml_fin.ml.optimization import Solver
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# fuzzy Choquet neural autoregressor
class FuzzyChoquetNeuralAutoRegressor(BaseForecaster):
    """Neural forecaster with automatic 2-additive fuzzy-lag selection.

    For every candidate lag order, chronological inner training data are used
    to learn a 2-additive Choquet capacity. Its scalar integral fuzzifies each
    lag window before a conventional one-hidden-layer tanh network with a
    linear regression output. The lag order minimizing validation RMSE is
    selected and the complete staged model is then refitted on all observations.

    Parameters
    ----------
    lag_candidates : tuple of int, default=(2, 3, 4, 5, 6, 7, 8)
        Autoregressive orders compared chronologically.
    validation_fraction : float, default=0.2
        Trailing fraction reserved for lag selection. It is never used to fit
        candidate models before their validation score is computed.
    hidden_layer_sizes : tuple of int, default=(10,)
        Hidden tanh layer widths. ``(10,)`` follows the best reported paper
        specification.
    alpha : float, default=1e-4
        Neural L2 regularization strength.
    max_iter : int, default=500
        Maximum neural optimization iterations.
    random_state : int, optional
        Reproducible neural initialization.
    mlp_solver : {"lbfgs", "sgd", "adam"}, default="lbfgs"
        scikit-learn neural optimizer.
    sparsity : CapacitySparsity, optional
        Capacity family. The default is 2-additive.
    solver : {"scipy", "pymoo"} or Solver, default="scipy"
        Capacity optimization backend.
    solver_options : dict, optional
        Capacity optimizer options.
    clip : bool, default=True
        Clip future lag values to the fitted normalization range.

    Attributes
    ----------
    best_lag_ : int
        Lag order with the lowest chronological validation RMSE.
    lag_scores_ : pandas.DataFrame
        Candidate lag orders, sample sizes and validation RMSE values.
    capacity_ : BaseCapacity
        Final learned 2-additive temporal capacity.
    neural_model_ : FuzzyChoquetNeuralRegressor
        Final staged fuzzy-input neural estimator.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:missing_values": False,
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        lag_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
        validation_fraction: float = 0.2,
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
        self.lag_candidates = lag_candidates
        self.validation_fraction = validation_fraction
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.mlp_solver = mlp_solver
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.clip = clip
        super().__init__()

    def _fit(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
        fh: ForecastingHorizon | None,
    ) -> "FuzzyChoquetNeuralAutoRegressor":
        if X is not None:
            raise ValueError(
                "FuzzyChoquetNeuralAutoRegressor does not use exogenous X."
            )
        candidates = self._validated_candidates()
        values = y.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("y must contain only finite values.")
        validation_size = max(2, int(np.ceil(values.size * self.validation_fraction)))
        split = values.size - validation_size
        if split <= max(candidates) + 1:
            raise ValueError(
                "y is too short for the largest lag candidate and validation fraction."
            )

        rows: list[dict[str, float | int]] = []
        for lag in candidates:
            inner_X, inner_y = lag_matrix(values[:split], lag)
            candidate = self._new_model().fit(inner_X, inner_y)
            validation_X = self._windows(values, start=split, lag=lag)
            validation_y = values[split:]
            prediction = candidate.predict(validation_X)
            rmse = float(np.sqrt(np.mean((validation_y - prediction) ** 2)))
            rows.append(
                {
                    "lags": lag,
                    "training_observations": inner_y.size,
                    "validation_observations": validation_y.size,
                    "validation_rmse": rmse,
                }
            )

        scores = pd.DataFrame(rows).sort_values(
            ["validation_rmse", "lags"], ignore_index=True
        )
        best_lag = int(scores.iloc[0]["lags"])
        final_X, final_y = lag_matrix(values, best_lag)
        final_model = self._new_model().fit(final_X, final_y)

        self.best_lag_ = best_lag
        self.lag_scores_ = scores
        self.neural_model_ = final_model
        self.capacity_ = final_model.capacity_
        self.capacity_model_ = final_model.capacity_model_
        self.normalizer_ = final_model.normalizer_
        self.fittedvalues_ = pd.Series(
            final_model.predict(final_X), index=y.index[best_lag:], name=y.name
        )
        self.resid_ = y.iloc[best_lag:] - self.fittedvalues_
        self.nobs_ = int(values.size)
        self._y_name_ = y.name
        return self

    def _predict(
        self,
        fh: ForecastingHorizon,
        X: pd.DataFrame | None,
    ) -> pd.Series:
        if X is not None:
            raise ValueError(
                "FuzzyChoquetNeuralAutoRegressor does not use exogenous X."
            )
        relative = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        if np.any(relative <= 0):
            raise NotImplementedError(
                "Only strictly out-of-sample forecasts are supported."
            )
        maximum = int(np.max(relative))
        history = self._y.to_numpy(dtype=float).tolist()
        predictions: dict[int, float] = {}
        for step in range(1, maximum + 1):
            window = np.asarray(
                [[history[-lag] for lag in range(1, self.best_lag_ + 1)]],
                dtype=float,
            )
            value = float(self.neural_model_.predict(window)[0])
            history.append(value)
            predictions[step] = value
        absolute = fh.to_absolute_index(self.cutoff)
        result = np.asarray([predictions[int(step)] for step in relative])
        return pd.Series(result, index=absolute, name=self._y_name_)

    def _update(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        update_params: bool = True,
    ) -> "FuzzyChoquetNeuralAutoRegressor":
        if update_params:
            return self._fit(y=self._y, X=None, fh=self._fh)
        self.nobs_ = len(self._y)
        return self

    def _new_model(self) -> FuzzyChoquetNeuralRegressor:
        return FuzzyChoquetNeuralRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            mlp_solver=self.mlp_solver,
            sparsity=self.sparsity,
            solver=self.solver,
            solver_options=self.solver_options,
            clip=self.clip,
        )

    def _validated_candidates(self) -> tuple[int, ...]:
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must lie strictly between 0 and 1.")
        candidates = tuple(self.lag_candidates)
        if not candidates:
            raise ValueError("lag_candidates cannot be empty.")
        if any(not isinstance(lag, (int, np.integer)) or lag < 1 for lag in candidates):
            raise ValueError("lag_candidates must contain positive integers.")
        return tuple(sorted(set(int(lag) for lag in candidates)))

    @staticmethod
    def _windows(values: np.ndarray, *, start: int, lag: int) -> np.ndarray:
        return np.asarray(
            [
                [values[time - offset] for offset in range(1, lag + 1)]
                for time in range(start, values.size)
            ],
            dtype=float,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict[str, Any]]:
        return [
            {
                "lag_candidates": (2,),
                "validation_fraction": 0.2,
                "hidden_layer_sizes": (2,),
                "max_iter": 50,
                "random_state": 0,
            }
        ]
