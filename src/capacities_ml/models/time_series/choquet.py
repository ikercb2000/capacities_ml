# imports
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

# modules
from capacities_ml.capacities import VariableUniverse
from capacities_ml.models.utils import capacity_design
from capacities_ml.optimization import (
    FullCapacity,
    Optimizer,
    ParameterBlock,
    ParameterLayout,
    Problem,
    Solver,
)
from capacities_ml.optimization.objectives import SquaredErrorObjective
from capacities_ml.optimization.sparsity import CapacitySparsity


# lagged supervised sample
def _lag_matrix(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(
        [[series[t - lag] for lag in range(1, lags + 1)] for t in range(lags, series.size)],
        dtype=float,
    )
    return rows, series[lags:].copy()


# Choquet autoregressive model
class ChoquetAutoRegressor(BaseEstimator):
    """Univariate Choquet autoregression with a pmdarima-like API.

    The conditional mean is ``intercept + phi * C_nu(y[t-1], ..., y[t-p])``.
    Optional exogenous variables enter additively.  When stationarity is
    enforced, ``abs(phi) <= stability_bound < 1`` as in the contraction
    condition for a Choquet autoregression.
    """

    def __init__(
        self,
        lags: int = 1,
        sparsity: CapacitySparsity | None = None,
        solver: Solver = Solver.SCIPY,
        solver_options: dict[str, Any] | None = None,
        fit_intercept: bool = True,
        enforce_stationarity: bool = True,
        stability_bound: float = 0.999,
    ) -> None:
        self.lags = lags
        self.sparsity = sparsity
        self.solver = solver
        self.solver_options = solver_options
        self.fit_intercept = fit_intercept
        self.enforce_stationarity = enforce_stationarity
        self.stability_bound = stability_bound

    def fit(self, y: ArrayLike, X: ArrayLike | None = None) -> "ChoquetAutoRegressor":
        """Fit from a univariate series and optional aligned exogenous data."""
        # series and hyperparameter validation
        series = np.asarray(column_or_1d(y), dtype=float)
        if not isinstance(self.lags, (int, np.integer)) or self.lags < 1:
            raise ValueError("lags must be a positive integer.")
        if series.size <= self.lags:
            raise ValueError("y must contain more observations than lags.")
        if np.any(~np.isfinite(series)):
            raise ValueError("y must contain only finite values.")
        if not isinstance(self.solver, Solver):
            raise TypeError("solver must be a Solver enum member.")
        if self.solver is Solver.CVXPY:
            raise ValueError("The phi-times-capacity model is non-convex; use SCIPY or PYMOO.")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise TypeError("fit_intercept must be boolean.")
        if not isinstance(self.enforce_stationarity, (bool, np.bool_)):
            raise TypeError("enforce_stationarity must be boolean.")
        if not 0.0 < float(self.stability_bound) < 1.0:
            raise ValueError("stability_bound must lie in (0, 1).")

        # lagged training matrix and aligned exogenous variables
        exogenous = self._validate_exogenous(X, series.size, fitting=True)
        lagged, target = _lag_matrix(series, int(self.lags))
        exogenous_train = None if exogenous is None else exogenous[self.lags :]

        # capacity design over the lag universe
        universe = VariableUniverse(tuple(f"lag_{lag}" for lag in range(1, self.lags + 1)))
        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(universe.n_vars)
        design = capacity_design(
            lagged,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        aggregate = design @ capacity_initial

        # linear warm start for phi, intercept and exogenous coefficients
        linear_columns = [aggregate]
        if self.fit_intercept:
            linear_columns.append(np.ones(target.size))
        if exogenous_train is not None:
            linear_columns.extend(exogenous_train[:, j] for j in range(exogenous_train.shape[1]))
        linear_design = np.column_stack(linear_columns)
        initial_linear, *_ = np.linalg.lstsq(linear_design, target, rcond=None)
        phi_initial = float(initial_linear[0])
        if self.enforce_stationarity:
            phi_initial = float(np.clip(phi_initial, -self.stability_bound, self.stability_bound))

        # complete optimization parameter layout
        blocks = [ParameterBlock("capacity", compilation.bundle.n_parameters)]
        phi_bound = float(self.stability_bound) if self.enforce_stationarity else np.inf
        blocks.append(ParameterBlock("phi", 1, lower=-phi_bound, upper=phi_bound))
        if self.fit_intercept:
            blocks.append(ParameterBlock("intercept", 1))
        if exogenous_train is not None:
            blocks.append(ParameterBlock("exogenous", exogenous_train.shape[1]))
        layout = ParameterLayout(*blocks)
        initial_parts = [capacity_initial, np.array([phi_initial])]
        cursor = 1
        if self.fit_intercept:
            initial_parts.append(np.array([initial_linear[cursor]]))
            cursor += 1
        if exogenous_train is not None:
            initial_parts.append(np.asarray(initial_linear[cursor:], dtype=float))

        capacity_slice = layout.slice("capacity")
        phi_slice = layout.slice("phi")

        # conditional-mean callback used by the optimizer
        def predictor(parameters: np.ndarray) -> np.ndarray:
            prediction = parameters[phi_slice][0] * (design @ parameters[capacity_slice])
            if self.fit_intercept:
                prediction = prediction + parameters[layout.slice("intercept")][0]
            if exogenous_train is not None:
                prediction = prediction + exogenous_train @ parameters[layout.slice("exogenous")]
            return prediction

        # constrained least-squares problem
        objective = SquaredErrorObjective(target=target, predictor=predictor)
        problem = Problem.from_capacity(
            universe=universe,
            objective=objective,
            sparsity=sparsity,
            parameter_layout=layout,
            initial_parameters=np.concatenate(initial_parts),
            name="choquet_autoregression",
        )
        options = {} if self.solver_options is None else dict(self.solver_options)
        result = Optimizer(solver=self.solver, **options).solve(problem)
        if not result.success:
            raise RuntimeError(f"Choquet autoregression optimization failed: {result.message}")

        # fitted values, residuals and public model state
        fitted = predictor(result.parameters)
        residuals = target - fitted
        blocks_result = layout.unpack(result.parameters)
        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.phi_ = float(blocks_result["phi"][0])
        self.intercept_ = float(blocks_result["intercept"][0]) if self.fit_intercept else 0.0
        self.exogenous_coef_ = blocks_result.get("exogenous", np.empty(0))
        self.n_exogenous_ = self.exogenous_coef_.size
        self.universe_ = universe
        self.sparsity_ = sparsity
        self.order_ = (int(self.lags), 0, 0)
        self.nobs_ = series.size
        self.n_params_ = result.parameters.size
        self.fittedvalues_ = fitted
        self.resid_ = residuals
        self.sigma2_ = float(np.sum(residuals**2) / max(1, residuals.size - self.n_params_))
        self._y = series.copy()
        self._X = None if exogenous is None else exogenous.copy()
        return self

    # exogenous input validation
    @staticmethod
    def _validate_exogenous(
        X: ArrayLike | None,
        n_rows: int,
        *,
        fitting: bool = False,
    ) -> np.ndarray | None:
        if X is None:
            return None
        matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=1)
        if matrix.shape[0] != n_rows:
            raise ValueError("X must have one row per observation in y.")
        if fitting and np.any(~np.isfinite(matrix)):
            raise ValueError("X must contain only finite values.")
        return matrix

    # future exogenous input validation
    def _future_exogenous(self, X: ArrayLike | None, n_periods: int) -> np.ndarray | None:
        if self.n_exogenous_ == 0:
            if X is not None:
                raise ValueError("X was supplied, but the fitted model has no exogenous terms.")
            return None
        if X is None:
            raise ValueError(
                "Future X is required because the model was fitted with exogenous data."
            )
        matrix = check_array(X, dtype=float, ensure_2d=True, ensure_min_samples=n_periods)
        if matrix.shape != (n_periods, self.n_exogenous_):
            raise ValueError(
                f"X must have shape ({n_periods}, {self.n_exogenous_}); got {matrix.shape}."
            )
        return matrix

    # one-step recursive forecast
    def _forecast_one(self, history: list[float], exogenous_row: np.ndarray | None) -> float:
        lagged = np.asarray([[history[-lag] for lag in range(1, self.lags + 1)]])
        design = capacity_design(
            lagged,
            self.problem_.parameter_masks,
            self.problem_.representation,
        )
        capacity_parameters = self.problem_.parameter_layout.unpack(
            self.result_.parameters
        )["capacity"]
        aggregate = (design @ capacity_parameters).item()
        value = self.intercept_ + self.phi_ * aggregate
        if exogenous_row is not None:
            value += float(exogenous_row @ self.exogenous_coef_)
        return value

    # out-of-sample forecasting
    def predict(
        self,
        n_periods: int = 10,
        X: ArrayLike | None = None,
        return_conf_int: bool = False,
        alpha: float = 0.05,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Recursively forecast future observations."""
        check_is_fitted(self, ["result_", "_y", "sigma2_"])
        if not isinstance(n_periods, (int, np.integer)) or n_periods < 1:
            raise ValueError("n_periods must be a positive integer.")
        future_X = self._future_exogenous(X, int(n_periods))
        history = self._y.tolist()
        forecasts = np.empty(n_periods, dtype=float)
        for step in range(n_periods):
            row = None if future_X is None else future_X[step]
            forecasts[step] = self._forecast_one(history, row)
            history.append(float(forecasts[step]))
        if not return_conf_int:
            return forecasts
        return forecasts, self._confidence_intervals(forecasts, alpha, recursive=True)

    # in-sample forecasting
    def predict_in_sample(
        self,
        X: ArrayLike | None = None,
        start: int | None = None,
        end: int | None = None,
        dynamic: bool = False,
        return_conf_int: bool = False,
        alpha: float = 0.05,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return one-step or dynamic forecasts over the observed sample."""
        check_is_fitted(self, ["result_", "_y", "sigma2_"])
        first = self.lags if start is None else int(start)
        last = self.nobs_ - 1 if end is None else int(end)
        if first < self.lags or last < first or last >= self.nobs_:
            raise ValueError("start/end must select observations from lags through nobs - 1.")
        observed_X = self._X if X is None else self._validate_exogenous(X, self.nobs_)
        if self.n_exogenous_ and observed_X is None:
            raise ValueError("X is required for a model fitted with exogenous data.")
        if not self.n_exogenous_ and observed_X is not None:
            raise ValueError("X was supplied, but the fitted model has no exogenous terms.")
        if observed_X is not None and observed_X.shape[1] != self.n_exogenous_:
            raise ValueError(
                f"X must have {self.n_exogenous_} columns; got {observed_X.shape[1]}."
            )
        history = self._y[:first].tolist()
        predictions = np.empty(last - first + 1, dtype=float)
        for offset, t in enumerate(range(first, last + 1)):
            row = None if observed_X is None else observed_X[t]
            predictions[offset] = self._forecast_one(history, row)
            history.append(float(predictions[offset] if dynamic else self._y[t]))
        if not return_conf_int:
            return predictions
        return predictions, self._confidence_intervals(predictions, alpha, recursive=dynamic)

    # Gaussian forecast intervals
    def _confidence_intervals(
        self,
        predictions: np.ndarray,
        alpha: float,
        *,
        recursive: bool,
    ) -> np.ndarray:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")
        z_score = float(norm.ppf(1.0 - alpha / 2.0))
        horizons = np.arange(1, predictions.size + 1) if recursive else np.ones(predictions.size)
        errors = z_score * np.sqrt(self.sigma2_ * horizons)
        return np.column_stack((predictions - errors, predictions + errors))

    # append observations and refit
    def update(
        self,
        y: ArrayLike,
        X: ArrayLike | None = None,
        maxiter: int | None = None,
    ) -> "ChoquetAutoRegressor":
        """Append observations and refit, mirroring pmdarima's update contract."""
        check_is_fitted(self, ["_y"])
        new_y = np.asarray(column_or_1d(y), dtype=float)
        if new_y.size < 1 or np.any(~np.isfinite(new_y)):
            raise ValueError("y must contain at least one finite observation.")
        if self.n_exogenous_:
            new_X = self._validate_exogenous(X, new_y.size)
            combined_X = np.vstack((self._X, new_X))
        else:
            if X is not None:
                raise ValueError("X was supplied, but the fitted model has no exogenous terms.")
            combined_X = None
        combined_y = np.concatenate((self._y, new_y))
        if maxiter is None:
            return self.fit(combined_y, combined_X)
        if self.solver is not Solver.SCIPY:
            raise ValueError("maxiter in update is currently supported only with Solver.SCIPY.")
        previous_options = self.solver_options
        options = {} if previous_options is None else dict(previous_options)
        nested = dict(options.get("options", {}))
        nested["maxiter"] = int(maxiter)
        options["options"] = nested
        self.solver_options = options
        try:
            return self.fit(combined_y, combined_X)
        finally:
            self.solver_options = previous_options

    # residual diagnostics
    def resid(self) -> np.ndarray:
        """Return in-sample one-step residuals."""
        check_is_fitted(self, ["resid_"])
        return self.resid_.copy()

    # Akaike information criterion
    def aic(self) -> float:
        """Return the Gaussian Akaike information criterion."""
        return self._information_criterion(penalty=2.0)

    # Bayesian information criterion
    def bic(self) -> float:
        """Return the Gaussian Bayesian information criterion."""
        check_is_fitted(self, ["resid_"])
        return self._information_criterion(penalty=float(np.log(self.resid_.size)))

    # shared Gaussian information criterion
    def _information_criterion(self, penalty: float) -> float:
        check_is_fitted(self, ["resid_", "n_params_"])
        n = self.resid_.size
        variance = max(float(np.mean(self.resid_**2)), np.finfo(float).tiny)
        negative_twice_log_likelihood = n * (np.log(2.0 * np.pi * variance) + 1.0)
        return float(negative_twice_log_likelihood + penalty * self.n_params_)
