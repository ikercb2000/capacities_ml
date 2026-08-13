# imports
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# modules
from capacities_ml_fin.base.capacities import VariableUniverse
from capacities_ml_fin.ml.models.time_series.utils import (
    autoregression_predictor,
    lag_matrix,
)
from capacities_ml_fin.ml.models.utils import capacity_design
from capacities_ml_fin.ml.optimization import (
    FullCapacity,
    Optimizer,
    ParameterBlock,
    ParameterLayout,
    Problem,
    Solver,
)
from capacities_ml_fin.ml.optimization.objectives import SquaredErrorObjective
from capacities_ml_fin.ml.optimization.sparsity import CapacitySparsity


# Choquet autoregressive forecaster
class ChoquetAutoRegressor(BaseForecaster):
    """Univariate Choquet autoregression compatible with ``sktime``.

    The conditional mean is ``intercept + phi * C_nu(y[t-1], ..., y[t-p])``.
    Optional exogenous variables enter additively. When stationarity is
    enforced, ``abs(phi) <= stability_bound < 1``.

    Parameters
    ----------
    lags : int, default=1
        Number of lagged observations aggregated by the Choquet integral.
    sparsity : CapacitySparsity or None, default=None
        Capacity parametrization. ``None`` uses a full capacity.
    solver : {"scipy", "pymoo"} or Solver, default="scipy"
        Numerical optimizer used to estimate the model.
    solver_options : dict or None, default=None
        Options forwarded to :class:`~capacities_ml_fin.ml.optimization.Optimizer`.
    fit_intercept : bool, default=True
        Whether to estimate an intercept.
    enforce_stationarity : bool, default=True
        Whether to constrain the autoregressive scale to a contraction.
    stability_bound : float, default=0.999
        Absolute upper bound for ``phi`` when stationarity is enforced.
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:missing_values": False,
        "capability:non_contiguous_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
    }

    def __init__(
        self,
        lags: int = 1,
        sparsity: CapacitySparsity | None = None,
        solver: Solver | str = "scipy",
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
        super().__init__()

    # fit hook called by BaseForecaster
    def _fit(
        self,
        y: pd.Series,
        X: pd.DataFrame | None,
        fh: ForecastingHorizon | None,
    ) -> "ChoquetAutoRegressor":
        self._validate_hyperparameters()
        series = y.to_numpy(dtype=float)
        if series.size <= self.lags:
            raise ValueError("y must contain more observations than lags.")

        exogenous = None if X is None else X.to_numpy(dtype=float)
        lagged, target = lag_matrix(series, int(self.lags))
        exogenous_train = None if exogenous is None else exogenous[self.lags :]

        # capacity design over the lag universe
        universe = VariableUniverse(
            tuple(f"lag_{lag}" for lag in range(1, self.lags + 1))
        )
        sparsity = self.sparsity if self.sparsity is not None else FullCapacity()
        compilation = sparsity.compile(universe.n_elements)
        design = capacity_design(
            lagged,
            compilation.bundle.parameter_masks,
            compilation.bundle.representation,
        )
        capacity_initial = compilation.initial_parameters
        aggregate = design @ capacity_initial

        # linear warm start for the non-capacity parameters
        linear_columns = [aggregate]
        if self.fit_intercept:
            linear_columns.append(np.ones(target.size))
        if exogenous_train is not None:
            linear_columns.extend(
                exogenous_train[:, column]
                for column in range(exogenous_train.shape[1])
            )
        initial_linear, *_ = np.linalg.lstsq(
            np.column_stack(linear_columns), target, rcond=None
        )
        phi_initial = float(initial_linear[0])
        if self.enforce_stationarity:
            phi_initial = float(
                np.clip(phi_initial, -self.stability_bound, self.stability_bound)
            )

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

        predictor = partial(
            autoregression_predictor,
            design=design,
            exogenous=exogenous_train,
            capacity_slice=layout.slice("capacity"),
            phi_slice=layout.slice("phi"),
            intercept_slice=(layout.slice("intercept") if self.fit_intercept else None),
            exogenous_slice=(
                layout.slice("exogenous") if exogenous_train is not None else None
            ),
        )
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
        result = Optimizer(solver=self.solver_, **options).solve(problem)
        if not result.success:
            raise RuntimeError(
                f"Choquet autoregression optimization failed: {result.message}"
            )

        # fitted state exposed with the sktime trailing-underscore convention
        fitted = np.asarray(predictor(result.parameters), dtype=float)
        residuals = target - fitted
        blocks_result = layout.unpack(result.parameters)
        fitted_index = y.index[self.lags :]
        target_name = y.name

        self.problem_ = problem
        self.result_ = result
        self.capacity_ = problem.decode_result(result)
        self.phi_ = float(blocks_result["phi"][0])
        self.intercept_ = (
            float(blocks_result["intercept"][0]) if self.fit_intercept else 0.0
        )
        self.exogenous_coef_ = blocks_result.get("exogenous", np.empty(0))
        self.n_exogenous_ = int(self.exogenous_coef_.size)
        self.exogenous_columns_ = tuple() if X is None else tuple(X.columns)
        self._y_name_ = y.name
        self.universe_ = universe
        self.sparsity_ = sparsity
        self.order_ = (int(self.lags), 0, 0)
        self.nobs_ = int(series.size)
        self.n_params_ = int(result.parameters.size)
        self.fittedvalues_ = pd.Series(fitted, index=fitted_index, name=target_name)
        self.resid_ = pd.Series(residuals, index=fitted_index, name=target_name)
        self.sigma2_ = float(
            np.sum(residuals**2) / max(1, residuals.size - self.n_params_)
        )
        return self

    # point forecast hook called by BaseForecaster
    def _predict(
        self,
        fh: ForecastingHorizon,
        X: pd.DataFrame | None,
    ) -> pd.Series:
        relative = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        if np.any(relative <= 0):
            raise NotImplementedError(
                "In-sample prediction is not defined before all autoregressive lags "
                "are available. Use fittedvalues_ for training diagnostics."
            )
        absolute = fh.to_absolute_index(self.cutoff)
        predictions: dict[int, float] = {}

        # positive horizons are generated recursively, including omitted steps
        positive = relative[relative > 0]
        if positive.size:
            maximum = int(np.max(positive))
            future_exogenous = self._future_exogenous(X, maximum)
            history = self._y.to_numpy(dtype=float).tolist()
            for step in range(1, maximum + 1):
                row = (
                    None
                    if future_exogenous is None
                    else future_exogenous.iloc[step - 1]
                )
                value = self._forecast_one(
                    history,
                    None if row is None else row.to_numpy(dtype=float),
                )
                history.append(value)
                predictions[step] = value

        values = np.asarray([predictions[int(step)] for step in relative], dtype=float)
        return pd.Series(values, index=absolute, name=self._y_name_)

    # Gaussian prediction interval hook called by BaseForecaster
    def _predict_interval(
        self,
        fh: ForecastingHorizon,
        X: pd.DataFrame | None,
        coverage: list[float],
    ) -> pd.DataFrame:
        prediction = self._predict(fh=fh, X=X)
        relative = np.asarray(fh.to_relative(self.cutoff), dtype=int)
        scale = np.sqrt(self.sigma2_ * np.maximum(relative, 1))
        columns: list[np.ndarray] = []
        for value in coverage:
            z_score = float(norm.ppf(0.5 + float(value) / 2.0))
            columns.extend(
                [
                    prediction.to_numpy() - z_score * scale,
                    prediction.to_numpy() + z_score * scale,
                ]
            )
        result = pd.DataFrame(np.column_stack(columns), index=prediction.index)
        result.columns = self._get_columns(
            method="predict_interval", coverage=coverage
        )
        return result

    # update hook called after BaseForecaster appends y and X
    def _update(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        update_params: bool = True,
    ) -> "ChoquetAutoRegressor":
        if update_params:
            return self._fit(y=self._y, X=self._X, fh=self._fh)
        self.nobs_ = len(self._y)
        return self

    # hyperparameter validation
    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.lags, (int, np.integer)) or self.lags < 1:
            raise ValueError("lags must be a positive integer.")
        try:
            self.solver_ = Solver(self.solver)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "solver must be 'scipy', 'pymoo', or a Solver member."
            ) from error
        if self.solver_ is Solver.CVXPY:
            raise ValueError(
                "The phi-times-capacity model is non-convex; use SCIPY or PYMOO."
            )
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise TypeError("fit_intercept must be boolean.")
        if not isinstance(self.enforce_stationarity, (bool, np.bool_)):
            raise TypeError("enforce_stationarity must be boolean.")
        if not 0.0 < float(self.stability_bound) < 1.0:
            raise ValueError("stability_bound must lie in (0, 1).")

    # aligned future exogenous values for all recursive steps
    def _future_exogenous(
        self,
        X: pd.DataFrame | None,
        maximum_step: int,
    ) -> pd.DataFrame | None:
        if self.n_exogenous_ == 0:
            if X is not None:
                raise ValueError(
                    "X was supplied, but the fitted model has no exogenous terms."
                )
            return None
        if X is None:
            raise ValueError(
                "Future X is required because the model was fitted with exogenous data."
            )
        if tuple(X.columns) != self.exogenous_columns_:
            raise ValueError(
                "Future X must have the same columns, in the same order, as fit X."
            )
        complete_horizon = ForecastingHorizon(
            np.arange(1, maximum_step + 1), is_relative=True
        )
        required_index = complete_horizon.to_absolute_index(self.cutoff)
        missing = required_index.difference(X.index)
        if len(missing):
            raise ValueError(
                "Future X must contain every intermediate recursive forecasting step; "
                f"missing index values: {list(missing)}."
            )
        return X.loc[required_index]

    # one-step recursive forecast
    def _forecast_one(
        self,
        history: list[float],
        exogenous_row: np.ndarray | None,
    ) -> float:
        lagged = np.asarray(
            [[history[-lag] for lag in range(1, self.lags + 1)]], dtype=float
        )
        design = capacity_design(
            lagged,
            self.problem_.parameter_masks,
            self.problem_.representation,
        )
        blocks = self.problem_.parameter_layout.unpack(self.result_.parameters)
        aggregate = float((design @ blocks["capacity"]).item())
        value = self.intercept_ + self.phi_ * aggregate
        if exogenous_row is not None:
            value += float(exogenous_row @ self.exogenous_coef_)
        return float(value)

    # residual diagnostics
    def resid(self) -> pd.Series:
        """Return one-step training residuals."""
        self.check_is_fitted()
        return self.resid_.copy()

    # Akaike information criterion
    def aic(self) -> float:
        """Return the Gaussian Akaike information criterion."""
        return self._information_criterion(penalty=2.0)

    # Bayesian information criterion
    def bic(self) -> float:
        """Return the Gaussian Bayesian information criterion."""
        self.check_is_fitted()
        return self._information_criterion(penalty=float(np.log(self.resid_.size)))

    # shared Gaussian information criterion
    def _information_criterion(self, penalty: float) -> float:
        self.check_is_fitted()
        n_observations = self.resid_.size
        variance = max(
            float(np.mean(self.resid_.to_numpy() ** 2)), np.finfo(float).tiny
        )
        negative_twice_log_likelihood = n_observations * (
            np.log(2.0 * np.pi * variance) + 1.0
        )
        return float(negative_twice_log_likelihood + penalty * self.n_params_)

    # lightweight parameter set used by sktime's estimator checks
    @classmethod
    def get_test_params(
        cls, parameter_set: str = "default"
    ) -> list[dict[str, Any]]:
        """Return a fast configuration for ``sktime`` estimator checks."""
        return [{"lags": 1}, {"lags": 2, "fit_intercept": False}]
