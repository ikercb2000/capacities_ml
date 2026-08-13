import pickle

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.transformations.series.detrend import Detrender
from sktime.utils.estimator_checks import check_estimator

from capacities_ml_fin.ml.models import ChoquetAutoRegressor


def _stable_ar_series(length: int = 41) -> pd.Series:
    values = [0.2]
    for _ in range(length - 1):
        values.append(0.1 + 0.6 * values[-1])
    index = pd.period_range("2020-01", periods=length, freq="M")
    return pd.Series(values, index=index, name="response")


def test_choquet_autoregressor_recovers_stable_ar_one_and_sktime_api():
    series = _stable_ar_series()
    model = ChoquetAutoRegressor(lags=1).fit(series)
    horizon = ForecastingHorizon([1, 2, 3], is_relative=True)

    forecast = model.predict(fh=horizon)
    intervals = model.predict_interval(fh=horizon, coverage=[0.8, 0.95])

    assert model.universe_.var_names == ("lag_1",)
    assert np.isclose(model.phi_, 0.6, atol=1e-4)
    assert np.isclose(model.intercept_, 0.1, atol=1e-4)
    assert forecast.index.equals(pd.period_range("2023-06", periods=3, freq="M"))
    assert forecast.name == series.name
    assert intervals.shape == (3, 4)
    assert model.fittedvalues_.shape == (len(series) - 1,)
    assert np.isfinite(model.aic()) and np.isfinite(model.bic())
    assert model.clone().lags == 1
    assert pickle.loads(pickle.dumps(model)).predict(fh=horizon).equals(forecast)


def test_choquet_autoregressor_supports_indexed_exogenous_data_and_update():
    index = pd.period_range("2020-01", periods=12, freq="M")
    exogenous = pd.DataFrame({"signal": np.arange(12) / 12.0}, index=index)
    series = pd.Series(index=index, dtype=float, name="response")
    series.iloc[0] = 0.1
    for time in range(1, len(series)):
        series.iloc[time] = (
            0.2 + 0.3 * series.iloc[time - 1] + 0.4 * exogenous.iloc[time, 0]
        )

    model = ChoquetAutoRegressor(lags=1).fit(series, X=exogenous)
    future_index = pd.period_range(index[-1] + 1, periods=2, freq="M")
    future_exogenous = pd.DataFrame({"signal": [1.0, 1.1]}, index=future_index)

    assert model.predict(fh=[1, 2], X=future_exogenous).shape == (2,)

    new_y = pd.Series([0.8], index=future_index[:1], name="response")
    model.update(new_y, X=future_exogenous.iloc[:1], update_params=True)
    assert model.nobs_ == 13
    assert model.cutoff.equals(future_index[:1])


def test_choquet_autoregressor_works_in_sktime_pipeline_and_grid_search():
    series = _stable_ar_series(32)
    horizon = ForecastingHorizon([1, 2], is_relative=True)
    pipeline = TransformedTargetForecaster(
        steps=[
            (
                "detrend",
                Detrender(forecaster=PolynomialTrendForecaster(degree=1)),
            ),
            ("forecast", ChoquetAutoRegressor(lags=1)),
        ]
    )
    pipeline.fit(series, fh=horizon)
    assert pipeline.predict().shape == (2,)

    search = ForecastingGridSearchCV(
        forecaster=ChoquetAutoRegressor(),
        cv=ExpandingWindowSplitter(initial_window=20, step_length=5, fh=[1, 2]),
        param_grid={"lags": [1, 2]},
    )
    search.fit(series)
    assert search.best_params_["lags"] in {1, 2}
    assert search.predict(fh=horizon).shape == (2,)


def test_choquet_autoregressor_passes_core_sktime_contract_checks():
    results = check_estimator(
        ChoquetAutoRegressor,
        tests_to_run=[
            "test_constructor",
            "test_get_test_params_coverage",
            "test_fit_returns_self",
            "test_fit_updates_state",
        ],
        verbose=False,
    )
    assert all(result == "PASSED" for result in results.values())
