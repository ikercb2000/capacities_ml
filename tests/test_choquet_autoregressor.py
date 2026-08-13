import numpy as np
from sklearn.base import clone

from capacities_ml_fin.ml.models import ChoquetAutoRegressor


def test_choquet_autoregressor_recovers_stable_ar_one_and_pmdarima_api():
    series = [0.2]
    for _ in range(40):
        series.append(0.1 + 0.6 * series[-1])

    model = ChoquetAutoRegressor(lags=1).fit(series)

    assert model.universe_.var_names == ("lag_1",)
    forecast, confidence = model.predict(3, return_conf_int=True)

    assert np.isclose(model.phi_, 0.6, atol=1e-4)
    assert np.isclose(model.intercept_, 0.1, atol=1e-4)
    assert forecast.shape == (3,)
    assert confidence.shape == (3, 2)
    assert model.predict_in_sample().shape == (len(series) - 1,)
    assert np.isfinite(model.aic()) and np.isfinite(model.bic())
    assert clone(model).lags == 1


def test_choquet_autoregressor_supports_exogenous_forecasts_and_update():
    exogenous = np.arange(12, dtype=float).reshape(-1, 1) / 12.0
    series = np.empty(12)
    series[0] = 0.1
    for t in range(1, series.size):
        series[t] = 0.2 + 0.3 * series[t - 1] + 0.4 * exogenous[t, 0]

    model = ChoquetAutoRegressor(lags=1).fit(series, exogenous)
    assert model.predict(2, X=np.array([[1.0], [1.1]])).shape == (2,)
    model.update([0.8], X=[[1.0]])
    assert model.nobs_ == 13
