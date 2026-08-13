import numpy as np
import pytest

from capacities_ml_fin.risk import (
    ExpectedShortfallDistortion,
    RollingRiskEstimator,
    block_bootstrap_interval,
    capital_backtest,
    christoffersen_independence_test,
    diversification_benefit,
    distortion_risk_measure,
    hac_mean_interval,
    kupiec_coverage_test,
    rolling_risk_estimates,
)


def test_rolling_estimator_uses_only_losses_before_forecast_date():
    losses = np.array([1.0, 2.0, 100.0, 4.0])
    estimator = RollingRiskEstimator(
        ExpectedShortfallDistortion(0.5),
        window=2,
    ).fit(losses)

    capital = estimator.predict_in_sample()

    np.testing.assert_allclose(capital, [np.nan, np.nan, 2.0, 100.0], equal_nan=True)
    assert estimator.predict(2).shape == (2,)


def test_rolling_estimator_supports_horizon_aggregation_and_expanding_windows():
    estimator = RollingRiskEstimator(
        lambda sample: np.mean(sample),
        window_type="expanding",
        window=None,
        min_periods=1,
        horizon=2,
    ).fit([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(
        estimator.predict_in_sample(),
        [np.nan, np.nan, 3.0, 4.0],
        equal_nan=True,
    )


def test_backtesting_reports_exceedances_and_likelihood_tests():
    losses = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    capital = np.array([2.0, 2.0, 2.0, 4.0, 2.0])
    result = capital_backtest(losses, capital)

    assert result.n_exceedances == 2
    assert result.exceedance_frequency == pytest.approx(0.4)
    assert result.mean_exceedance == pytest.approx(1.0)
    assert np.isfinite(kupiec_coverage_test(losses, capital, 0.8).p_value)
    assert np.isfinite(christoffersen_independence_test(losses, capital).p_value)


def test_diversification_and_block_bootstrap_helpers():
    measure = lambda sample: distortion_risk_measure(
        sample,
        ExpectedShortfallDistortion(0.5),
    )
    first = np.array([0.0, 4.0, 0.0, 4.0])
    second = np.array([4.0, 0.0, 4.0, 0.0])

    assert diversification_benefit(first, second, measure) >= 0.0
    lower, upper = block_bootstrap_interval(
        np.arange(10.0),
        n_resamples=50,
        block_size=2,
        random_state=0,
    )
    assert lower <= upper

    hac = hac_mean_interval(np.arange(10.0), max_lags=2)
    assert hac.lower_bound <= hac.mean <= hac.upper_bound


def test_multiple_rolling_measures_return_aligned_columns():
    estimates = rolling_risk_estimates(
        np.arange(1.0, 7.0),
        {
            "mean": np.mean,
            "tail": ExpectedShortfallDistortion(0.5),
        },
        window=3,
    )

    assert list(estimates.columns) == ["mean", "tail"]
    assert estimates.shape == (6, 2)
