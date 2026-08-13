import numpy as np

from capacities_ml_fin.finance import (
    amihud_illiquidity,
    drawdown,
    max_drawdown,
    momentum,
    realized_volatility,
    relative_bid_ask_spread,
    turnover,
)


def test_trailing_market_features_do_not_use_future_observations():
    returns = np.array([0.10, 0.20, -0.10, 0.05])

    volatility = realized_volatility(returns, window=2)
    trend = momentum(returns, lookback=3, skip=1, method="simple")

    assert np.isnan(volatility[0])
    np.testing.assert_allclose(volatility[1:], np.sqrt([0.05, 0.05, 0.0125]))
    assert np.isnan(trend[:2]).all()
    np.testing.assert_allclose(trend[2:], [0.32, 0.08])


def test_drawdown_and_liquidity_features_follow_financial_conventions():
    returns = np.array([0.10, -0.20, 0.05])

    np.testing.assert_allclose(drawdown(returns), [0.0, -0.20, -0.16])
    np.testing.assert_allclose(max_drawdown(returns), -0.20)
    np.testing.assert_allclose(relative_bid_ask_spread(99.0, 101.0), 0.02)
    np.testing.assert_allclose(turnover([10.0, 20.0], 100.0), [0.10, 0.20])
    np.testing.assert_allclose(
        amihud_illiquidity([0.02, -0.01], [1000.0, 2000.0]),
        [2e-5, 5e-6],
    )
