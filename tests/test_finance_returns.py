import numpy as np
import pandas as pd

from capacities_ml_fin.finance import (
    aggregate_returns,
    excess_returns,
    forward_losses,
    forward_returns,
    price_returns,
    to_losses,
    wealth_index,
)


def test_price_returns_preserve_pandas_alignment():
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2024-01-01", periods=3),
        name="asset",
    )

    result = price_returns(prices, method="simple")

    assert result.index.equals(prices.index)
    assert result.name == "asset"
    np.testing.assert_allclose(result.iloc[1:], [0.1, 0.1])
    assert np.isnan(result.iloc[0])


def test_historical_and_forward_returns_use_the_expected_windows():
    returns = np.array([0.10, 0.20, -0.10, 0.05])

    historical = aggregate_returns(returns, horizon=2, method="simple")
    forward = forward_returns(returns, horizon=2, method="simple")

    np.testing.assert_allclose(historical[1:], [0.32, 0.08, -0.055])
    assert np.isnan(historical[0])
    np.testing.assert_allclose(forward[:2], [0.08, -0.055])
    assert np.isnan(forward[2:]).all()
    np.testing.assert_allclose(
        forward_losses(returns, horizon=2, method="simple")[:2],
        [-0.08, 0.055],
    )


def test_loss_excess_return_and_wealth_conventions():
    returns = pd.Series([0.10, -0.10], name="return")

    losses = to_losses(returns)
    excess = excess_returns(returns, 0.01)
    wealth = wealth_index(returns)

    assert losses.name == "return_loss"
    np.testing.assert_allclose(losses, [-0.10, 0.10])
    np.testing.assert_allclose(excess, [0.09, -0.11])
    np.testing.assert_allclose(wealth, [1.10, 0.99])
