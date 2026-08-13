import numpy as np
import pandas as pd

from capacities_ml_fin.finance import (
    equal_weights,
    lag_weights,
    market_cap_weights,
    normalize_weights,
    portfolio_losses,
    portfolio_returns,
)


def test_portfolio_weight_construction_and_lagging():
    named = equal_weights(["asset_a", "asset_b"])
    capitalized = market_cap_weights([1.0, 3.0])
    normalized = normalize_weights([[1.0, 1.0], [1.0, 3.0]])
    lagged = lag_weights(normalized)

    np.testing.assert_allclose(named, [0.5, 0.5])
    assert list(named.index) == ["asset_a", "asset_b"]
    np.testing.assert_allclose(capitalized, [0.25, 0.75])
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    assert np.isnan(lagged[0]).all()
    np.testing.assert_allclose(lagged[1], [0.5, 0.5])


def test_portfolio_returns_align_weights_by_asset_name():
    returns = pd.DataFrame(
        {"asset_a": [0.10, -0.05], "asset_b": [0.00, 0.15]}
    )
    weights = pd.Series({"asset_b": 0.75, "asset_a": 0.25})

    result = portfolio_returns(returns, weights)
    losses = portfolio_losses(returns, weights)

    assert result.name == "portfolio_return"
    np.testing.assert_allclose(result, [0.025, 0.10])
    np.testing.assert_allclose(losses, -result)
