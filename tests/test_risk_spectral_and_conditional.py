import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from capacities_ml.risk import (
    KusuokaRiskMeasure,
    ResidualBootstrapDistribution,
    SpectralRiskMeasure,
    expected_shortfall,
)


def test_spectral_measure_is_a_normalized_quantile_mixture():
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    measure = SpectralRiskMeasure(
        levels=(0.25, 0.75),
        weights=(1.0, 3.0),
        quantile="lower",
    )

    assert measure(losses) == pytest.approx(2.5)


def test_kusuoka_measure_mixes_tail_var_and_worst_case_loss():
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    measure = KusuokaRiskMeasure(
        levels=(0.5,),
        weights=(1.0,),
        worst_case_weight=0.25,
    )

    expected = 0.25 * 4.0 + 0.75 * expected_shortfall(losses, 0.5)
    assert measure(losses) == pytest.approx(expected)


def test_residual_bootstrap_wraps_a_point_regressor():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = 1.0 + 2.0 * X[:, 0] + np.array([-1.0, 1.0] * 4)
    model = ResidualBootstrapDistribution(LinearRegression()).fit(X, y)

    scenarios = model.predict_scenarios([[8.0], [9.0]])
    distributions = model.predict_distribution([[8.0]])

    assert scenarios.shape == (2, 8)
    assert len(distributions) == 1
    assert distributions[0].losses.shape == (8,)

