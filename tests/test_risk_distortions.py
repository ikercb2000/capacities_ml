import numpy as np
import pytest

from capacities_ml_fin.risk import (
    CustomDistortion,
    ExpectedShortfallDistortion,
    IdentityDistortion,
    PiecewiseLinearDistortion,
    ProportionalHazardsDistortion,
    ValueAtRiskDistortion,
    validate_distortion,
)


def test_standard_distortions_match_their_definitions():
    probabilities = np.array([0.0, 0.1, 0.5, 1.0])

    np.testing.assert_allclose(IdentityDistortion()(probabilities), probabilities)
    np.testing.assert_allclose(
        ExpectedShortfallDistortion(0.8)(probabilities),
        [0.0, 0.5, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        ProportionalHazardsDistortion(0.5)(probabilities),
        np.sqrt(probabilities),
    )
    np.testing.assert_allclose(
        ValueAtRiskDistortion(0.8)(probabilities),
        [0.0, 0.0, 1.0, 1.0],
    )


def test_piecewise_and_custom_distortions_are_validated():
    piecewise = PiecewiseLinearDistortion(
        probabilities=(0.0, 0.5, 1.0),
        values=(0.0, 0.75, 1.0),
    )
    custom = CustomDistortion(lambda probability: np.sqrt(probability))

    assert piecewise(0.25) == pytest.approx(0.375)
    assert piecewise.is_concave()
    assert custom.is_concave()
    validate_distortion(custom)


def test_invalid_custom_distortion_is_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        CustomDistortion(
            lambda probability: probability + 0.2 * np.sin(2.0 * np.pi * probability)
        )
