import numpy as np

from capacities_ml.risk import (
    DistortionRiskMeasure,
    ExpectedShortfallDistortion,
    check_risk_measure_axioms,
    is_comonotonic,
)


def test_expected_shortfall_satisfies_risk_axioms_on_comonotonic_losses():
    first = np.array([0.0, 1.0, 2.0, 3.0])
    second = np.array([1.0, 2.0, 4.0, 8.0])
    measure = DistortionRiskMeasure(ExpectedShortfallDistortion(0.5))

    report = check_risk_measure_axioms(measure, first, second)

    assert is_comonotonic(first, second)
    assert report.monotonicity
    assert report.cash_invariance
    assert report.positive_homogeneity
    assert report.subadditivity
    assert report.convexity
    assert report.comonotonic_additivity
