import numpy as np

from capacities_ml.risk import (
    DistortedCapacity,
    DistortionRiskMeasure,
    ExpectedShortfallDistortion,
    ProbabilityCapacity,
    ProportionalHazardsDistortion,
    check_risk_measure_axioms,
    is_comonotonic,
    is_concave_event_capacity,
    is_convex_event_capacity,
)


def test_concave_distortion_of_probability_is_concave_capacity():
    probability = ProbabilityCapacity([0.2, 0.3, 0.5])
    capacity = DistortedCapacity(probability, ProportionalHazardsDistortion(0.5))

    assert is_concave_event_capacity(capacity)
    assert not is_convex_event_capacity(capacity)


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
