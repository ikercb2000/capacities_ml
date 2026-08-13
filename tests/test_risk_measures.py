import numpy as np
import pytest

from capacities_ml.capacities import (
    ExplicitCapacity,
    MobiusCapacity,
    VariableUniverse,
)
from capacities_ml.integrals.choquet import ordered_choquet
from capacities_ml.risk import (
    DistortionRiskMeasure,
    ExpectedShortfallDistortion,
    IdentityDistortion,
    ProbabilityCapacity,
    UpperEnvelopeCapacity,
    choquet_risk_measure,
    distortion_risk_measure,
    expected_shortfall,
    generalized_tail_value_at_risk,
    generalized_value_at_risk,
    risk_contributions,
    value_at_risk,
)


def test_identity_distortion_equals_weighted_mean_for_signed_losses():
    losses = np.array([-2.0, 1.0, 4.0])
    weights = np.array([0.2, 0.3, 0.5])

    risk = distortion_risk_measure(
        losses,
        IdentityDistortion(),
        sample_weight=weights,
    )

    assert risk == pytest.approx(np.average(losses, weights=weights))


def test_var_and_expected_shortfall_match_empirical_quantiles():
    losses = np.array([1.0, 2.0, 3.0, 4.0])

    assert value_at_risk(losses, 0.75) == 3.0
    assert expected_shortfall(losses, 0.5) == pytest.approx(3.5)


def test_generalized_risk_measures_reduce_to_classical_probability_case():
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    probability = ProbabilityCapacity(np.ones(losses.size))

    assert generalized_value_at_risk(losses, 0.75, probability) == 3.0
    assert generalized_tail_value_at_risk(losses, 0.5, probability) == pytest.approx(3.5)


def test_upper_envelope_generalized_var_is_worst_prior_var():
    losses = np.array([0.0, 10.0, 20.0])
    priors = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    capacity = UpperEnvelopeCapacity(priors)

    generalized = generalized_value_at_risk(losses, 0.5, capacity)

    assert generalized == 20.0


def test_contributions_sum_to_choquet_risk():
    losses = np.array([-1.0, 2.0, 5.0])
    capacity = ProbabilityCapacity([0.2, 0.3, 0.5])
    contributions = risk_contributions(losses, capacity)

    assert contributions["contribution"].sum() == pytest.approx(
        choquet_risk_measure(losses, capacity)
    )
    assert DistortionRiskMeasure(ExpectedShortfallDistortion(0.5))(losses) == pytest.approx(
        expected_shortfall(losses, 0.5)
    )


def test_risk_measure_accepts_original_tabular_capacity():
    capacity = ExplicitCapacity(
        universe=VariableUniverse(("scenario_1", "scenario_2", "scenario_3")),
        values={
            ("scenario_1",): 0.2,
            ("scenario_2",): 0.3,
            ("scenario_3",): 0.5,
            ("scenario_1", "scenario_2"): 0.5,
            ("scenario_1", "scenario_3"): 0.7,
            ("scenario_2", "scenario_3"): 0.8,
            ("scenario_1", "scenario_2", "scenario_3"): 1.0,
        },
    )
    losses = np.array([0.1, 0.6, 0.8])

    assert choquet_risk_measure(losses, capacity) == pytest.approx(
        ordered_choquet(capacity, losses)
    )


def test_risk_measure_accepts_mobius_capacity_without_materialization():
    capacity = MobiusCapacity(
        universe=VariableUniverse(("scenario_1", "scenario_2", "scenario_3")),
        coefficients={
            ("scenario_1",): 0.2,
            ("scenario_2",): 0.3,
            ("scenario_3",): 0.5,
        },
    )
    losses = np.array([0.1, 0.6, 0.8])

    assert choquet_risk_measure(losses, capacity) == pytest.approx(0.6)
