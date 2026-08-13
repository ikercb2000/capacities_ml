import pickle

import numpy as np
import pytest

from capacities_ml_fin.base.capacities import validate_capacity
from capacities_ml_fin.risk import (
    DistortedCapacity,
    EmpiricalLossDistribution,
    ExpectedShortfallDistortion,
    LowerEnvelopeCapacity,
    ProbabilityCapacity,
    UpperEnvelopeCapacity,
)


def test_empirical_distribution_supports_weighted_lower_and_upper_quantiles():
    distribution = EmpiricalLossDistribution(
        [1.0, 2.0, 3.0],
        sample_weight=[0.2, 0.3, 0.5],
    )

    assert distribution.distribution(2.0) == pytest.approx(0.5)
    assert distribution.survival(2.0) == pytest.approx(0.5)
    assert distribution.lower_quantile(0.5) == 2.0
    assert distribution.upper_quantile(0.5) == 3.0


def test_probability_and_distorted_capacities_evaluate_event_masks():
    probability = ProbabilityCapacity([1.0, 2.0, 1.0])
    distorted = DistortedCapacity(probability, ExpectedShortfallDistortion(0.5))

    with pytest.raises(AttributeError, match="immutable"):
        probability.weights = np.ones(3)
    with pytest.raises(ValueError, match="read-only"):
        probability.weights[0] = 0.9
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        probability.weights.setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        distorted.base_capacity = ProbabilityCapacity([1.0, 1.0, 1.0])

    assert probability.event_value([True, False, True]) == pytest.approx(0.5)
    assert distorted.event_value([True, False, False]) == pytest.approx(0.5)
    validate_capacity(distorted)

    restored = pickle.loads(pickle.dumps(distorted))
    assert restored.event_value([True, False, False]) == pytest.approx(0.5)
    with pytest.raises(AttributeError, match="immutable"):
        restored.distortion = ExpectedShortfallDistortion(0.75)


def test_probability_envelopes_capture_model_uncertainty():
    priors = np.array([[0.8, 0.2], [0.3, 0.7]])
    upper = UpperEnvelopeCapacity(priors)
    lower = LowerEnvelopeCapacity(priors)

    with pytest.raises(ValueError, match="read-only"):
        upper.prior_weights[0, 0] = 0.1
    with pytest.raises(ValueError, match="read-only"):
        lower.prior_weights[0, 0] = 0.1

    assert upper.event_value([True, False]) == pytest.approx(0.8)
    assert lower.event_value([True, False]) == pytest.approx(0.3)
