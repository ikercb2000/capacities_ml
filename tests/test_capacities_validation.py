from capacities_ml_fin.base.capacities import (
    is_concave_capacity,
    is_convex_capacity,
    validate_capacity,
)
from capacities_ml_fin.risk import (
    DistortedCapacity,
    ProbabilityCapacity,
    ProportionalHazardsDistortion,
)


def test_concave_distortion_of_probability_is_concave_capacity():
    probability = ProbabilityCapacity([0.2, 0.3, 0.5])
    capacity = DistortedCapacity(probability, ProportionalHazardsDistortion(0.5))

    validate_capacity(capacity)
    assert is_concave_capacity(capacity)
    assert not is_convex_capacity(capacity)
