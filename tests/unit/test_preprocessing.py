import numpy as np
import polars as pl

from capacities_ml.preprocessing import CapacityNormalizer


def test_capacity_normalizer_scales_and_reverses_cost_features():
    X = np.array(
        [
            [10.0, 100.0],
            [20.0, 50.0],
            [30.0, 0.0],
        ]
    )
    normalizer = CapacityNormalizer(cost_features=["cost"])
    normalizer.fit(
        pl.DataFrame(
            {
                "benefit": X[:, 0],
                "cost": X[:, 1],
            }
        )
    )

    transformed = normalizer.transform(X)

    np.testing.assert_allclose(transformed[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(transformed[:, 1], [0.0, 0.5, 1.0])


def test_capacity_normalizer_round_trips_values():
    X = np.array([[1.0, 100.0], [2.0, 50.0], [3.0, 0.0]])
    normalizer = CapacityNormalizer(cost_features=[1]).fit(X)

    np.testing.assert_allclose(normalizer.inverse_transform(normalizer.transform(X)), X)
