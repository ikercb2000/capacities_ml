# imports
import numpy as np

# modules
from capacities_ml.capacities import CapacityMap, CoalitionValue, KAdditiveCapacity
from capacities_ml.integrals.batch_integrals import (
    batch_choquet_integral,
    batch_choquet_integral_mobius,
)
from capacities_ml.integrals.choquet import mobius_choquet, ordered_choquet

# main function example
def main():
    capacity = KAdditiveCapacity(
        k=2,
        subset_values=CapacityMap(
            capacities=[
                CoalitionValue(frozenset({0}), 0.2),
                CoalitionValue(frozenset({1}), 0.3),
                CoalitionValue(frozenset({2}), 0.1),
                CoalitionValue(frozenset({0, 1}), 0.6),
                CoalitionValue(frozenset({0, 2}), 0.45),
                CoalitionValue(frozenset({1, 2}), 0.55),
                CoalitionValue(frozenset({0, 1, 2}), 1.0),
            ]
        ),
        n_features=3,
        feature_names=("x0", "x1", "x2"),
    )
    x = np.array([0.3, 0.8, 0.5])
    X = np.array(
        [
            [0.3, 0.8, 0.5],
            [0.6, 0.2, 0.9],
            [0.4, 0.4, 0.7],
        ]
    )

    capacity.validate()

    choquet_value = ordered_choquet(capacity, x)
    mobius_value = mobius_choquet(capacity, x)
    batch_values = batch_choquet_integral(X, capacity)
    batch_mobius_values = batch_choquet_integral_mobius(X, capacity)

    print("2-additive capacity with n=3")
    print(f"x = {x}")
    print(f"Choquet integral = {choquet_value:.6f}")
    print(f"Möbius Choquet integral = {mobius_value:.6f}")
    print()
    print("Batch input X:")
    print(X)
    print(f"Batch Choquet values = {batch_values}")
    print(f"Batch Möbius Choquet values = {batch_mobius_values}")


if __name__ == "__main__":
    main()
