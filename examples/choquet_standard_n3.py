
# imports
from pathlib import Path
import sys
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# modules
from capacities_ml.capacities import Capacity, VariableUniverse
from capacities_ml.mobius import mobius_transform
from capacities_ml.integrals.batch_integrals import (
    batch_choquet_integral,
    batch_choquet_integral_mobius,
)
from capacities_ml.integrals.choquet import mobius_choquet, ordered_choquet

# main function example
def main():
    universe = VariableUniverse(
        var_names=("price", "quality", "reliability")
    )
    capacity = Capacity(
        universe=universe,
        values={
            "price": 0.2,
            "quality": 0.3,
            "reliability": 0.4,
            ("price", "quality"): 0.6,
            ("price", "reliability"): 0.7,
            ("quality", "reliability"): 0.8,
            ("price", "quality", "reliability"): 1.0,
        },
    )
    x = np.array([0.3, 0.8, 0.5])
    X = np.array(
        [
            [0.3, 0.8, 0.5],
            [0.6, 0.2, 0.9],
            [0.4, 0.4, 0.7],
        ]
    )

    mobius_rep = mobius_transform(capacity)
    choquet_value = ordered_choquet(capacity, x)
    mobius_value = mobius_choquet(mobius_rep, x)
    batch_values = batch_choquet_integral(X, capacity)
    batch_mobius_values = batch_choquet_integral_mobius(X, mobius_rep)
    named_mobius = {
        coalition: round(value, 6)
        for coalition, value in mobius_rep.to_named_dict().items()
    }

    print("Standard capacity with n=3")
    print(f"Variables = {capacity.var_names}")
    print(f"Möbius coefficients = {named_mobius}")
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
