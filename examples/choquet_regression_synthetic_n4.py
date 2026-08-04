# imports
from pathlib import Path
import sys

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# modules
from capacities_ml.capacities import VariableUniverse
from capacities_ml.integrals.batch_integrals import batch_choquet_integral_mobius
from capacities_ml.mobius import MobiusRepresentation, inverse_mobius_transform
from capacities_ml.optimization import KAdditivity, Solver
from capacities_ml.models import ChoquetRegressor


# synthetic regression data
def build_data(
    n_observations: int = 50,
    seed: int = 21,
) -> tuple[pl.DataFrame, VariableUniverse, MobiusRepresentation]:
    """Create four-feature data from a normalized 2-additive capacity."""
    rng = np.random.default_rng(seed)
    universe = VariableUniverse(("x0", "x1", "x2", "x3"))
    X = rng.uniform(0.0, 1.0, size=(n_observations, universe.n_vars))

    coefficients = {
        frozenset({0}): 0.15,
        frozenset({1}): 0.15,
        frozenset({2}): 0.15,
        frozenset({3}): 0.15,
        frozenset({0, 1}): 0.0666666667,
        frozenset({0, 2}): 0.0666666667,
        frozenset({0, 3}): 0.0666666667,
        frozenset({1, 2}): 0.0666666667,
        frozenset({1, 3}): 0.0666666667,
        frozenset({2, 3}): 0.0666666667,
    }
    true_mobius = MobiusRepresentation(universe, coefficients)
    target = 0.35 + batch_choquet_integral_mobius(X, true_mobius)
    target += rng.normal(0.0, 0.02, size=n_observations)

    data = pl.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "x3": X[:, 3],
            "target": target,
        }
    )
    return data, universe, true_mobius


# regression example
def main() -> None:
    data, universe, true_mobius = build_data()
    feature_columns = list(universe.var_names)
    X = data.select(feature_columns).to_numpy()
    y = data["target"].to_numpy()

    model = ChoquetRegressor(
        universe=universe,
        sparsity=KAdditivity(order=2),
        solver=Solver.SCIPY,
        solver_options={"options": {"maxiter": 1_000, "ftol": 1e-12}},
    ).fit(X, y)

    predictions = model.predict(X)
    mse = float(np.mean((predictions - y) ** 2))
    print("Choquet regression with synthetic data")
    print(f"Observations: {data.height}")
    print(f"Features: {universe.var_names}")
    print(f"Optimized parameters: {model.result_.parameters.size}")
    print(f"Intercept: {model.intercept_:.6f}")
    print(f"Training MSE: {mse:.6f}")
    print(f"Solver success: {model.result_.success}")
    print()
    print("First three observations")
    print(data.head(3).write_csv())
    print()
    print("True Mobius coefficients")
    print(true_mobius.to_named_dict())
    print()
    print("Fitted Mobius coefficients")
    print(model.capacity_.to_named_dict())
    print()
    print("Fitted capacity values")
    print(inverse_mobius_transform(model.capacity_).to_named_dict())


if __name__ == "__main__":
    main()
