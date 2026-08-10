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
from capacities_ml.mobius import MobiusRepresentation
from capacities_ml.optimization import KAdditivity, Solver
from capacities_ml.models import ChoquetClassifier


# synthetic classification data
def build_data(
    n_observations: int = 50,
    seed: int = 22,
) -> tuple[pl.DataFrame, VariableUniverse, MobiusRepresentation]:
    """Create binary labels from a thresholded 2-additive Choquet score."""
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
    scores = batch_choquet_integral_mobius(X, true_mobius)
    labels = (scores >= 0.5).astype(int)

    data = pl.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "x3": X[:, 3],
            "label": labels,
        }
    )
    return data, universe, true_mobius


# classifier example
def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    data, universe, true_mobius = build_data()
    feature_columns = list(universe.var_names)
    X = data.select(feature_columns).to_numpy()
    y = data["label"].to_numpy()

    model = ChoquetClassifier(
        universe=universe,
        sparsity=KAdditivity(order=2),
        solver=Solver.PYMOO,
        solver_options={
            "population_size": 120,
            "n_generations": 150,
            "seed": 22,
            "verbose": False,
            "equality_tolerance": 1e-4,
        },
    ).fit(X, y)

    scores = model.decision_function(X)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    accuracy = float(np.mean(predictions == y))
    X_new = np.array(
        [
            [0.15, 0.20, 0.10, 0.25],
            [0.80, 0.75, 0.90, 0.70],
            [0.45, 0.60, 0.55, 0.50],
        ]
    )
    new_scores = model.decision_function(X_new)
    new_predictions = model.predict(X_new)

    print("Choquet linear classifier with synthetic data")
    print(f"Observations: {data.height}")
    print(f"Features: {universe.var_names}")
    print(f"Class counts: {np.bincount(y, minlength=2).tolist()}")
    print(f"Optimized parameters: {model.result_.parameters.size}")
    print(f"Threshold: {model.threshold_:.6f}")
    print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Solver success: {model.result_.success}")
    print(f"Maximum constraint violation: {model.result_.diagnostics['maximum_constraint_violation']:.6g}")
    print()
    print("First five rows of X")
    print(X[:5])
    print("First five labels of y")
    print(y[:5])
    print("First five scores, predictions and positive-class probabilities")
    print(np.column_stack((scores[:5], predictions[:5], probabilities[:5, 1])))
    print()
    print("True Mobius coefficients")
    print(true_mobius.to_named_dict())
    print()
    print("Fitted Mobius coefficients")
    print(model.capacity_.to_named_dict())
    print()
    print("New observations X_new")
    print(X_new)
    print("Scores for X_new")
    print(new_scores)
    print("Predicted classes for X_new")
    print(new_predictions)


if __name__ == "__main__":
    main()
