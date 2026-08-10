# imports
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# modules
from capacities_ml.capacities import VariableUniverse
from capacities_ml.mobius import inverse_mobius_transform
from capacities_ml.models import ChoquisticRegression
from capacities_ml.optimization import KAdditivity, Solver


# choquistic regression example
def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # Rows are observations and columns are normalized criteria.
    X = np.array(
        [
            [0.10, 0.20, 0.15],
            [0.20, 0.15, 0.30],
            [0.25, 0.40, 0.20],
            [0.35, 0.30, 0.45],
            [0.40, 0.55, 0.35],
            [0.50, 0.45, 0.60],
            [0.55, 0.70, 0.50],
            [0.65, 0.60, 0.75],
            [0.70, 0.80, 0.65],
            [0.80, 0.70, 0.90],
            [0.90, 0.85, 0.80],
            [0.95, 0.90, 0.95],
        ]
    )
    y = np.array(
        ["negative", "negative", "negative", "negative", "positive", "negative",
         "positive", "positive", "negative", "positive", "positive", "positive"]
    )
    universe = VariableUniverse(("profitability", "liquidity", "solvency"))

    model = ChoquisticRegression(
        universe=universe,
        sparsity=KAdditivity(order=2),
        solver=Solver.SCIPY,
        solver_options={"options": {"maxiter": 2_000, "ftol": 1e-12}},
        class_weight="balanced",
    ).fit(X, y)

    utilities = model.utility_function(X)
    logits = model.decision_function(X)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    accuracy = float(np.mean(predictions == y))
    fitted_capacity = inverse_mobius_transform(model.capacity_)

    X_new = np.array(
        [
            [0.30, 0.25, 0.35],
            [0.60, 0.65, 0.55],
            [0.85, 0.75, 0.90],
        ]
    )

    print("Choquistic regression with several observations")
    print(f"Features: {universe.var_names}")
    print("X")
    print(X)
    print("y")
    print(y)
    print()
    print(f"Learned beta threshold: {model.beta_:.6f}")
    print(f"Learned gamma scale: {model.gamma_:.6f}")
    print(f"Training accuracy: {accuracy:.4f}")
    print("Utility, logit, positive probability and correct prediction")
    print(np.column_stack((utilities, logits, probabilities, predictions == y)))
    print()
    print("Fitted Mobius coefficients")
    print(model.capacity_.to_named_dict())
    print("Fitted capacity values")
    print(fitted_capacity.to_named_dict())
    print()
    print("New observations X_new")
    print(X_new)
    print("Positive probabilities for X_new")
    print(model.predict_proba(X_new)[:, 1])
    print("Predicted classes for X_new")
    print(model.predict(X_new))


if __name__ == "__main__":
    main()
