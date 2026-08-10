import numpy as np

from capacities_ml.capacities import VariableUniverse
from capacities_ml.models import ChoquetClassifier
from capacities_ml.optimization import Solver


def test_choquet_classifier_optimizes_capacity_and_threshold():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = ChoquetClassifier(
        universe=VariableUniverse(("x0", "x1")),
        solver=Solver.PYMOO,
        solver_options={
            "population_size": 30,
            "n_generations": 20,
            "seed": 4,
        },
    ).fit(X, y)

    assert model.result_.success
    assert 0.0 <= model.threshold_ <= 1.0
    assert model.problem_.parameter_layout.slice("threshold") == slice(4, 5)


def test_choquet_classifier_accepts_arbitrary_binary_labels():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array(["negative", "negative", "positive", "positive"])

    model = ChoquetClassifier(
        universe=VariableUniverse(("x0", "x1")),
        solver=Solver.PYMOO,
        solver_options={"population_size": 20, "n_generations": 10, "seed": 5},
    ).fit(X, y)

    assert model.classes_.tolist() == ["negative", "positive"]
    assert set(model.predict(X)).issubset(set(model.classes_))
    assert model.predict_proba(X).shape == (4, 2)
