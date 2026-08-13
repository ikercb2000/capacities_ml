import numpy as np

from capacities_ml_fin.ml.models import ChoquetClassifier
from capacities_ml_fin.ml.optimization import L1Penalty, Solver


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

    penalty = L1Penalty(weight=0.001, selection=[0])
    model = ChoquetClassifier(
        solver=Solver.PYMOO,
        penalty=penalty,
        solver_options={
            "population_size": 30,
            "n_generations": 20,
            "seed": 4,
        },
    ).fit(X, y)

    assert model.result_.success
    assert 0.0 <= model.threshold_ <= 1.0
    assert model.problem_.parameter_layout.slice("threshold") == slice(4, 5)
    assert model.problem_.objective.penalty is penalty
    assert model.universe_.var_names == ("x0", "x1")


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
        solver=Solver.PYMOO,
        solver_options={"population_size": 20, "n_generations": 10, "seed": 5},
    ).fit(X, y)

    assert model.classes_.tolist() == ["negative", "positive"]
    assert set(model.predict(X)).issubset(set(model.classes_))
    assert model.predict_proba(X).shape == (4, 2)


def test_choquet_classifier_is_serializable(estimator_roundtrip):
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
        solver=Solver.PYMOO,
        solver_options={
            "population_size": 20,
            "n_generations": 10,
            "seed": 7,
        },
    ).fit(X, y)

    restored = estimator_roundtrip(model)

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_allclose(
        restored.decision_function(X),
        model.decision_function(X),
    )
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    assert restored.problem_.name == model.problem_.name
    assert restored.result_.diagnostics == model.result_.diagnostics
    assert restored.capacity_.to_named_dict() == model.capacity_.to_named_dict()
