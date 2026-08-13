import numpy as np
import pytest
from sklearn.base import clone

from capacities_ml_fin.base.integrals.batch_integrals import batch_choquet_integral
from capacities_ml_fin.ml.models import ChoquetClassifier
from capacities_ml_fin.ml.optimization import KAdditivity, L1Penalty, Solver


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
    assert model.problem_.parameter_layout.slice("feature_scales") == slice(4, 6)
    assert model.problem_.parameter_layout.slice("threshold") == slice(6, 7)
    assert np.all((0.0 <= model.feature_scales_) & (model.feature_scales_ <= 1.0))
    assert np.max(model.feature_scales_) == pytest.approx(1.0)
    np.testing.assert_allclose(
        model.decision_function(X),
        batch_choquet_integral(X * model.feature_scales_, model.capacity_)
        - model.threshold_,
    )
    assert model.decision_function([[0.2, 0.2]])[0] <= (
        model.decision_function([[0.8, 0.8]])[0]
    )
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
    assert not hasattr(model, "predict_proba")
    np.testing.assert_array_equal(
        model.predict(X),
        model.classes_[(model.decision_function(X) >= 0.0).astype(int)],
    )
    np.testing.assert_allclose(
        model.decision_function(X),
        model.choquet_score(X) - model.threshold_,
    )


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
    np.testing.assert_allclose(restored.feature_scales_, model.feature_scales_)
    assert restored.problem_.name == model.problem_.name
    assert restored.result_.diagnostics == model.result_.diagnostics
    assert restored.capacity_.to_named_dict() == model.capacity_.to_named_dict()


def test_choquet_classifier_can_disable_feature_scale_learning():
    model = ChoquetClassifier(learn_feature_scales=False)

    assert clone(model).learn_feature_scales is False

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model.set_params(
        solver=Solver.PYMOO,
        solver_options={"population_size": 20, "n_generations": 10, "seed": 8},
    ).fit(X, y)

    assert model.feature_scales_.tolist() == [1.0, 1.0]
    assert all(
        block.name != "feature_scales"
        for block in model.problem_.parameter_layout.blocks
    )


def test_choquet_classifier_requires_unit_interval_inputs():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = ChoquetClassifier(
        solver=Solver.PYMOO,
        solver_options={"population_size": 20, "n_generations": 10, "seed": 9},
    )

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.fit(2.0 * X, y)

    model.fit(X, y)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.predict(2.0 * X)


def test_pymoo_repairs_an_approximately_feasible_mobius_capacity():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 1.0],
            [0.3, 0.8, 0.4],
            [0.8, 0.3, 0.6],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = ChoquetClassifier(
        sparsity=KAdditivity(2),
        solver="pymoo",
        solver_options={
            "population_size": 50,
            "n_generations": 40,
            "seed": 23,
            "equality_tolerance": 1e-4,
        },
    ).fit(X, y)

    assert model.result_.success
    assert model.result_.diagnostics["maximum_constraint_violation"] <= 1e-10
    assert sum(model.capacity_.to_named_dict().values()) == pytest.approx(
        1.0,
        abs=1e-9,
    )
