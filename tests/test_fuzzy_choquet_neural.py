import numpy as np
import pandas as pd

from capacities_ml_fin.ml.models import (
    FuzzyChoquetInputLayer,
    FuzzyChoquetNeuralAutoRegressor,
    FuzzyChoquetNeuralClassifier,
    FuzzyChoquetNeuralRegressor,
)


def test_fuzzy_choquet_input_layer_and_regressor_are_staged_and_two_additive():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(80, 3))
    y = 0.5 * X[:, 0] + 0.3 * np.minimum(X[:, 0], X[:, 1]) + 0.1

    layer = FuzzyChoquetInputLayer().fit(X, y)
    model = FuzzyChoquetNeuralRegressor(
        hidden_layer_sizes=(4,), max_iter=300, random_state=0
    ).fit(X, y)

    assert layer.transform(X).shape == (80, 1)
    assert layer.sparsity_.order == 2
    layer.capacity_.validate()
    assert model.fuzzy_transform(X).shape == (80, 1)
    assert np.mean((model.predict(X) - y) ** 2) < 1e-3
    model.capacity_.validate()


def test_fuzzy_choquet_neural_classifier_exposes_probabilities_and_capacity():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(160, 2))
    y = np.where(X[:, 0] + X[:, 1] > 1.0, "high", "low")
    model = FuzzyChoquetNeuralClassifier(
        hidden_layer_sizes=(10,), max_iter=600, random_state=0
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (160, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(model.predict(X) == y) > 0.85
    model.capacity_.validate()


def test_fuzzy_choquet_neural_autoregressor_selects_lags_chronologically():
    values = [0.2, 0.25]
    for _ in range(38):
        values.append(0.05 + 0.75 * values[-1] + 0.1 * values[-2])
    series = pd.Series(
        values,
        index=pd.period_range("2020-01", periods=len(values), freq="M"),
        name="response",
    )
    model = FuzzyChoquetNeuralAutoRegressor(
        lag_candidates=(2, 3),
        validation_fraction=0.2,
        hidden_layer_sizes=(3,),
        max_iter=300,
        random_state=0,
    ).fit(series)

    assert model.best_lag_ in {2, 3}
    assert set(model.lag_scores_["lags"]) == {2, 3}
    assert model.lag_scores_["validation_observations"].nunique() == 1
    assert model.predict(fh=[1, 2]).shape == (2,)
    assert model.fittedvalues_.shape == (len(series) - model.best_lag_,)
    model.capacity_.validate()
