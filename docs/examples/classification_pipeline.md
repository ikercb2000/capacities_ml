# Example: deterministic and probabilistic Choquet classification

This example highlights the difference between the package's two binary classification APIs.

## Shared preprocessing

```python
from sklearn.pipeline import Pipeline

from capacities_ml_fin.ml.models import (
    ChoquetClassifier,
    ChoquisticRegression,
)
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer
```

Assume `X_train` has benefit criteria plus a cost criterion named `volatility`.

## Deterministic threshold classifier

```python
hard_pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquetClassifier(
                sparsity=KAdditivity(order=2),
                learn_feature_scales=True,
                solver="pymoo",
                solver_options={"seed": 5},
            ),
        ),
    ]
).set_output(transform="pandas")

hard_pipeline.fit(X_train, y_train)
hard_labels = hard_pipeline.predict(X_test)
```

Inspect the learned threshold and feature scales:

```python
hard_model = hard_pipeline.named_steps["model"]
print(hard_model.threshold_)
print(hard_model.feature_scales_)
print(hard_model.decision_function(X_test))
```

`decision_function` returns the signed difference between the scaled Choquet score and the learned threshold.

## Probabilistic Choquistic regression

```python
prob_pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        (
            "model",
            ChoquisticRegression(
                sparsity=KAdditivity(order=2),
                solver="scipy",
                class_weight="balanced",
            ),
        ),
    ]
).set_output(transform="pandas")

prob_pipeline.fit(X_train, y_train)
probabilities = prob_pipeline.predict_proba(X_test)
prob_labels = prob_pipeline.predict(X_test)
```

Inspect the latent utility and link parameters:

```python
prob_model = prob_pipeline.named_steps["model"]
utility = prob_model.utility_function(
    prob_pipeline.named_steps["normalize"].transform(X_test)
)
print(prob_model.gamma_)
print(prob_model.beta_)
```

## Interpret both capacities

```python
from capacities_ml_fin.base.interpretation import (
    shapley_indices,
    pairwise_interactions,
)

print("hard model")
print(shapley_indices(hard_model.capacity_))
print(pairwise_interactions(hard_model.capacity_))

print("probabilistic model")
print(shapley_indices(prob_model.capacity_))
print(pairwise_interactions(prob_model.capacity_))
```

## Which output should be reported?

For `ChoquetClassifier`, report classification metrics, threshold, feature scales and capacity interpretation. Do not treat margins as calibrated probabilities.

For `ChoquisticRegression`, probability-sensitive metrics such as log loss or Brier score are meaningful in addition to classification metrics because the model explicitly defines a logistic probability link.
