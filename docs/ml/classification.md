# Classification

The package provides two conceptually different binary Choquet classifiers:

1. `ChoquetClassifier` — a deterministic threshold classifier optimized directly for classification decisions;
2. `ChoquisticRegression` — a probabilistic model in which a Choquet utility is passed through a logistic link.

Both expect criteria in $[0,1]$ and assume that larger values favor the positive class.

## Choosing between the two

| Property | `ChoquetClassifier` | `ChoquisticRegression` |
|---|---|---|
| Binary only | yes | yes |
| Requires normalized/oriented inputs | yes | yes |
| Predicts hard labels | yes | yes |
| `decision_function()` | threshold margin | fitted log-odds |
| `predict_proba()` | intentionally absent | yes |
| Objective | direct 0–1 loss | logistic negative log-likelihood |
| Default solver | PYMOO | SciPy only |
| Optional learned feature scales | yes | no |

## Deterministic `ChoquetClassifier`

The classifier first forms a Choquet projection and compares it with a learned threshold:

\[
\widehat y
=
\mathbf 1\left\{
C_\mu(s\odot x)\ge \tau
\right\},
\]

where $s$ is an optional vector of non-negative feature scales and $\tau\in[0,1]$ is fitted jointly with the capacity.

```python
from capacities_ml_fin.ml.models import ChoquetClassifier
from capacities_ml_fin.ml.optimization import KAdditivity

classifier = ChoquetClassifier(
    sparsity=KAdditivity(order=2),
    learn_feature_scales=True,
    solver="pymoo",
    solver_options={"seed": 7},
)
classifier.fit(X_train, y_train)
```

### Scores and margins

```python
choquet_scores = classifier.choquet_score(X_test)
margins = classifier.decision_function(X_test)
labels = classifier.predict(X_test)
```

The margin is

\[
C_\mu(s\odot x)-\tau.
\]

The model deliberately has no `predict_proba()`: a signed distance from a fitted threshold is not automatically a probability model.

### Fitted attributes

- `capacity_`
- `threshold_`
- `feature_scales_`
- `classes_`
- `problem_`
- `result_`
- `sparsity_`

Feature scales are normalized after optimization so that their maximum equals one.

## Probabilistic `ChoquisticRegression`

`ChoquisticRegression` follows the two-stage utility/link structure implemented from the cited Choquistic formulation. Its latent utility is

\[
u(x)=C_\mu(x),
\]

and its fitted log-odds are

\[
\eta(x)
=
\gamma\left(C_\mu(x)-\beta\right),
\qquad \gamma>0,
\quad \beta\in[0,1].
\]

The positive-class probability is

\[
P(Y=1\mid x)=\sigma(\eta(x)).
\]

```python
from capacities_ml_fin.ml.models import ChoquisticRegression

model = ChoquisticRegression(
    sparsity=KAdditivity(order=2),
    solver="scipy",
)
model.fit(X_train, y_train)
```

### Inspect each stage

```python
utility = model.utility_function(X_test)
log_odds = model.decision_function(X_test)
probabilities = model.predict_proba(X_test)
labels = model.predict(X_test)
```

The fitted parameters are exposed as:

```python
model.capacity_
model.gamma_
model.beta_
```

### Class and sample weights

`class_weight` follows the scikit-learn class-weight convention and `fit()` additionally accepts `sample_weight`.

```python
model = ChoquisticRegression(class_weight="balanced")
model.fit(X_train, y_train, sample_weight=weights)
```

The two sources of weight are multiplied.

## Recommended preprocessing pipeline

```python
from sklearn.pipeline import Pipeline
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", ChoquisticRegression(sparsity=KAdditivity(order=2))),
    ]
).set_output(transform="pandas")
```

Because both classification models explicitly validate the unit interval, normalization is not merely cosmetic.

## Model selection

Use stratified cross-validation for i.i.d. classification problems:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
search = GridSearchCV(
    pipeline,
    capacity_parameter_grid(
        parameter_name="model__sparsity",
        orders=(1, 2),
    ),
    scoring="balanced_accuracy",
    cv=cv,
)
```

For temporal classification, use a chronology-preserving splitter instead.

## Interpretation

Both models expose `capacity_`, so the same exact interpretation API applies:

```python
from capacities_ml_fin.base.interpretation import shapley_indices, pairwise_interactions

shapley_indices(model.capacity_)
pairwise_interactions(model.capacity_)
```

For `ChoquetClassifier`, the learned feature scales are an additional source of model structure and should be reported separately from capacity importance.

## API

- [`ChoquetClassifier`](../api/ml_models.md#choquetclassifier)
- [`ChoquisticRegression`](../api/ml_models.md#choquisticregression)
