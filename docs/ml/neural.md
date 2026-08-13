# Neural Choquet models

The neural estimators add a nonlinear hidden layer while retaining a capacity inside each hidden unit. They are intended for problems where a single Choquet integral is too restrictive but capacity-based aggregation remains useful as an interpretable building block.

## Architecture

For hidden unit $h$, the model computes a Choquet aggregate

\[
z_h=C_{\mu_h}(x)+b_h,
\]

applies the selected activation,

\[
h_h=\sigma(z_h),
\]

and combines hidden activations linearly at the output.

The regressor uses a continuous output. The classifier interprets the output as a logit and applies a logistic function for `predict_proba()`.

## `ChoquetNeuralRegressor`

```python
from capacities_ml_fin.ml.models import ChoquetNeuralRegressor
from capacities_ml_fin.ml.optimization import KAdditivity

model = ChoquetNeuralRegressor(
    n_hidden=4,
    activation="tanh",
    sparsity=KAdditivity(order=2),
    alpha=1e-4,
    max_iter=300,
    random_state=0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The most important fitted attributes include:

- `capacities_` — one decoded capacity per hidden unit;
- `capacity_parameters_` — numerical capacity parameter matrix;
- `problem_` and `result_` — optimization diagnostics;
- `loss_` — final objective value;
- `n_iter_` — number of optimization iterations when available.

## `ChoquetNeuralClassifier`

```python
from capacities_ml_fin.ml.models import ChoquetNeuralClassifier

classifier = ChoquetNeuralClassifier(
    n_hidden=4,
    activation="tanh",
    sparsity=KAdditivity(order=2),
    class_weight="balanced",
    random_state=0,
)
classifier.fit(X_train, y_train)

logits = classifier.decision_function(X_test)
probabilities = classifier.predict_proba(X_test)
labels = classifier.predict(X_test)
```

The classifier is binary and exposes `classes_` after fitting.

## Capacity interpretation in a neural model

There is no single `capacity_` because each hidden neuron has its own capacity:

```python
from capacities_ml_fin.base.interpretation import shapley_indices

for hidden_index, capacity in enumerate(model.capacities_):
    print(hidden_index, shapley_indices(capacity))
```

This interpretation is at the hidden-unit level. The final network output also depends on activation values, output weights and biases, so a capacity's Shapley indices should not be presented as a complete local explanation of the entire neural network.

## Regularization

The `alpha` hyperparameter controls the neural model's internal regularization. The capacity family itself can additionally be restricted through `sparsity`, for example with `KAdditivity(order=2)`.

These mechanisms control different types of complexity:

- `sparsity` limits the structural interaction order of each hidden capacity;
- `n_hidden` controls network width;
- `alpha` regularizes numerical network parameters.

## Optimization behavior

The current implementation builds a constrained numerical problem and solves it through SciPy/SLSQP. If optimization does not converge, fitting emits a scikit-learn `ConvergenceWarning` and still exposes the resulting fitted state for inspection.

`random_state` controls the stochastic initialization. Estimator tags mark the model as non-deterministic when no seed is supplied.

## Sample weighting

Both neural estimators accept `sample_weight` in `fit()`. The classifier also accepts `class_weight`; class and sample weights are combined.

## When to use neural Choquet models

Use a single `ChoquetRegressor` or `ChoquisticRegression` when global interpretability and a direct capacity representation are primary. Use a neural Choquet model when:

- one global monotone Choquet surface is too restrictive;
- interactions may differ across hidden representations;
- predictive flexibility is more important than having one globally interpretable capacity.

## API

- [`ChoquetNeuralRegressor`](../api/ml_models.md#choquetneuralregressor)
- [`ChoquetNeuralClassifier`](../api/ml_models.md#choquetneuralclassifier)
