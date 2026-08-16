# Choquet model aggregation

Prediction aggregation learns a capacity whose elements are fitted models, not
the original explanatory variables:

\[
(\hat y_1,\ldots,\hat y_K)
\longrightarrow C_\mu
\longrightarrow \hat y_{\mathrm{final}}.
\]

Rows correspond to observations and columns correspond to source models. Fit
predictions must be genuinely out-of-sample with respect to those source
models—for example, cross-validated or earlier validation predictions—to avoid
training the aggregator on overly optimistic in-sample outputs.

## Regression

```python
from capacities_ml_fin.ml.aggregation import aggregate_regression_predictions
from capacities_ml_fin.ml.optimization import KAdditivity

result = aggregate_regression_predictions(
    fit_predictions,
    y_fit,
    test_predictions,
    sparsity=KAdditivity(order=2),
)

final_prediction = result.predictions
capacity = result.capacity
```

Every input column must use the target's prediction scale. Internally,
`ChoquetRegressor(fit_intercept=False)` is used, so the final values are exactly

\[
C_\mu(\hat y_1,\ldots,\hat y_K)
\]

without an additive intercept. `KAdditivity(order=1)` gives an additive
monotone ensemble, while `order=2` permits pairwise complementarity and
redundancy. Existing penalties can be passed through `penalty`.

## Binary probabilities

```python
from capacities_ml_fin.ml.aggregation import aggregate_binary_probabilities

result = aggregate_binary_probabilities(
    fit_positive_probabilities,
    y_fit,
    test_positive_probabilities,
    sparsity=KAdditivity(order=2),
    class_weight="balanced",
)

positive_probability = result.probabilities
print(result.positive_class)
```

Each column must contain the probability of the same positive class and all
values must lie in `[0, 1]`. This first implementation uses
`ChoquisticRegression`; multiclass aggregation and specialized literature
methods are intentionally outside its scope.

## Named models and interpretation

DataFrame columns become the capacity universe. Fit and prediction DataFrames
must contain the same names in the same order.

```python
from capacities_ml_fin.base.interpretation import (
    pairwise_interactions,
    shapley_indices,
)

importance = shapley_indices(result.capacity)
interactions = pairwise_interactions(result.capacity)
```

The returned Shapley values and interactions therefore describe models:

- a Shapley value measures a source model's importance in the ensemble;
- a positive interaction indicates complementarity between two models;
- a negative interaction indicates redundancy.

Both result objects also expose `model_names`, `optimization_result`, and
`fitted_model` for diagnostics and reproducibility.

