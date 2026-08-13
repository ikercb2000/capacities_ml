# Preprocessing and model selection

Capacity models are especially sensitive to **criterion scale**, **criterion direction**, and **interaction complexity**. The package therefore provides a dedicated transformer and capacity-aware model-selection grids.

## `CapacityNormalizer`

The transformer learns per-feature minimum and maximum values and maps them to `feature_range`, which defaults to $(0,1)$.

```python
from capacities_ml_fin.ml.preprocessing import CapacityNormalizer

normalizer = CapacityNormalizer()
X_scaled = normalizer.fit_transform(X_train)
```

For feature $j$, ordinary min-max scaling is used. Constant training features use a safe unit denominator rather than dividing by zero.

### Cost criteria

A capacity model is easiest to interpret when larger transformed values always mean “more” of the common preferred direction. Cost criteria can be reversed after scaling:

```python
normalizer = CapacityNormalizer(
    cost_features=["volatility", "drawdown"],
)
```

If the fitted feature range is $[a,b]$, a cost feature value $z$ after scaling is transformed to

\[
a+b-z.
\]

String cost-feature names require named input columns. Alternatively, integer indices can be supplied.

### Clipping

With the default `clip=True`, out-of-range values encountered during transform are clipped to the configured interval. This prevents extrapolated min-max values from leaving the capacity input range.

### Inverse transform

```python
X_original = normalizer.inverse_transform(X_scaled)
```

The cost reversal and scaling are both undone.

## Keep preprocessing inside the pipeline

Do not fit normalization on the complete dataset before cross-validation. Use a scikit-learn `Pipeline` so each fold estimates preprocessing only from its training subset.

```python
from sklearn.pipeline import Pipeline
from capacities_ml_fin.ml.models import ChoquetRegressor

pipeline = Pipeline(
    [
        ("normalize", CapacityNormalizer(cost_features=["volatility"])),
        ("model", ChoquetRegressor()),
    ]
).set_output(transform="pandas")
```

## Capacity order grids

`capacity_sparsity_grid()` creates a list of `KAdditivity` candidates:

```python
from capacities_ml_fin.ml.model_selection import capacity_sparsity_grid

candidates = capacity_sparsity_grid((1, 2, 3))
```

`capacity_parameter_grid()` wraps the same candidates in a dictionary suitable for `GridSearchCV`:

```python
from capacities_ml_fin.ml.model_selection import capacity_parameter_grid

param_grid = capacity_parameter_grid(
    parameter_name="model__sparsity",
    orders=(1, 2, 3),
)
```

## Capacity shape grids

```python
from capacities_ml_fin.ml.model_selection import capacity_shape_grid

candidates = capacity_shape_grid(order=2)
```

The default comparison includes general, convex and concave 2-additive capacities.

## Pairwise interaction grids

To compare models where selected interactions are fixed:

```python
from capacities_ml_fin.ml.model_selection import pairwise_interaction_grid

candidates = pairwise_interaction_grid(
    orders=(2,),
    pairs=((0, 1),),
    target=0.0,
)
```

This is useful for testing whether suppressing a particular interaction harms predictive performance.

## Choosing the cross-validation splitter

The grid helpers only build model parameters; they do not decide how data should be split.

- For i.i.d. regression: `KFold` or a problem-appropriate alternative.
- For binary i.i.d. classification: `StratifiedKFold`.
- For financial/temporal prediction: chronological splitters such as scikit-learn `TimeSeriesSplit` or sktime forecasting splitters.

The split design is part of the empirical model and should match the prediction setting.

## Feature names through pipelines

`CapacityNormalizer.get_feature_names_out()` preserves names. With `.set_output(transform="pandas")`, the downstream Choquet estimator can retain DataFrame column names in its fitted `VariableUniverse`. This makes `capacity_`, Shapley values and interaction outputs easier to interpret.

## API

See [Preprocessing & model-selection API](../api/preprocessing_model_selection.md).
