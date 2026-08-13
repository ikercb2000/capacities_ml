# Preprocessing and model-selection API

## `CapacityNormalizer`

Min-max transformer with optional cost-criterion reversal. It is scikit-learn compatible and preserves feature names through `get_feature_names_out()`.

::: capacities_ml_fin.ml.preprocessing.normalization.CapacityNormalizer

## Capacity sparsity grids

### `capacity_sparsity_grid`

::: capacities_ml_fin.ml.model_selection.grids.capacity_sparsity_grid

### `capacity_shape_grid`

::: capacities_ml_fin.ml.model_selection.grids.capacity_shape_grid

### `interaction_order_grid`

::: capacities_ml_fin.ml.model_selection.grids.interaction_order_grid

### `pairwise_interaction_grid`

::: capacities_ml_fin.ml.model_selection.grids.pairwise_interaction_grid

### `capacity_parameter_grid`

Convenience wrapper returning a dictionary ready for scikit-learn `GridSearchCV`.

::: capacities_ml_fin.ml.model_selection.grids.capacity_parameter_grid
