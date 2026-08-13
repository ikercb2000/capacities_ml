# Integrals API

## `ordered_choquet`

Representation-independent discrete Choquet integral. It sorts one observation, asks the capacity for the corresponding nested event values, and combines those values with ordered increments.

::: capacities_ml_fin.base.integrals.choquet.ordered_choquet

## `mobius_choquet`

Direct Möbius evaluation using coalition minima.

::: capacities_ml_fin.base.integrals.choquet.mobius_choquet

## `batch_choquet_integral`

Primary batch dispatcher. It selects optimized paths for explicit and Möbius capacities and falls back to row-wise generic evaluation for other `BaseCapacity` implementations.

::: capacities_ml_fin.base.integrals.batch_integrals.batch_choquet_integral

## `batch_choquet_integral_mobius`

Vectorized batch evaluation for a `MobiusCapacity`.

::: capacities_ml_fin.base.integrals.batch_integrals.batch_choquet_integral_mobius

## Design matrices

These functions are also used internally by the learning layer.

### `capacity_design_matrix`

Builds the direct-capacity design matrix indexed by subset bitmasks.

::: capacities_ml_fin.base.integrals.batch_integrals.capacity_design_matrix

### `mobius_design_matrix`

Builds one column per coalition using the row-wise coalition minimum.

::: capacities_ml_fin.base.integrals.batch_integrals.mobius_design_matrix
