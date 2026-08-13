# Interpretation API

The interpretation functions are exact and operate through the generic `BaseCapacity.event_value()` interface.

## `shapley_index`

Computes one exact Shapley importance value. The element may be an integer index or a variable name when the capacity has a named universe.

::: capacities_ml_fin.base.interpretation.indices.shapley_index

## `shapley_indices`

Returns all exact Shapley indices as a dictionary keyed by names or indices.

::: capacities_ml_fin.base.interpretation.indices.shapley_indices

## `pairwise_interaction_index`

Computes the exact Shapley interaction index of two distinct elements.

::: capacities_ml_fin.base.interpretation.indices.pairwise_interaction_index

## `pairwise_interactions`

Returns every pairwise interaction.

::: capacities_ml_fin.base.interpretation.indices.pairwise_interactions

## `pairwise_interaction_matrix`

Returns the same information in a symmetric NumPy matrix with a zero diagonal.

::: capacities_ml_fin.base.interpretation.indices.pairwise_interaction_matrix

## `interaction_signs`

Maps pairwise interactions to `-1`, `0`, or `1` for redundancy, numerical neutrality, or complementarity.

::: capacities_ml_fin.base.interpretation.indices.interaction_signs
