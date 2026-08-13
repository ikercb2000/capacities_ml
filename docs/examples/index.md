# Examples

These examples are shorter documentation versions of the repository notebooks. They are written as complete workflows rather than isolated API calls.

| Example | What it demonstrates |
|---|---|
| [Capacity workflow](capacity_workflow.md) | explicit capacity, $k$-additivity, Möbius transform, scalar/batch Choquet integration, interpretation |
| [Choquet regression pipeline](regression_pipeline.md) | preprocessing, order selection, interaction regularization, held-out comparison, interpretation |
| [Classification pipeline](classification_pipeline.md) | deterministic versus probabilistic Choquet classification |
| [Time-series pipeline](time_series_pipeline.md) | statsmodels diagnostics, sktime transformations, temporal CV, lag-capacity interpretation |
| [Finance dataset pipeline](finance_pipeline.md) | returns, trailing features, forward targets, point-in-time fundamentals, portfolio losses |
| [Risk pipeline](risk_pipeline.md) | weighted empirical distributions, distortions, generalized risk, rolling capital and backtesting |
| [Optimization and model selection](optimization_model_selection.md) | sparsity families, shape grids, selected interactions and solver choices |

The code intentionally uses the public package API. For exact signatures and edge-case validation, consult the [API reference](../api/index.md).
