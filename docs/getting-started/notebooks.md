# Notebooks

The repository notebooks are executable, end-to-end examples. They are useful as a bridge between the conceptual guide and the object-level API reference.

| Notebook | Main topics |
|---|---|
| `01_capacities_and_choquet_integral.ipynb` | explicit and $k$-additive capacities, Möbius transform, scalar/batch Choquet integration, Shapley values, interactions |
| `02_choquet_regression.ipynb` | normalization, `ChoquetRegressor`, capacity-order selection, L1 interaction regularization, comparison with linear regression |
| `03_choquet_classifier.ipynb` | deterministic Choquet classification, feature scales, capacity selection, classification metrics |
| `04_choquistic_regression.ipynb` | probabilistic Choquistic model, likelihood-based fitting, class probabilities |
| `05_optimization_backends.ipynb` | capacity parameterizations, objectives, constraints and solver backends |
| `06_risk_pipeline.ipynb` | finite loss distributions, distortions, generalized risk measures, rolling capital, conditional scenarios, backtesting |
| `07_choquet_time_series.ipynb` | statsmodels diagnostics, sktime transformations, temporal CV, Choquet autoregression, prediction intervals, lag interpretation |

## Suggested reading order

If the mathematical objects are new, begin with notebook 01. For machine learning, continue with 02–04. Notebook 05 is useful when extending the optimization layer. Notebook 07 demonstrates how `ChoquetAutoRegressor` fits into a standard sktime workflow. Notebook 06 covers the risk layer independently of the supervised estimators.

The [Examples](../examples/index.md) section of this documentation distills the same workflows into shorter, copyable pages.
