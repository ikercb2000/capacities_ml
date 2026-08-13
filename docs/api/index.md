# API reference

The API reference is the object-level counterpart to the User Guide. Each section contains two layers:

1. a short manual explanation of the object's role and important behavioral details;
2. a `mkdocstrings` block that reads the current Python signature and docstring from the source tree.

This means signatures stay synchronized with the code, while conceptual explanations can remain richer than a docstring should be.

## Import conventions

The package root intentionally exports the main fitted estimators:

```python
from capacities_ml_fin import (
    ChoquetRegressor,
    ChoquetClassifier,
    ChoquisticRegression,
    ChoquetNeuralRegressor,
    ChoquetNeuralClassifier,
    ChoquetAutoRegressor,
)
```

Other functionality is grouped by domain:

```python
from capacities_ml_fin.base.capacities import ExplicitCapacity, MobiusCapacity
from capacities_ml_fin.base.interpretation import shapley_indices
from capacities_ml_fin.ml.optimization import KAdditivity
from capacities_ml_fin.finance import price_returns
from capacities_ml_fin.risk import ExpectedShortfallDistortion
```

## Sections

- [Capacities](capacities.md)
- [Integrals](integrals.md)
- [Interpretation](interpretation.md)
- [ML models](ml_models.md)
- [Optimization](optimization.md)
- [Preprocessing & model selection](preprocessing_model_selection.md)
- [Finance](finance.md)
- [Risk](risk.md)

!!! note
    Objects whose names begin with `_` are implementation details and are intentionally excluded from the public documentation by the MkDocs configuration.
