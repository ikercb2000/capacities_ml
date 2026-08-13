# Getting started

This section covers the shortest path from a clone of the repository to a fitted capacity model. The project uses a `src/` package layout and Poetry for dependency management.

## Recommended path

1. [Install the package](installation.md).
2. Run the [quickstart](quickstart.md).
3. Read the [theory overview](../theory/index.md) if capacities or Möbius representations are new to you.
4. Follow one of the complete [examples](../examples/index.md).
5. Keep the [API reference](../api/index.md) open while developing.

## Package layout

The repository is organized approximately as follows:

```text
capacities_ml_fin/
├── src/capacities_ml_fin/
│   ├── base/       # capacity objects, integrals, interpretation
│   ├── ml/         # estimators, optimization, preprocessing
│   ├── finance/    # returns, features, portfolio, alignment
│   └── risk/       # distributions, distortions, measures, backtesting
├── tests/
├── notebooks/
├── docs/
├── pyproject.toml
└── mkdocs.yml
```

The package intentionally keeps the mathematical layer independent from any single estimator. For example, a `MobiusCapacity` can be created manually, recovered from an `ExplicitCapacity`, produced by a fitted `ChoquetRegressor`, interpreted with Shapley indices, or used as a finite event capacity.

## Typical development loop

```powershell
poetry install --with dev,docs
poetry run pytest
poetry run mkdocs serve
```

When working on the documentation, `mkdocs serve` watches both `docs/` and the package source configured in `mkdocs.yml`, so API changes can be reflected immediately.
