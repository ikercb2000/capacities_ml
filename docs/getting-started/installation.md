# Installation

## Python version

The current project metadata declares:

```text
Python >= 3.12, < 3.15
```

## Poetry installation

From the repository root:

```powershell
poetry install
```

For the complete development and documentation environment:

```powershell
poetry install --with dev,docs
```

The package can then be imported inside the Poetry environment:

```powershell
poetry run python
```

```python
from capacities_ml_fin import ChoquetRegressor
```

## Local pip installation

A local checkout can also be installed with pip:

```powershell
pip install .
```

Because the package uses a `src/` layout, importing it directly from an arbitrary working directory without installing it is not the recommended workflow.

## Documentation dependencies

The `docs` dependency group contains `mkdocs-material` and `mkdocstrings-python`. Start the development site with:

```powershell
poetry run mkdocs serve
```

Build the static site with:

```powershell
poetry run mkdocs build
```

The generated HTML is written to `site/`; the source files under `docs/` are the files that should be version-controlled.

## Test the installation

Run the full test suite:

```powershell
poetry run pytest
```

The current tests cover capacity representations, validation, Choquet integrals, machine-learning estimators, optimization, preprocessing, finance utilities, risk measures, rolling estimation, and backtesting.

## Optional troubleshooting

### Import errors after changing branches

Reinstall the environment if `pyproject.toml` or `poetry.lock` changed:

```powershell
poetry install --with dev,docs
```

### MkDocs cannot import package objects

The provided `mkdocs.yml` configures the Python handler with `paths: [src]`. If you move the package away from the `src/` layout, this setting must be updated.

### Heavy optimization dependencies

The project exposes SciPy, PYMOO, and CVXPY backends. A normal Poetry installation installs all project dependencies, so the API documentation can import the complete package when `mkdocstrings` builds the site.
