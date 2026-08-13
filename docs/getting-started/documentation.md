# Documentation website

The website is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and [mkdocstrings](https://mkdocstrings.github.io/). Narrative guides live in `docs/`; API pages render signatures and NumPy-style docstrings directly from `src/capacities_ml_fin`.

## Preview locally

Install the documentation group and start the live-reloading server from the repository root:

```powershell
poetry install --with docs
poetry run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Changes to Markdown pages, `mkdocs.yml`, and Python docstrings trigger a rebuild.

To perform the same strict build used by continuous integration:

```powershell
poetry run mkdocs build --strict
```

The static website is written to `site/`.

## Docstring convention

Public estimators and domain objects use NumPy-style docstrings, following the structure commonly used by scikit-learn:

```python
class Estimator:
    """One-line summary followed by the estimator semantics.

    Parameters
    ----------
    parameter : type, default=value
        Meaning and accepted domain.

    Attributes
    ----------
    learned_attribute_ : type
        Attribute created by ``fit``.

    Notes
    -----
    Mathematical or behavioral details that affect interpretation.
    """
```

Use `Parameters` for constructor or function inputs, `Returns` for function outputs, `Attributes` for public object state, `Raises` for important validation failures, and `Notes` for mathematical conventions. Learned scikit-learn attributes end in an underscore.

## Publish with GitHub Pages

The `docs.yml` workflow validates the website for pull requests and deploys the `main` branch. In the GitHub repository, choose **Settings → Pages → Build and deployment → Source: GitHub Actions** once. After the workflow succeeds, the site is available at:

<https://ikercb2000.github.io/capacities_ml_fin/>
