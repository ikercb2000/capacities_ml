import ast
import importlib
import inspect
import re
from pathlib import Path


API_DIRECTIVE = re.compile(r"^::: (.+)$", re.MULTILINE)


def _api_references():
    api_root = Path(__file__).parents[1] / "docs" / "api"
    for page in sorted(api_root.glob("*.md")):
        for reference in API_DIRECTIVE.findall(page.read_text(encoding="utf-8")):
            yield page, reference


def test_documented_api_objects_resolve_and_have_docstrings():
    references = list(_api_references())
    assert references, "No mkdocstrings API directives were found."

    for page, reference in references:
        module_name, separator, object_name = reference.rpartition(".")
        assert separator, f"Invalid API reference in {page}: {reference}"
        module = importlib.import_module(module_name)
        assert hasattr(module, object_name), (
            f"Unknown API object in {page}: {reference}"
        )
        documented_object = getattr(module, object_name)
        assert inspect.getdoc(documented_object), (
            f"Public API object has no docstring: {reference}"
        )

def test_public_top_level_objects_have_docstrings():
    source_root = Path(__file__).parents[1] / "src" / "capacities_ml_fin"
    missing = []

    for path in sorted(source_root.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            is_public_object = isinstance(node, (ast.ClassDef, ast.FunctionDef))
            if is_public_object and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(source_root)}:{node.name}")

    assert not missing, "Public objects without docstrings:\n" + "\n".join(missing)
