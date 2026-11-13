"""Static check that every `from app.x import name` names something app/x.py really defines.

CI installs a slim dependency set (no torch, reportlab, asyncpg, fastapi), so the tests never import
app.main or app.output. A removed name (memory._model, still imported by output.py) therefore passed CI
and crashed the app at startup on AWS. Parsing the source catches that without the heavy dependencies.
"""
import ast
import pathlib
import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app"


def _module_path(module: str) -> pathlib.Path | None:
    rel = pathlib.Path(*module.split("."))
    for candidate in (APP.parent / rel.with_suffix(".py"), APP.parent / rel / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _top_level_names(path: pathlib.Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _app_imports():
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "app":
                for alias in node.names:
                    yield pytest.param(path.relative_to(APP.parent).as_posix(), node.module, alias.name,
                                       id=f"{path.name}:{node.module}.{alias.name}")


@pytest.mark.parametrize("importer, module, name", list(_app_imports()))
def test_app_import_targets_exist(importer, module, name):
    target = _module_path(module)
    assert target is not None, f"{importer}: module {module} not found"
    if (APP.parent / pathlib.Path(*module.split(".")) / name).with_suffix(".py").exists():
        return   # `from app.tools import search` - a submodule
    assert name in _top_level_names(target), f"{importer} imports {name!r} from {module}, which doesn't define it"
