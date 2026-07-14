"""Guards on the frozen build.

Freezing this application has one structural hazard, and it comes directly from the
thing that makes the engine good: the registry loads architectures *dynamically*, by
string, so that adding a network is one file and one decorator. A static analyser
cannot see through that, so PyInstaller has to be told each architecture by name.

The failure that follows is nasty precisely because it is quiet. Forget an entry and:
the build succeeds; the app starts; `/api/v1/models` lists the model; its weights
download and checksum; and then, the first time a user actually selects it, the
architecture cannot be built and the job fails. Nothing catches it earlier — not the
type checker, not the test suite, not a smoke test that happens to use a different
model.

So the list is asserted against the registry here, in the ordinary test suite, where it
is checked on every run rather than only when someone thinks to rebuild the installer.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = REPO_ROOT / "packaging" / "visionsr-server.spec"
ENTRY = REPO_ROOT / "packaging" / "server_entry.py"


def _spec_architectures() -> list[str]:
    """The ARCHITECTURES list from the spec file.

    Parsed rather than imported: a PyInstaller spec is only valid inside PyInstaller's
    own execution context (it relies on injected globals like SPECPATH and EXE), so
    importing it would raise NameError long before reaching the list.
    """
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "ARCHITECTURES" in targets:
            return [
                element.value
                for element in node.value.elts  # type: ignore[attr-defined]
                if isinstance(element, ast.Constant)
            ]

    pytest.fail("the spec file no longer defines ARCHITECTURES")


def test_every_dynamically_loaded_architecture_is_a_hidden_import():
    """Every module the registry imports by string must be named in the spec.

    ``models/__init__.py`` lists them relative (``.sr.rrdbnet``); the spec needs them
    absolute (``visionsr.models.sr.rrdbnet``). Compare after normalising.
    """
    from visionsr.models import _ARCH_MODULES

    required = {f"visionsr.models{module}" for module in _ARCH_MODULES}
    declared = set(_spec_architectures())

    missing = required - declared
    assert not missing, (
        f"{len(missing)} architecture module(s) are loaded dynamically but not declared "
        f"as hidden imports in packaging/visionsr-server.spec: {sorted(missing)}. "
        "The frozen build will start fine and then fail the first time anyone selects "
        "one of these models."
    )


def test_the_spec_declares_the_package_itself():
    """`visionsr.models` runs the imports; without it, nothing else is reached."""
    assert "visionsr.models" in _spec_architectures()


def test_registered_architectures_all_resolve():
    """Every architecture a registered model names must actually be buildable.

    Catches the other half of the same class of bug: a YAML entry pointing at an
    architecture nobody wrote, or one whose module stopped being imported.
    """
    from visionsr.core.registry import ONNX_GRAPH, architectures
    from visionsr.core.registry import models as registry
    from visionsr.inference.engine import _ensure_registry

    _ensure_registry()

    for spec in registry.all():
        if spec.architecture == ONNX_GRAPH:
            # No torch architecture by definition — the checkpoint is the graph.
            continue
        assert spec.architecture in architectures, (
            f"model {spec.id!r} names architecture {spec.architecture!r}, which is not "
            f"registered. Known: {architectures.names()}"
        )


def test_the_frozen_entry_point_guards_against_the_fork_bomb():
    """`multiprocessing.freeze_support()` is not optional on Windows.

    Windows has no fork: a frozen child process re-executes the .exe, which re-runs the
    entry module, which spawns another child. Without the guard the app forks itself
    exponentially on launch — and the symptom (the machine locking up) looks nothing
    like a packaging mistake.
    """
    source = ENTRY.read_text(encoding="utf-8")
    assert re.search(r"multiprocessing\.freeze_support\(\)", source), (
        "packaging/server_entry.py must call multiprocessing.freeze_support() before "
        "anything else."
    )


def test_the_frozen_server_does_not_import_the_app_by_string():
    """uvicorn's ``"module:app"`` form resolves against the filesystem.

    A frozen build has no filesystem to resolve against — its modules live in a bundled
    archive — so the string form fails at startup with an import error that reads like
    the app is broken. The entry point must hand uvicorn the app object it already has.
    """
    source = ENTRY.read_text(encoding="utf-8")

    assert "from backend.app.main import app" in source
    assert not re.search(r'uvicorn\.run\(\s*["\']', source), (
        "the frozen entry point passes uvicorn an import string; it must pass the app object"
    )
