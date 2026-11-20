"""Load the 5070 Ti smoke scripts by path: they live outside the package and import each other by name."""
import importlib.util
import sys
from pathlib import Path

import pytest

SMOKE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "m6-smoke"
_NAMES = ("smoke_common", "adapters", "probes", "smoke_report", "smoke")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SMOKE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def smoke_modules():
    """The adapters, probes and smoke modules, registered under their import names for the session."""
    saved = {name: sys.modules.get(name) for name in _NAMES}
    modules = {name: _load(name) for name in _NAMES}
    yield modules
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


@pytest.fixture
def smoke(smoke_modules):
    return smoke_modules["smoke"]


@pytest.fixture
def common(smoke_modules):
    return smoke_modules["smoke_common"]


@pytest.fixture
def adapters(smoke_modules):
    return smoke_modules["adapters"]


@pytest.fixture
def probes(smoke_modules):
    return smoke_modules["probes"]
