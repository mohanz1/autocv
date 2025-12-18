import ctypes
import importlib
import platform
import runpy
import types
from pathlib import Path

import pytest
from mock import MagicMock

import autocv

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_AUTO_CV_INIT = _PACKAGE_ROOT / "autocv" / "__init__.py"


def test_import_non_windows_raises(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError):
        runpy.run_path(str(_AUTO_CV_INIT))


def test_import_windows_10_sets_process_dpi_awareness(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "10")

    shcore = types.SimpleNamespace(SetProcessDpiAwareness=MagicMock())
    user32 = types.SimpleNamespace(SetProcessDPIAware=MagicMock())
    windll = types.SimpleNamespace(shcore=shcore, user32=user32)
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    importlib.reload(autocv)

    shcore.SetProcessDpiAwareness.assert_called_once_with(2)
    user32.SetProcessDPIAware.assert_not_called()


def test_import_windows_other_sets_process_dpi_aware(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "7")

    shcore = types.SimpleNamespace(SetProcessDpiAwareness=MagicMock())
    user32 = types.SimpleNamespace(SetProcessDPIAware=MagicMock())
    windll = types.SimpleNamespace(shcore=shcore, user32=user32)
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    importlib.reload(autocv)

    user32.SetProcessDPIAware.assert_called_once_with()
    shcore.SetProcessDpiAwareness.assert_not_called()
