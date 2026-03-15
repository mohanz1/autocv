from __future__ import annotations

import ctypes
import importlib
import platform
import sys
import types
from unittest.mock import MagicMock

import pytest


def _reload_autocv() -> object:
    module = importlib.import_module("autocv")
    return importlib.reload(module)


def test_import_non_windows_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    _reload_autocv()


def test_import_does_not_eager_import_heavy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    sys.modules.pop("autocv.autocv", None)
    sys.modules.pop("autocv.auto_color_aid", None)

    module = _reload_autocv()
    assert "autocv.autocv" not in sys.modules
    assert "autocv.auto_color_aid" not in sys.modules

    _ = module.__version__
    assert "autocv.autocv" not in sys.modules


def test_import_windows_10_sets_process_dpi_awareness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "10")

    shcore = types.SimpleNamespace(SetProcessDpiAwareness=MagicMock())
    user32 = types.SimpleNamespace(SetProcessDPIAware=MagicMock())
    windll = types.SimpleNamespace(shcore=shcore, user32=user32)
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    _reload_autocv()

    shcore.SetProcessDpiAwareness.assert_called_once_with(2)
    user32.SetProcessDPIAware.assert_not_called()


def test_import_windows_other_sets_process_dpi_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "7")

    shcore = types.SimpleNamespace(SetProcessDpiAwareness=MagicMock())
    user32 = types.SimpleNamespace(SetProcessDPIAware=MagicMock())
    windll = types.SimpleNamespace(shcore=shcore, user32=user32)
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    _reload_autocv()

    user32.SetProcessDPIAware.assert_called_once_with()
    shcore.SetProcessDpiAwareness.assert_not_called()
