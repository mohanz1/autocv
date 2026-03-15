from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "install_ocr.py"
    spec = importlib.util.spec_from_file_location("install_ocr_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_for_cpu_backend_avoids_gpu_sources():
    module = _load_script_module()

    command = module.build_command(module.BACKENDS["cpu"], retry=False)

    assert command == [
        "uv",
        "sync",
        "--locked",
        "--extra",
        "ocr",
        "--extra",
        "paddle-cpu",
        "--no-sources-package",
        "paddlepaddle-gpu",
    ]


def test_build_command_for_retry_reinstalls_target_packages():
    module = _load_script_module()

    command = module.build_command(module.BACKENDS["gpu"], retry=True)

    assert "--refresh-package" in command
    assert "--reinstall-package" in command
    assert command.count("paddleocr") == 2
    assert command.count("paddlepaddle-gpu") == 2


def test_install_ocr_retries_after_failure(monkeypatch):
    module = _load_script_module()
    calls: list[list[str]] = []
    results = iter([SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)])

    monkeypatch.setattr(module.shutil, "which", lambda _: "uv")

    def fake_run(command, cwd, check):  # noqa: ANN001
        calls.append(command)
        assert cwd == module.PROJECT_ROOT
        assert check is False
        return next(results)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.install_ocr("cpu", max_attempts=2)

    assert result == 0
    assert len(calls) == 2
    assert "--no-sources-package" in calls[0]
    assert "--reinstall-package" in calls[1]


def test_install_ocr_fails_cleanly_when_uv_is_missing(monkeypatch, capsys):
    module = _load_script_module()
    monkeypatch.setattr(module.shutil, "which", lambda _: None)

    result = module.install_ocr("cpu", max_attempts=2)

    assert result == 1
    assert "uv is required" in capsys.readouterr().err
