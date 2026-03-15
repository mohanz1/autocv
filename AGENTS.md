# Repository Guidelines
## Project Structure & Module Organization
AutoCV's runtime code lives in `autocv/`, organized by domain: `core/` for capture, vision, and input primitives; `utils/` for generic helpers; `models/` for shared dataclasses and exceptions; `data/` for shipped assets; and `prebuilt/` for version-specific native binaries such as `antigcp`. The class stack is intentional: `WindowCapture -> Vision -> Input -> AutoCV`. Keep that layering clear when adding functionality.

Interactive tools live in `autocv/auto_color_aid.py`, `autocv/color_picker.py`, `autocv/image_picker.py`, and `autocv/image_filter.py`. Follow the existing pattern of keeping Tk/OpenCV widget code thin and moving testable state or orchestration into controller/dataclass helpers. CLI entry points sit in `autocv/cli.py` and `autocv/__main__.py`. CI lives under `.github/workflows/`, while `docs/` hosts the Sphinx configuration for the public site. Tests belong in `tests/`, mirroring the package structure; keep the legacy `test/` directory untouched unless you are updating the vendored C shim.

`autocv/__init__.py` is intentionally lazy and must stay lightweight at import time. Do not eagerly import heavy GUI, OCR, or Win32-dependent modules there. `.pyi` files and `autocv/py.typed` are part of the package's typed public surface; when public exports or signatures change, keep the matching stub files in sync. If you add support for a new Python minor version, check whether `autocv/prebuilt/python*/` needs a matching `antigcp` binary.

## Environment & Dependencies
We standardize on uv to create environments and lock dependencies. For the normal non-OCR development workflow, sync
the full dev environment with `uv sync --locked --no-sources --group dev`, or install only what you need with
`uv sync --locked --no-sources --group test`, `--group lint`, `--group type`, or `--group docs`.

The runtime library is Windows-first because capture/input depend on `pywin32`, but CI also runs linting, type checking, and docs on Ubuntu. Keep imports and module initialization resilient when optional or platform-specific dependencies are unavailable unless the module is explicitly Windows-only. Optional extras are:
- `autocv[gui]` for desktop theming helpers used by interactive tools
- `autocv[ocr]` for PaddleOCR integration
- `autocv[ocr,paddle-cpu]` or `autocv[ocr,paddle-gpu]` for complete OCR backends

If you add new optional dependencies or imports that Sphinx cannot import in CI, update `docs/conf.py` mocks/stubs accordingly. PaddleOCR may perform model-host checks or downloads at runtime; document any environment variables or setup changes when that behavior is affected. Prefer `python scripts/install_ocr.py --backend cpu` or `--backend gpu` over ad-hoc OCR setup commands; the helper script applies the current repo-supported `uv sync` flags and retries targeted Paddle installs once.

## Required Validation After Changes
After syncing the relevant groups, run the same commands enforced in GitHub Actions from the repository root:
- `uv run --no-sync --group lint ruff format --check .`
- `uv run --no-sync --group lint ruff check .`
- `uv run --no-sync --group type mypy`
- `uv run --no-sync --group test pytest -q tests --cov=autocv --cov-config=pyproject.toml --cov-report=term --cov-report=xml`
- `uv run --no-sync --group docs sphinx-build -W docs public/docs`

If a command cannot run because of missing platform features, GUI support, or external dependencies, say so explicitly in the handoff and include the exact failure.

## Coding Style & Naming Conventions
Follow Ruff with a 120-column limit. Python files should start with `from __future__ import annotations` (`ruff` enforces this). Use 4-space indentation, `snake_case` functions, `PascalCase` classes, and avoid abbreviations unless they are already established in the module.

Type hints are required and `mypy` runs in CI. Prefer the existing style of explicit `TypeAlias`, `Final`, small `Protocol` helpers, and typed dataclasses where they improve clarity. Keep docstrings in Google style where meaningful.

Be precise about color semantics: public-facing color APIs use RGB tuples, while OpenCV image buffers are stored in BGR order. Preserve that boundary in code, tests, and documentation. Maintain compatibility-oriented behavior when refactoring public modules: lazy imports, best-effort DPI awareness, and backwards-compatible properties/helpers are all covered by tests.

## Testing Guidelines
Author unit tests under `tests/` using `pytest` naming (`test_*.py`, `Test*` classes). CI runs tests on Windows with Python 3.11 and 3.12, and the 3.12 job is the one that records coverage.

Favor deterministic fixtures over sleeps or GUI timing. Follow the current testing style for Windows/Tk/OpenCV-heavy code: mock `win32*`, `Tk`, `ImageTk`, `Vision`, or `AutoCV` boundaries and test controller/state logic directly where possible. Keep new code injectable enough that it can be exercised without a real desktop session. Aim to keep statement coverage comparable to existing modules; flag any deliberate gaps with `# pragma: no cover`.

After changes that touch live capture, window attachment, or interactive Tk tools, run the local smoke test on Windows when possible:
- `python scripts/windows_smoke_test.py --exercise-cursor`
This launches a temporary Tk harness window and validates attachment, capture, color search, `ImagePickerController`, and optional `ColorPickerController` sampling against a real window.

## Docs & Consistency Notes
Sphinx builds on Linux with mocked Win32/Tk/Paddle imports and a small tkinter stub in `docs/conf.py`. If you add imports that would break docs builds, update those mocks/stubs in the same change. Treat `pyproject.toml` and `.github/workflows/ci.yml` as the source of truth for supported Python versions, dependency groups, and validation commands, then keep `README.md` and `docs/` aligned with them.

`ImagePicker.rect` is a compatibility field containing the full target-window bounds. `ImagePicker.selection_rect` is the authoritative selected ROI. Preserve `AutoCV.image_picker()` for backwards compatibility and use `AutoCV.image_picker_capture()` for new code that needs explicit ROI metadata.

## Commit & Pull Request Guidelines
Commit messages are short, present-tense summaries (`Add prebuilts`, `Fix trailing whitespace`). Group related changes into a single commit when practical. Pull requests should describe the change, link issues, and call out any manual setup or platform constraints, especially around GUI behavior, Win32 assumptions, or OCR dependencies. Attach screenshots or logs when touching GUI capture or coverage metrics, and confirm the CI-equivalent `ruff`, `mypy`, `pytest`, and docs commands passed locally.
