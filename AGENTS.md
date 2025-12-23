# Repository Guidelines
## Project Structure & Module Organization
AutoCV's runtime code lives in `autocv/`, organized by domain: `core/` for window automation primitives, `utils/` for cross-cutting helpers, `models/` and `data/` for shared types and assets, and `prebuilt/` for compiled extensions such as `antigcp`. CLI entry points sit in `autocv/cli.py` and `__main__.py`. CI recipes and reproducible dev tasks are defined under `pipelines/`, while `docs/` hosts the Sphinx configuration used for the public site. Tests belong in `tests/`, mirroring the package structure; keep the legacy `test/` directory untouched unless you are updating the C shim it vendors.

## Environment & Dependencies
We standardize on uv to create environments and lock dependencies. Sync the dev environment with `uv sync --locked --group pytest --group ruff --group ty` (add `--group sphinx` when touching docs). External tooling such as Tesseract OCR must be available on PATH for runtime validation; document any platform-specific setup in PR descriptions.

## Build, Test, and Development Commands
Run the default automation suite with `uv run nox`, which maps to `reformat-code`, `codespell`, `pytest`, `ruff`, and `ty` sessions defined in `pipelines/`. Examples:
- `uv run nox -s reformat-code` strips trailing whitespace and applies `ruff format` plus import sorting.
- `uv run nox -s pytest -- --coverage` executes `pytest` with HTML/XML coverage written to `public/coverage/html`.
- `uv run nox -s sphinx` builds docs into `public/docs`.

## Coding Style & Naming Conventions
Follow the Ruff formatter with a 120-column limit; Python files should start with `from __future__ import annotations` (`ruff` enforces this). Use 4-space indentation, `snake_case` functions, `PascalCase` classes, and avoid abbreviations unless established in the module. Type hints are required (ty runs in CI); keep docstrings in Google style where meaningful.

## Testing Guidelines
Author unit tests under `tests/` using `pytest` naming (`test_*.py`, `Test*` classes). Favour deterministic fixtures over sleeps or GUI timing. Aim to keep statement coverage comparable to existing modules; flag any deliberate gaps with `# pragma: no cover`. Validate coverage artifacts before merging.

## Commit & Pull Request Guidelines
Commit messages are short, present-tense summaries (`Add prebuilts`, `Fix trailing whitespace`). Group related changes into a single commit when practical. Pull requests must describe the change, link issues, and call out manual setup steps (e.g., Tesseract requirements). Attach screenshots or logs when touching GUI capture or coverage metrics, and confirm the relevant `uv run nox` sessions passed locally.
