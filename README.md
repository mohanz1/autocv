# AutoCV
![Supported Python versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
[![CI](https://github.com/mohanz1/autocv/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/mohanz1/autocv/actions/workflows/ci.yml)
![Ty Checked](https://img.shields.io/badge/ty-checked-green.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://pypi.org/project/ruff)
[![codecov.io](https://codecov.io/github/mohanz1/autocv/coverage.svg?branch=main)](https://app.codecov.io/github/mohanz1/autocv)

AutoCV is a Windows-first computer vision automation toolkit for capturing game or desktop windows, analyzing them with OpenCV and Tesseract, and steering human-like input back into the client. It ships with a Win32 injection shim (`antigcp`) so you can neutralize `GetCursorPos` checks, color and contour search utilities, and interactive pickers that make it easy to tune bot logic live.

## Highlights
- Zero-copy window capture paired with OCR, color, template, and contour search APIs that return typed results.
- Human-style mouse and keyboard simulation with adjustable wind/acceleration profiles and ghost mouse support.
- uv + nox based automation that keeps formatting, linting, type checking, docs, and tests one command away.

## Installation
AutoCV is not published on PyPI. Install it directly from GitHub.

### Requirements
- Windows 10/11 with a recent Visual C++ runtime.
- Python 3.10-3.12 (matching the prebuilt `prebuilt/python<version>/antigcp.pyd`).
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 5.5+ available as `tesseract` on `PATH` (and optional tessconfigs when your OS package omits them).

### Install from GitHub
```powershell
# Clone the project
git clone https://github.com/mohanz1/autocv.git
cd autocv

# Create a managed environment with uv (recommended)
uv sync --locked --group pytest --group ruff --group ty

# Verify the install
uv run python -m autocv
```
Prefer pip? Use `python -m pip install -e .` inside the cloned directory after ensuring the prerequisites above.

## Quick Start
```python
from autocv import AutoCV
import cv2

bot = AutoCV()
if not bot.set_hwnd_by_title("RuneLite"):
    raise SystemExit("RuneLite window not found")
while bot.set_inner_hwnd_by_title("SunAwtCanvas"):
    pass
bot.antigcp()  # patch GetCursorPos so the client cannot detect synthetic input

bot.refresh()
contours = bot.find_contours((0, 255, 0), tolerance=40, min_area=50)
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    bot.move_mouse(x + w // 2, y + h // 2)
    bot.click_mouse()
```

```python
# Extract sparse overlays and react when values change
bot.refresh()
overlays = bot.get_text(rect=(20, 40, 320, 120), confidence=0.75)
for match in overlays:
    print(match["text"], match["rect"], match["confidence"])

if bot.get_pixel_change(area=(500, 300, 120, 80)) > 1500:
    bot.save_backbuffer_to_file("alerts/combat.png")
```

More GUI helpers live in `autocv/color_picker.py`, `autocv/image_picker.py`, and `autocv/image_filter.py`; call `bot.color_picker()`, `bot.image_picker()`, or `bot.image_filter()` to tune colors or filters in real time.

## Development Workflow
- `uv run nox` (or `uvx nox` if you don't have `nox` installed) runs the default suite: `reformat-code`, `codespell`, `pytest`, `ruff`, and `ty`.
- `uv run nox -s pytest -- --coverage` writes HTML/XML coverage to `public/coverage/html`.
- `uv run nox -s sphinx` builds the docs to `public/docs` (served on GitHub Pages).

Project code lives in `autocv/`, reusable automation lives in `pipelines/`, and tests mirror the package under `tests/`. See `AGENTS.md` for contributor guidelines.

## Resources
- Documentation: https://mohanz1.github.io/autocv/
- Issues & feature requests: https://github.com/mohanz1/autocv/issues
- Discord: https://discord.gg/Jx4cNGG
- License: MIT
