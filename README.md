# AutoCV
![Supported Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
[![CI](https://github.com/mohanz1/autocv/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/mohanz1/autocv/actions/workflows/ci.yml)
[![codecov.io](https://codecov.io/github/mohanz1/autocv/coverage.svg?branch=main)](https://app.codecov.io/github/mohanz1/autocv)

AutoCV is a Windows-first computer-vision automation toolkit for window capture, image analysis, and Win32 input simulation.

## Installation
AutoCV is not published on PyPI. Install from source:

```powershell
git clone https://github.com/mohanz1/autocv.git
cd autocv
uv sync --locked --group dev
```

Optional extras:
- `autocv[gui]`: interactive desktop tools (`AutoColorAid`)
- `autocv[ocr]`: PaddleOCR integration
- `autocv[ocr,paddle-cpu]` or `autocv[ocr,paddle-gpu]`: OCR + Paddle backend

For OCR installs, use the helper script instead of retrying `uv sync` manually:

```powershell
python scripts/install_ocr.py --backend cpu
# or:
python scripts/install_ocr.py --backend gpu
```

The helper retries once with a targeted Paddle/PaddleOCR reinstall if the first install flakes.
The `gpu` backend uses Paddle's official Windows x64 CUDA 12.9 package index.

## Quick Start
```python
from autocv import AutoCV

bot = AutoCV()
if not bot.set_hwnd_by_title("RuneLite"):
    raise SystemExit("RuneLite window not found")

while bot.set_inner_hwnd_by_title("SunAwtCanvas"):
    pass

bot.refresh()
points = bot.find_color((0, 255, 0), tolerance=40)
if points:
    bot.move_mouse(*points[0])
    bot.click_mouse()
```

## Development Workflow
```powershell
# Lint
uv run --no-sync --group lint ruff format --check .
uv run --no-sync --group lint ruff check .

# Type check
uv run --no-sync --group type mypy

# Test
uv run --no-sync --group test pytest -q tests

# Docs
uv run --no-sync --group docs sphinx-build -W docs public/docs

# Optional Windows smoke test for capture/UI changes
python scripts/windows_smoke_test.py --exercise-cursor
```

## Image Picker Bounds
`AutoCV.image_picker()` preserves the legacy return shape: `(image, window_rect)`, where `window_rect` is the full
target-window bounds. For new code that needs the actual selected ROI, use `AutoCV.image_picker_capture()` and read
`selection_rect`.

## Resources
- Documentation: https://mohanz1.github.io/autocv/
- Issues: https://github.com/mohanz1/autocv/issues
- License: MIT
