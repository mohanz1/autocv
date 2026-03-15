from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COLOR_IMAGE_NDIMS = 3
REGION_X1 = 20
REGION_Y1 = 20
REGION_X2 = 80
REGION_Y2 = 80
EXPECTED_REGION_SIZE = REGION_X2 - REGION_X1

HARNESS_TITLE = "AutoCV Smoke Harness"
HARNESS_GREEN = (0, 255, 0)

if sys.platform == "win32":
    import win32api
    import win32gui

    from autocv import AutoCV
    from autocv.color_picker import PIXELS, ColorPickerController
    from autocv.image_picker import ImagePickerController


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


class SmokeHarness:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(HARNESS_TITLE)
        self.root.geometry("420x320+160+160")
        self.root.configure(bg="#00ff00")

        self.canvas = tk.Canvas(self.root, width=320, height=180, bg="#00ff00", highlightthickness=0)
        self.canvas.pack(padx=16, pady=16)
        self.canvas.create_rectangle(20, 20, 140, 140, fill="#00ff00", outline="")
        self.canvas.create_rectangle(180, 20, 300, 140, fill="#0000ff", outline="")

        self.entry = tk.Entry(self.root)
        self.entry.insert(0, "autocv smoke harness")
        self.entry.pack(pady=(0, 12))

    def pump(self, delay: float) -> None:
        self.root.update_idletasks()
        self.root.update()
        time.sleep(delay)
        self.root.update_idletasks()
        self.root.update()

    def destroy(self) -> None:
        if self.root.winfo_exists():
            self.root.destroy()


def _record(results: list[CheckResult], name: str, *, passed: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=passed, detail=detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def _exercise_cursor_sampling(autocv: AutoCV, harness: SmokeHarness, results: list[CheckResult]) -> None:
    original_cursor = win32gui.GetCursorPos()
    try:
        left, top, _, _ = win32gui.GetWindowRect(autocv.get_hwnd())
        target_cursor = (left + 90, top + 90)
        win32api.SetCursorPos(target_cursor)
        harness.pump(0.1)

        default_canvas = np.full((PIXELS * 2 + 1, PIXELS * 2 + 1, 3), 127, dtype=np.uint8)
        color_picker = ColorPickerController(autocv.get_hwnd(), default_canvas)
        patch, x, y, _ = color_picker.capture_cursor_patch()
        patch_ok = (
            x >= 0
            and y >= 0
            and patch.shape == default_canvas.shape
            and not np.array_equal(
                patch,
                default_canvas,
            )
        )
        _record(results, "color picker patch", passed=patch_ok, detail=f"point=({x}, {y}), shape={tuple(patch.shape)}")
    finally:
        win32api.SetCursorPos(original_cursor)


def run_smoke_test(*, settle_seconds: float, exercise_cursor: bool) -> int:
    harness = SmokeHarness()
    results: list[CheckResult] = []
    autocv: AutoCV | None = None

    try:
        harness.pump(settle_seconds)
        autocv = AutoCV()

        attached = autocv.set_hwnd_by_title(HARNESS_TITLE)
        _record(results, "attach main window", passed=attached, detail=f"title={HARNESS_TITLE!r}")
        if not attached:
            return 1

        width, height = autocv.get_window_size()
        _record(results, "window size", passed=width > 0 and height > 0, detail=f"size=({width}, {height})")

        child_windows = autocv.get_child_windows()
        _record(
            results,
            "child enumeration",
            passed=isinstance(child_windows, list),
            detail=f"count={len(child_windows)}",
        )

        autocv.refresh()
        frame = autocv.opencv_image
        frame_ok = frame.ndim == COLOR_IMAGE_NDIMS and frame.shape[0] > 0 and frame.shape[1] > 0
        _record(results, "frame capture", passed=frame_ok, detail=f"shape={tuple(frame.shape)}")

        green_points = autocv.find_color(HARNESS_GREEN, tolerance=20)
        _record(results, "find known color", passed=bool(green_points), detail=f"matches={len(green_points)}")

        image_picker = ImagePickerController(autocv.get_hwnd())
        bounds = image_picker.full_rect_as_bounds()
        bounds_ok = bounds[2] > 0 and bounds[3] > 0
        _record(results, "image picker bounds", passed=bounds_ok, detail=f"bounds={bounds}")

        region = image_picker.capture_region(REGION_X1, REGION_Y1, REGION_X2, REGION_Y2)
        region_ok = region.ndim in {2, COLOR_IMAGE_NDIMS} and region.shape[0] == EXPECTED_REGION_SIZE
        region_ok = region_ok and region.shape[1] == EXPECTED_REGION_SIZE
        _record(results, "image picker region", passed=region_ok, detail=f"shape={tuple(region.shape)}")

        if exercise_cursor:
            _exercise_cursor_sampling(autocv, harness, results)

        return 0 if all(result.passed for result in results) else 1
    finally:
        if autocv is not None:
            autocv.hwnd = -1
        harness.destroy()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Windows smoke test against a temporary Tk window.")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.3,
        help="Delay used while waiting for the harness window to paint. Defaults to 0.3 seconds.",
    )
    parser.add_argument(
        "--exercise-cursor",
        action="store_true",
        help="Temporarily move the cursor into the harness window and validate ColorPickerController sampling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "win32":
        print("windows_smoke_test.py only runs on Windows.", file=sys.stderr)
        return 2

    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.settle_seconds < 0:
        print("--settle-seconds must be non-negative.", file=sys.stderr)
        return 2

    return run_smoke_test(settle_seconds=args.settle_seconds, exercise_cursor=args.exercise_cursor)


if __name__ == "__main__":
    raise SystemExit(main())
