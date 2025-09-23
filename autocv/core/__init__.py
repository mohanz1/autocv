"""Core automation primitives: window capture, vision, and input."""

from __future__ import annotations

__all__ = ("Input", "Vision", "WindowCapture", "check_valid_hwnd", "check_valid_image", "filter_colors")

from .decorators import check_valid_hwnd, check_valid_image
from .image_processing import filter_colors
from .input import Input
from .vision import Vision
from .window_capture import WindowCapture
