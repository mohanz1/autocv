"""The core package of the autocv project provides essential classes and functions for vision-based automation tasks.

It includes modules for direct input simulation, image processing, advanced vision techniques, and window capturing
functionalities.

This package serves as the backbone for computer vision-driven interactions with the user interface, enabling tasks such
as image recognition, mouse and keyboard automation, and window management.

Modules:
- filter_colors: Functions for filtering and manipulating colors in images.
- Input: Class for simulating user input.
- Vision: Class for performing image processing and optical character recognition.
- WindowCapture: Class for capturing window images.
"""

__all__ = ("Input", "Vision", "WindowCapture", "check_valid_hwnd", "check_valid_image", "filter_colors")


from .decorators import check_valid_hwnd, check_valid_image
from .image_processing import filter_colors
from .input import Input
from .vision import Vision
from .window_capture import WindowCapture
