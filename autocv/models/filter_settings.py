"""This module defines the FilterSettings dataclass.

FilterSettings stores configuration settings for various image processing filters.
These settings include ranges for hue, saturation, and value, parameters for edge detection,
and kernel sizes for erosion and dilation.
"""

from __future__ import annotations

__all__ = ("FilterSettings",)

from dataclasses import dataclass

# Use slots if supported by the dataclass implementation.
SLOTS_DATACLASS = {"slots": True} if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(**SLOTS_DATACLASS)
class FilterSettings:
    """Stores settings for image processing filters.

    Attributes:
        h_min (int): Minimum hue value. Defaults to 0.
        h_max (int): Maximum hue value. Defaults to 179.
        s_min (int): Minimum saturation value. Defaults to 0.
        s_max (int): Maximum saturation value. Defaults to 255.
        v_min (int): Minimum value (brightness). Defaults to 0.
        v_max (int): Maximum value (brightness). Defaults to 255.
        s_add (int): Amount to add to the saturation value. Defaults to 0.
        s_subtract (int): Amount to subtract from the saturation value. Defaults to 0.
        v_add (int): Amount to add to the value (brightness). Defaults to 0.
        v_subtract (int): Amount to subtract from the value (brightness). Defaults to 0.
        canny_threshold1 (int): First threshold for the Canny edge detection algorithm. Defaults to 0.
        canny_threshold2 (int): Second threshold for the Canny edge detection algorithm. Defaults to 0.
        erode_kernel_size (int): Kernel size for the erosion filter. Defaults to 0.
        dilate_kernel_size (int): Kernel size for the dilation filter. Defaults to 0.
    """

    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    s_add: int = 0
    s_subtract: int = 0
    v_add: int = 0
    v_subtract: int = 0
    canny_threshold1: int = 0
    canny_threshold2: int = 0
    erode_kernel_size: int = 0
    dilate_kernel_size: int = 0
