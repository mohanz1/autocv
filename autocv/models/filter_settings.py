"""Filter configuration dataclass used throughout AutoCV.

The :class:`~autocv.models.filter_settings.FilterSettings` model acts as a
lightweight container for user-adjustable HSV, Canny, and morphology parameters.
"""

from __future__ import annotations

__all__ = ("FilterSettings",)

from dataclasses import dataclass


@dataclass(slots=True)
class FilterSettings:
    """Persist HSV, Canny, and morphology parameters for image filtering.

    Attributes:
        h_min: Minimum hue value. Defaults to 0.
        h_max: Maximum hue value. Defaults to 179.
        s_min: Minimum saturation value. Defaults to 0.
        s_max: Maximum saturation value. Defaults to 255.
        v_min: Minimum value (brightness). Defaults to 0.
        v_max: Maximum value (brightness). Defaults to 255.
        s_add: Amount added to saturation during adjustment. Defaults to 0.
        s_subtract: Amount subtracted from saturation during adjustment. Defaults to 0.
        v_add: Amount added to brightness during adjustment. Defaults to 0.
        v_subtract: Amount subtracted from brightness during adjustment. Defaults to 0.
        canny_threshold1: Lower threshold used in Canny edge detection. Defaults to 0.
        canny_threshold2: Upper threshold used in Canny edge detection. Defaults to 0.
        erode_kernel_size: Kernel size used for erosion passes. Defaults to 0.
        dilate_kernel_size: Kernel size used for dilation passes. Defaults to 0.
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
