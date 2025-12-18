"""Helpers for image manipulation.

Exposes color-filtering utilities tailored to OpenCV BGR images.
"""

from __future__ import annotations

__all__ = ("filter_colors",)

import logging
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

Color: TypeAlias = tuple[int, int, int]  # RGB ordering for user-facing APIs
ColorArray: TypeAlias = npt.NDArray[np.int16]
Mask: TypeAlias = npt.NDArray[np.uint8]
ImageU8: TypeAlias = npt.NDArray[np.uint8]

_CHANNELS: Final[int] = 3


def _normalize_colors(colors: Color | Sequence[Color]) -> ColorArray:
    """Normalize a single color or sequence of colors into an ``(N, 3)`` array.

    Args:
        colors: Single RGB tuple or iterable of RGB tuples.

    Returns:
        Normalized array of shape ``(N, 3)`` in RGB order.

    Raises:
        ValueError: If no colors are provided or channel count is not three.
    """
    color_values: ColorArray = np.asarray(colors, dtype=np.int16)
    if color_values.size == 0:
        msg = "At least one color is required."
        raise ValueError(msg)
    if color_values.ndim == 1:
        color_values = color_values[np.newaxis, :]
    if color_values.shape[1] != _CHANNELS:
        msg = f"Each color must have exactly {_CHANNELS} channels (received {color_values.shape[1]})."
        raise ValueError(msg)
    return color_values


def filter_colors(
    opencv_image: ImageU8,
    colors: Color | Sequence[Color],
    tolerance: int = 0,
    *,
    keep_original_colors: bool = False,
) -> ImageU8:
    """Filter an OpenCV image to retain only the specified colors within a tolerance.

    Args:
        opencv_image: Input image in BGR channel order.
        colors: Single RGB tuple or iterable of RGB tuples to keep.
        tolerance: Allowed per-channel deviation (0-255). Must be non-negative.
        keep_original_colors: When ``True``, return the input image masked to matching pixels;
            otherwise return a binary mask.

    Returns:
        Filtered image if ``keep_original_colors`` is ``True``; otherwise a binary mask.

    Raises:
        ValueError: If ``colors`` is empty, has invalid channel length, or ``tolerance`` is negative.
    """
    if tolerance < 0:
        msg = "tolerance must be non-negative."
        raise ValueError(msg)

    color_values = _normalize_colors(colors)
    num_colors = int(color_values.shape[0])

    logger.debug(
        "Filtering with %d color(s) with tolerance=%d and keep_original_colors=%s.",
        num_colors,
        tolerance,
        keep_original_colors,
    )

    # Convert colors from RGB to BGR order for OpenCV processing.
    color_values = color_values[..., ::-1]

    # Create lower and upper bounds for each target color.
    lower_bounds = np.clip(color_values - tolerance, 0, 255).astype(np.uint8)
    upper_bounds = np.clip(color_values + tolerance, 0, 255).astype(np.uint8)

    # Build the mask using the first color.
    mask = cast("Mask", cv.inRange(opencv_image, lower_bounds[0], upper_bounds[0]))
    logger.debug(
        "Filtering color 1 of %d with lower bound %s and upper bound %s.",
        num_colors,
        lower_bounds[0],
        upper_bounds[0],
    )

    # Combine masks for remaining colors using bitwise OR.
    for i, (lb, ub) in enumerate(zip(lower_bounds[1:], upper_bounds[1:], strict=True), start=2):
        logger.debug(
            "Filtering color %d of %d with lower bound %s and upper bound %s.",
            i,
            num_colors,
            lb,
            ub,
        )
        color_mask = cast("Mask", cv.inRange(opencv_image, lb, ub))
        mask = cast("Mask", cv.bitwise_or(mask, color_mask))

    if keep_original_colors:
        logger.debug("Returning filtered image with original colors preserved for matching pixels.")
        return cast("ImageU8", cv.bitwise_and(opencv_image, opencv_image, mask=mask))

    return mask
