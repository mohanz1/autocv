"""Utility helpers for colour filtering in OpenCV images."""

from __future__ import annotations

__all__ = ("filter_colors",)

import logging
from typing import TYPE_CHECKING, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_RGBColor = tuple[int, int, int]
_RGB_CHANNELS = 3
_RGB_DIMENSIONS = 2


def _normalize_colours(colours: Iterable[_RGBColor] | _RGBColor) -> npt.NDArray[np.int16]:
    """Return colours as a ``(n, 3)`` RGB array."""
    if isinstance(colours, tuple) and len(colours) == _RGB_CHANNELS:
        values = np.asarray([colours], dtype=np.int16)
    else:
        values = np.asarray(list(colours), dtype=np.int16)
    if values.ndim != _RGB_DIMENSIONS or values.shape[1] != _RGB_CHANNELS:
        msg = "Expected colours to consist of RGB triplets."
        raise ValueError(msg)
    return values


def filter_colors(
    opencv_image: npt.NDArray[np.uint8],
    colours: Iterable[_RGBColor] | _RGBColor,
    tolerance: int = 0,
    *,
    keep_original_colors: bool = False,
) -> npt.NDArray[np.uint8]:
    """Return a mask (or filtered image) for ``colours`` within ``tolerance``."""
    if tolerance < 0:
        msg = "tolerance must be non-negative"
        raise ValueError(msg)

    rgb_values = _normalize_colours(colours)

    logger.debug(
        "Filtering %d colour(s) with tolerance=%d keep_original_colors=%s",
        len(rgb_values),
        tolerance,
        keep_original_colors,
    )

    bgr_values = rgb_values[:, ::-1]
    lower_bounds = np.clip(bgr_values - tolerance, 0, 255).astype(np.uint8)
    upper_bounds = np.clip(bgr_values + tolerance, 0, 255).astype(np.uint8)

    mask = cv.inRange(opencv_image, lower_bounds[0], upper_bounds[0])
    for lower, upper in zip(lower_bounds[1:], upper_bounds[1:], strict=False):
        colour_mask = cv.inRange(opencv_image, lower, upper)
        mask = cv.bitwise_or(mask, colour_mask)

    if keep_original_colors:
        filtered = np.zeros_like(opencv_image)
        filtered[mask > 0] = opencv_image[mask > 0]
        return filtered

    return cast("npt.NDArray[np.uint8]", mask)
