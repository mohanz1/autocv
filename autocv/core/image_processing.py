"""The image_processing module provides utility functions for processing images.

This includes filtering colors within specific tolerance ranges. It primarily supports operations
related to color manipulation and filtering in OpenCV images.
"""

from __future__ import annotations

__all__ = ("filter_colors",)

import logging
from typing import cast

import cv2 as cv
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def filter_colors(
    opencv_image: npt.NDArray[np.uint8],
    colors: tuple[int, int, int] | list[tuple[int, int, int]],
    tolerance: int = 0,
    *,
    keep_original_colors: bool = False,
) -> npt.NDArray[np.uint8]:
    """Filters an OpenCV image to retain only the specified colors within a given tolerance.

    This function creates a mask by checking if each pixel's color is within the tolerance range
    of any of the target colors (provided as RGB tuples). Optionally, it returns a filtered version
    of the original image in which all non-matching pixels are set to black.

    Args:
        opencv_image (npt.NDArray[np.uint8]): The input image in BGR format.
        colors (tuple[int, int, int] | list[tuple[int, int, int]]):
            A single RGB tuple or a list of RGB tuples representing the target colors (specified in RGB order).
        tolerance (int, optional): The allowed deviation (0-255) for each color channel. Defaults to 0.
        keep_original_colors (bool, optional): If True, returns an image with non-matching pixels set to black.
            If False, returns a binary mask. Defaults to False.

    Returns:
        npt.NDArray[np.uint8]: The filtered image if `keep_original_colors` is True;
            otherwise, a binary mask where matching pixels have the maximum value (255).
    """
    # Convert the input colors to a NumPy array of shape (N, 3)
    color_values = np.array(colors, dtype=np.int16)
    if color_values.ndim == 1:
        color_values = color_values[np.newaxis, :]

    logger.debug(
        "Filtering with %d color(s) with tolerance=%d and keep_original_colors=%s.",
        len(color_values),
        tolerance,
        keep_original_colors,
    )

    # Convert colors from RGB to BGR order for OpenCV processing
    color_values = color_values[..., ::-1]

    # Create lower and upper bounds for each target color
    lower_bounds = np.clip(color_values - tolerance, 0, 255)
    upper_bounds = np.clip(color_values + tolerance, 0, 255)

    # Build the mask using the first color
    mask = cv.inRange(opencv_image, lower_bounds[0].astype(np.uint8), upper_bounds[0].astype(np.uint8))
    logger.debug(
        "Filtering color 1 of %d with lower bound %s and upper bound %s.",
        len(color_values),
        lower_bounds[0],
        upper_bounds[0],
    )

    # Combine masks for remaining colors using bitwise OR
    for i, (lb, ub) in enumerate(zip(lower_bounds[1:], upper_bounds[1:], strict=False), start=2):
        logger.debug(
            "Filtering color %d of %d with lower bound %s and upper bound %s.",
            i,
            len(color_values),
            lb,
            ub,
        )
        color_mask = cv.inRange(opencv_image, lb.astype(np.uint8), ub.astype(np.uint8))
        mask = cv.bitwise_or(mask, color_mask)

    # Optionally return a filtered image preserving original colors
    if keep_original_colors:
        logger.debug("Returning filtered image with original colors preserved for matching pixels.")
        filtered_image = np.zeros_like(opencv_image)
        filtered_image[mask > 0] = opencv_image[mask > 0]
        return filtered_image

    return cast("npt.NDArray[np.uint8]", mask)
