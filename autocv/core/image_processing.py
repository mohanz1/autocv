import logging
from collections.abc import Sequence
from typing import cast

import cv2 as cv
import numpy as np
import numpy.typing as npt

__all__ = ("filter_colors",)


logger = logging.getLogger(__name__)


def filter_colors(
    opencv_image: npt.NDArray[np.uint8],
    colors: tuple[int, int, int] | Sequence[tuple[int, int, int]],
    tolerance: int = 0,
    keep_original_colors: bool = False,
) -> npt.NDArray[np.uint8]:
    """Filter out all colors from the image that are not in the specified list of colors with a given tolerance.

    Args:
    ----
        opencv_image (npt.NDArray[np.uint8]): The image to filter.
        colors (Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]): A sequence of RGB tuples or a sequence of
            sequences containing RGB tuples.
        tolerance (int): The color tolerance in the range of 0-255.
        keep_original_colors (bool): If True, the returned value will be a copy of the input image where all
            non-matching pixels are set to black.

    Returns:
    -------
        npt.NDArray[np.uint8]: The mask of the filtered image.
    """
    # Convert color_values to numpy array
    color_values = np.array(colors, dtype=np.int16)

    # If color_values has only one dimension, add another dimension to make it 2D
    if color_values.ndim == 1:
        color_values = color_values[np.newaxis, :]

    logger.debug(f"Filtering with {len(color_values)} color(s) with {tolerance=} and {keep_original_colors=}.")

    # Convert color_values from RGB to BGR order
    color_values = color_values[..., ::-1]

    # Create lower and upper bounds for each color
    lower_bounds = np.clip(color_values - tolerance, 0, 255)
    upper_bounds = np.clip(color_values + tolerance, 0, 255)

    # Create a mask for each color and combine them using bitwise OR
    mask = cv.inRange(opencv_image, lower_bounds[0], upper_bounds[0])
    logger.debug(
        f"Filtering color 1/{len(color_values)} with lower bound {lower_bounds[0]} and upper bound {upper_bounds[0]}.",
    )
    for i, (lb, ub) in enumerate(zip(lower_bounds[1:], upper_bounds[1:]), start=2):
        logger.debug(f"Filtering color {i}/{len(color_values)} with lower bound {lb} and upper bound {ub}.")
        color_mask = cv.inRange(opencv_image, lb, ub)
        mask = cv.bitwise_or(mask, color_mask)

    # If keep_original_colors is True, create a copy of the image and set all non-matching pixels to black
    if keep_original_colors:
        logger.debug("Reverting to original colors.")
        filtered_image = np.zeros_like(opencv_image)
        filtered_image[mask > 0] = opencv_image[mask > 0]
        return filtered_image
    return cast(npt.NDArray[np.uint8], mask)
