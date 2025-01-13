"""This module provides utility functions for working with shapes."""

from __future__ import annotations

__all__ = ("get_center", "get_random_point", "sort_shapes")

from typing import Literal

import cv2
import cv2 as cv
import numpy as np
import numpy.typing as npt


def get_center(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """Returns the center (x, y) of a shape.

    - If `shape` is a rect (x, y, width, height), the function returns:
        (x + width // 2, y + height // 2)

    - If `shape` is a contour (np.ndarray of points), the function returns
      the centroid calculated by image moments (M10 / M00, M01 / M00).

    Args:
        shape: Either
            - A tuple of four integers (x, y, w, h).
            - A NumPy array (contour) of shape (N, 2) or (N, 1, 2).

    Returns:
        A tuple (center_x, center_y) as integers.

    Raises:
        ValueError: If the area (M00) of the contour is zero or
                    if the shape is not recognized as a tuple of length 4
                    or a valid contour.
    """
    if isinstance(shape, np.ndarray):
        # Convert contour to the correct shape if needed (N, 2)
        # Many OpenCV contours come in shape (N, 1, 2), so let's reshape if needed
        contour = shape.reshape(-1, 2) if shape.ndim == 3 and shape.shape[1] == 1 else shape

        # Calculate centroid via moments
        moments = cv.moments(contour)
        if moments["m00"] == 0:
            raise ValueError
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """Returns a random point (x, y) from a shape.

    - If `shape` is a rect (x, y, width, height), it returns a random point
      within that bounding box.

    - If `shape` is a contour (np.ndarray of points), it returns a random
      point chosen from the contour points themselves.

    Args:
        shape: Either
            - A tuple of four integers (x, y, w, h).
            - A NumPy array (contour) of shape (N, 2) or (N, 1, 2).

    Returns:
        A tuple (rand_x, rand_y) as integers.

    Raises:
        ValueError: If the contour array is empty or invalid,
                    or if the shape cannot be unpacked into (x, y, w, h).
    """
    if isinstance(shape, np.ndarray):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(shape.astype(np.int32))

        while True:
            # Generate random candidate in bounding box
            rx = np.random.randint(x, x + w)
            ry = np.random.randint(y, y + h)

            if cv2.pointPolygonTest(shape.astype(np.int32), (rx, ry), measureDist=False) > 0:
                return rx, ry

    # Rect case
    x, y, w, h = shape
    rand_x = np.random.randint(x, x + w)
    rand_y = np.random.randint(y, y + h)
    return rand_x, rand_y


def sort_shapes(
    window_size: tuple[int, int],
    shapes: list[tuple[int, int, int, int] | npt.NDArray[np.uintp]],
    sort_by: Literal[
        "inner_outer",
        "outer_inner",
        "left_right",
        "right_left",
        "top_bottom",
        "bottom_top",
    ],
) -> list[tuple[int, int, int, int] | npt.NDArray[np.uintp]]:
    """Sorts a list of shapes according to the specified criterion.

    The shapes can be either rectangles (x, y, w, h) or contours (NumPy arrays of points).
    The sorting is based on the center of each shape, obtained via ``get_center``.

    The available sorting methods are:

    - ``inner_outer``: Sort by ascending distance from the window center.
    - ``outer_inner``: Sort by descending distance from the window center.
    - ``left_right``:  Sort by ascending x-coordinate of the shape center.
    - ``right_left``:  Sort by descending x-coordinate of the shape center.
    - ``top_bottom``:  Sort by ascending y-coordinate of the shape center.
    - ``bottom_top``:  Sort by descending y-coordinate of the shape center.

    Args:
        window_size (tuple[int, int]):
            The (width, height) of the window. Used to determine the window's center
            for distance-based sorting.
        shapes (list[tuple[int, int, int, int] | numpy.typing.NDArray[numpy.uintp]]):
            A list of shapes, where each shape is either:
             - A tuple of four integers (x, y, w, h).
             - A NumPy array (contour) of shape (N, 2) or (N, 1, 2).
        sort_by (Literal["inner_outer", "outer_inner", "left_right", "right_left", "top_bottom", "bottom_top"]):
            The sorting criterion.

    Returns:
        list[tuple[int, int, int, int] | numpy.typing.NDArray[numpy.uintp]]:
            The sorted list of shapes.

    Raises:
        ValueError: If any of the shapes cannot be recognized or has an area of zero
            when calculating the center (e.g., for empty contours).
    """
    shapes_sorted = shapes
    # Helper to compute distance from window center.

    def distance_from_window_center(s: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> float:
        center_x, center_y = get_center(s)
        # Window center:
        window_center_x = window_size[0] // 2
        window_center_y = window_size[1] // 2
        return (center_x - window_center_x) ** 2 + (center_y - window_center_y) ** 2

    # Decide which key function to use for sorting, and whether to reverse.
    if sort_by in {"inner_outer", "outer_inner"}:
        # Sort by distance from window center.
        reverse_sort = sort_by == "outer_inner"
        shapes_sorted = sorted(
            shapes,
            key=distance_from_window_center,
            reverse=reverse_sort,
        )
    elif sort_by in {"left_right", "right_left"}:
        # Sort by center x-coordinate.
        reverse_sort = sort_by == "right_left"
        shapes_sorted = sorted(
            shapes,
            key=lambda s: get_center(s)[0],
            reverse=reverse_sort,
        )
    elif sort_by in {"top_bottom", "bottom_top"}:
        # Sort by center y-coordinate.
        reverse_sort = sort_by == "bottom_top"
        shapes_sorted = sorted(
            shapes,
            key=lambda s: get_center(s)[1],
            reverse=reverse_sort,
        )

    return shapes_sorted
