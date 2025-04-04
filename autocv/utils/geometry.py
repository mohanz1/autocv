"""Utility functions for working with shapes.

This module provides helper functions to get the center of a shape, obtain a random point within a shape,
and sort shapes based on various criteria.
"""

from __future__ import annotations

__all__ = ("get_center", "get_random_point", "sort_shapes")

from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt


def get_center(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """Return the center (x, y) of a shape.

    If the shape is a rectangle (x, y, width, height), returns:
        (x + width // 2, y + height // 2).

    If the shape is a contour (a NumPy array of points), returns the centroid computed
    from image moments (m10/m00, m01/m00).

    Args:
        shape: Either a tuple of four integers (x, y, w, h) representing a rectangle,
            or a NumPy array representing a contour with shape (N, 2) or (N, 1, 2).

    Returns:
        A tuple (center_x, center_y) as integers.

    Raises:
        ValueError: If the contour area (m00) is zero.
    """
    if isinstance(shape, np.ndarray):
        # Reshape contour to (N, 2) if necessary (e.g., when shape is (N, 1, 2))
        contour = shape.reshape(-1, 2) if shape.ndim == 3 and shape.shape[1] == 1 else shape

        # Calculate centroid using image moments
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            raise ValueError
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """Return a random point (x, y) within a shape.

    If the shape is a rectangle (x, y, width, height), returns a random point within that rectangle.
    If the shape is a contour (a NumPy array of points), returns a random point that lies inside the contour.

    Args:
        shape: Either a tuple of four integers (x, y, w, h) representing a rectangle,
            or a NumPy array representing a contour with shape (N, 2) or (N, 1, 2).

    Returns:
        A tuple (rand_x, rand_y) as integers representing a random point inside the shape.

    Raises:
        ValueError: If the contour is empty or invalid, or if the shape cannot be unpacked into (x, y, w, h).
    """
    if isinstance(shape, np.ndarray):
        # Convert contour to int32 and get its bounding rectangle
        int_shape = shape.astype(np.int32)
        x, y, w, h = cv2.boundingRect(int_shape)

        while True:
            # Generate a random candidate point within the bounding rectangle
            rx = np.random.randint(x, x + w)
            ry = np.random.randint(y, y + h)

            # Check if the point is inside the contour
            if cv2.pointPolygonTest(int_shape, (rx, ry), measureDist=False) > 0:
                return rx, ry

    # Rectangle case
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
    """Sort shapes based on the specified criterion.

    Shapes can be either rectangles (x, y, w, h) or contours (NumPy arrays of points).
    Sorting is performed based on the center of each shape, computed using `get_center`.

    Sorting options:
        - "inner_outer": Sort in ascending order of distance from the window center.
        - "outer_inner": Sort in descending order of distance from the window center.
        - "left_right":  Sort in ascending order of the x-coordinate of the shape center.
        - "right_left":  Sort in descending order of the x-coordinate of the shape center.
        - "top_bottom":  Sort in ascending order of the y-coordinate of the shape center.
        - "bottom_top":  Sort in descending order of the y-coordinate of the shape center.

    Args:
        window_size: A tuple (width, height) representing the size of the window. Used to compute the window center.
        shapes: A list of shapes, each of which is either a tuple (x, y, w, h) or a NumPy array representing a contour.
        sort_by: The sorting criterion. Must be one of "inner_outer", "outer_inner", "left_right",
            "right_left", "top_bottom", or "bottom_top".

    Returns:
        A sorted list of shapes according to the specified criterion.

    Raises:
        ValueError: If any shape is unrecognized or has an area of zero when calculating the center.
    """

    def distance_from_window_center(s: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> float:
        """Compute the squared distance from the shape's center to the window center."""
        center_x, center_y = get_center(s)
        window_center_x = window_size[0] // 2
        window_center_y = window_size[1] // 2
        return (center_x - window_center_x) ** 2 + (center_y - window_center_y) ** 2

    if sort_by in {"inner_outer", "outer_inner"}:
        reverse_sort = sort_by == "outer_inner"
        shapes_sorted = sorted(shapes, key=distance_from_window_center, reverse=reverse_sort)
    elif sort_by in {"left_right", "right_left"}:
        reverse_sort = sort_by == "right_left"
        shapes_sorted = sorted(shapes, key=lambda s: get_center(s)[0], reverse=reverse_sort)
    elif sort_by in {"top_bottom", "bottom_top"}:
        reverse_sort = sort_by == "bottom_top"
        shapes_sorted = sorted(shapes, key=lambda s: get_center(s)[1], reverse=reverse_sort)
    else:
        shapes_sorted = shapes

    return shapes_sorted
