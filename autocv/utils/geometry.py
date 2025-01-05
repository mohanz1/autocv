from __future__ import annotations

__all__ = "get_center"

import numpy as np
import numpy.typing as npt
import cv2 as cv


def get_center(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """
    Returns the center (x, y) of a shape.

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
        if len(shape.shape) < 2 or shape.size == 0:
            raise ValueError("Contour array is empty or invalid.")

        # Convert contour to the correct shape if needed (N, 2)
        # Many OpenCV contours come in shape (N, 1, 2), so let's reshape if needed
        if shape.ndim == 3 and shape.shape[1] == 1:
            contour = shape.reshape(-1, 2)
        else:
            contour = shape

        # Calculate centroid via moments
        M = cv.moments(contour)
        if M["m00"] == 0:
            raise ValueError("Contour area (m00) is zero; cannot compute center.")
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    x, y, w, h = shape
    return x + w // 2, y + h // 2


def get_random_point(shape: tuple[int, int, int, int] | npt.NDArray[np.uintp]) -> tuple[int, int]:
    """
    Returns a random point (x, y) from a shape.

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
        # Contour case
        if len(shape.shape) < 2 or shape.size == 0:
            raise ValueError("Contour array is empty or invalid.")

        # Convert contour to shape (N, 2) if it is (N, 1, 2)
        if shape.ndim == 3 and shape.shape[1] == 1:
            contour = shape.reshape(-1, 2)
        else:
            contour = shape

        # Pick a random index and return the point
        idx = np.random.randint(0, contour.shape[0])
        point = contour[idx]
        return int(point[0]), int(point[1])

    # Rect case
    x, y, w, h = shape
    rand_x = np.random.randint(x, x + w)
    rand_y = np.random.randint(y, y + h)
    return rand_x, rand_y
