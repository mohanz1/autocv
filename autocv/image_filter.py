"""This module defines the ImageFilter class which is used for applying various image processing techniques.

This includes HSV filtering, Canny edge detection, eroding, and dilating operations. It uses OpenCV for image
manipulation and provides a user interface using trackbars to adjust the filter settings in real-time.
"""

from __future__ import annotations

__all__ = ("ImageFilter",)

from typing import cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .models import FilterSettings

ESC_CODE = 27


def nothing(_: int) -> None:
    pass


class ImageFilter:
    """A class for applying an HSV filter, Canny edge detection, erode, and dilate operations to an image."""

    def __init__(self: Self, image: npt.NDArray[np.uint8]) -> None:
        """Initialize the HSV filter with an input image.

        Args:
            image: The input image to be filtered.
        """
        self.image = image
        self.hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Initialize trackbar values for HSV filter
        self.filter_settings = FilterSettings()

        # Create GUI with trackbars
        cv.namedWindow("Image Filter")
        cv.namedWindow("Trackbars", cv.WINDOW_NORMAL)
        cv.createTrackbar("H min", "Trackbars", self.filter_settings.h_min, 179, nothing)
        cv.createTrackbar("H max", "Trackbars", self.filter_settings.h_max, 179, nothing)
        cv.createTrackbar("S min", "Trackbars", self.filter_settings.s_min, 255, nothing)
        cv.createTrackbar("S max", "Trackbars", self.filter_settings.s_max, 255, nothing)
        cv.createTrackbar("V min", "Trackbars", self.filter_settings.v_min, 255, nothing)
        cv.createTrackbar("V max", "Trackbars", self.filter_settings.v_max, 255, nothing)
        cv.createTrackbar("S add", "Trackbars", self.filter_settings.s_add, 255, nothing)
        cv.createTrackbar("S subtract", "Trackbars", self.filter_settings.s_subtract, 255, nothing)
        cv.createTrackbar("V add", "Trackbars", self.filter_settings.v_add, 255, nothing)
        cv.createTrackbar("V subtract", "Trackbars", self.filter_settings.v_subtract, 255, nothing)
        cv.createTrackbar("Canny 1", "Trackbars", self.filter_settings.canny_threshold1, 500, nothing)
        cv.createTrackbar("Canny 2", "Trackbars", self.filter_settings.canny_threshold2, 500, nothing)
        cv.createTrackbar("Erode", "Trackbars", self.filter_settings.erode_kernel_size, 10, nothing)
        cv.createTrackbar("Dilate", "Trackbars", self.filter_settings.dilate_kernel_size, 10, nothing)

        # Initialize filtered image
        self.filtered_image = self.get_filtered_image()

        # Display filtered image
        cv.imshow("Image Filter", self.filtered_image)

        # Start event loop
        while cv.getWindowProperty("Trackbars", 0) >= 0:
            k = cv.waitKey(1)
            if k == ESC_CODE:
                # Exit on ESC key press
                break

            # check if Trackbars was closed
            try:
                cv.getWindowProperty("Trackbars", 0)
            except cv.error:
                break

            # Get current trackbar values for HSV filter
            self.filter_settings.h_min = cv.getTrackbarPos("H min", "Trackbars")
            self.filter_settings.h_max = cv.getTrackbarPos("H max", "Trackbars")
            self.filter_settings.s_min = cv.getTrackbarPos("S min", "Trackbars")
            self.filter_settings.s_max = cv.getTrackbarPos("S max", "Trackbars")
            self.filter_settings.v_min = cv.getTrackbarPos("V min", "Trackbars")
            self.filter_settings.v_max = cv.getTrackbarPos("V max", "Trackbars")

            self.filter_settings.s_add = cv.getTrackbarPos("S add", "Trackbars")
            self.filter_settings.s_subtract = cv.getTrackbarPos("S subtract", "Trackbars")
            self.filter_settings.v_add = cv.getTrackbarPos("V add", "Trackbars")
            self.filter_settings.v_subtract = cv.getTrackbarPos("V subtract", "Trackbars")

            # Get current trackbar values for Canny edge detection
            self.filter_settings.canny_threshold1 = cv.getTrackbarPos("Canny 1", "Trackbars")
            self.filter_settings.canny_threshold2 = cv.getTrackbarPos("Canny 2", "Trackbars")

            # Get current trackbar values for erode and dilate operations
            self.filter_settings.erode_kernel_size = cv.getTrackbarPos("Erode", "Trackbars")
            self.filter_settings.dilate_kernel_size = cv.getTrackbarPos("Dilate", "Trackbars")

            # Update filtered image
            self.filtered_image = self.get_filtered_image()

            # Display filtered image
            cv.imshow("Image Filter", self.filtered_image)

        # Release resources
        cv.destroyAllWindows()

    def update_filter_settings(self: Self) -> None:
        """Update filter settings based on current trackbar values."""
        self.filter_settings.h_min = cv.getTrackbarPos("H min", "Trackbars")
        self.filter_settings.h_max = cv.getTrackbarPos("H max", "Trackbars")
        self.filter_settings.s_min = cv.getTrackbarPos("S min", "Trackbars")
        self.filter_settings.s_max = cv.getTrackbarPos("S max", "Trackbars")
        self.filter_settings.v_min = cv.getTrackbarPos("V min", "Trackbars")
        self.filter_settings.v_max = cv.getTrackbarPos("V max", "Trackbars")
        self.filter_settings.s_add = cv.getTrackbarPos("S add", "Trackbars")
        self.filter_settings.s_subtract = cv.getTrackbarPos("S subtract", "Trackbars")
        self.filter_settings.v_add = cv.getTrackbarPos("V add", "Trackbars")
        self.filter_settings.v_subtract = cv.getTrackbarPos("V subtract", "Trackbars")
        self.filter_settings.canny_threshold1 = cv.getTrackbarPos("Canny 1", "Trackbars")
        self.filter_settings.canny_threshold2 = cv.getTrackbarPos("Canny 2", "Trackbars")
        self.filter_settings.erode_kernel_size = cv.getTrackbarPos("Erode", "Trackbars")
        self.filter_settings.dilate_kernel_size = cv.getTrackbarPos("Dilate", "Trackbars")

    def get_filtered_image(self: Self) -> npt.NDArray[np.uint8]:
        """Get the image with filters applied.

        Returns:
            The filtered image.
        """
        # Update filter settings
        self.update_filter_settings()

        # Apply HSV filter
        hsv_mask = cv.inRange(
            self.hsv_image,
            np.array([
                self.filter_settings.h_min,
                self.filter_settings.s_min,
                self.filter_settings.v_min,
            ]),
            np.array([
                self.filter_settings.h_max,
                self.filter_settings.s_max,
                self.filter_settings.v_max,
            ]),
        )
        hsv_filtered = cv.bitwise_and(self.hsv_image, self.hsv_image, mask=hsv_mask)
        hsv_filtered[..., 1] = np.clip(
            hsv_filtered[..., 1] + self.filter_settings.s_add - self.filter_settings.s_subtract,
            0,
            255,
        )
        hsv_filtered[..., 2] = np.clip(
            hsv_filtered[..., 2] + self.filter_settings.v_add - self.filter_settings.v_subtract,
            0,
            255,
        )

        bgr_filtered = cv.cvtColor(hsv_filtered, cv.COLOR_HSV2BGR)

        # Apply Canny edge detection
        if self.filter_settings.canny_threshold1 > 0 and self.filter_settings.canny_threshold2 > 0:
            edges = cv.Canny(
                bgr_filtered,
                self.filter_settings.canny_threshold1,
                self.filter_settings.canny_threshold2,
            )
        else:
            edges = bgr_filtered

        # Apply erode and dilate operations
        if self.filter_settings.erode_kernel_size > 0:
            erode_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (
                    self.filter_settings.erode_kernel_size,
                    self.filter_settings.erode_kernel_size,
                ),
            )
            edges = cv.erode(edges, erode_kernel)
        if self.filter_settings.dilate_kernel_size > 0:
            dilate_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (
                    self.filter_settings.dilate_kernel_size,
                    self.filter_settings.dilate_kernel_size,
                ),
            )
            edges = cv.dilate(edges, dilate_kernel)

        # Return
        return cast("npt.NDArray[np.uint8]", edges)
