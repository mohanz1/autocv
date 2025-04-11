"""This module defines the ImageFilter class used for applying various image processing techniques.

It supports HSV filtering, Canny edge detection, erosion, and dilation operations using OpenCV.
A user interface with trackbars is provided for real-time adjustment of filter settings.
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
    """Dummy callback function for trackbar events."""


class ImageFilter:
    """Applies HSV filtering, Canny edge detection, and morphological operations to an image.

    Provides a GUI with trackbars to adjust filter settings in real-time.
    """

    def __init__(self: Self, image: npt.NDArray[np.uint8]) -> None:
        """Initializes the ImageFilter with an input image and sets up the GUI.

        Args:
            image: The input image (in BGR format) to be filtered.
        """
        self.image = image
        self.hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Initialize filter settings from the FilterSettings dataclass.
        self.filter_settings = FilterSettings()

        # Create windows for displaying the filtered image and trackbars.
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

        # Initially update and display the filtered image.
        self.filtered_image = self.get_filtered_image()
        cv.imshow("Image Filter", self.filtered_image)

        # Start the event loop.
        while cv.getWindowProperty("Trackbars", 0) >= 0:
            k = cv.waitKey(1)
            if k == ESC_CODE:
                break

            # Update the filtered image (get_filtered_image() updates the filter settings internally).
            self.filtered_image = self.get_filtered_image()
            cv.imshow("Image Filter", self.filtered_image)

        cv.destroyAllWindows()

    def update_filter_settings(self: Self) -> None:
        """Updates filter settings based on current trackbar positions."""
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
        """Applies the current filter settings to the image and returns the result.

        The method updates the filter settings, applies an HSV filter, performs Canny edge detection,
        and applies erosion and dilation if specified.

        Returns:
            The filtered image.
        """
        # Update the filter settings from trackbar positions.
        self.update_filter_settings()

        # Apply HSV filtering.
        hsv_mask = cv.inRange(
            self.hsv_image,
            np.array(
                [
                    self.filter_settings.h_min,
                    self.filter_settings.s_min,
                    self.filter_settings.v_min,
                ]
            ),
            np.array(
                [
                    self.filter_settings.h_max,
                    self.filter_settings.s_max,
                    self.filter_settings.v_max,
                ]
            ),
        )
        hsv_filtered = cv.bitwise_and(self.hsv_image, self.hsv_image, mask=hsv_mask)

        # Adjust saturation and value channels.
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

        # Convert the filtered HSV image back to BGR format.
        bgr_filtered = cv.cvtColor(hsv_filtered, cv.COLOR_HSV2BGR)

        # Apply Canny edge detection if thresholds are set.
        if self.filter_settings.canny_threshold1 > 0 and self.filter_settings.canny_threshold2 > 0:
            edges = cv.Canny(
                bgr_filtered,
                self.filter_settings.canny_threshold1,
                self.filter_settings.canny_threshold2,
            )
        else:
            edges = bgr_filtered

        # Apply erosion.
        if self.filter_settings.erode_kernel_size > 0:
            erode_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (self.filter_settings.erode_kernel_size, self.filter_settings.erode_kernel_size),
            )
            edges = cv.erode(edges, erode_kernel)
        # Apply dilation.
        if self.filter_settings.dilate_kernel_size > 0:
            dilate_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (self.filter_settings.dilate_kernel_size, self.filter_settings.dilate_kernel_size),
            )
            edges = cv.dilate(edges, dilate_kernel)

        return cast("npt.NDArray[np.uint8]", edges)
