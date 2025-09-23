"""Interactive HSV/Canny filter tuning widget backed by AutoCV frames."""

from __future__ import annotations

__all__ = ("ImageFilter",)

from typing import cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .models import FilterSettings

ESC_CODE = 27
WINDOW_NAME_IMAGE = "Image Filter"
WINDOW_NAME_TRACKBARS = "Trackbars"
TRACKBAR_SPECS: tuple[tuple[str, str, int], ...] = (
    ("H min", "h_min", 179),
    ("H max", "h_max", 179),
    ("S min", "s_min", 255),
    ("S max", "s_max", 255),
    ("V min", "v_min", 255),
    ("V max", "v_max", 255),
    ("S add", "s_add", 255),
    ("S subtract", "s_subtract", 255),
    ("V add", "v_add", 255),
    ("V subtract", "v_subtract", 255),
    ("Canny 1", "canny_threshold1", 500),
    ("Canny 2", "canny_threshold2", 500),
    ("Erode", "erode_kernel_size", 10),
    ("Dilate", "dilate_kernel_size", 10),
)


def nothing(_: int) -> None:
    """Dummy callback function for trackbar events."""


class ImageFilter:
    """Applies HSV filtering, Canny edge detection, and morphological operations to an image."""

    def __init__(self: Self, image: npt.NDArray[np.uint8]) -> None:
        """Initializes the ImageFilter with an input image and sets up the GUI."""
        self.image = image
        self.hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.filter_settings = FilterSettings()

        cv.namedWindow(WINDOW_NAME_IMAGE)
        cv.namedWindow(WINDOW_NAME_TRACKBARS, cv.WINDOW_NORMAL)
        for name, attribute, maximum in TRACKBAR_SPECS:
            cv.createTrackbar(
                name,
                WINDOW_NAME_TRACKBARS,
                getattr(self.filter_settings, attribute),
                maximum,
                nothing,
            )

        self.filtered_image = self.get_filtered_image()
        cv.imshow(WINDOW_NAME_IMAGE, self.filtered_image)

        while cv.getWindowProperty(WINDOW_NAME_TRACKBARS, cv.WND_PROP_VISIBLE) >= 1:
            key = cv.waitKey(1)
            if key == ESC_CODE:
                break
            self.filtered_image = self.get_filtered_image()
            cv.imshow(WINDOW_NAME_IMAGE, self.filtered_image)

        try:
            cv.destroyWindow(WINDOW_NAME_IMAGE)
            cv.destroyWindow(WINDOW_NAME_TRACKBARS)
        except cv.error:
            pass
        cv.destroyAllWindows()

    def update_filter_settings(self: Self) -> None:
        """Synchronise the stored settings with the active trackbar positions."""
        for name, attribute, _ in TRACKBAR_SPECS:
            setattr(self.filter_settings, attribute, cv.getTrackbarPos(name, WINDOW_NAME_TRACKBARS))

    def get_filtered_image(self: Self) -> npt.NDArray[np.uint8]:
        """Applies the current filter settings to the image and returns the result."""
        self.update_filter_settings()

        lower = np.array(
            [
                self.filter_settings.h_min,
                self.filter_settings.s_min,
                self.filter_settings.v_min,
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                self.filter_settings.h_max,
                self.filter_settings.s_max,
                self.filter_settings.v_max,
            ],
            dtype=np.uint8,
        )
        hsv_mask = cv.inRange(self.hsv_image, lower, upper)
        hsv_filtered = cv.bitwise_and(self.hsv_image, self.hsv_image, mask=hsv_mask)

        saturation = hsv_filtered[..., 1].astype(np.int16)
        saturation += self.filter_settings.s_add
        saturation -= self.filter_settings.s_subtract
        hsv_filtered[..., 1] = np.clip(saturation, 0, 255).astype(np.uint8)

        value = hsv_filtered[..., 2].astype(np.int16)
        value += self.filter_settings.v_add
        value -= self.filter_settings.v_subtract
        hsv_filtered[..., 2] = np.clip(value, 0, 255).astype(np.uint8)

        bgr_filtered = cv.cvtColor(hsv_filtered, cv.COLOR_HSV2BGR)

        if self.filter_settings.canny_threshold1 > 0 and self.filter_settings.canny_threshold2 > 0:
            gray = cv.cvtColor(bgr_filtered, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(
                gray,
                self.filter_settings.canny_threshold1,
                self.filter_settings.canny_threshold2,
            )
            output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        else:
            output = bgr_filtered

        for size, operation in (
            (self.filter_settings.erode_kernel_size, cv.erode),
            (self.filter_settings.dilate_kernel_size, cv.dilate),
        ):
            if size > 0:
                kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
                output = operation(output, kernel)

        return cast("npt.NDArray[np.uint8]", output)
