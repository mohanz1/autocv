"""Interactive HSV/Canny filter tuning widget backed by AutoCV frames."""

from __future__ import annotations

__all__ = ("ImageFilter",)

from typing import Any, Final, Self, cast

import numpy as np
import numpy.typing as npt

import cv2 as cv

from .models import FilterSettings

ESC_CODE: Final[int] = 27
WINDOW_FILTER: Final[str] = "Image Filter"
WINDOW_TRACKBARS: Final[str] = "Trackbars"
COLOR_IMAGE_NDIMS: Final[int] = 3
BGR_CHANNELS: Final[int] = 3

ImageArray = npt.NDArray[np.uint8]


def nothing(_: int) -> None:
    """Dummy callback function for trackbar events."""
    return


def _require_color_image(image: ImageArray) -> ImageArray:
    """Return ``image`` when it is a 3-channel BGR array, otherwise raise ``ValueError``."""
    if image.ndim != COLOR_IMAGE_NDIMS or image.shape[-1] != BGR_CHANNELS:
        msg = "ImageFilter requires a BGR image with 3 channels."
        raise ValueError(msg)
    return image


def _apply_channel_offset(channel: ImageArray, delta: int) -> ImageArray:
    """Apply a signed offset to an HSV channel with saturating uint8 arithmetic."""
    adjusted = channel.astype(np.int16) + delta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


class FilterEngine:
    """Encapsulate filtering operations based on current settings."""

    def __init__(self, image: ImageArray) -> None:
        self.image = _require_color_image(image)
        self.hsv_image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        self.filter_settings = FilterSettings()

    def apply(self) -> ImageArray:
        """Apply HSV, Canny, and morphology filters."""
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
        hsv_mask = cv.inRange(self.hsv_image, cast("Any", lower), cast("Any", upper))
        hsv_filtered = cv.bitwise_and(self.hsv_image, self.hsv_image, mask=hsv_mask)

        hsv_filtered[..., 1] = _apply_channel_offset(
            cast("ImageArray", hsv_filtered[..., 1]),
            self.filter_settings.s_add - self.filter_settings.s_subtract,
        )
        hsv_filtered[..., 2] = _apply_channel_offset(
            cast("ImageArray", hsv_filtered[..., 2]),
            self.filter_settings.v_add - self.filter_settings.v_subtract,
        )

        bgr_filtered = cv.cvtColor(hsv_filtered, cv.COLOR_HSV2BGR).astype(np.uint8, copy=False)
        return self._apply_morphology(self._apply_canny(bgr_filtered))

    def _apply_canny(self, image: ImageArray) -> ImageArray:
        """Apply Canny edge detection if thresholds are set."""
        if self.filter_settings.canny_threshold1 > 0 and self.filter_settings.canny_threshold2 > 0:
            return cast(
                "ImageArray",
                cv.Canny(
                    image,
                    self.filter_settings.canny_threshold1,
                    self.filter_settings.canny_threshold2,
                ),
            )
        return image

    def _apply_morphology(self, image: ImageArray) -> ImageArray:
        """Apply erosion then dilation if configured."""
        result: ImageArray = image
        if self.filter_settings.erode_kernel_size > 0:
            erode_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (self.filter_settings.erode_kernel_size, self.filter_settings.erode_kernel_size),
            )
            result = cv.erode(result, erode_kernel)

        if self.filter_settings.dilate_kernel_size > 0:
            dilate_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (self.filter_settings.dilate_kernel_size, self.filter_settings.dilate_kernel_size),
            )
            result = cv.dilate(result, dilate_kernel)

        return result


class ImageFilter:
    """Applies HSV filtering, Canny edge detection, and morphological operations to an image.

    Provides a GUI with trackbars to adjust filter settings in real-time.
    """

    def __init__(self: Self, image: ImageArray) -> None:
        """Initialise the ImageFilter with an input image and set up the GUI.

        Args:
            image: Input image in BGR format to be filtered.
        """
        self.engine = FilterEngine(image)
        self.filter_settings = self.engine.filter_settings
        self.image = image

        self._create_windows_and_trackbars()

        # Initially update and display the filtered image.
        self.filtered_image = self.get_filtered_image()
        cv.imshow(WINDOW_FILTER, self.filtered_image)

        # Start the event loop.
        while cv.getWindowProperty(WINDOW_TRACKBARS, 0) >= 0:
            k = cv.waitKey(1)
            if k == ESC_CODE:
                break

            # Update the filtered image (get_filtered_image() updates the filter settings internally).
            self.filtered_image = self.get_filtered_image()
            cv.imshow(WINDOW_FILTER, self.filtered_image)

        cv.destroyAllWindows()

    def _ensure_engine(self: Self) -> FilterEngine:
        """Lazily construct the filter engine for tests that bypass ``__init__``."""
        if not hasattr(self, "engine"):
            if not hasattr(self, "image"):
                msg = "ImageFilter requires an image to initialise the filter engine."
                raise ValueError(msg)
            self.engine = FilterEngine(self.image)
        # Keep filter_settings in sync when manually assigned in tests.
        if hasattr(self, "filter_settings") and self.filter_settings is not self.engine.filter_settings:
            self.engine.filter_settings = self.filter_settings
        else:
            self.filter_settings = self.engine.filter_settings
        return self.engine

    def _create_windows_and_trackbars(self: Self) -> None:
        """Create UI windows and trackbars for interactive tuning."""
        cv.namedWindow(WINDOW_FILTER)
        cv.namedWindow(WINDOW_TRACKBARS, cv.WINDOW_NORMAL)

        trackbars: tuple[tuple[str, int, int], ...] = (
            ("H min", self.engine.filter_settings.h_min, 179),
            ("H max", self.engine.filter_settings.h_max, 179),
            ("S min", self.engine.filter_settings.s_min, 255),
            ("S max", self.engine.filter_settings.s_max, 255),
            ("V min", self.engine.filter_settings.v_min, 255),
            ("V max", self.engine.filter_settings.v_max, 255),
            ("S add", self.engine.filter_settings.s_add, 255),
            ("S subtract", self.engine.filter_settings.s_subtract, 255),
            ("V add", self.engine.filter_settings.v_add, 255),
            ("V subtract", self.engine.filter_settings.v_subtract, 255),
            ("Canny 1", self.engine.filter_settings.canny_threshold1, 500),
            ("Canny 2", self.engine.filter_settings.canny_threshold2, 500),
            ("Erode", self.engine.filter_settings.erode_kernel_size, 10),
            ("Dilate", self.engine.filter_settings.dilate_kernel_size, 10),
        )

        for name, initial, max_value in trackbars:
            cv.createTrackbar(name, WINDOW_TRACKBARS, initial, max_value, nothing)

    def update_filter_settings(self: Self) -> None:
        """Synchronise the stored settings with the active trackbar positions."""
        engine = self._ensure_engine()
        get = cv.getTrackbarPos
        settings = engine.filter_settings
        settings.h_min = get("H min", WINDOW_TRACKBARS)
        settings.h_max = get("H max", WINDOW_TRACKBARS)
        settings.s_min = get("S min", WINDOW_TRACKBARS)
        settings.s_max = get("S max", WINDOW_TRACKBARS)
        settings.v_min = get("V min", WINDOW_TRACKBARS)
        settings.v_max = get("V max", WINDOW_TRACKBARS)
        settings.s_add = get("S add", WINDOW_TRACKBARS)
        settings.s_subtract = get("S subtract", WINDOW_TRACKBARS)
        settings.v_add = get("V add", WINDOW_TRACKBARS)
        settings.v_subtract = get("V subtract", WINDOW_TRACKBARS)
        settings.canny_threshold1 = get("Canny 1", WINDOW_TRACKBARS)
        settings.canny_threshold2 = get("Canny 2", WINDOW_TRACKBARS)
        settings.erode_kernel_size = get("Erode", WINDOW_TRACKBARS)
        settings.dilate_kernel_size = get("Dilate", WINDOW_TRACKBARS)

    def get_filtered_image(self: Self) -> ImageArray:
        """Apply the current filter settings and return the filtered image.

        The method updates the filter settings, applies an HSV filter, performs Canny edge detection,
        and applies erosion and dilation if specified.

        Returns:
            Filtered image after applying HSV/Canny and morphology.
        """
        # Update the filter settings from trackbar positions.
        self.update_filter_settings()
        engine = self._ensure_engine()
        return engine.apply()

    @property
    def filter_settings(self) -> FilterSettings:
        """Expose active filter settings for compatibility with the AutoCV API."""
        return self._filter_settings

    @filter_settings.setter
    def filter_settings(self, value: FilterSettings) -> None:
        self._filter_settings = value
        if hasattr(self, "engine"):
            self.engine.filter_settings = value
