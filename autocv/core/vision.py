"""This module provides the Vision class and related utilities for image capture, processing, and OCR.

It integrates with OpenCV, pytesseract, and other libraries to perform image analysis,
text extraction, and various image manipulation operations.
"""

from __future__ import annotations

__all__ = ("Vision",)

import io
import logging
import pathlib
from typing import TYPE_CHECKING, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
import polars as pl
import win32con
import win32gui
import win32ui
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI
from typing_extensions import Self

from .decorators import check_valid_hwnd, check_valid_image
from .image_processing import filter_colors
from .window_capture import WindowCapture

if TYPE_CHECKING:
    from collections.abc import Sequence

RECTANGLE_SIDES = 4
GRESCALE_CHANNELS = 3
MAX_COLOR_VALUE = 255
ONE_HUNDRED = 100

logger = logging.getLogger(__name__)


class Vision(WindowCapture):
    """A class for image processing and optical character recognition (OCR).

    Extends the WindowCapture class to provide methods for capturing a window,
    processing the image, and extracting text and color information.
    """

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes a Vision object.

        Args:
            hwnd (int): Window handle of the target window. Defaults to -1.
        """
        super().__init__(hwnd)
        # Holds the image in OpenCV format.
        self.opencv_image: npt.NDArray[np.uint8] = np.empty(0, dtype=np.uint8)

        # Define the path to the tessdata directory and set up Tesseract configuration.
        absolute_directory = pathlib.Path(__file__).parents[1] / "data" / "traineddata"
        # noinspection PyArgumentList
        self.api = PyTessBaseAPI(
            path=str(absolute_directory),
            lang="runescape",
            psm=PSM.SPARSE_TEXT,
            oem=OEM.LSTM_ONLY,
        )
        self._config = rf"--tessdata-dir {absolute_directory} --oem 1 --psm 11"

    def set_backbuffer(self: Self, image: npt.NDArray[np.uint8] | Image.Image) -> None:
        """Sets the image buffer to the provided NumPy array or PIL Image.

        Args:
            image (npt.NDArray[np.uint8] | Image.Image): Image data used to refresh the backbuffer.
        """
        if isinstance(image, Image.Image):
            self.opencv_image = cast(
                "npt.NDArray[np.uint8]",
                cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR),
            )
        else:
            self.opencv_image = image

    @check_valid_hwnd
    def refresh(self: Self, *, set_backbuffer: bool = True) -> npt.NDArray[np.uint8] | None:
        """Captures the current window image and converts it to an OpenCV-compatible format.

        Args:
            set_backbuffer (bool): When ``True``, persist the capture to ``self.opencv_image``.
                          If False, returns the captured image.

        Raises:
            InvalidHandleError: Raised when ``self.hwnd`` is not a valid window handle.

        Returns:
            npt.NDArray[np.uint8] | None: Captured frame when ``set_backbuffer`` is ``False``; ``None`` otherwise.
        """
        # Get window dimensions.
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Get device context and compatible bitmap.
        window_dc = win32gui.GetWindowDC(self.hwnd)
        mem_dc = win32ui.CreateDCFromHandle(window_dc)
        bmp_dc = mem_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        bmp_dc.SelectObject(bitmap)

        # Copy window image data onto bitmap.
        bmp_dc.BitBlt((0, 0), (width, height), mem_dc, (0, 0), win32con.SRCCOPY)

        # Convert raw data into a format that OpenCV can read.
        signed_ints_array = bitmap.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype="uint8")
        img.shape = (height, width, 4)

        # Free resources.
        mem_dc.DeleteDC()
        bmp_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, window_dc)
        win32gui.DeleteObject(bitmap.GetHandle())

        # Ensure the image is contiguous and drop the alpha channel.
        image = np.ascontiguousarray(img[..., :3])

        if set_backbuffer:
            self.set_backbuffer(image)
            return None
        return image

    @check_valid_image
    def save_backbuffer_to_file(self: Self, file_name: str) -> None:
        """Saves the backbuffer image to a file.

        Args:
            file_name (str): Path where the backbuffer snapshot is stored.
        """
        cv.imwrite(file_name, self.opencv_image)

    @check_valid_hwnd
    def get_pixel_change(self: Self, area: tuple[int, int, int, int] | None = None) -> int:
        """Calculates the number of pixels that have changed between the current image and a newly captured image.

        Args:
            area (tuple[int, int, int, int] | None): Region of interest expressed as (x, y, width, height); ``None``
                inspects the full frame. If None, the entire image is used.

        Raises:
            InvalidImageError: Raised when the capture buffer is empty.

        Returns:
            int: Count of pixels with different intensities between frames.
        """
        # Crop and convert the current image to grayscale.
        image = self._crop_image(area, self.opencv_image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Capture a new image and convert it to grayscale.
        updated_image = self.refresh(set_backbuffer=False)
        updated_image = self._crop_image(area, updated_image)
        updated_image = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)

        # Compute and return the difference.
        diff = cv.absdiff(image, updated_image)
        return int(np.count_nonzero(diff))

    def _get_grouped_text(
        self: Self,
        image: npt.NDArray[np.uint8],
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
    ) -> pl.LazyFrame:
        """Preprocesses the image and extracts text using Tesseract OCR, grouping text data by block.

        Args:
            image (npt.NDArray[np.uint8]): Image to preprocess before OCR.
            rect (tuple[int, int, int, int] | None): Optional region of interest described as
                (x, y, width, height).
            colors (tuple[int, int, int] | list[tuple[int, int, int]] | None): Colours (RGB) to isolate before OCR.
            tolerance (int): Per-channel tolerance when matching the colour filter.

        Returns:
            pl.LazyFrame: Grouped OCR output including text and bounding boxes.
        """
        if colors:
            image = filter_colors(image, colors, tolerance)

        # Resize image to double the size.
        resized_img = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
        img = cv.bilateralFilter(resized_img, 9, 75, 75)

        # Convert to grayscale and apply thresholding.
        if len(img.shape) == GRESCALE_CHANNELS and img.shape[-1] == GRESCALE_CHANNELS:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.threshold(img, 0, MAX_COLOR_VALUE, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        img = cv.bitwise_not(img)

        # Perform OCR.
        self.api.SetImage(Image.fromarray(img))
        if rect:
            self.api.SetRectangle(*rect)
        text_data = self.api.GetTSVText(0)
        self.api.Clear()

        text = pl.scan_csv(
            io.StringIO(text_data),
            has_header=False,
            separator="\t",
            quote_char=None,
            new_columns=[
                "level",
                "page_num",
                "block_num",
                "par_num",
                "line_num",
                "word_num",
                "left",
                "top",
                "width",
                "height",
                "conf",
                "text",
            ],
        )

        # Filter out invalid or low-confidence text.
        text = text.filter((pl.col("conf") > 0) & (pl.col("conf") < ONE_HUNDRED))
        text = text.with_columns(
            [
                pl.col("text").cast(pl.Utf8),
                (pl.col("conf") / 100).alias("conf"),
            ]
        )
        text = text.filter(pl.col("text").str.strip_chars().str.len_chars() > 0)

        grouped_text = text.group_by("block_num").agg(
            [
                pl.col("word_num").max().alias("word_num"),
                pl.col("left").min().alias("left"),
                pl.col("top").min().alias("top"),
                pl.col("height").max().alias("height"),
                pl.col("conf").max().alias("confidence"),
                pl.col("text").str.concat(" ").alias("text"),
                ((pl.col("left") + pl.col("width")).max() - pl.col("left").min()).alias("width"),
            ]
        )

        sorted_text = grouped_text.sort("confidence", descending=True)
        return sorted_text.with_columns(
            [
                (pl.col("top") // 2).alias("top"),
                (pl.col("left") // 2).alias("left"),
                (pl.col("height") // 2).alias("height"),
                (pl.col("width") // 2).alias("width"),
            ]
        )

    def _crop_image(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        image: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Crops the current image or a provided image to the specified rectangle.

        Args:
            rect (tuple[int, int, int, int] | None): Region to crop in
                (x, y, width, height) form.
            image (npt.NDArray[np.uint8] | None): Explicit image to operate on; defaults to ``self.opencv_image``.

        Returns:
            npt.NDArray[np.uint8]: Cropped slice of the source image.
        """
        image = image if image is not None else self.opencv_image
        if rect:
            x, y, w, h = rect
            return image[y : y + h, x : x + w]
        return image

    @check_valid_image
    def get_text(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
        confidence: float | None = 0.8,
    ) -> list[dict[str, str | int | float | list[int]]]:
        """Extracts text from the backbuffer using Tesseract OCR.

        Only text with confidence greater than or equal to the provided threshold is returned.

        Args:
            rect (tuple[int, int, int, int] | None): Search region specified as
                (x, y, width, height).
            colors (tuple[int, int, int] | list[tuple[int, int, int]] | None): Colours (RGB) to isolate before OCR.
            tolerance (int): Per-channel tolerance when matching the colour filter.
            confidence (float | None): Minimum acceptable OCR confidence between 0 and 1.

        Returns:
            list[dict[str, str | int | float | list[int]]]: Text entries with bounding boxes and confidence levels.
        """
        sorted_text = self._get_grouped_text(self.opencv_image, rect, colors, tolerance)
        acceptable_text = sorted_text.filter(pl.col("confidence") >= confidence)
        acceptable_text = acceptable_text.select("text", "left", "top", "width", "height", "confidence")
        if rect:
            acceptable_text = acceptable_text.with_columns(pl.col("left") + rect[0], pl.col("top") + rect[1])
        result_df = acceptable_text.with_columns(pl.concat_list(["left", "top", "width", "height"]).alias("rect"))
        result_df = result_df.select(["text", "rect", "confidence"])
        return cast("list[dict[str, str | int | float | list[int]]]", result_df.collect().to_dicts())

    @check_valid_image
    def get_color(self: Self, point: tuple[int, int]) -> tuple[int, int, int]:
        """Returns the color of the pixel at the specified coordinates.

        Args:
            point (tuple[int, int]): Pixel coordinates expressed as (x, y).

        Returns:
            tuple[int, int, int]: Pixel colour as (R, G, B).

        Raises:
            InvalidImageError: Raised when ``self.opencv_image`` is empty.
            IndexError: If the coordinates are out of bounds.
        """
        x, y = point
        if not (0 <= x < self.opencv_image.shape[1] and 0 <= y < self.opencv_image.shape[0]):
            raise IndexError
        return tuple(np.flip(self.opencv_image[y, x]))

    @check_valid_image
    def find_color(
        self: Self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
    ) -> list[tuple[int, int]]:
        """Finds all pixel coordinates matching the given color within the specified tolerance.

        Args:
            color (tuple[int, int, int]): Target colour expressed as (R, G, B).
            rect (tuple[int, int, int, int] | None): Optional search region
                described as (x, y, width, height).
            tolerance (int): Allowed per-channel delta when searching for colour matches.

        Returns:
            list[tuple[int, int]]: Pixel coordinates in image space that match the colour constraint.
        """
        image = self._crop_image(rect)
        mask = filter_colors(image, color, tolerance)
        points = np.stack(np.where(mask == MAX_COLOR_VALUE)[::-1], axis=1)
        if rect:
            points = np.add(points, rect[0:2])
        return cast("list[tuple[int, int]]", points.tolist())

    @check_valid_image
    def get_average_color(self: Self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
        """Calculates the average color within a specified region.

        Args:
            rect (tuple[int, int, int, int] | None): Region to average;
                ``None`` uses the full image.

        Returns:
            tuple[int, int, int]: Average RGB value inside the requested region.
        """
        return cast("tuple[int, int, int]", tuple(int(x) for x in self._get_average_color(self.opencv_image, rect)))

    def _get_average_color(
        self: Self,
        image: npt.NDArray[np.uint8],
        rect: tuple[int, int, int, int] | None = None,
    ) -> npt.NDArray[np.int16]:
        """Calculates the average color of the specified image region.

        Args:
            image (npt.NDArray[np.uint8]): Image to preprocess before OCR.
            rect (tuple[int, int, int, int] | None): Region to sample;
                defaults to the entire frame.

        Returns:
            npt.NDArray[np.int16]: Average colour in BGR order prior to flipping to RGB.
        """
        image = self._crop_image(rect, image)
        avg_color = cv.mean(image)
        avg_color_rgb = np.array(avg_color[:3], dtype=np.int16)
        return avg_color_rgb[::-1]

    @check_valid_image
    def get_most_common_color(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        index: int = 1,
        ignore_colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[int, int, int]:
        """Determines the most common color in the specified region.

        Args:
            rect (tuple[int, int, int, int] | None): Region to average;
                ``None`` uses the full image.
            index (int): Rank of the dominant colour to extract (1-based).
            ignore_colors (tuple[int, int, int] | list[tuple[int, int, int]] | None): Colours to skip while ranking.

        Returns:
            tuple[int, int, int]: Most common RGB colour in the region.
        """
        cropped_image = self._crop_image(rect)
        reshaped_image = cropped_image.reshape(-1, 3)
        if ignore_colors is not None:
            color_values = np.array(ignore_colors, dtype=np.int16)
            if color_values.ndim == 1:
                color_values = color_values[np.newaxis, :]
            ignore_mask = np.isin(reshaped_image, color_values).any(axis=1)
            reshaped_image = np.array(reshaped_image[~ignore_mask])
        unique, counts = np.unique(reshaped_image, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        desired_index = min(index - 1, len(sorted_indices) - 1)
        most_common_color = unique[sorted_indices[desired_index]][::-1]
        return (int(most_common_color[0]), int(most_common_color[1]), int(most_common_color[2]))

    @check_valid_image
    def get_count_of_color(
        self: Self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int | None = 0,
    ) -> int:
        """Counts the number of pixels matching a given color within a tolerance.

        Args:
            color (tuple[int, int, int]): Target colour expressed as (R, G, B).
            rect (tuple[int, int, int, int] | None): Region to average;
                ``None`` uses the full image.
            tolerance (int): Allowed per-channel difference; defaults to 0 for exact matches.

        Returns:
            int: Number of pixels matching the specified colour.
        """
        cropped_image = self._crop_image(rect)
        match_mask = filter_colors(cropped_image, color, tolerance or 0)
        return int(np.count_nonzero(match_mask))

    @check_valid_image
    def get_all_colors_with_counts(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[tuple[int, int, int], int]]:
        """Retrieves all colors in the specified region along with their pixel counts.

        Args:
            rect (tuple[int, int, int, int] | None): Region to average;
                ``None`` uses the full image.

        Returns:
            list[tuple[tuple[int, int, int], int]]: Colour counts ordered by frequency.
        """
        cropped_image = self._crop_image(rect)
        reshaped_image = cropped_image.reshape(-1, 3)
        unique, counts = np.unique(reshaped_image, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_unique = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]
        return [
            ((int(bgr[2]), int(bgr[1]), int(bgr[0])), int(c))
            for bgr, c in zip(sorted_unique, sorted_counts, strict=False)
        ]

    @check_valid_image
    def get_median_color(self: Self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
        """Calculates the median color of the specified region.

        Args:
            rect (tuple[int, int, int, int] | None): Region to average;
                ``None`` uses the full image.

        Returns:
            tuple[int, int, int]: Median RGB colour inside the region.
        """
        cropped_image = self._crop_image(rect)
        reshaped_image = cropped_image.reshape(-1, 3)
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)
        return tuple(median_color[::-1].tolist())

    @staticmethod
    def _get_dominant_color(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Returns the dominant color in the given image.

        Args:
            image (npt.NDArray[np.uint8]): Input frame in BGR order.

        Returns:
            npt.NDArray[np.uint8]: Dominant colour in BGR order.
        """
        reshaped_image = image.reshape(-1, 3)
        return cast("npt.NDArray[np.uint8]", np.median(reshaped_image, axis=0).astype(np.uint8))

    @check_valid_image
    def maximize_color_match(
        self: Self,
        rect: tuple[int, int, int, int],
        initial_tolerance: int = 100,
        tolerance_step: int = 1,
    ) -> tuple[tuple[int, int, int], int]:
        """Finds the color and tolerance that best match the region's dominant color.

        Args:
            rect (tuple[int, int, int, int] | None): Region to evaluate when
                computing the dominant colour.
            initial_tolerance (int): Initial tolerance applied when searching.
            tolerance_step (int): Amount to decrease tolerance when broadening the search.

        Returns:
            tuple[tuple[int, int, int], int]: Matched RGB colour and the tolerance applied.
        """
        cropped_image = self._crop_image(rect)
        dominant_color = cast("tuple[int, int, int]", tuple(int(x) for x in self._get_dominant_color(cropped_image)))
        best_color, best_tolerance = self._find_best_color_match(
            cropped_image, dominant_color, initial_tolerance, tolerance_step
        )
        return best_color, best_tolerance

    def _find_best_color_match(
        self: Self,
        cropped_image: npt.NDArray[np.uint8],
        dominant_color: tuple[int, int, int],
        initial_tolerance: int,
        tolerance_step: int,
    ) -> tuple[tuple[int, int, int], int]:
        """Searches for the best color match within the specified tolerance range.

        Args:
            cropped_image (npt.NDArray[np.uint8]): Image region under evaluation.
            dominant_color (tuple[int, int, int]): Dominant colour used for comparison.
            initial_tolerance (int): Starting tolerance before decrements.
            tolerance_step (int): Amount to reduce the tolerance between attempts.

        Returns:
            tuple[tuple[int, int, int], int]: Matched RGB colour with the tolerance applied.
        """
        tolerance = initial_tolerance
        best_tolerance = 0
        best_ratio = -1.0
        inner_total_pixels = cropped_image.size // 3
        outer_total_pixels = (self.opencv_image.size // 3) - inner_total_pixels

        while tolerance >= 0:
            lower_bound, upper_bound = self._get_color_bounds(dominant_color, tolerance)
            pixel_count, outside_pixel_count = self._get_pixel_counts(cropped_image, lower_bound, upper_bound)
            inner_ratio = pixel_count / inner_total_pixels
            outer_ratio = outside_pixel_count / outer_total_pixels
            ratio = inner_ratio / (outer_ratio + 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_tolerance = tolerance
            tolerance -= tolerance_step

        best_color_rgb = (dominant_color[2], dominant_color[1], dominant_color[0])  # Convert BGR to RGB.
        return best_color_rgb, best_tolerance

    @staticmethod
    def _get_color_bounds(
        dominant_color: tuple[int, int, int], tolerance: int
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Calculates lower and upper bounds for a color given a tolerance.

        Args:
            dominant_color (tuple[int, int, int]): Target colour used to refine matches.
            tolerance (int): Channel tolerance currently applied to the search.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: Lower and upper BGR bounds.
        """
        lower_bound = np.array(
            [
                max(dominant_color[0] - tolerance, 0),
                max(dominant_color[1] - tolerance, 0),
                max(dominant_color[2] - tolerance, 0),
            ]
        )
        upper_bound = np.array(
            [
                min(dominant_color[0] + tolerance, 255),
                min(dominant_color[1] + tolerance, 255),
                min(dominant_color[2] + tolerance, 255),
            ]
        )
        return lower_bound, upper_bound

    def _get_pixel_counts(
        self: Self,
        cropped_image: npt.NDArray[np.uint8],
        lower_bound: npt.NDArray[np.uint8],
        upper_bound: npt.NDArray[np.uint8],
    ) -> tuple[int, int]:
        """Counts pixels within the specified color bounds in the cropped and main images.

        Args:
            cropped_image (npt.NDArray[np.uint8]): Image region extracted for analysis.
            lower_bound (tuple[int, int, int]): Lower inclusive colour bound (BGR).
            upper_bound (tuple[int, int, int]): Upper inclusive colour bound (BGR).

        Returns:
            tuple[int, int]: Pixel counts inside the region and outside it.
        """
        mask = cv.inRange(cropped_image, lower_bound, upper_bound)
        pixel_count = int(np.sum(mask == MAX_COLOR_VALUE))
        outside_mask = cv.inRange(self.opencv_image, lower_bound, upper_bound)
        outside_pixel_count = int(np.sum(outside_mask == MAX_COLOR_VALUE)) - pixel_count
        return pixel_count, outside_pixel_count

    @staticmethod
    def _calculate_median_difference(
        image1: npt.NDArray[np.uint8],
        image2: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None = None,
    ) -> int:
        """Calculates the median absolute difference between two images, optionally ignoring masked areas.

        Args:
            image1: The first image.
            image2: The second image.
            mask (npt.NDArray[np.uint8] | None): Optional mask of pixels to ignore.

        Returns:
            int: Median colour difference, or ``-1`` when shapes mismatch.
        """
        if image1.shape != image2.shape:
            return -1
        if mask is not None:
            mask_expanded = np.expand_dims(mask, axis=2)
            image1 = np.where(mask_expanded, image1, np.nan)
            image2 = np.where(mask_expanded, image2, np.nan)
        diff = np.abs(image1 - image2)
        median_diff = np.nanmedian(diff)
        return int(median_diff)

    @check_valid_image
    def erode_image(self, iterations: int = 1, kernel: npt.NDArray[np.uint8] | None = None) -> None:
        """Applies morphological erosion to the backbuffer image.

        Args:
            iterations (int): Number of erosion passes to run. Defaults to 1.
            kernel (npt.NDArray[np.uint8] | None): Structuring element to use; defaults to a 3x3 ones matrix.
        """
        kernel = kernel or np.ones((3, 3), np.uint8)
        self.opencv_image = cast(
            "npt.NDArray[np.uint8]",
            cv.erode(self.opencv_image, kernel, iterations=iterations),
        )

    @check_valid_image
    def dilate_image(self, iterations: int = 1, kernel: npt.NDArray[np.uint8] | None = None) -> None:
        """Applies morphological dilation to the backbuffer image.

        Args:
            iterations (int): Number of dilation passes to run. Defaults to 1.
            kernel (npt.NDArray[np.uint8] | None): Structuring element to use; defaults to a 3x3 ones matrix.
        """
        kernel = kernel or np.ones((3, 3), np.uint8)
        self.opencv_image = cast(
            "npt.NDArray[np.uint8]",
            cv.dilate(self.opencv_image, kernel, iterations=iterations),
        )

    @check_valid_image
    def find_image(
        self: Self,
        sub_image: npt.NDArray[np.uint8] | Image.Image,
        rect: tuple[int, int, int, int] | None = None,
        confidence: float = 0.95,
        median_tolerance: int | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Finds occurrences of a subimage within the main image using template matching.

        Args:
            sub_image (npt.NDArray[np.uint8] | Image.Image): Subimage to locate inside the backbuffer.
            rect (tuple[int, int, int, int] | None): Search region specified as
                (x, y, width, height). If None, the entire image is used.
            confidence (float): Matching confidence threshold (default ``0.95``).
            median_tolerance (int | None): Optional per-channel median colour tolerance for matches.

        Returns:
            list[tuple[int, int, int, int]]: Bounding boxes locating the subimage.
        """
        image = self._crop_image(rect)
        sub_image_bgr, mask = self._prepare_sub_image(sub_image)
        main_image_gray, sub_image_gray = self._convert_to_grayscale(image, sub_image_bgr)
        res = self._perform_template_matching(main_image_gray, sub_image_gray, mask, confidence)
        rects = self._process_matching_results(res, image, sub_image_bgr, mask, rect, median_tolerance)
        return self._group_and_convert_to_shape_list(rects)

    @staticmethod
    def _prepare_sub_image(
        sub_image: npt.NDArray[np.uint8] | Image.Image,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]:
        """Prepares the subimage for template matching.

        Converts the subimage to BGR format and extracts the alpha channel as a mask if present.

        Args:
            sub_image (npt.NDArray[np.uint8] | Image.Image): Image used as a template when matching.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]: Prepared template and optional mask.
        """
        if isinstance(sub_image, Image.Image):
            sub_image = np.array(sub_image.convert("RGBA"))
        if sub_image.shape[-1] == RECTANGLE_SIDES:
            sub_alpha = sub_image[..., 3]
            sub_image_bgr = cv.cvtColor(sub_image[..., :3], cv.COLOR_RGB2BGR)
            mask = sub_alpha.astype(np.uint8)
        else:
            sub_image_bgr = cv.cvtColor(sub_image, cv.COLOR_RGB2BGR)
            mask = None
        return cast("npt.NDArray[np.uint8]", sub_image_bgr), cast("npt.NDArray[np.uint8]", mask)

    @staticmethod
    def _convert_to_grayscale(
        main_image: npt.NDArray[np.uint8], sub_image_bgr: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Converts the main and subimages to grayscale.

        Args:
            main_image (npt.NDArray[np.uint8]): Reference image to search within.
            sub_image_bgr (npt.NDArray[np.uint8]): Template in BGR colour space.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: Grayscale main image and template.
        """
        main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
        sub_image_gray = cv.cvtColor(sub_image_bgr, cv.COLOR_BGR2GRAY)
        return cast("npt.NDArray[np.uint8]", main_image_gray), cast("npt.NDArray[np.uint8]", sub_image_gray)

    @staticmethod
    def _perform_template_matching(
        main_image_gray: npt.NDArray[np.uint8],
        sub_image_gray: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None,
        confidence: float,
    ) -> npt.NDArray[np.uint8]:
        """Performs template matching between the main image and subimage.

        Args:
            main_image_gray (npt.NDArray[np.uint8]): Main frame in grayscale.
            sub_image_gray (npt.NDArray[np.uint8]): Template converted to grayscale for matching.
            mask (npt.NDArray[np.uint8] | None): Optional template mask.
            confidence (float): Minimum required match confidence.

        Returns:
            npt.NDArray[np.bool_]: Mask where template scores exceed the confidence threshold.
        """
        res = cv.matchTemplate(main_image_gray, sub_image_gray, cv.TM_CCORR_NORMED, mask=mask)
        return cast(
            "npt.NDArray[np.uint8]", np.logical_and(res >= confidence, np.logical_not(np.isinf(res))).astype(np.uint8)
        )

    def _process_matching_results(
        self: Self,
        res: npt.NDArray[np.uint8],
        main_image: npt.NDArray[np.uint8],
        sub_image_bgr: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None,
        rect: tuple[int, int, int, int] | None,
        median_tolerance: int | None,
    ) -> list[tuple[int, int, int, int]]:
        """Processes template matching results to extract matching rectangles.

        Args:
            res (npt.NDArray[np.uint8]): Binary mask produced by the template comparison.
            main_image (npt.NDArray[np.uint8]): Reference image to search within.
            sub_image_bgr (npt.NDArray[np.uint8]): Template in BGR colour space.
            mask (npt.NDArray[np.uint8] | None): Optional mask applied to the subimage.
            rect (tuple[int, int, int, int] | None): Region to constrain the
                template search.
            median_tolerance (int | None): Optional tolerance applied to colour medians.

        Returns:
            list[tuple[int, int, int, int]]: Bounding boxes for detected matches.
        """
        rects = []
        w, h = sub_image_bgr.shape[1::-1]
        locations = np.column_stack(np.where(res))
        for i in range(locations.shape[0]):
            y = int(locations[i, 0])
            x = int(locations[i, 1])
            main_image_region = main_image[y : y + h, x : x + w]
            found_rect = (
                x + (rect[0] if rect else 0),
                y + (rect[1] if rect else 0),
                w,
                h,
            )
            if median_tolerance is not None:
                found_median_diff = self._calculate_median_difference(main_image_region, sub_image_bgr, mask)
                if found_median_diff < median_tolerance:
                    rects.append(found_rect)
            else:
                rects.append(found_rect)
        return rects

    @staticmethod
    def _group_and_convert_to_shape_list(
        rects: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Groups similar rectangles and returns a consolidated list.

        Args:
            rects (list[tuple[int, int, int, int]]): Bounding boxes emitted by the matcher.

        Returns:
            list[tuple[int, int, int, int]]: Grouped rectangles merged by OpenCV's clustering.
        """
        rects_arr = np.repeat(np.array(rects), 2, axis=0)
        grouped_rects, _ = cv.groupRectangles(rects_arr, groupThreshold=1, eps=0.1)  # type: ignore[arg-type]
        return cast("list[tuple[int, int, int, int]]", grouped_rects)

    @check_valid_image
    def find_contours(
        self: Self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
        min_area: int = 10,
        vertices: int | None = None,
    ) -> list[npt.NDArray[np.uintp]]:
        """Finds contours in the image that match the specified color.

        Args:
            color (tuple[int, int, int]): Target colour expressed as (R, G, B).
            rect (tuple[int, int, int, int] | None): Search region specified as
                (x, y, width, height). If None, the entire image is used.
            tolerance (int): Allowed deviation per colour channel.
            min_area (int): Minimum area in pixels squared for a contour to qualify.
            vertices (int | None): Required vertex count for returned contours.

        Returns:
            list[npt.NDArray[np.uintp]]: Contours matching the search criteria.
        """
        image = self._crop_image(rect)
        image = filter_colors(image, color, tolerance)
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [c + np.array([rect[0], rect[1]]) if rect else c for c in contours]
        contours = [c for c in contours if cv.moments(c)["m00"] != 0 and cv.contourArea(c) >= min_area]
        if vertices is not None:
            contours = [
                c
                for c in contours
                if vertices == len(cv.approxPolyDP(c, 0.01 * cv.arcLength(c, closed=True), closed=True))
            ]
        return cast("list[npt.NDArray[np.uintp]]", contours)

    @check_valid_image
    def draw_points(
        self: Self,
        points: Sequence[tuple[int, int]],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws points on the backbuffer image.

        Args:
            points (Sequence[tuple[int, int]]): Coordinates to mark on the backbuffer.
            color (tuple[int, int, int]): Drawing colour (RGB). Defaults to red.
        """
        points_arr = np.array(points)
        self.opencv_image[points_arr[:, 1], points_arr[:, 0]] = color[::-1]

    @check_valid_image
    def draw_contours(
        self: Self,
        contours: tuple[tuple[tuple[tuple[int, int]]]],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws contours on the backbuffer image.

        Args:
            contours (tuple[tuple[tuple[tuple[int, int]]]]): Contour sequences as produced by OpenCV.
            color (tuple[int, int, int]): Drawing colour (RGB). Defaults to red.
        """
        cv.drawContours(self.opencv_image, contours, -1, color[::-1], 2)  # type: ignore[arg-type]

    @check_valid_image
    def draw_circle(
        self: Self,
        circle: tuple[int, int, int],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a circle on the backbuffer image.

        Args:
            circle (tuple[int, int, int]): Circle definition (x, y, radius).
            color (tuple[int, int, int]): Drawing colour (RGB). Defaults to red.
        """
        x, y, r = circle
        cv.circle(self.opencv_image, (x, y), r, color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def draw_rectangle(
        self: Self,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a rectangle on the backbuffer image.

        Args:
            rect (tuple[int, int, int, int]): Rectangle specified as (x, y, width, height).
            color (tuple[int, int, int]): Drawing colour (RGB). Defaults to red.
        """
        x, y, w, h = rect
        cv.rectangle(self.opencv_image, (x, y), (x + w, y + h), color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def filter_colors(
        self: Self,
        colors: tuple[int, int, int] | list[tuple[int, int, int]],
        tolerance: int = 0,
        *,
        keep_original_colors: bool = False,
    ) -> None:
        """Filters the backbuffer image to retain only specified colors within a given tolerance.

        Args:
            colors (tuple[int, int, int] | list[tuple[int, int, int]]): Colours to keep while filtering.
            tolerance (int): Per-channel tolerance threshold (0-255).
            keep_original_colors (bool): When ``True``, retain source colours for matching pixels;
                otherwise, update the backbuffer in place.
        """
        grey_image = filter_colors(self.opencv_image, colors, tolerance, keep_original_colors=keep_original_colors)
        self.opencv_image = grey_image
