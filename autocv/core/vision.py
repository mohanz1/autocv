"""This module provides the Vision class and related utilities for image capture, processing, and OCR.

It includes decorators for validating image and window handle states and integrates with OpenCV, pytesseract, and other
libraries to provide comprehensive image analysis and text extraction functionalities.
"""

from __future__ import annotations

__all__ = ("Vision", "check_valid_hwnd", "check_valid_image")

import functools
import io
import logging
import pathlib
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
import polars as pl
import win32con
import win32gui
import win32ui
from PIL import Image
from tesserocr import OEM, PSM, PyTessBaseAPI  # type: ignore[import-untyped]
from typing_extensions import Self

from autocv.models import InvalidHandleError, InvalidImageError

from .image_processing import filter_colors
from .window_capture import WindowCapture

RECTANGLE_SIDES = 4
GRESCALE_CHANNELS = 3
MAX_COLOR_VALUE = 255

logger = logging.getLogger(__name__)
FuncT = TypeVar("FuncT", bound=Callable[..., Any])
# Define a type variable for the return type
R = TypeVar("R")
# Define a parameter specification variable for the function parameters
P = ParamSpec("P")
SelfWindowCapture = TypeVar("SelfWindowCapture", bound="WindowCapture")
SelfVision = TypeVar("SelfVision", bound="Vision")


def check_valid_hwnd(
    func: Callable[Concatenate[SelfWindowCapture, P], R],
) -> Callable[Concatenate[SelfWindowCapture, P], R]:
    """Decorator that checks if the `hwnd` attribute is set before calling the decorated method."""

    @functools.wraps(func)
    def wrapper(self: SelfWindowCapture, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.hwnd == -1:
            raise InvalidHandleError(self.hwnd)
        return func(self, *args, **kwargs)

    return cast(Callable[Concatenate[SelfWindowCapture, P], R], wrapper)


def check_valid_image(func: Callable[Concatenate[SelfVision, P], R]) -> Callable[Concatenate[SelfVision, P], R]:
    """Decorator that checks if the `opencv_image` attribute is set before calling the decorated method."""

    @functools.wraps(func)
    def wrapper(self: SelfVision, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.opencv_image.size == 0:
            raise InvalidImageError
        return func(self, *args, **kwargs)

    return cast(Callable[Concatenate[SelfVision, P], R], wrapper)


class Vision(WindowCapture):
    """Class for performing image processing and optical character recognition (OCR)."""

    def __init__(self: Self, hwnd: int = -1) -> None:
        """Initializes a Vision object.

        Args:
            hwnd (int | None): The window handle of the window to capture. Defaults to None.

        Returns:
            None
        """
        super().__init__(hwnd)
        # Holds the image in OpenCV format
        self.opencv_image: npt.NDArray[np.uint8] = np.empty(0, dtype=np.uint8)

        # Define the path to the tessdata directory and set up Tesseract configuration
        absolute_directory = pathlib.Path(__file__).parents[1] / "data" / "traineddata"
        self.api = PyTessBaseAPI(path=str(absolute_directory), lang="runescape", psm=PSM.SPARSE_TEXT, oem=OEM.LSTM_ONLY)
        self._config = rf"--tessdata-dir {absolute_directory} --oem 1 --psm 11"

    def set_backbuffer(self: Self, image: npt.NDArray[np.uint8] | Image.Image) -> None:
        """Sets the image buffer to the provided numpy array or PIL Image object.

        Args:
            image (npt.NDArray[np.uint8] | Image.Image]): The object to set as the image back buffer.

        Returns:
            None
        """
        if isinstance(image, Image.Image):
            self.opencv_image = cast(npt.NDArray[np.uint8], cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR))
        else:
            self.opencv_image = cast(npt.NDArray[np.uint8], image)

    @check_valid_hwnd
    def refresh(self: Self, *, set_backbuffer: bool = True) -> npt.NDArray[np.uint8] | None:
        """Captures the current window image and converts it to an OpenCV format.

        Args:
            set_backbuffer (bool | None): If True, set the captured image as the window's backbuffer.

        Raises:
            InvalidHandleError: If the window handle is not valid.

        Returns:
            npt.NDArray[np.uint8] | None: The captured image as a NumPy array with shape (height, width, 3), or None if
                set_backbuffer is True.
        """
        # Get window dimensions
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Get device context and compatible bitmap
        window_dc = win32gui.GetWindowDC(self.hwnd)
        mem_dc = win32ui.CreateDCFromHandle(window_dc)
        bmp_dc = mem_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        bmp_dc.SelectObject(bitmap)

        # Copy window image data onto bitmap
        bmp_dc.BitBlt((0, 0), (width, height), mem_dc, (0, 0), win32con.SRCCOPY)

        # Convert raw data into a format OpenCV can read
        signed_ints_array = bitmap.GetBitmapBits(True)  # noqa:FBT003
        img = np.fromstring(signed_ints_array, dtype="uint8")  # type: ignore[call-overload]
        img.shape = (height, width, 4)

        # Free resources
        mem_dc.DeleteDC()
        bmp_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, window_dc)
        win32gui.DeleteObject(bitmap.GetHandle())

        # Make image C_CONTIGUOUS to avoid errors and drop alpha channel
        image = np.ascontiguousarray(img[..., :3])

        if set_backbuffer:
            # Set the captured image as the window's backbuffer
            self.set_backbuffer(image)
        return None if set_backbuffer else image

    @check_valid_image
    def save_backbuffer_to_file(self: Self, file_name: str) -> None:
        """Save the backbuffer image to a file.

        Args:
            file_name (str): The name of the file to save the image to.

        Returns:
            None

        """
        cv.imwrite(file_name, self.opencv_image)

    @check_valid_hwnd
    def get_pixel_change(self: Self, area: tuple[int, int, int, int] | None = None) -> int:
        """Calculates the number of pixels that have changed between the current image and a later version of the image.

        Args:
            area (tuple[int, int, int, int] | None): The region of the image to consider. If not specified, the entire
                image will be used. Defaults to None.

        Raises:
            InvalidImageError: If the image data is invalid.

        Returns:
            int: The number of pixels that have changed between the two images.
        """
        # Gray and crop image
        image = self._crop_image(area, self.opencv_image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Create second image grayed with same size
        updated_image = self.refresh(set_backbuffer=False)
        updated_image = self._crop_image(area, updated_image)
        updated_image = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)

        # Compute the difference between the images
        diff = cv.absdiff(image, updated_image)

        # Count the number of non-zero pixels in the difference image
        return np.count_nonzero(diff)

    def _get_grouped_text(
        self: Self,
        image: npt.NDArray[np.uint8],
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
    ) -> pl.LazyFrame:
        """Applies pre-processing to the image and extracts text from the image using Tesseract OCR.

        Groups the text data by block and returns a DataFrame with the extracted text and relevant columns.

        Args:
            image (npt.NDArray[np.uint8]): The input image to search.
            colors (tuple[int, int, int] | Sequence[tuple[int, int, int]] | None): A sequence of RGB tuples or a
                sequence of sequences containing RGB tuples.
            rect (tuple[int, int, int, int]): A rect in the form of left, top, width, height.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.

        Returns:
            pl.DataFrame:
                A DataFrame with the extracted text and relevant columns.
        """
        if colors:
            image = filter_colors(image, colors, tolerance)

        # Resize the image to double the original size
        resized_img = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        # Apply bilateral filter for noise reduction
        img = cv.bilateralFilter(resized_img, 9, 75, 75)

        # Convert the image to grayscale and apply thresholding
        if len(img.shape) == GRESCALE_CHANNELS and img.shape[-1] == GRESCALE_CHANNELS:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.threshold(img, 0, MAX_COLOR_VALUE, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        img = cv.bitwise_not(img)

        # Extract text from the thresholded image using Tesseract OCR
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

        # Filter out invalid text data and text with low confidence
        text = text.filter((pl.col("conf") > 0) & (pl.col("conf") < 100))

        # Ensure 'text' column is string and remove empty strings
        text = text.with_columns([
            pl.col("text").cast(pl.Utf8),
            (pl.col("conf") / 100).alias("conf"),
        ])
        text = text.filter(pl.col("text").str.strip_chars())

        # Group text data by block and extract relevant columns
        grouped_text = text.group_by("block_num").agg([
            pl.col("word_num").max().alias("word_num"),
            pl.col("left").min().alias("left"),
            pl.col("top").min().alias("top"),
            pl.col("height").max().alias("height"),
            pl.col("conf").max().alias("confidence"),
            pl.col("text").str.concat(" ").alias("text"),
            ((pl.col("left") + pl.col("width")).max() - pl.col("left").min()).alias("width"),
        ])

        # Sort values in descending order based on confidence
        sorted_text = grouped_text.sort("confidence", descending=True)

        # Convert coordinates back to original size
        return sorted_text.with_columns([
            (pl.col("top") // 2).alias("top"),
            (pl.col("left") // 2).alias("left"),
            (pl.col("height") // 2).alias("height"),
            (pl.col("width") // 2).alias("width"),
        ])

    def _crop_image(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        image: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Crop the current OpenCV image.

        Args:
            rect (Sequence[int, int, int, int] | None): A tuple specifying a rectangular region in the image to crop.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is returned.
            image (npt.NDArray[np.uint8] | None): An optional input image to crop. If not provided, the current OpenCV
                image is used.

        Returns:
            npt.NDArray[np.uint8]: A numpy array of unsigned integers of shape (height, width, channels), representing
                an image in the OpenCV format.
        """
        image = image if image is not None else self.opencv_image

        if rect:
            # Extract the coordinates and dimensions from the rectangle
            x, y, w, h = rect
            # Crop the image using the specified region
            return image[y : y + h, x : x + w]
        # If no rectangle is specified, return the entire image
        return image

    @check_valid_image
    def get_text(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
        confidence: float | None = 0.8,
    ) -> list[dict[str, str | int | float]]:
        """Extracts text from the image using Tesseract OCR with confidence greater than or equal to the confidence arg.

        Args:
            rect (Sequence[int, int, int, int] | None): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            colors (Sequence[int, int, int] | Sequence[Sequence[int, int, int]]): A sequence of RGB tuples or a sequence
                of sequences containing RGB tuples.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel
                color.
            confidence (float | None): The minimum confidence level for text to be included in the output. Must be a
                value between 0 and 1. Defaults to 0.8.

        Returns:
            list[dict[str, str | int | float]]: A list of TextInfo objects for the acceptable text in the image.

        Raises:
            InvalidImageError: If the input image is invalid or None.
        """
        sorted_text = self._get_grouped_text(self.opencv_image, rect, colors, tolerance)

        # Filter out text with low confidence and format the output data
        acceptable_text = sorted_text.filter(pl.col("confidence") >= confidence)

        acceptable_text = acceptable_text.select("text", "left", "top", "width", "height", "confidence")

        if rect:
            acceptable_text = acceptable_text.with_columns(pl.col("left") + rect[0], pl.col("top") + rect[1])

        return acceptable_text.collect().to_dicts()

    @check_valid_image
    def get_color(self: Self, point: tuple[int, int]) -> tuple[int, int, int]:
        """Returns the color of a pixel in the image at the specified coordinates.

        Args:
            point (Sequence[int, int]): The a tuple containing the x and y coordinate of the pixel.

        Returns:
            Color: The color of the pixel.

        Raises:
            InvalidImageError: If the input image is invalid or None.
            IndexError: If the coordinates are out of bounds.
        """
        x, y = point

        # Check if the coordinates are within the image bounds
        if not (0 <= x < self.opencv_image.shape[1] and 0 <= y < self.opencv_image.shape[0]):
            raise IndexError

        # Get the color of the pixel at the specified coordinates
        return cast(tuple[int, int, int], tuple(*np.flip(self.opencv_image[y, x])))

    @check_valid_image
    def find_color(
        self: Self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
    ) -> list[tuple[int, int]]:
        """Finds all x, y coordinates in a given OpenCV image that match the given color and tolerance.

        Args:
            color (Sequence[int, int, int]): The color to search for, in RGB format.
            rect (Sequence[int, int, int, int] | None): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel
                color.

        Returns:
            list[tuple[int, int]]: A ShapeList instance containing the x, y coordinates of all pixels in the image that
                match the given color within the specified tolerance.

        Raises:
            InvalidImageError: If the input image is invalid or None.
        """
        image = self._crop_image(rect)
        mask = filter_colors(image, color, tolerance)

        # Stack x and y arrays into a single (2, num_points) array
        points = np.stack(np.where(mask == MAX_COLOR_VALUE)[::-1], axis=1)
        if rect:
            points = np.add(points, rect[0:2])

        return cast(list[tuple[int, int]], points.tolist())

    @check_valid_image
    def get_average_color(self: Self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
        """Get the average color of an image within a specified rectangular region.

        Args:
            rect (Sequence[int, int, int, int] | None): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.

        Returns:
            tuple[int, int, int]: The average color within the specified region.

        """
        return cast(tuple[int, int, int], tuple(*self._get_average_color(self.opencv_image, rect)))

    def _get_average_color(
        self: Self,
        image: npt.NDArray[np.uint8],
        rect: tuple[int, int, int, int] | None = None,
    ) -> npt.NDArray[np.int16]:
        """Calculate the average color of an image within a specified rectangular region.

        Args:
            image (npt.NDArray[np.uint8]): The input image to calculate the average color from.
            rect (tuple[int, int, int, int] | None): A tuple specifying a rectangular region in the image to search. The
                tuple contains the coordinates of the top-left corner (x, y) and the width and height of the rectangle
                (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.

        Returns:
            Color: The average color within the specified region.

        """
        image = self._crop_image(rect, image)

        # Calculate the average color using cv2.mean
        avg_color = cv.mean(image)

        # Convert the average color from a tuple to a numpy array
        avg_color = np.array(avg_color[:3], dtype=np.int16)

        return avg_color[::-1]

    @check_valid_image
    def get_most_common_color(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
        index: int = 1,
        ignore_colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[int, int, int]:
        """Returns the most common color in the given image.

        Args:
            rect (Sequence[int, int, int, int] | None): A Sequence containing the x, y coordinates and the width and
                height of the area to analyze.
            index (int): The index of the desired popular color (1 for the most common, 2 for the second most
                common, and so on).
            ignore_colors (Sequence[int, int, int] | Sequence[Sequence[int, int, int]] | None): A sequence of RGB
                Sequence or a sequence of sequences containing RGB tuples.

        Returns:
          tuple[int, int, int]: Most common color.
        """
        cropped_image = self._crop_image(rect)

        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = cropped_image.reshape(-1, 3)

        # Filter out the specified ignore colors
        if ignore_colors is not None:
            # Convert ignore_colors to numpy array for mask creation
            color_values = np.array(ignore_colors, dtype=np.int16)

            # If color_values has only one dimension, add another dimension to make it 2D
            if color_values.ndim == 1:
                color_values = color_values[np.newaxis, :]

            # Create a mask to filter out ignore_colors
            ignore_mask = np.isin(reshaped_image, color_values).any(axis=1)

            # Apply the mask to reshaped_image
            reshaped_image = reshaped_image[~ignore_mask]

        unique, counts = np.unique(reshaped_image, axis=0, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        desired_index = min(index - 1, len(sorted_indices) - 1)
        most_common_color = unique[sorted_indices[desired_index]][::-1]

        return cast(tuple[int, int, int], tuple(*most_common_color))

    @check_valid_image
    def get_all_colors_with_counts(
        self: Self,
        rect: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[tuple[int, int, int], int]]:
        """Returns all colors in a given image or area of an image with the respective counts.

        Args:
            rect (Sequence[int, int, int, int] | None): A tuple containing the x, y coordinates and the width and height
                of the area to analyze.


        Returns:
          Color: Most common color.
        """
        cropped_image = self._crop_image(rect)

        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = cropped_image.reshape(-1, 3)

        unique, counts = np.unique(reshaped_image, axis=0, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        sorted_unique = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        return [((r, g, b), count) for (b, g, r), count in zip(sorted_unique, sorted_counts, strict=False)]

    @check_valid_image
    def get_median_color(self: Self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]:
        """Returns the dominant color in the given image.

        Args:
          rect (Sequence[int, int, int, int] | None): A Sequence containing the x, y coordinates and the width and
            height of the area to analyze.

        Returns:
          tuple[int, int, int]: Dominant color.
        """
        cropped_image = self._crop_image(rect)

        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = cropped_image.reshape(-1, 3)

        # Calculate the median value for each channel
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)

        # Calculate the median value for each channel
        median_color = median_color[::-1].tolist()

        return cast(tuple[int, int, int], tuple(*median_color))

    @staticmethod
    def _get_dominant_color(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Returns the dominant color in the given image.

        Args:
          image (npt.NDArray[np.uint8]): Input image (in BGR format).

        Returns:
          npt.NDArray[np.uint8]: Dominant color in the image (in BGR format).
        """
        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = image.reshape(-1, 3)

        # Calculate the median value for each channel
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)

        return cast(npt.NDArray[np.uint8], median_color)

    @check_valid_image
    def maximize_color_match(
        self: Self,
        rect: tuple[int, int, int, int],
        initial_tolerance: int = 100,
        tolerance_step: int = 1,
    ) -> tuple[tuple[int, int, int], int]:
        """Finds the best color and tolerance to maximize the colors in a given area of the image.

        Args:
            rect (Sequence[int, int, int, int]): A Sequence containing the x, y coordinates and the width and height of
                the area to analyze.
            initial_tolerance (int): The initial tolerance value to use when searching for the best color match.
                Defaults to 100.
            tolerance_step (int): The step size to use when decreasing the tolerance value during the search. Defaults
                to 1.

        Returns:
            tuple[Color, int]: A tuple containing the RGB values of the best color match and the tolerance value used to
                obtain that match.
        """
        cropped_image = self._crop_image(rect)
        dominant_color = self._get_dominant_color(cropped_image)
        best_color, best_tolerance = self._find_best_color_match(
            cropped_image, dominant_color.tolist(), initial_tolerance, tolerance_step
        )
        return best_color, best_tolerance

    def _find_best_color_match(
        self: Self,
        cropped_image: npt.NDArray[np.uint8],
        dominant_color: tuple[int, int, int],
        initial_tolerance: int,
        tolerance_step: int,
    ) -> tuple[tuple[int, int, int], int]:
        """Find the best color match with the given tolerance and step settings.

        Args:
            cropped_image (npt.NDArray[np.uint8]): The image region to analyze.
            dominant_color (Sequence[int, int, int]): The dominant color of the cropped image as RGB a Sequence.
            initial_tolerance (int): Starting tolerance for color matching.
            tolerance_step (int): Step size to decrement tolerance.

        Returns:
            A tuple of the best matching Color and the tolerance used to find it.
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

        best_color_rgb = (dominant_color[2], dominant_color[1], dominant_color[0])  # Convert BGR to RGB
        return best_color_rgb, best_tolerance

    @staticmethod
    def _get_color_bounds(
        dominant_color: tuple[int, int, int], tolerance: int
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Calculate the lower and upper bounds for a color based on the given tolerance.

        Args:
            dominant_color (Sequence[int, int, int]): The dominant color as an RGB tuple.
            tolerance (int): The tolerance value for calculating bounds.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: A tuple containing the lower and upper bounds as numpy
                arrays.
        """
        lower_bound = np.array(
            [
                max(dominant_color[0] - tolerance, 0),
                max(dominant_color[1] - tolerance, 0),
                max(dominant_color[2] - tolerance, 0),
            ],
        )
        upper_bound = np.array(
            [
                min(dominant_color[0] + tolerance, 255),
                min(dominant_color[1] + tolerance, 255),
                min(dominant_color[2] + tolerance, 255),
            ],
        )
        return lower_bound, upper_bound

    def _get_pixel_counts(
        self: Self,
        cropped_image: npt.NDArray[np.uint8],
        lower_bound: npt.NDArray[np.uint8],
        upper_bound: npt.NDArray[np.uint8],
    ) -> tuple[int, int]:
        """Count the number of pixels within the color range for inner and outer regions.

        Args:
            cropped_image (npt.NDArray[np.uint8]): The cropped section of the image for analysis.
            lower_bound (npt.NDArray[np.uint8]): The lower bound of the color range.
            upper_bound (npt.NDArray[np.uint8]): The upper bound of the color range.

        Returns:
            tuple[int, int]: A tuple containing the number of pixels within the color range for the inner and outer
                regions.
        """
        mask = cv.inRange(cropped_image, lower_bound, upper_bound)
        pixel_count = np.sum(mask == MAX_COLOR_VALUE)
        outside_mask = cv.inRange(self.opencv_image, lower_bound, upper_bound)
        outside_pixel_count = np.sum(outside_mask == MAX_COLOR_VALUE) - pixel_count
        return pixel_count, outside_pixel_count

    @staticmethod
    def _calculate_median_difference(
        image1: npt.NDArray[np.uint8],
        image2: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None = None,
    ) -> int:
        """Calculate the median difference between two images while optionally ignoring a mask.

        Args:
            image1 (npt.NDArray[np.uint8]): The first image.
            image2 (npt.NDArray[np.uint8]): The second image.
            mask (Optional[npt.NDArray[np.uint8]): The mask indicating the pixels to ignore. Defaults to None.

        Returns:
            int: The median difference between the images.
        """
        if image1.shape != image2.shape:
            return -1

        # Apply the mask if provided
        if mask is not None:
            mask_expanded = np.expand_dims(mask, axis=2)
            image1 = np.where(mask_expanded, image1, np.nan)
            image2 = np.where(mask_expanded, image2, np.nan)

        # Calculate the difference between the images
        diff = np.abs(image1 - image2)

        # Calculate the median difference
        median_diff = np.nanmedian(diff)

        return int(median_diff)

    @check_valid_image
    def find_image(
        self: Self,
        sub_image: npt.NDArray[np.uint8] | Image.Image,
        rect: tuple[int, int, int, int] | None = None,
        confidence: float = 0.95,
        median_tolerance: int | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Find a subimage in a larger image given a confidence level and optional color tolerance.

        Args:
            sub_image (npt.NDArray[np.uint8] | Image.Image): The subimage to search for.
            rect (Sequence[int, int, int, int] | None): A Sequence specifying a rectangular region in the image to
                search. The Sequence contains the coordinates of the top-left corner (x, y) and the width and height of
                the rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            confidence (float): The confidence level to use for the template matching. Defaults to 0.95.
            median_tolerance (int | None): The maximum difference allowed between each channel of the average color of
                the subimage and the average color of the matched region in the main image. If not provided, no color
                tolerance check is performed.

        Returns:
            ShapeList[Rectangle]: An object containing the coordinates of the subimage matches.

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
        """Prepare the sub-image for template matching by ensuring correct format and creating a mask if needed.

        Args:
            sub_image (npt.NDArray[np.uint8] | Image.Image): The image to prepare, which can be a numpy array or an
                Image object.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]: A tuple of the prepared sub-image and the mask
                (if applicable, otherwise None).
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
        return cast(npt.NDArray[np.uint8], sub_image_bgr), cast(npt.NDArray[np.uint8], mask)

    @staticmethod
    def _convert_to_grayscale(
        main_image: npt.NDArray[np.uint8], sub_image_bgr: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Convert main and sub images to grayscale for template matching.

        Args:
            main_image (npt.NDArray[np.uint8]): The main image where the sub-image will be searched for.
            sub_image_bgr (npt.NDArray[np.uint8]): The sub-image to be searched in the main image.

        Returns:
            tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]: A tuple of the main and sub-images converted to
                grayscale.
        """
        main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
        sub_image_gray = cv.cvtColor(sub_image_bgr, cv.COLOR_BGR2GRAY)
        return cast(npt.NDArray[np.uint8], main_image_gray), cast(npt.NDArray[np.uint8], sub_image_gray)

    @staticmethod
    def _perform_template_matching(
        main_image_gray: npt.NDArray[np.uint8],
        sub_image_gray: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None,
        confidence: float,
    ) -> npt.NDArray[np.uint8]:
        """Perform template matching using the grayscale images and mask.

        Args:
            main_image_gray (npt.NDArray[np.uint8]): The grayscale version of the main image.
            sub_image_gray (npt.NDArray[np.uint8]): The grayscale version of the sub-image.
            mask (npt.NDArray[np.uint8] | None): The mask for the sub-image (if applicable).
            confidence (float): The confidence level to use for matching.

        Returns:
            A numpy array of matching results.
        """
        res = cv.matchTemplate(main_image_gray, sub_image_gray, cv.TM_CCORR_NORMED, mask=mask)
        return np.logical_and(res >= confidence, np.logical_not(np.isinf(res)))

    def _process_matching_results(
        self: Self,
        res: npt.NDArray[np.uint8],
        main_image: npt.NDArray[np.uint8],
        sub_image_bgr: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None,
        rect: tuple[int, int, int, int] | None,
        median_tolerance: int | None,
    ) -> list[tuple[int, int, int, int]]:
        """Process the template matching results to extract matching rectangles.

        Args:
            res (npt.NDArray[np.uint8]): The result from the template matching.
            main_image (npt.NDArray[np.uint8]): The main image where the sub-image was searched for.
            sub_image_bgr (npt.NDArray[np.uint8]): The BGR version of the sub-image.
            mask (npt.NDArray[np.uint8] | None): The mask for the sub-image (if applicable).
            rect (tuple[int, int, int, int] | None): The rectangle specifying the area in the main image where the
                sub-image was searched.
            median_tolerance (int | None): The tolerance for color differences.

        Returns:
            list[tuple[int, int, int, int]]: A list of tuples representing the found rectangles in the format
                (x, y, width, height).
        """
        rects = []
        w, h = sub_image_bgr.shape[1::-1]
        for loc in np.fliplr(np.transpose(np.where(res))):
            x, y = loc
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
        """Group similar rectangles and convert them to a ShapeList for easier manipulation.

        Args:
            rects (list[tuple[int, int, int, int]]): The list of rectangles to group and convert.

        Returns:
            ShapeList[Rectangle]: A ShapeList of Rectangles after grouping similar ones.
        """
        rects = np.repeat(np.array(rects), 2, axis=0)
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.1)  # type: ignore[arg-type]
        return cast(list[tuple[int, int, int, int]], rects)

    @check_valid_image
    def find_contours(
        self: Self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
        min_area: int = 10,
        vertices: int | None = None,
    ) -> list[npt.NDArray[np.uintp]]:
        """Find contours of the regions in the image that match a given color.

        Args:
            color (tuple[int, int, int]): The target color as a sequence of (R,G,B) values.
            rect (tuple[int, int, int, int] | None): A tuple specifying a rectangular region in the image to search. The
                tuple contains the coordinates of the top-left corner (x, y) and the width and height of the rectangle
                (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            tolerance (int): The tolerance value for color match.
            min_area (int): The minimum area of a contour to be included in the result.
            vertices (int | None): The number of vertices to look for.

        Returns:
            ShapeList[Contour]: A ShapeList representing the contours of the regions that match the given color.

        Raises:
            InvalidImageError: If the input image is invalid or None.
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

        return cast(list[npt.NDArray[np.uintp]], contours)

    @check_valid_image
    def draw_points(
        self: Self,
        points: tuple[tuple[int, int]],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draw points on the image with the specified color.

        Args:
            points (Sequence[tuple[int, int]]): The points to be drawn.
            color (tuple[int, int, int]): The color to use for drawing the points, specified as a tuple of (R,G,B)
                values. Default is red (255, 0, 0).

        Returns:
            None

        Raises:
            InvalidImageError: If the input image is invalid or None.
        """
        points = np.array(points)

        self.opencv_image[points[:, 1], points[:, 0]] = color[::-1]

    @check_valid_image
    def draw_contours(
        self: Self,
        contours: tuple[tuple[tuple[tuple[int, int]]]],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draw a contour on the image with the specified color.

        Args:
            contours (Sequence[Sequence[Sequence[tuple[int, int]]]]): The contours to be drawn.
            color (tuple[int, int, int]): The color to use for drawing the contours, specified as a tuple of
                (R,G,B) values. Default is red (255, 0, 0).

        Returns:
            None

        Raises:
            InvalidImageError: If the input image is invalid or None.
        """
        cv.drawContours(self.opencv_image, contours, -1, color[::-1], 2)  # type: ignore[arg-type]

    @check_valid_image
    def draw_circle(
        self: Self,
        circle: tuple[int, int, int],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a rectangle onto the backbuffer.

        Args:
            circle (tuple[int, int, int]): A tuple specifying an x, y, radius in the image. The tuple contains the
                coordinates of the top-left corner (x, y) and the width and height of the rectangle (w, h) in the format
                (x, y, w, h).
            color (tuple[int, int, int]): The color of the rectangle. Defaults to red (255, 0, 0).
        """
        # Convert the top-left and bottom-right coordinates to OpenCV format
        x, y, r = circle

        # Draw the rectangle onto the image
        cv.circle(self.opencv_image, (x, y), r, color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def draw_rectangle(
        self: Self,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int] = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a rectangle onto the backbuffer.

        Args:
            rect (tuple[int, int, int, int]): A tuple specifying a rectangular region. The tuple contains the
                coordinates of the top-left corner (x, y) and the width and height of the rectangle (w, h) in the format
                (x, y, w, h).
            color (tuple[int, int, int]): The color of the rectangle. Defaults to red (255, 0, 0).
        """
        # Convert the top-left and bottom-right coordinates to OpenCV format
        x, y, w, h = rect

        # Draw the rectangle onto the image
        cv.rectangle(self.opencv_image, (x, y), (x + w, y + h), color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def filter_colors(
        self: Self,
        colors: tuple[int, int, int] | list[tuple[int, int, int]],
        tolerance: int = 0,
        *,
        keep_original_colors: bool = False,
    ) -> None:
        """Filter out all colors from the image that are not in the specified list of colors with a given tolerance.

        Updates the backbuffer image of the window with the filtered image.

        Args:
            colors (Utuple[int, int, int] | Sequence[tuple[int, int, int]]): A sequence of RGB tuples or a sequence of
                sequences containing RGB tuples.
            tolerance (int): The color tolerance in the range of 0-255.
            keep_original_colors (bool): If True, the returned value will be a copy of the input image where all
                non-matching pixels are set to black.

        Returns:
            None

        Raises:
            InvalidImageError: If the input image is invalid or None.
        """
        grey_image = filter_colors(self.opencv_image, colors, tolerance, keep_original_colors=keep_original_colors)
        self.opencv_image = cast(npt.NDArray[np.uint8], grey_image)
