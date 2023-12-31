import functools
import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytesseract  # type: ignore[import-untyped]
import win32con
import win32gui
import win32ui
from PIL import Image

from autocv.models import Color, Contour, InvalidHandleError, InvalidImageError, Point, Rectangle, ShapeList, TextInfo

from .image_processing import filter_colors
from .window_capture import WindowCapture

__all__ = ("Vision", "check_valid_hwnd", "check_valid_image")

logger = logging.getLogger(__name__)
FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def check_valid_hwnd(func: FuncT) -> FuncT:
    """Decorator that checks if the `hwnd` attribute is set before calling the decorated method.

    If `hwnd` is `None`, raises an `InvalidHandle` exception.

    Args:
    ----
        func (FuncT): The function to decorate.

    Returns:
    -------
        FuncT: The decorated function.
    """

    @functools.wraps(func)
    def wrapper(self: WindowCapture, *args: Any, **kwargs: Any) -> Any:
        """Wrapper function that checks if the `hwnd` attribute is set before calling the decorated method.

        If `hwnd` is `None`, raises an `InvalidHandle` exception.

        Args:
        ----
            self (Any): The instance of the decorated class.
            *args (Any): Positional arguments passed to the decorated function.
            **kwargs (Any): Keyword arguments passed to the decorated function.

        Raises:
        ------
            InvalidHandle: If `hwnd` is `None`.

        Returns:
        -------
            Any: The return value of the decorated function.
        """
        if not self.hwnd:
            raise InvalidHandleError(f"Invalid handle: {self.hwnd}. Please set handle before calling this method.")
        return func(self, *args, **kwargs)

    return cast(FuncT, wrapper)


def check_valid_image(func: FuncT) -> FuncT:
    """Decorator that checks if the `opencv_image` attribute is set before calling the decorated method. If `opencv_image`
    is None, raises an `InvalidImage` exception.

    Args:
    ----
        func (FuncT): The function being decorated.

    Returns:
    -------
        FuncT: The decorated function.
    """

    @functools.wraps(func)
    def wrapper(self: "Vision", *args: Any, **kwargs: Any) -> Any:
        """Wrapper function that checks if the `opencv_image` attribute is set before calling the decorated method.

        If `opencv_image` is `None`, raises an `InvalidImage` exception.

        Args:
        ----
            self (Any): The instance of the decorated class.
            *args (Any): Positional arguments passed to the decorated function.
            **kwargs (Any): Keyword arguments passed to the decorated function.

        Raises:
        ------
            InvalidImage: If `opencv_image` is `None`.

        Returns:
        -------
            Any: The return value of the decorated function.
        """
        if self.opencv_image is None:
            raise InvalidImageError("Invalid image. Please call refresh() before calling this method.")
        return func(self, *args, **kwargs)

    return cast(FuncT, wrapper)


class Vision(WindowCapture):
    """Class for performing image processing and optical character recognition (OCR)
    on a specific window identified by its window handle (hwnd).
    Inherits from the WindowCapture class.
    """

    def __init__(self, hwnd: int | None = None) -> None:
        """Initializes a Vision object.

        Args:
        ----
            hwnd (int, optional): The window handle of the window to capture. Defaults to None.

        Returns:
        -------
            None
        """
        super().__init__(hwnd)
        self.opencv_image: npt.NDArray[np.uint8] | None = None  # Holds the image in OpenCV format

        # Define the path to the tessdata directory and set up Tesseract configuration
        absolute_directory = pathlib.Path(__file__).parents[1] / "data" / "traineddata"
        self._config = rf'--tessdata-dir "{absolute_directory}" --oem 1 --psm 11 -l runescape'

    def set_backbuffer(self, image: npt.NDArray[np.uint8] | Image.Image) -> None:
        """Sets the image buffer to the provided numpy array or PIL Image object.

        Args:
        ----
            image (Union[npt.NDArray[np.uint8], Image.Image]): The Object to set as the image back buffer.

        Returns:
        -------
            None
        """
        if isinstance(image, Image.Image):
            self.opencv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        else:
            self.opencv_image = image

    @check_valid_hwnd
    def refresh(self, set_backbuffer: bool | None = True) -> npt.NDArray[np.uint8] | None:
        """Captures the current window image and converts it to an OpenCV format.

        Args:
        ----
            set_backbuffer (Optional[bool]): If True, set the captured image as the window's backbuffer.

        Raises:
        ------
            InvalidHandle: If the window handle is not valid.

        Returns:
        -------
            Optional[npt.NDArray[np.uint8]]: The captured image as a NumPy array with shape (height, width, 3),
                or None if set_backbuffer is True.
        """
        # Get window dimensions
        assert self.hwnd
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Get device context and compatible bitmap
        window_dc = win32gui.GetWindowDC(self.hwnd)
        mem_dc = win32ui.CreateDCFromHandle(window_dc)
        bmp_dc = mem_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mem_dc, width, height)
        bmp_dc.SelectObject(bitmap)  # type: ignore[no-untyped-call]

        # Copy window image data onto bitmap
        bmp_dc.BitBlt((0, 0), (width, height), mem_dc, (0, 0), win32con.SRCCOPY)

        # Convert raw data into a format OpenCV can read
        signed_ints_array = bitmap.GetBitmapBits(True)
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
    def save_backbuffer_to_file(self, file_name: str) -> None:
        """Save the backbuffer image to a file.

        Args:
        ----
            file_name (str): The name of the file to save the image to.

        Returns:
        -------
            None

        """
        cv.imwrite(file_name, self.opencv_image)

    @check_valid_hwnd
    def get_pixel_change(self, area: tuple[int, int, int, int] | None = None) -> int:
        """Calculates the number of pixels that have changed between the current image and a later version of the image.

        Args:
        ----
            area (Optional[Optional[Tuple[int, int, int, int]]]): The region of the image to consider. If not specified,
                the entire image will be used. Defaults to None.

        Raises:
        ------
            InvalidImage: If the image data is invalid.

        Returns:
        -------
            int: The number of pixels that have changed between the two images.
        """
        # Gray and crop image
        image = self._crop_image(area, self.opencv_image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Create second image grayed with same size
        updated_image = self.refresh(False)
        updated_image = self._crop_image(area, updated_image)
        updated_image = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)

        # Compute the difference between the images
        diff = cv.absdiff(image, updated_image)

        # Count the number of non-zero pixels in the difference image
        num_pixels = np.count_nonzero(diff)

        return num_pixels

    # @check_valid_hwnd
    # def get_pixel_change(self, area: Optional[Tuple[int, int, int, int]] = None, delay: int = 600) -> int:
    #     """
    #     Calculates the number of pixels that have changed between the current image and a later version of the image.
    #
    #     Args:
    #         area (Optional[Optional[Tuple[int, int, int, int]]]): The region of the image to consider. If not specified,
    #             the entire image will be used. Defaults to None.
    #         delay (Optional[int]): The delay in milliseconds to wait before capturing the second image. Defaults to 600.
    #
    #     Raises:
    #         InvalidImage: If the image data is invalid.
    #
    #     Returns:
    #         int: The number of pixels that have changed between the two images.
    #     """
    #     image = self.refresh(False)
    #
    #     # Gray and crop image
    #     image = self._crop_image(area, image)
    #     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #
    #     sleep(delay / 1_000)
    #
    #     # Create second image grayed with same size
    #     updated_image = self.refresh(False)
    #     updated_image = self._crop_image(area, updated_image)
    #     updated_image = cv.cvtColor(updated_image, cv.COLOR_BGR2GRAY)
    #
    #     # Compute the difference between the images
    #     diff = cv.absdiff(image, updated_image)
    #
    #     # Count the number of non-zero pixels in the difference image
    #     num_pixels = np.count_nonzero(diff)
    #
    #     return num_pixels

    def _get_grouped_text(
        self,
        image: npt.NDArray[np.uint8],
        colors: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
    ) -> pd.DataFrame:
        """Applies pre-processing to the image and extracts text from the image using Tesseract OCR.
        Groups the text data by block and returns a DataFrame with the extracted text and relevant columns.

        Args:
        ----
            image (npt.NDArray[np.uint8]): The input image to search.
            colors (Optional[Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]]): A sequence of RGB tuples or
                a sequence of sequences containing RGB tuples.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.

        Returns:
        -------
            pd.DataFrame:
                A DataFrame with the extracted text and relevant columns.
        """
        if colors:
            image = filter_colors(image, colors, tolerance)

        # Resize the image to double the original size
        resized_img = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        # Apply bilateral filter for noise reduction
        img = cv.bilateralFilter(resized_img, 9, 75, 75)

        # Convert the image to grayscale and apply thresholding
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        img = cv.bitwise_not(img)

        # Extract text from the thresholded image using Tesseract OCR
        text: pd.DataFrame = pytesseract.image_to_data(
            img,
            config=self._config,
            output_type=pytesseract.Output.DATAFRAME,
        )

        # Filter out invalid text data and text with low confidence
        text = text[(text.conf > 0) & (text.conf < 100)]

        if text.empty:
            return text

        text["text"] = text["text"].astype(str)
        text = text[text.text.str.strip() != ""]
        text["conf"] = text["conf"] / 100

        # Group text data by block and extract relevant columns
        grouped_text = (
            text.groupby("block_num", as_index=False)
            .agg(
                {
                    "word_num": "max",
                    "left": "min",
                    "top": "min",
                    "height": "max",
                    "conf": "median",
                    "text": " ".join,
                }
            )
            .rename(columns={"conf": "confidence", "text": "text"})
        )

        # Extract the last word from each group of words and calculate the width of the group
        last_word_per_group = pd.merge(
            grouped_text[["word_num", "block_num", "left"]],
            text[["word_num", "block_num", "left", "width"]],
            on=["block_num", "word_num"],
        )
        last_word_per_group["width"] = last_word_per_group.eval("left_y - left_x + width")

        # Merge the width data with the grouped text data
        grouped_text = pd.merge(grouped_text, last_word_per_group[["block_num", "width"]], on="block_num")

        # Sort values in descending order based on confidence
        sorted_text = grouped_text.sort_values("confidence", ascending=False)

        # Convert coordinates back to original size
        sorted_text[["top", "left", "height", "width"]] //= 2

        return sorted_text

    def _crop_image(
        self,
        rect: tuple[int, int, int, int] | None = None,
        image: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Crop the current OpenCV image.

        Args:
        ----
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to crop.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is returned.
            image (Optional[npt.NDArray[np.uint8]]): An optional input image to crop. If not provided, the current
                OpenCV image is used.

        Returns:
        -------
            npt.NDArray[np.uint8]: A numpy array of unsigned integers of shape (height, width, channels), representing
            an image in the OpenCV format.
        """
        image = image if image is not None else self.opencv_image
        assert image is not None

        if rect:
            # Extract the coordinates and dimensions from the rectangle
            x, y, w, h = rect
            # Crop the image using the specified region
            return image[y : y + h, x : x + w]
        # If no rectangle is specified, return the entire image
        return image

    @check_valid_image
    def get_text(
        self,
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
        confidence: float = 0.8,
    ) -> Sequence[TextInfo]:
        """Extracts text from the image using Tesseract OCR and returns a list of TextInfo objects for the text with
        confidence greater than or equal to the specified value.

        Args:
        ----
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            colors (Optional[Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]]): A sequence of RGB tuples or
                a sequence of sequences containing RGB tuples.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.
            confidence (float): The minimum confidence level for text to be included in the output. Must be a
                value between 0 and 1. Defaults to 0.8.

        Returns:
        -------
            Sequence[TextInfo]: A list of TextInfo objects for the acceptable text in the image.

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        image = self._crop_image(rect)
        sorted_text = self._get_grouped_text(image, colors, tolerance)

        if sorted_text.empty:
            return []

        # Filter out text with low confidence and format the output data
        acceptable_text = sorted_text[sorted_text["confidence"] >= confidence][
            ["text", "left", "top", "width", "height", "confidence"]
        ]

        if rect:
            acceptable_text.loc[:, ["left", "top"]] += rect[0:2]

        # Return `TextInfo` objects for the acceptable text in the image
        return cast(Sequence[TextInfo], acceptable_text.apply(TextInfo.from_row, axis=1).tolist())  # type: ignore[call-overload]

    @check_valid_image
    def find_text(
        self,
        search_text: str,
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
        confidence: float = 0.8,
    ) -> Sequence[TextInfo]:
        """Extracts text from the image using Tesseract OCR and returns a list of TextInfo objects for the text with
        confidence greater than or equal to the specified value and containing the specified search text.

        Args:
        ----
            search_text (str): The text to search for in the image.
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            colors (Optional[Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]]): A sequence of RGB tuples or
                a sequence of sequences containing RGB tuples.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.
            confidence (float, optional): The minimum confidence level for text to be included in the output. Must be a
                value between 0 and 1. Defaults to 0.8.

        Returns:
        -------
            List[TextInfo]: A list of TextInfo objects for the acceptable text in the image.

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
            ValueError: If the `search_text` argument is empty or None.
        """
        image = self._crop_image(rect)
        sorted_text = self._get_grouped_text(image, colors, tolerance)

        if sorted_text.empty:
            return []

        # Filter out text with low confidence and format the output data
        acceptable_text = sorted_text[
            (sorted_text["confidence"] >= confidence) & (sorted_text["text"].str.contains(search_text))
        ][["text", "left", "top", "width", "height", "confidence"]]

        if rect:
            acceptable_text.loc[:, ["left", "top"]] += rect[0:2]

        # Return TextInfo objects for the acceptable text in the image
        return cast(Sequence[TextInfo], acceptable_text.apply(TextInfo.from_row, axis=1).tolist())  # type: ignore[call-overload]

    @check_valid_image
    def get_color(self, point: tuple[int, int]) -> Color:
        """Returns the color of a pixel in the image at the specified coordinates.

        Args:
        ----
            point (Tuple[int, int]): The a tuple containing the x and y coordinate of the pixel.

        Returns:
        -------
            Color: The color of the pixel.

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
            IndexError: If the coordinates are out of bounds.
        """
        assert self.opencv_image is not None
        x, y = point

        # Check if the coordinates are within the image bounds
        if not (0 <= x < self.opencv_image.shape[1] and 0 <= y < self.opencv_image.shape[0]):
            raise IndexError(f"Coordinates ({x},{y}) are out of bounds. Please click within the selected window.")

        # Get the color of the pixel at the specified coordinates
        color = Color(*np.flip(self.opencv_image[y, x]))

        return color

    @check_valid_image
    def find_color(
        self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
    ) -> ShapeList[Point]:
        """Finds all x, y coordinates in a given OpenCV image that match the given color and tolerance.

        Args:
        ----
            color (Tuple[int, int, int]): The color to search for, in RGB format.
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.

        Returns:
        -------
            ShapeList[Point]: A ShapeList instance containing the x, y coordinates of all pixels in the image that match
            the given color within the specified tolerance.

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        assert self.opencv_image is not None
        image = self._crop_image(rect)
        mask = filter_colors(image, color, tolerance)

        # Stack x and y arrays into a single (2, num_points) array
        points = np.stack(np.where(mask == 255)[::-1], axis=1)
        if rect:
            points = np.add(points, rect[0:2])

        # Combine coordinates into (x, y) tuples
        height, width = self.opencv_image.shape[:2]
        center = Point(width // 2, height // 2)
        return ShapeList(Point, points, center)  # type: ignore[arg-type]

    @check_valid_image
    def get_average_color(self, rect: tuple[int, int, int, int] | None = None) -> Color:
        """Get the average color of an image within a specified rectangular region.

        Args:
        ----
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.

        Returns:
        -------
            Color: The average color within the specified region.

        """
        assert self.opencv_image is not None
        return Color(*self._get_average_color(self.opencv_image, rect))

    def _get_average_color(
        self,
        image: npt.NDArray[np.uint8],
        rect: tuple[int, int, int, int] | None = None,
    ) -> npt.NDArray[np.int16]:
        """Calculate the average color of an image within a specified rectangular region.

        Args:
        ----
            image (npt.NDArray[np.uint8]): The input image to calculate the average color from.
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.

        Returns:
        -------
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
        self,
        rect: tuple[int, int, int, int] | None = None,
        index: int = 1,
        ignore_colors: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None = None,
    ) -> Color:
        """Returns the most common color in the given image.

        Args:
        ----
            rect (Optional[Tuple[int, int, int, int]]): A tuple containing the x, y coordinates and the width and height
                of the area to analyze.
            index (int): The index of the desired popular color (1 for the most common, 2 for the second most common,
                and so on).
            ignore_colors (Optional[Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]]): A sequence of RGB
                tuples or a sequence of sequences containing RGB tuples.

        Returns:
        -------
          Color: Most common color.
        """
        assert self.opencv_image is not None
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

        return Color(*most_common_color)

    @check_valid_image
    def get_all_colors_with_counts(
        self,
        rect: tuple[int, int, int, int] | None = None,
    ) -> Sequence[tuple[Color, int]]:
        """Returns all colors in a given image or area of an image with the respective counts.

        Args:
        ----
            rect (Optional[Tuple[int, int, int, int]]): A tuple containing the x, y coordinates and the width and height
                of the area to analyze.


        Returns:
        -------
          Color: Most common color.
        """
        assert self.opencv_image is not None
        cropped_image = self._crop_image(rect)

        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = cropped_image.reshape(-1, 3)

        unique, counts = np.unique(reshaped_image, axis=0, return_counts=True)

        sorted_indices = np.argsort(counts)[::-1]
        sorted_unique = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        colors_with_counts = [(Color(r, g, b), count) for (b, g, r), count in zip(sorted_unique, sorted_counts)]
        return colors_with_counts

    @check_valid_image
    def get_median_color(self, rect: tuple[int, int, int, int] | None = None) -> Color:
        """Returns the dominant color in the given image.

        Args:
        ----
          rect (Optional[Tuple[int, int, int, int]]): A tuple containing the x, y coordinates and the width and height
            of the area to analyze.

        Returns:
        -------
          Color: Dominant color.
        """
        assert self.opencv_image is not None
        cropped_image = self._crop_image(rect)

        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = cropped_image.reshape(-1, 3)

        # Calculate the median value for each channel
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)

        # Calculate the median value for each channel
        median_color = median_color[::-1].tolist()

        return Color(*median_color)

    @staticmethod
    def _get_dominant_color(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Returns the dominant color in the given image.

        Args:
        ----
          image (np.ndarray): Input image (in BGR format).

        Returns:
        -------
          np.ndarray: Dominant color in the image (in BGR format).
        """
        # Reshape the image to a 2D array (height*width, channels)
        reshaped_image = image.reshape(-1, 3)

        # Calculate the median value for each channel
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)

        return cast(npt.NDArray[np.uint8], median_color)

    @check_valid_image
    def maximize_color_match(
        self,
        rect: tuple[int, int, int, int],
        initial_tolerance: int = 100,
        tolerance_step: int = 1,
    ) -> tuple[Color, int]:
        """Finds the best color and tolerance to maximize the colors in a given area of the image, while minimizing
            outside that area.

        Args:
        ----
            rect (Tuple[int, int, int, int]): A tuple containing the x, y coordinates and the width and height of the
                area to analyze.
            initial_tolerance (int): The initial tolerance value to use when searching for the best color match.
            tolerance_step (int): The step size to use when decreasing the tolerance value during the search.

        Returns:
        -------
            A tuple containing the RGB values of the best color match and the tolerance value used to obtain that match.
        """
        assert self.opencv_image is not None
        x, y, w, h = rect

        # Crop the image to the specified region of interest (ROI)
        cropped_image = self.opencv_image[y : y + h, x : x + w]

        # Get the dominant color in the ROI
        dominant_color = self._get_dominant_color(cropped_image)

        # Initialize variables to keep track of the best color match and tolerance value found so far
        tolerance = initial_tolerance
        best_tolerance = 0
        best_ratio = -1

        # Calculate the total number of pixels in the inner (ROI) and outer (rest of the image) regions
        inner_total_pixels = cropped_image.size // 3
        outer_total_pixels = (self.opencv_image.size // 3) - inner_total_pixels

        # Iterate over different tolerance values to find the best color match
        while tolerance >= 0:
            # Define the color range to consider based on the dominant color and current tolerance value
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

            # Create a binary mask of the pixels within the color range in the ROI
            mask = cv.inRange(cropped_image, lower_bound, upper_bound)

            # Count the number of pixels within the color range in the ROI and outside the ROI
            pixel_count = np.sum(mask == 255)
            outside_mask = cv.inRange(self.opencv_image, lower_bound, upper_bound)
            outside_pixel_count = np.sum(outside_mask == 255) - pixel_count

            # Calculate the ratio of pixels within the color range in the ROI to the pixels outside the ROI
            inner_ratio = pixel_count / inner_total_pixels
            outer_ratio = outside_pixel_count / outer_total_pixels
            ratio = inner_ratio / (outer_ratio + 1)

            # Update the best color match and tolerance value if a better match is found
            if ratio >= best_ratio:
                best_ratio = ratio
                best_tolerance = tolerance

            # Decrease the tolerance value
            tolerance -= tolerance_step

        # Convert the best color match from BGR to RGB and return it along with the tolerance value
        best_color_rgb = Color(dominant_color[2], dominant_color[1], dominant_color[0])  # Convert BGR to RGB

        return best_color_rgb, best_tolerance

    @staticmethod
    def _calculate_median_difference(
        image1: npt.NDArray[np.uint8],
        image2: npt.NDArray[np.uint8],
        mask: npt.NDArray[np.uint8] | None = None,
    ) -> int:
        """Calculate the median difference between two images while optionally ignoring a mask.

        Args:
        ----
            image1 (npt.NDArray[np.uint8]): The first image.
            image2 (npt.NDArray[np.uint8]): The second image.
            mask (Optional[npt.NDArray[np.uint8]): The mask indicating the pixels to ignore. Defaults to None.

        Returns:
        -------
            int: The median difference between the images.
        """
        assert image1.shape == image2.shape, "Image shapes must match."

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
        self,
        sub_image: npt.NDArray[np.uint8] | Image.Image,
        rect: tuple[int, int, int, int] | None = None,
        confidence: float | None = 0.95,
        median_tolerance: int | None = None,
    ) -> ShapeList[Rectangle]:
        """Find a subimage in a larger image given a confidence level and optional color tolerance.

        Args:
        ----
            sub_image (Union[npt.NDArray[np.uint8], Image.Image]): The subimage to search for.
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            confidence (Optional[float]): The confidence level to use for the template matching. Defaults to 0.95.
            median_tolerance (Optional[int]): The maximum difference allowed between each channel of the average color
                of the subimage and the average color of the matched region in the main image. If not provided, no color
                tolerance check is performed.

        Returns:
        -------
            ShapeList[Rectangle]: An object containing the coordinates of the subimage matches.

        """
        assert self.opencv_image is not None
        image = self._crop_image(rect)

        # Convert sub_image to BGR color space if needed
        if isinstance(sub_image, Image.Image):
            sub_image = np.array(sub_image.convert("RGBA"))

        if sub_image.shape[-1] == 4:
            sub_alpha = sub_image[..., 3]
            sub_image_bgr = cv.cvtColor(sub_image[..., :3], cv.COLOR_RGB2BGR)
            mask = sub_alpha.astype(np.uint8)
        else:
            sub_image_bgr = cv.cvtColor(sub_image[..., :3], cv.COLOR_RGB2BGR)
            mask = None

        # Convert the images to grayscale
        main_image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        sub_image_gray = cv.cvtColor(sub_image_bgr, cv.COLOR_RGB2GRAY)

        # Perform template matching with mask if available
        res = cv.matchTemplate(main_image_gray, sub_image_gray, cv.TM_CCORR_NORMED, mask=mask)

        # Apply confidence threshold to filter matches
        res = np.logical_and(res >= confidence, np.logical_not(np.isinf(res)))

        rects = []
        w, h = sub_image_bgr.shape[1::-1]

        # Apply mask to sub_image if available
        if mask is not None:
            sub_image_bgr = cv.bitwise_and(sub_image_bgr, sub_image_bgr, mask=mask)
            sub_image_bgr = sub_image_bgr[..., :3]  # Remove the alpha channel if present

        for loc in np.fliplr(np.transpose(np.where(res))):
            x, y = loc
            main_image_region = image[y : y + h, x : x + w]
            found_rect = (
                x + (rect[0] if rect else 0),
                y + (rect[1] if rect else 0),
                w,
                h,
            )

            if median_tolerance is not None:
                found_median_diff = self._calculate_median_difference(main_image_region, sub_image_bgr, mask)
                print(found_median_diff)

                if found_median_diff < median_tolerance:
                    rects.append(found_rect)
            else:
                rects.append(found_rect)

        # double the rectangles for groupRectangles
        rects = np.repeat(np.array(rects), 2, axis=0)

        # Group the rectangle coordinates
        rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.1)

        center_of_bitmap = Point(self.opencv_image.shape[1] // 2, self.opencv_image.shape[0] // 2)

        return ShapeList(Rectangle, rects, center_of_bitmap)

    # @check_valid_image
    # def find_image(
    #     self,
    #     sub_image: Union[npt.NDArray[np.uint8], Image.Image],
    #     rect: Optional[Tuple[int, int, int, int]] = None,
    #     confidence: Optional[float] = 0.95,
    #     tolerance: int = 0
    # ) -> ShapeList[Rectangle]:
    #     """
    #     Find a subimage in a larger image given a confidence level.
    #
    #     Args:
    #         sub_image (Union[npt.NDArray[np.uint8], Image.Image]): The subimage to search for.
    #         rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
    #             The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
    #             rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
    #         confidence (Optional[float]): The confidence level to use for the template matching. Defaults to 0.95.
    #         tolerance (int): The maximum difference allowed between each channel of the given color and the pixel color.
    #             Defaults to 0 for an exact match.
    #
    #     Returns:
    #         np.ndarray: An array containing the coordinates of the subimage matches.
    #     """
    #     assert self.opencv_image is not None
    #     image = self._crop_image(rect)
    #
    #     if isinstance(sub_image, Image.Image):
    #         sub_image = np.array(sub_image.convert("RGBA"))
    #
    #     if sub_image.shape[-1] == 4:
    #         sub_alpha = sub_image[..., 3]
    #         sub_image_bgr = cv.cvtColor(sub_image[..., :3], cv.COLOR_RGB2BGR)
    #         mask = sub_alpha.astype(np.uint8)
    #     else:
    #         sub_image_bgr = cv.cvtColor(sub_image[..., :3], cv.COLOR_RGB2BGR)
    #         mask = None
    #
    #     main_image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #     sub_image_gray = cv.cvtColor(sub_image_bgr, cv.COLOR_RGB2GRAY)
    #
    #     # Perform template matching with mask if available
    #     res = cv.matchTemplate(main_image_gray, sub_image_gray, cv.TM_CCORR_NORMED, mask=mask)
    #
    #     res = np.logical_and(res >= confidence, np.logical_not(np.isinf(res)))
    #
    #     loc = np.repeat(np.fliplr(np.transpose(np.where(res))), 2, axis=0)  # type: ignore[arg-type]
    #     if rect:
    #         loc = np.add(loc, rect[0:2])
    #
    #     assert isinstance(sub_image, np.ndarray)
    #     rects = np.hstack((loc, np.full((len(loc), 2), (sub_image.shape[1], sub_image.shape[0]))))
    #
    #     # Group the rectangle coordinates
    #     rects, _ = cv.groupRectangles(rects, groupThreshold=1, eps=0.1)
    #
    #     center_of_bitmap = Point(self.opencv_image.shape[1] // 2, self.opencv_image.shape[0] // 2)
    #
    #     return ShapeList(Rectangle, rects, center_of_bitmap)

    @check_valid_image
    def find_contours(
        self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
        min_area: int = 10,
        vertices: int | None = None,
    ) -> ShapeList[Contour]:
        """Find contours of the regions in the image that match a given color.

        Args:
        ----
            color (Tuple[int, int, int]): The target color as a sequence of (R,G,B) values.
            rect (Optional[Tuple[int, int, int, int]]): A tuple specifying a rectangular region in the image to search.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h). If not provided, the whole image is searched.
            tolerance (int): The tolerance value for color match.
            min_area (int): The minimum area of a contour to be included in the result.
            vertices (int): The number of vertices to look for.

        Returns:
        -------
            ShapeList[Contour]: A ShapeList representing the contours of the regions that match the given color.

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        assert self.opencv_image is not None
        image = self._crop_image(rect)

        image = filter_colors(image, color, tolerance)

        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours = [c + np.array([rect[0], rect[1]]) if rect else c for c in contours]
        contours = [c for c in contours if cv.moments(c)["m00"] != 0 and cv.contourArea(c) >= min_area]

        if vertices is not None:
            contours = [c for c in contours if vertices == len(cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True))]

        center_of_bitmap = Point(self.opencv_image.shape[1] // 2, self.opencv_image.shape[0] // 2)

        return ShapeList(Contour, contours, center_of_bitmap)  # type: ignore[type-abstract]

    @check_valid_image
    def draw_points(
        self,
        points: Sequence[tuple[int, int]],
        color: tuple[int, int, int] = Color(255, 0, 0),
    ) -> None:
        """Draw points on the image with the specified color.

        Args:
        ----
            points (Sequence[Tuple[int, int]]): The points to be drawn.
            color (Tuple[int, int, int]): The color to use for drawing the points, specified as a tuple of
                (R,G,B) values. Default is red (255, 0, 0).

        Returns:
        -------
            None

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        assert self.opencv_image is not None

        if not isinstance(points, ShapeList):
            center_of_bitmap = Point(self.opencv_image.shape[1] // 2, self.opencv_image.shape[0] // 2)
            points = ShapeList(Point, np.array(points), center_of_bitmap)

        self.opencv_image[points.data[:, 1], points.data[:, 0]] = color[::-1]  # type: ignore[call-overload]

    @check_valid_image
    def draw_contours(
        self,
        contours: Sequence[Sequence[Sequence[tuple[int, int]]]],
        color: tuple[int, int, int] = Color(255, 0, 0),
    ) -> None:
        """Draw a contour on the image with the specified color.

        Args:
        ----
            contours (Sequence[Sequence[Sequence[Tuple[int, int]]]]): The contours to be drawn.
            color (Tuple[int, int, int]): The color to use for drawing the contours, specified as a tuple of
                (R,G,B) values. Default is red (255, 0, 0).

        Returns:
        -------
            None

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        assert self.opencv_image is not None

        if not isinstance(contours, ShapeList):
            center_of_bitmap = Point(self.opencv_image.shape[1] // 2, self.opencv_image.shape[0] // 2)
            contours = ShapeList(Contour, [np.array(c) for c in contours], center_of_bitmap)  # type: ignore[type-abstract]

        cv.drawContours(self.opencv_image, contours.data, -1, color[::-1], 2)

    @check_valid_image
    def draw_circle(
        self,
        circle: tuple[int, int, int],
        color: tuple[int, int, int] = Color(255, 0, 0),
    ) -> None:
        """Draws a rectangle onto the backbuffer.

        Args:
        ----
            circle (Tuple[int, int, int]): A tuple specifying an x, y, radius in the image.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h).
            color (Tuple[int, int, int]): The color of the rectangle. Defaults to red (255, 0, 0).
        """
        # Convert the top-left and bottom-right coordinates to OpenCV format
        assert self.opencv_image is not None
        x, y, r = circle

        # Draw the rectangle onto the image
        cv.circle(self.opencv_image, (x, y), r, color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def draw_rectangle(
        self,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int] = Color(255, 0, 0),
    ) -> None:
        """Draws a rectangle onto the backbuffer.

        Args:
        ----
            rect (Tuple[int, int, int, int]): A tuple specifying a rectangular region.
                The tuple contains the coordinates of the top-left corner (x, y) and the width and height of the
                rectangle (w, h) in the format (x, y, w, h).
            color (Tuple[int, int, int]): The color of the rectangle. Defaults to red (255, 0, 0).
        """
        # Convert the top-left and bottom-right coordinates to OpenCV format
        assert self.opencv_image is not None
        x, y, w, h = rect

        # Draw the rectangle onto the image
        cv.rectangle(self.opencv_image, (x, y), (x + w, y + h), color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def filter_colors(
        self,
        colors: tuple[int, int, int] | Sequence[tuple[int, int, int]],
        tolerance: int = 0,
        keep_original_colors: bool = False,
    ) -> None:
        """Filter out all colors from the image that are not in the specified list of colors with a given tolerance.
        Updates the backbuffer image of the window with the filtered image.

        Args:
        ----
            colors (Union[Tuple[int, int, int], Sequence[Tuple[int, int, int]]]): A sequence of RGB tuples or a sequence
                of sequences containing RGB tuples.
            tolerance (int): The color tolerance in the range of 0-255.
            keep_original_colors (bool): If True, the returned value will be a copy of the input image where all
                non-matching pixels are set to black.

        Returns:
        -------
            None

        Raises:
        ------
            autocv.exceptions.InvalidImage: If the input image is invalid or None.
        """
        assert self.opencv_image is not None
        grey_image = filter_colors(self.opencv_image, colors, tolerance, keep_original_colors)
        self.opencv_image = cv.cvtColor(grey_image, cv.COLOR_GRAY2BGR)
