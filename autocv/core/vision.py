"""Window capture, image processing, and OCR utilities.

This module provides :class:`~autocv.core.vision.Vision`, which extends
:class:`~autocv.core.window_capture.WindowCapture` with a persistent image
backbuffer (``opencv_image``) and common operations needed by AutoCV:

- capturing frames into the backbuffer,
- extracting text via PaddleOCR,
- sampling, filtering, and analyzing colors,
- basic morphology, template matching, and contour helpers.

All image operations use OpenCV's BGR channel order unless explicitly stated.
Public color inputs/outputs use RGB tuples for consistency across the library.
"""

from __future__ import annotations

__all__ = ("Vision",)

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

import cv2 as cv
import numpy as np
import numpy.typing as npt
from PIL import Image
from typing_extensions import Self

from paddleocr import PaddleOCR

from .decorators import check_valid_hwnd, check_valid_image
from .image_processing import filter_colors
from .window_capture import WindowCapture

if TYPE_CHECKING:
    from collections.abc import Sequence

Color = tuple[int, int, int]  # RGB channel order
Point = tuple[int, int]
Rect = tuple[int, int, int, int]  # x, y, width, height
Contour = npt.NDArray[np.int32]
MaskArray = npt.NDArray[np.uint8]
NDArrayUint8 = npt.NDArray[np.uint8]
NDArrayInt16 = npt.NDArray[np.int16]


class OcrTextEntry(TypedDict):
    """Structured OCR output returned by :meth:`Vision.get_text`.

    The ``rect`` field is ``[left, top, width, height]`` in backbuffer coordinates.
    """

    text: str
    rect: list[int]
    confidence: float


RGBA_CHANNELS: Final[int] = 4
COLOR_IMAGE_NDIMS: Final[int] = 3
RGB_CHANNELS: Final[int] = 3
MAX_COLOR_VALUE: Final[int] = 255


SpeedPreset = Literal["fast", "balanced", "accurate"]

_OCR_CPU_THREADS: Final[int] = 8
_OCR_UPSCALE_FACTOR: Final[float] = 2.0
_OCR_BOX_LENGTH: Final[int] = 4
_OCR_BILATERAL_DIAMETER: Final[int] = 5
_OCR_BILATERAL_SIGMA_COLOR: Final[int] = 50
_OCR_BILATERAL_SIGMA_SPACE: Final[int] = 50
_GROUP_RECTANGLES_EPS: Final[float] = 0.1
_GROUP_RECTANGLES_THRESHOLD: Final[int] = 1
_DEFAULT_MORPH_KERNEL_SIZE: Final[int] = 3


@dataclass(frozen=True, slots=True)
class _OcrDetectionPreset:
    """PaddleOCR detection tuning for a speed preset."""

    det_side_len: int
    det_box_thresh: float
    det_unclip_ratio: float


_OCR_PRESETS: Final[dict[str, _OcrDetectionPreset]] = {
    "fast": _OcrDetectionPreset(det_side_len=640, det_box_thresh=0.55, det_unclip_ratio=1.6),
    "balanced": _OcrDetectionPreset(det_side_len=960, det_box_thresh=0.50, det_unclip_ratio=1.8),
    "accurate": _OcrDetectionPreset(det_side_len=1280, det_box_thresh=0.45, det_unclip_ratio=1.9),
}


class Vision(WindowCapture):
    """Capture windows, process images, and perform OCR.

    The class maintains a persistent OpenCV-compatible backbuffer in
    :attr:`opencv_image`. Most image processing routines operate on this buffer.

    Notes:
        - Frames returned by :meth:`refresh` and stored in :attr:`opencv_image`
          are in BGR channel order.
        - Public color values are expressed as RGB tuples.
    """

    def __init__(
        self: Self,
        hwnd: int = -1,
        lang: str = "en",
        device: str | None = None,
        conf_threshold: float = 0.60,
        speed: SpeedPreset = "balanced",
        *,
        disable_model_source_check: bool = False,
    ) -> None:
        """Initialise a Vision object.

        Args:
            hwnd: Window handle of the target window. Defaults to -1.
            lang: PaddleOCR language code.
            device: PaddleOCR device override (e.g. ``"cpu"`` / ``"gpu"``); ``None`` uses PaddleOCR defaults.
            conf_threshold: OCR recognition confidence threshold between 0 and 1.
            speed: Preset that tunes detection settings.
            disable_model_source_check: When ``True``, disables PaddleOCR/PaddleX model host connectivity checks
                via the ``DISABLE_MODEL_SOURCE_CHECK`` environment variable.
        """
        super().__init__(hwnd)
        self.opencv_image: NDArrayUint8 = np.empty(0, dtype=np.uint8)
        self.api: PaddleOCR | None = None

        if disable_model_source_check:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        preset = _OCR_PRESETS.get(speed, _OCR_PRESETS["balanced"])
        self._ocr_lang: str = lang
        self._ocr_device: str | None = device
        self._ocr_conf_threshold: float = float(conf_threshold)
        self._ocr_det_side_len: int = int(preset.det_side_len)
        self._ocr_det_box_thresh: float = float(preset.det_box_thresh)
        self._ocr_det_unclip_ratio: float = float(preset.det_unclip_ratio)

    def _ensure_ocr(self: Self) -> PaddleOCR:
        """Return an initialised PaddleOCR instance, creating one on demand."""
        if self.api is not None:
            return self.api

        try:
            self.api = PaddleOCR(
                # Core selection
                lang=self._ocr_lang,
                ocr_version="PP-OCRv5",
                # OSRS/game UI: disable doc features
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                # Detection tuning
                text_det_limit_type="max",
                text_det_limit_side_len=self._ocr_det_side_len,
                text_det_box_thresh=self._ocr_det_box_thresh,
                text_det_unclip_ratio=self._ocr_det_unclip_ratio,
                # Recognition filtering (TTS-friendly)
                text_rec_score_thresh=self._ocr_conf_threshold,
                # Runtime
                device=self._ocr_device,
                cpu_threads=_OCR_CPU_THREADS,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "paddle":
                raise
            msg = "PaddleOCR requires PaddlePaddle. Install `autocv[paddle-cpu]` or `autocv[paddle-gpu]`."
            raise RuntimeError(msg) from exc

        return self.api

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> NDArrayUint8:
        """Convert a PIL image to an OpenCV-compatible BGR array."""
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        return cast("NDArrayUint8", cv.cvtColor(rgb, cv.COLOR_RGB2BGR))

    def set_backbuffer(self: Self, image: NDArrayUint8 | Image.Image) -> None:
        """Set the image buffer to the provided NumPy array or PIL Image.

        Args:
            image: Image data used to refresh the backbuffer.
        """
        if isinstance(image, Image.Image):
            self.opencv_image = self._pil_to_bgr(image)
        else:
            self.opencv_image = image

    @check_valid_hwnd
    def refresh(self: Self, *, set_backbuffer: bool = True) -> NDArrayUint8 | None:
        """Capture the current window image and optionally persist it to ``opencv_image``.

        Args:
            set_backbuffer: When ``True``, persist the capture to ``self.opencv_image``;
                otherwise return the captured frame.

        Raises:
            InvalidHandleError: Raised when ``self.hwnd`` is not a valid window handle.

        Returns:
            Captured frame when ``set_backbuffer`` is ``False``; ``None`` otherwise.
        """
        frame = self.capture_frame(persist=False)
        if set_backbuffer:
            self.set_backbuffer(frame)
            return None
        return frame

    @check_valid_image
    def save_backbuffer_to_file(self: Self, file_name: str) -> None:
        """Save the backbuffer image to a file.

        Args:
            file_name: Path where the backbuffer snapshot is stored.
        """
        cv.imwrite(file_name, self.opencv_image)

    @check_valid_hwnd
    @check_valid_image
    def get_pixel_change(self: Self, area: Rect | None = None) -> int:
        """Calculate how many pixels changed between current and refreshed frames.

        Args:
            area: Region of interest expressed as (x, y, width, height); ``None``
                inspects the full frame.

        Raises:
            InvalidImageError: Raised when the capture buffer is empty.

        Returns:
            int: Count of pixels with different intensities between frames.
        """
        current_region = self._crop_image(area, self.opencv_image)
        current_gray = cv.cvtColor(current_region, cv.COLOR_BGR2GRAY)

        updated_frame = self.capture_frame(persist=False)
        updated_region = self._crop_image(area, updated_frame)
        updated_gray = cv.cvtColor(updated_region, cv.COLOR_BGR2GRAY)

        diff = cv.absdiff(current_gray, updated_gray)
        return int(np.count_nonzero(diff))

    def _crop_image(
        self: Self,
        rect: Rect | None = None,
        image: NDArrayUint8 | None = None,
    ) -> NDArrayUint8:
        """Crop the current image or a provided image to the specified rectangle.

        Args:
            rect: Region to crop in ``(x, y, width, height)`` form.
            image: Explicit image to operate on; defaults to ``self.opencv_image``.

        Returns:
            Cropped slice of the source image.

        Raises:
            ValueError: If ``rect`` has non-positive dimensions, negative coordinates, or lies outside the image.
        """
        image = image if image is not None else self.opencv_image
        if rect is None:
            return image

        x, y, w, h = rect
        if w <= 0 or h <= 0:
            msg = "Crop rectangle must have positive width and height."
            raise ValueError(msg)
        if x < 0 or y < 0:
            msg = "Crop rectangle coordinates must be non-negative."
            raise ValueError(msg)

        height, width = image.shape[:2]
        if x >= width or y >= height:
            msg = "Crop rectangle lies outside the image bounds."
            raise ValueError(msg)

        right = min(x + w, width)
        bottom = min(y + h, height)
        if right <= x or bottom <= y:
            msg = "Crop rectangle lies outside the image bounds."
            raise ValueError(msg)

        return image[y:bottom, x:right]

    @staticmethod
    def _poly_to_bbox(poly: object) -> tuple[int, int, int, int]:
        """Convert a PaddleOCR polygon into an ``(x_min, y_min, x_max, y_max)`` box."""
        points = np.asarray(poly, dtype=np.int32)
        xs = points[:, 0]
        ys = points[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _box_to_bbox(box: Sequence[float | int]) -> tuple[int, int, int, int]:
        """Convert a PaddleOCR box into integer ``(x_min, y_min, x_max, y_max)`` coordinates."""
        if len(box) != _OCR_BOX_LENGTH:
            msg = "OCR box must have four elements (x_min, y_min, x_max, y_max)."
            raise ValueError(msg)
        x_min, y_min, x_max, y_max = box
        return int(x_min), int(y_min), int(x_max), int(y_max)

    @staticmethod
    def _extract_ocr_payload(prediction: object) -> dict[str, object] | None:
        """Normalize PaddleOCR predictions into a single JSON-like dictionary."""
        payload: object = prediction
        json_payload: object | None = getattr(prediction, "json", None)
        if json_payload is not None:
            payload = json_payload
        if isinstance(payload, list):
            payload_list = cast("list[object]", payload)
            if not payload_list:
                return None
            payload = payload_list[0]
        return cast("dict[str, object]", payload) if isinstance(payload, dict) else None

    @staticmethod
    def _prepare_ocr_input(image: NDArrayUint8) -> NDArrayUint8:
        """Preprocess an image for PaddleOCR inference."""
        if image.ndim == COLOR_IMAGE_NDIMS and image.shape[-1] == RGB_CHANNELS:
            gray: NDArrayUint8 = cast("NDArrayUint8", cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        else:
            gray = image

        gray = cast(
            "NDArrayUint8",
            cv.resize(
                gray,
                None,
                fx=_OCR_UPSCALE_FACTOR,
                fy=_OCR_UPSCALE_FACTOR,
                interpolation=cv.INTER_LANCZOS4,
            ),
        )
        gray = cast(
            "NDArrayUint8",
            cv.bilateralFilter(gray, _OCR_BILATERAL_DIAMETER, _OCR_BILATERAL_SIGMA_COLOR, _OCR_BILATERAL_SIGMA_SPACE),
        )
        gray = cast("NDArrayUint8", cv.normalize(gray, gray, 0, MAX_COLOR_VALUE, cv.NORM_MINMAX))
        return cast("NDArrayUint8", cv.cvtColor(gray, cv.COLOR_GRAY2BGR))

    @check_valid_image
    def get_text(
        self: Self,
        rect: Rect | None = None,
        colors: Color | Sequence[Color] | None = None,
        tolerance: int = 0,
        confidence: float | None = 0.8,
    ) -> list[OcrTextEntry]:
        """Extract text from the backbuffer using PaddleOCR.

        Args:
            rect: Search region (x, y, width, height).
            colors: RGB colour(s) to isolate before OCR.
            tolerance: Per-channel tolerance when matching the colour filter.
            confidence: Minimum acceptable OCR confidence between 0 and 1. If None, no filtering.

        Returns:
            Text entries with bounding boxes and confidence levels, using backbuffer coordinates.
        """
        image = self._crop_image(rect, self.opencv_image)
        if colors:
            image = filter_colors(image, colors, tolerance, keep_original_colors=True)

        work = self._prepare_ocr_input(image)
        payload = self._extract_ocr_payload(cast("object", self._ensure_ocr().predict(work)))
        if payload is None:
            return []

        rec_texts = cast("list[str]", payload.get("rec_texts") or [])
        rec_scores = cast("list[float]", payload.get("rec_scores") or [])
        rec_boxes = cast("Sequence[Sequence[float | int]] | None", payload.get("rec_boxes"))
        rec_polys = cast("Sequence[object] | None", payload.get("rec_polys"))

        min_confidence = float(confidence) if confidence is not None else None
        results: list[OcrTextEntry] = []

        n = min(len(rec_texts), len(rec_scores))
        for i in range(n):
            text = str(rec_texts[i]).strip()
            if not text:
                continue

            score = float(rec_scores[i])
            if min_confidence is not None and score < min_confidence:
                continue

            bbox: tuple[int, int, int, int] | None = None
            if rec_boxes is not None and i < len(rec_boxes):
                bbox = self._box_to_bbox(rec_boxes[i])
            elif rec_polys is not None and i < len(rec_polys):
                bbox = self._poly_to_bbox(rec_polys[i])

            if bbox is None:
                continue

            x_min_i, y_min_i, x_max_i, y_max_i = bbox
            left = int(x_min_i / _OCR_UPSCALE_FACTOR)
            top = int(y_min_i / _OCR_UPSCALE_FACTOR)
            right = int(x_max_i / _OCR_UPSCALE_FACTOR)
            bottom = int(y_max_i / _OCR_UPSCALE_FACTOR)
            width = max(0, right - left)
            height = max(0, bottom - top)

            if rect is not None:
                left += rect[0]
                top += rect[1]

            results.append(OcrTextEntry(text=text, rect=[left, top, width, height], confidence=score))

        return results

    @check_valid_image
    def get_color(self: Self, point: Point) -> Color:
        """Return the color of the pixel at the specified coordinates.

        Args:
            point: Pixel coordinates expressed as ``(x, y)``.

        Returns:
            Pixel colour as ``(R, G, B)``.

        Raises:
            InvalidImageError: Raised when ``self.opencv_image`` is empty.
            ValueError: Raised when the backbuffer is not a 3-channel BGR image.
            IndexError: If the coordinates are out of bounds.
        """
        if self.opencv_image.ndim != COLOR_IMAGE_NDIMS or self.opencv_image.shape[-1] != RGB_CHANNELS:
            msg = "Backbuffer must be a 3-channel BGR image."
            raise ValueError(msg)
        x, y = point
        if not (0 <= x < self.opencv_image.shape[1] and 0 <= y < self.opencv_image.shape[0]):
            raise IndexError(point)
        b, g, r = self.opencv_image[y, x].tolist()
        return int(r), int(g), int(b)

    @check_valid_image
    def find_color(
        self: Self,
        color: Color,
        rect: Rect | None = None,
        tolerance: int = 0,
    ) -> list[Point]:
        """Find pixel coordinates matching a color within a tolerance.

        Args:
            color: Target colour expressed as ``(R, G, B)``.
            rect: Optional search region described as ``(x, y, width, height)``.
            tolerance: Allowed per-channel delta when searching for colour matches.

        Returns:
            Pixel coordinates in image space that match the colour constraint.
        """
        image = self._crop_image(rect)
        mask = filter_colors(image, color, tolerance)
        points = np.column_stack(np.where(mask == MAX_COLOR_VALUE)[::-1])
        if points.size == 0:
            return []
        if rect is not None:
            points = points + np.array(rect[0:2], dtype=points.dtype)
        return [(int(point[0]), int(point[1])) for point in points]

    @check_valid_image
    def get_average_color(self: Self, rect: Rect | None = None) -> Color:
        """Calculate the average color within a specified region.

        Args:
            rect: Region to average; ``None`` uses the full image.

        Returns:
            Average RGB value inside the requested region.
        """
        r, g, b = (int(x) for x in self._get_average_color(self.opencv_image, rect))
        return r, g, b

    def _get_average_color(
        self: Self,
        image: NDArrayUint8,
        rect: Rect | None = None,
    ) -> NDArrayInt16:
        """Calculate the average color of the specified image region.

        Args:
            image: Image to sample in BGR order.
            rect: Region to sample; defaults to the entire frame.

        Returns:
            Average colour in RGB channel order.
        """
        image = self._crop_image(rect, image)
        avg_color = cv.mean(image)
        avg_color_bgr: NDArrayInt16
        if isinstance(avg_color, float):
            avg_color_bgr = np.array([avg_color, avg_color, avg_color], dtype=np.int16)
        else:
            avg_color_bgr = np.array(avg_color[:3], dtype=np.int16)
        return avg_color_bgr[::-1]

    @staticmethod
    def _pack_bgr_frame(frame: NDArrayUint8) -> npt.NDArray[np.uint32]:
        """Pack a BGR frame into 24-bit integers for fast uniqueness queries."""
        pixels_bgr = frame.reshape(-1, RGB_CHANNELS)
        packed = (
            pixels_bgr[:, 0].astype(np.uint32)
            | (pixels_bgr[:, 1].astype(np.uint32) << 8)
            | (pixels_bgr[:, 2].astype(np.uint32) << 16)
        )
        return cast("npt.NDArray[np.uint32]", packed)

    @staticmethod
    def _unpack_packed_bgr(value: int) -> Color:
        """Unpack a 24-bit packed BGR integer into an RGB tuple."""
        b = value & 0xFF
        g = (value >> 8) & 0xFF
        r = (value >> 16) & 0xFF
        return int(r), int(g), int(b)

    @check_valid_image
    def get_most_common_color(
        self: Self,
        rect: Rect | None = None,
        index: int = 1,
        ignore_colors: Color | Sequence[Color] | None = None,
    ) -> Color:
        """Determines the most common color in the specified region.

        Args:
            rect: Region to sample; ``None`` uses the full image.
            index: Rank of the dominant colour to extract (1-based).
            ignore_colors: RGB colour(s) to skip while ranking.

        Returns:
            Most common RGB colour in the region.

        Raises:
            ValueError: If ``index`` is less than 1 or the requested region contains no pixels after filtering.
        """
        if index < 1:
            msg = "index must be >= 1."
            raise ValueError(msg)

        cropped_image = self._crop_image(rect)
        packed = self._pack_bgr_frame(cropped_image)

        if ignore_colors is not None:
            ignore_arr = np.asarray(ignore_colors, dtype=np.uint32)
            if ignore_arr.size:
                if ignore_arr.ndim == 1:
                    ignore_arr = ignore_arr[np.newaxis, :]
                if ignore_arr.shape[1] != RGB_CHANNELS:
                    msg = "ignore_colors must contain RGB triples."
                    raise ValueError(msg)
                ignore_packed = ignore_arr[:, 2] | (ignore_arr[:, 1] << 8) | (ignore_arr[:, 0] << 16)
                packed = packed[~np.isin(packed, ignore_packed)]

        if packed.size == 0:
            msg = "No pixels available in the requested region."
            raise ValueError(msg)

        unique, counts = np.unique(packed, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        desired_index = min(index - 1, len(sorted_indices) - 1)
        chosen = int(unique[sorted_indices[desired_index]])
        return self._unpack_packed_bgr(chosen)

    @check_valid_image
    def get_count_of_color(
        self: Self,
        color: Color,
        rect: Rect | None = None,
        tolerance: int | None = 0,
    ) -> int:
        """Counts the number of pixels matching a given color within a tolerance.

        Args:
            color: Target colour expressed as ``(R, G, B)``.
            rect: Region to sample; ``None`` uses the full image.
            tolerance: Allowed per-channel difference; ``None`` is treated as 0 for exact matches.

        Returns:
            Number of pixels matching the specified colour.
        """
        cropped_image = self._crop_image(rect)
        match_mask = filter_colors(cropped_image, color, tolerance or 0)
        return int(np.count_nonzero(match_mask))

    @check_valid_image
    def get_all_colors_with_counts(
        self: Self,
        rect: Rect | None = None,
    ) -> list[tuple[Color, int]]:
        """Retrieves all colors in the specified region along with their pixel counts.

        Args:
            rect: Region to sample; ``None`` uses the full image.

        Returns:
            Colour counts ordered by frequency.
        """
        cropped_image = self._crop_image(rect)
        packed = self._pack_bgr_frame(cropped_image)
        unique, counts = np.unique(packed, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]

        results: list[tuple[Color, int]] = []
        for packed_color, count in zip(unique[sorted_indices], counts[sorted_indices], strict=False):
            results.append((self._unpack_packed_bgr(int(packed_color)), int(count)))
        return results

    @check_valid_image
    def get_median_color(self: Self, rect: Rect | None = None) -> Color:
        """Calculate the median color of the specified region.

        Args:
            rect: Region to sample; ``None`` uses the full image.

        Returns:
            Median RGB colour inside the region.
        """
        cropped_image = self._crop_image(rect)
        reshaped_image = cropped_image.reshape(-1, RGB_CHANNELS)
        median_color = np.median(reshaped_image, axis=0).astype(np.uint8)
        b, g, r = median_color.tolist()
        return int(r), int(g), int(b)

    @staticmethod
    def _get_dominant_color(image: NDArrayUint8) -> NDArrayUint8:
        """Return the median colour (a proxy for a dominant colour) in the given image.

        Args:
            image: Input frame in BGR order.

        Returns:
            Median colour in BGR order.
        """
        reshaped_image = image.reshape(-1, RGB_CHANNELS)
        return cast("NDArrayUint8", np.median(reshaped_image, axis=0).astype(np.uint8))

    @check_valid_image
    def maximize_color_match(
        self: Self,
        rect: Rect,
        initial_tolerance: int = 100,
        tolerance_step: int = 1,
    ) -> tuple[Color, int]:
        """Finds the color and tolerance that best match the region's dominant color.

        Args:
            rect: Region to evaluate when computing the dominant colour.
            initial_tolerance: Initial tolerance applied when searching.
            tolerance_step: Amount to decrease tolerance when narrowing the search.

        Returns:
            Matched RGB colour and the tolerance applied.
        """
        if initial_tolerance < 0:
            msg = "initial_tolerance must be non-negative."
            raise ValueError(msg)
        if tolerance_step <= 0:
            msg = "tolerance_step must be >= 1."
            raise ValueError(msg)

        cropped_image = self._crop_image(rect)
        b, g, r = self._get_dominant_color(cropped_image).tolist()
        dominant_color = (int(b), int(g), int(r))
        best_color, best_tolerance = self._find_best_color_match(
            cropped_image, dominant_color, initial_tolerance, tolerance_step
        )
        return best_color, best_tolerance

    def _find_best_color_match(
        self: Self,
        cropped_image: NDArrayUint8,
        dominant_color: tuple[int, int, int],
        initial_tolerance: int,
        tolerance_step: int,
    ) -> tuple[Color, int]:
        """Searches for the best color match within the specified tolerance range.

        Args:
            cropped_image: Image region under evaluation (BGR).
            dominant_color: Dominant colour used for comparison (BGR).
            initial_tolerance: Starting tolerance before decrements.
            tolerance_step: Amount to reduce the tolerance between attempts.

        Returns:
            Matched RGB colour with the tolerance applied.
        """
        if cropped_image.size == 0:
            msg = "cropped_image must not be empty."
            raise ValueError(msg)
        if initial_tolerance < 0:
            msg = "initial_tolerance must be non-negative."
            raise ValueError(msg)
        if tolerance_step <= 0:
            msg = "tolerance_step must be >= 1."
            raise ValueError(msg)

        tolerance = initial_tolerance
        best_tolerance = 0
        best_ratio = -1.0
        inner_total_pixels = cropped_image.size // RGB_CHANNELS
        outer_total_pixels = (self.opencv_image.size // RGB_CHANNELS) - inner_total_pixels

        while tolerance >= 0:
            lower_bound, upper_bound = self._get_color_bounds(dominant_color, tolerance)
            pixel_count, outside_pixel_count = self._get_pixel_counts(cropped_image, lower_bound, upper_bound)
            inner_ratio = pixel_count / inner_total_pixels
            outer_ratio = (outside_pixel_count / outer_total_pixels) if outer_total_pixels else 0.0
            ratio = inner_ratio / (outer_ratio + 1.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_tolerance = tolerance
            tolerance -= tolerance_step

        best_color_rgb = (dominant_color[2], dominant_color[1], dominant_color[0])  # Convert BGR to RGB.
        return best_color_rgb, best_tolerance

    @staticmethod
    def _get_color_bounds(dominant_color: tuple[int, int, int], tolerance: int) -> tuple[NDArrayUint8, NDArrayUint8]:
        """Calculates lower and upper bounds for a color given a tolerance.

        Args:
            dominant_color: Target colour used to refine matches (BGR).
            tolerance: Channel tolerance currently applied to the search.

        Returns:
            Lower and upper BGR bounds.
        """
        dominant = np.asarray(dominant_color, dtype=np.int16)
        lower_bound: NDArrayUint8 = np.clip(dominant - tolerance, 0, MAX_COLOR_VALUE).astype(np.uint8)
        upper_bound: NDArrayUint8 = np.clip(dominant + tolerance, 0, MAX_COLOR_VALUE).astype(np.uint8)
        return lower_bound, upper_bound

    def _get_pixel_counts(
        self: Self,
        cropped_image: NDArrayUint8,
        lower_bound: NDArrayUint8,
        upper_bound: NDArrayUint8,
    ) -> tuple[int, int]:
        """Counts pixels within the specified color bounds in the cropped and main images.

        Args:
            cropped_image: Image region extracted for analysis (BGR).
            lower_bound: Lower inclusive colour bound (BGR).
            upper_bound: Upper inclusive colour bound (BGR).

        Returns:
            tuple[int, int]: Pixel counts inside the region and outside it.
        """
        mask = cv.inRange(cropped_image, lower_bound, upper_bound)
        pixel_count = int(np.count_nonzero(mask))
        outside_mask = cv.inRange(self.opencv_image, lower_bound, upper_bound)
        outside_pixel_count = int(np.count_nonzero(outside_mask)) - pixel_count
        return pixel_count, outside_pixel_count

    @staticmethod
    def _calculate_median_difference(
        image1: NDArrayUint8,
        image2: NDArrayUint8,
        mask: NDArrayUint8 | None = None,
    ) -> int:
        """Calculate the median absolute difference between two images.

        Args:
            image1: The first image.
            image2: The second image.
            mask: Optional inclusion mask; pixels where ``mask`` is zero are ignored.

        Returns:
            int: Median colour difference, or ``-1`` when shapes mismatch or no pixels remain after masking.
        """
        if image1.shape != image2.shape:
            return -1

        image1_f = image1.astype(np.float32)
        image2_f = image2.astype(np.float32)

        if mask is not None:
            mask_expanded = mask.astype(bool)[..., None]
            image1_f = np.where(mask_expanded, image1_f, np.nan)
            image2_f = np.where(mask_expanded, image2_f, np.nan)

        diff = np.abs(image1_f - image2_f)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            median_diff = np.nanmedian(diff)
        if np.isnan(median_diff):
            return -1
        return int(median_diff)

    @check_valid_image
    def erode_image(self: Self, iterations: int = 1, kernel: NDArrayUint8 | None = None) -> None:
        """Applies morphological erosion to the backbuffer image.

        Args:
            iterations: Number of erosion passes to run. Defaults to 1.
            kernel: Structuring element to use; defaults to a 3x3 ones matrix.
        """
        if kernel is None:
            kernel = np.ones((_DEFAULT_MORPH_KERNEL_SIZE, _DEFAULT_MORPH_KERNEL_SIZE), np.uint8)
        self.opencv_image = cast(
            "NDArrayUint8",
            cv.erode(self.opencv_image, kernel, iterations=iterations),
        )

    @check_valid_image
    def dilate_image(self: Self, iterations: int = 1, kernel: NDArrayUint8 | None = None) -> None:
        """Applies morphological dilation to the backbuffer image.

        Args:
            iterations: Number of dilation passes to run. Defaults to 1.
            kernel: Structuring element to use; defaults to a 3x3 ones matrix.
        """
        if kernel is None:
            kernel = np.ones((_DEFAULT_MORPH_KERNEL_SIZE, _DEFAULT_MORPH_KERNEL_SIZE), np.uint8)
        self.opencv_image = cast(
            "NDArrayUint8",
            cv.dilate(self.opencv_image, kernel, iterations=iterations),
        )

    @check_valid_image
    def find_image(
        self: Self,
        sub_image: NDArrayUint8 | Image.Image,
        rect: Rect | None = None,
        confidence: float = 0.95,
        median_tolerance: int | None = None,
    ) -> list[Rect]:
        """Finds occurrences of a subimage within the main image using template matching.

        Args:
            sub_image: Template image in RGB/RGBA ordering.
            rect: Search region specified as ``(x, y, width, height)``. If ``None``, the entire image is used.
            confidence: Matching confidence threshold (default ``0.95``).
            median_tolerance: Optional per-channel median colour tolerance for matches.

        Returns:
            Bounding boxes locating the subimage.
        """
        image = self._crop_image(rect)
        sub_image_bgr, mask = self._prepare_sub_image(sub_image)
        main_image_gray, sub_image_gray = self._convert_to_grayscale(image, sub_image_bgr)
        res = self._perform_template_matching(main_image_gray, sub_image_gray, mask, confidence)
        rects = self._process_matching_results(res, image, sub_image_bgr, mask, rect, median_tolerance)
        return self._group_and_convert_to_shape_list(rects)

    @staticmethod
    def _prepare_sub_image(
        sub_image: NDArrayUint8 | Image.Image,
    ) -> tuple[NDArrayUint8, NDArrayUint8 | None]:
        """Prepares the subimage for template matching.

        Converts the subimage to BGR format and extracts the alpha channel as a mask if present.

        Args:
            sub_image: Template image in RGB/RGBA ordering.

        Returns:
            Prepared template and optional mask.

        Raises:
            ValueError: If ``sub_image`` does not have shape ``(H, W, 3)`` or ``(H, W, 4)``.
        """
        if isinstance(sub_image, Image.Image):
            sub_image_arr: NDArrayUint8 = np.array(sub_image.convert("RGBA"), dtype=np.uint8)
        else:
            sub_image_arr = sub_image

        if sub_image_arr.ndim != COLOR_IMAGE_NDIMS or sub_image_arr.shape[-1] not in (RGB_CHANNELS, RGBA_CHANNELS):
            msg = "sub_image must have shape (H, W, 3) or (H, W, 4)."
            raise ValueError(msg)

        if sub_image_arr.shape[-1] == RGBA_CHANNELS:
            sub_alpha = sub_image_arr[..., 3]
            sub_image_bgr = cv.cvtColor(sub_image_arr[..., :3], cv.COLOR_RGB2BGR)
            mask = sub_alpha
        else:
            sub_image_bgr = cv.cvtColor(sub_image_arr, cv.COLOR_RGB2BGR)
            mask = None
        return cast("NDArrayUint8", sub_image_bgr), mask

    @staticmethod
    def _convert_to_grayscale(
        main_image: NDArrayUint8, sub_image_bgr: NDArrayUint8
    ) -> tuple[NDArrayUint8, NDArrayUint8]:
        """Converts the main and subimages to grayscale.

        Args:
            main_image: Reference image to search within.
            sub_image_bgr: Template in BGR colour space.

        Returns:
            Grayscale main image and template.
        """
        main_image_gray = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
        sub_image_gray = cv.cvtColor(sub_image_bgr, cv.COLOR_BGR2GRAY)
        return cast("NDArrayUint8", main_image_gray), cast("NDArrayUint8", sub_image_gray)

    @staticmethod
    def _perform_template_matching(
        main_image_gray: NDArrayUint8,
        sub_image_gray: NDArrayUint8,
        mask: NDArrayUint8 | None,
        confidence: float,
    ) -> MaskArray:
        """Performs template matching between the main image and subimage.

        Args:
            main_image_gray: Main frame in grayscale.
            sub_image_gray: Template converted to grayscale for matching.
            mask: Optional template mask.
            confidence: Minimum required match confidence.

        Returns:
            MaskArray: Mask where template scores exceed the confidence threshold.
        """
        res = cv.matchTemplate(main_image_gray, sub_image_gray, cv.TM_CCORR_NORMED, mask=mask)
        mask_arr: MaskArray = np.logical_and(res >= confidence, np.logical_not(np.isinf(res))).astype(np.uint8)
        return mask_arr

    def _process_matching_results(
        self: Self,
        res: MaskArray,
        main_image: NDArrayUint8,
        sub_image_bgr: NDArrayUint8,
        mask: NDArrayUint8 | None,
        rect: Rect | None,
        median_tolerance: int | None,
    ) -> list[Rect]:
        """Processes template matching results to extract matching rectangles.

        Args:
            res: Binary mask produced by the template comparison.
            main_image: Reference image to search within (BGR).
            sub_image_bgr: Template in BGR colour space.
            mask: Optional template mask.
            rect: Region to constrain the template search.
            median_tolerance: Optional tolerance applied to colour medians.

        Returns:
            Bounding boxes for detected matches.
        """
        rects: list[Rect] = []
        template_height, template_width = sub_image_bgr.shape[:2]
        offset_x = rect[0] if rect is not None else 0
        offset_y = rect[1] if rect is not None else 0

        for y, x in np.column_stack(np.where(res)):
            y_i = int(y)
            x_i = int(x)
            main_image_region = main_image[y_i : y_i + template_height, x_i : x_i + template_width]
            found_rect = (x_i + offset_x, y_i + offset_y, template_width, template_height)
            if median_tolerance is not None:
                found_median_diff = self._calculate_median_difference(main_image_region, sub_image_bgr, mask)
                if found_median_diff < median_tolerance:
                    rects.append(found_rect)
            else:
                rects.append(found_rect)
        return rects

    @staticmethod
    def _group_and_convert_to_shape_list(
        rects: list[Rect],
    ) -> list[Rect]:
        """Group similar rectangles and return a consolidated list.

        Args:
            rects: Bounding boxes emitted by the matcher.

        Returns:
            Grouped rectangles merged by OpenCV's clustering.
        """
        if not rects:
            return []

        rects_list: list[list[int]] = []
        for rect in rects:
            x, y, w, h = rect
            expanded = [int(x), int(y), int(w), int(h)]
            rects_list.append(expanded)
            rects_list.append(expanded.copy())

        grouped_rects, _ = cv.groupRectangles(
            rects_list,
            groupThreshold=_GROUP_RECTANGLES_THRESHOLD,
            eps=_GROUP_RECTANGLES_EPS,
        )
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in grouped_rects]

    @check_valid_image
    def find_contours(
        self: Self,
        color: Color,
        rect: Rect | None = None,
        tolerance: int = 0,
        min_area: int = 10,
        vertices: int | None = None,
    ) -> list[Contour]:
        """Find contours in the backbuffer that match a color.

        Args:
            color: Target colour expressed as ``(R, G, B)``.
            rect: Search region specified as ``(x, y, width, height)``. If ``None``, the entire image is used.
            tolerance: Allowed deviation per colour channel.
            min_area: Minimum area in pixels squared for a contour to qualify.
            vertices: Required vertex count for returned contours.

        Returns:
            Contours matching the search criteria.
        """
        image = self._crop_image(rect)
        image = filter_colors(image, color, tolerance)
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if rect is not None:
            offset = np.array([rect[0], rect[1]], dtype=np.int32)
            contours = [c + offset for c in contours]
        contours = [c for c in contours if cv.contourArea(c) >= min_area]
        if vertices is not None:
            contours = [
                c
                for c in contours
                if vertices == len(cv.approxPolyDP(c, 0.01 * cv.arcLength(c, closed=True), closed=True))
            ]
        return cast("list[Contour]", contours)

    @check_valid_image
    def draw_points(
        self: Self,
        points: Sequence[Point],
        color: Color = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws points on the backbuffer image.

        Args:
            points: Coordinates to mark on the backbuffer.
            color: Drawing colour (RGB). Defaults to red.
        """
        if not points:
            return
        points_arr = np.asarray(points, dtype=np.int64)
        self.opencv_image[points_arr[:, 1], points_arr[:, 0]] = color[::-1]

    @check_valid_image
    def draw_contours(
        self: Self,
        contours: Contour | Sequence[Contour],
        color: Color = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws contours on the backbuffer image.

        Args:
            contours: Contour(s) as produced by OpenCV.
            color: Drawing colour (RGB). Defaults to red.
        """
        contours_to_draw = [contours] if isinstance(contours, np.ndarray) else list(contours)
        if not contours_to_draw:
            return
        cv.drawContours(self.opencv_image, contours_to_draw, -1, color[::-1], 2)

    @check_valid_image
    def draw_circle(
        self: Self,
        circle: tuple[int, int, int],
        color: Color = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a circle on the backbuffer image.

        Args:
            circle: Circle definition ``(x, y, radius)``.
            color: Drawing colour (RGB). Defaults to red.
        """
        x, y, r = circle
        cv.circle(self.opencv_image, (x, y), r, color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def draw_rectangle(
        self: Self,
        rect: Rect,
        color: Color = (MAX_COLOR_VALUE, 0, 0),
    ) -> None:
        """Draws a rectangle on the backbuffer image.

        Args:
            rect: Rectangle specified as ``(x, y, width, height)``.
            color: Drawing colour (RGB). Defaults to red.
        """
        x, y, w, h = rect
        cv.rectangle(self.opencv_image, (x, y), (x + w, y + h), color[::-1], 2, cv.LINE_4)

    @check_valid_image
    def filter_colors(
        self: Self,
        colors: Color | Sequence[Color],
        tolerance: int = 0,
        *,
        keep_original_colors: bool = False,
    ) -> None:
        """Filters the backbuffer image to retain only specified colors within a given tolerance.

        Args:
            colors: Colours to keep while filtering.
            tolerance: Per-channel tolerance threshold (0-255).
            keep_original_colors: When ``True``, retain source colours for matching pixels; otherwise replace the
                backbuffer with a binary mask.
        """
        filtered_image = filter_colors(self.opencv_image, colors, tolerance, keep_original_colors=keep_original_colors)
        self.opencv_image = filtered_image
