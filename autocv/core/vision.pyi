from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from .decorators import check_valid_hwnd, check_valid_image
from .window_capture import WindowCapture

__all__ = ["Vision"]

class Vision(WindowCapture):
    opencv_image: npt.NDArray[np.uint8]
    api: Any

    def __init__(self, hwnd: int = -1) -> None: ...
    def set_backbuffer(self, image: npt.NDArray[np.uint8] | Image.Image) -> None: ...
    @check_valid_hwnd
    def refresh(self, *, set_backbuffer: bool = True) -> npt.NDArray[np.uint8] | None: ...
    @check_valid_image
    def save_backbuffer_to_file(self, file_name: str) -> None: ...
    @check_valid_hwnd
    def get_pixel_change(self, area: tuple[int, int, int, int] | None = None) -> int: ...
    @check_valid_image
    def get_text(
        self,
        rect: tuple[int, int, int, int] | None = None,
        colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
        tolerance: int = 0,
        confidence: float | None = 0.8,
    ) -> list[dict[str, str | int | float | list[int]]]: ...
    @check_valid_image
    def get_color(self, point: tuple[int, int]) -> tuple[int, int, int]: ...
    @check_valid_image
    def find_color(
        self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
    ) -> list[tuple[int, int]]: ...
    @check_valid_image
    def get_average_color(self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]: ...
    @check_valid_image
    def get_most_common_color(
        self,
        rect: tuple[int, int, int, int] | None = None,
        index: int = 1,
        ignore_colors: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[int, int, int]: ...
    @check_valid_image
    def get_count_of_color(
        self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int | None = 0,
    ) -> int: ...
    @check_valid_image
    def get_all_colors_with_counts(
        self,
        rect: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[tuple[int, int, int], int]]: ...
    @check_valid_image
    def get_median_color(self, rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int]: ...
    @check_valid_image
    def maximize_color_match(
        self,
        rect: tuple[int, int, int, int],
        initial_tolerance: int = 100,
        tolerance_step: int = 1,
    ) -> tuple[tuple[int, int, int], int]: ...
    @check_valid_image
    def erode_image(
        self,
        iterations: int = 1,
        kernel: npt.NDArray[np.uint8] | None = None,
    ) -> None: ...
    @check_valid_image
    def dilate_image(
        self,
        iterations: int = 1,
        kernel: npt.NDArray[np.uint8] | None = None,
    ) -> None: ...
    @check_valid_image
    def find_image(
        self,
        sub_image: npt.NDArray[np.uint8] | Image.Image,
        rect: tuple[int, int, int, int] | None = None,
        confidence: float = 0.95,
        median_tolerance: int | None = None,
    ) -> list[tuple[int, int, int, int]]: ...
    @check_valid_image
    def find_contours(
        self,
        color: tuple[int, int, int],
        rect: tuple[int, int, int, int] | None = None,
        tolerance: int = 0,
        min_area: int = 10,
        vertices: int | None = None,
    ) -> list[npt.NDArray[np.uintp]]: ...
    @check_valid_image
    def draw_points(
        self,
        points: Sequence[tuple[int, int]],
        color: tuple[int, int, int] = ...,
    ) -> None: ...
    @check_valid_image
    def draw_contours(
        self,
        contours: tuple[tuple[tuple[tuple[int, int]]]],
        color: tuple[int, int, int] = ...,
    ) -> None: ...
    @check_valid_image
    def draw_circle(
        self,
        circle: tuple[int, int, int],
        color: tuple[int, int, int] = ...,
    ) -> None: ...
    @check_valid_image
    def draw_rectangle(
        self,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int] = ...,
    ) -> None: ...
    @check_valid_image
    def filter_colors(
        self,
        colors: tuple[int, int, int] | list[tuple[int, int, int]],
        tolerance: int = 0,
        *,
        keep_original_colors: bool = False,
    ) -> None: ...
