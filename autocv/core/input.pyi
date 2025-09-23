from .decorators import check_valid_hwnd
from .vision import Vision

__all__ = ["Input"]

class Input(Vision):
    def __init__(self, hwnd: int = -1) -> None: ...
    def get_last_moved_point(self) -> tuple[int, int]: ...
    @check_valid_hwnd
    def _get_topmost_hwnd(self) -> int: ...
    @check_valid_hwnd
    def move_mouse(
        self,
        x: int,
        y: int,
        *,
        human_like: bool = True,
        ghost_mouse: bool = True,
    ) -> None: ...
    @check_valid_hwnd
    def _move_mouse(
        self,
        x: int,
        y: int,
        *,
        ghost_mouse: bool = True,
    ) -> None: ...
    @check_valid_hwnd
    def click_mouse(
        self,
        button: int = 1,
        *,
        send_message: bool = False,
    ) -> None: ...
    @check_valid_hwnd
    def press_vk_key(self, vk_code: int) -> None: ...
    @check_valid_hwnd
    def release_vk_key(self, vk_code: int) -> None: ...
    @check_valid_hwnd
    def send_vk_key(self, vk_code: int) -> None: ...
    @check_valid_hwnd
    def send_keys(self, characters: str) -> None: ...
