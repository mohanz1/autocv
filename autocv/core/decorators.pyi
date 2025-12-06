from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")

def check_valid_hwnd(func: Callable[_P, _R]) -> Callable[_P, _R]: ...
def check_valid_image(func: Callable[_P, _R]) -> Callable[_P, _R]: ...
