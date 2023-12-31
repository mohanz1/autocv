from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt

from autocv.models import Contour, Point, Rectangle

__all__ = ("OrderBy", "ShapeList")

SLOTS_DATACLASS = dict(slots=True) if "slots" in dataclass.__kwdefaults__ else {}
T = TypeVar("T", Point, Rectangle, Contour)


class OrderBy(Enum):
    LEFT_TO_RIGHT = auto()
    RIGHT_TO_LEFT = auto()
    TOP_TO_BOTTOM = auto()
    BOTTOM_TO_TOP = auto()
    INNER_TO_OUTER = auto()
    OUTER_TO_INNER = auto()


_KEY_FUNCTIONS: dict[OrderBy, Callable[[Point, T], int | float]] = {
    OrderBy.LEFT_TO_RIGHT: lambda _, shape: shape.center().x,
    OrderBy.RIGHT_TO_LEFT: lambda _, shape: -shape.center().x,
    OrderBy.TOP_TO_BOTTOM: lambda _, shape: shape.center().y,
    OrderBy.BOTTOM_TO_TOP: lambda _, shape: -shape.center().y,
    OrderBy.INNER_TO_OUTER: lambda center_of_bitmap, shape: float(np.linalg.norm(center_of_bitmap - shape.center())),
    OrderBy.OUTER_TO_INNER: lambda center_of_bitmap, shape: -float(np.linalg.norm(center_of_bitmap - shape.center())),
}


@dataclass(**SLOTS_DATACLASS)
class ShapeList(Sequence[T]):
    """A class representing a list of shapes, including points, rectangles, and contours.

    Args:
    ----
        shape_cls: The class of the shapes in the list, where each shape can be a Point, Rectangle, or Contour instance.
        data: An ndarray representing the bitmap.
        center_of_bitmap: The center point of the bitmap.
    """

    shape_cls: type[T]
    data: npt.NDArray[np.uintp] | Sequence[npt.NDArray[np.uintp]]
    center_of_bitmap: Point = field(repr=False)

    def __len__(self) -> int:
        return len(self.data)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, int):
            return self.shape_cls.from_ndarray(self.data[index])

        return tuple(self.shape_cls.from_ndarray(self.data[i]) for i in range(*index.indices(len(self))))

    def order_by(self, by: OrderBy) -> "ShapeList[T]":
        """Order the shapes in the list according to the given algorithm. Orders in place.

        Args:
        ----
            by: An OrderBy enum specifying the algorithm to use.

        Returns:
        -------
            ShapeList(Sequence[T]): The sorted ShapeList.
        """
        if isinstance(self.data, np.ndarray):
            sorted_indices = np.argsort([_KEY_FUNCTIONS[by](self.center_of_bitmap, shape) for shape in self])
            self.data = self.data[sorted_indices]
            print("HERE")
        elif isinstance(self[0], Contour):
            self.data = [
                shape.data
                for shape in sorted(
                    self,
                    key=lambda shape: _KEY_FUNCTIONS[by](self.center_of_bitmap, shape),
                )
            ]

        return self
