"""Represents a list of shapes including points, rectangles, and contours.

This class provides functionality to manage a collection of shape objects. It allows for accessing individual shapes,
retrieving the length of the list, and ordering the shapes based on various criteria defined in the OrderBy enum.
"""

from __future__ import annotations

__all__ = ("OrderBy", "ShapeList")

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TypeVar, overload
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from autocv.models import Contour, Point, Rectangle

SLOTS_DATACLASS = {"slots": True} if "slots" in dataclass.__kwdefaults__ else {}
T = TypeVar("T", Point, Rectangle, Contour)


class OrderBy(Enum):
    """Defines constants for different ordering strategies for shapes.

    This Enum class provides different options for specifying the order in which shapes
    should be arranged or processed. It is typically used in sorting functions where the
    arrangement of shapes matters, such as graphical representations or spatial analysis.

    Attributes:
        LEFT_TO_RIGHT: Order shapes from left to right based on their x-coordinates.
        RIGHT_TO_LEFT: Order shapes from right to left based on their x-coordinates.
        TOP_TO_BOTTOM: Order shapes from top to bottom based on their y-coordinates.
        BOTTOM_TO_TOP: Order shapes from bottom to top based on their y-coordinates.
        INNER_TO_OUTER: Order shapes from innermost to outermost based on their distance to a center point.
        OUTER_TO_INNER: Order shapes from outermost to innermost based on their distance to a center point.
    """

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
    """Represents a list of shapes including points, rectangles, and contours.

    This class provides functionality to manage a collection of shape objects. It allows for accessing individual
    shapes, retrieving the length of the list, and ordering the shapes based on various criteria defined in the OrderBy
    enum.

    Attributes:
        shape_cls (type[T]): The class type of the shapes in the list, such as Point, Rectangle, or Contour.
        data (npt.NDArray[np.uintp] | Sequence[npt.NDArray[np.uintp]]): The shape data as an array-like structure.
            Can be an ndarray or a sequence of ndarrays.
        center_of_bitmap (Point): The center point of the bitmap, used for calculating distances in certain orderings.

    Methods:
        __len__() -> int: Returns the number of shapes in the list.
        __getitem__(index: int | slice) -> T | Sequence[T]: Retrieves a shape or a sequence of shapes from the list by
            index.
        order_by(by: OrderBy) -> Self: Orders the shapes in the list according to a specified criterion.
    """

    shape_cls: type[T]
    data: npt.NDArray[np.uintp] | Sequence[npt.NDArray[np.uintp]]
    center_of_bitmap: Point = field(repr=False)

    def __len__(self: Self) -> int:
        """Return the number of shapes in the list.

        This method provides the length of the list, enabling functions like len() to work on ShapeList objects.

        Returns:
            int: The number of shapes in the list.
        """
        return len(self.data)

    @overload
    def __getitem__(self: Self, index: int) -> T: ...

    @overload
    def __getitem__(self: Self, index: slice) -> Sequence[T]: ...

    def __getitem__(self: Self, index: int | slice) -> T | Sequence[T]:
        """Retrieve a shape or a sequence of shapes from the list by index.

        Args:
            index (int | slice): The index of the shape to retrieve or the slice of the list to retrieve.

        Returns:
            T | Sequence[T]: The requested shape or sequence of shapes.
        """
        if isinstance(index, int):
            return self.shape_cls.from_ndarray(self.data[index])

        return tuple(self.shape_cls.from_ndarray(self.data[i]) for i in range(*index.indices(len(self))))

    def order_by(self: Self, by: OrderBy) -> ShapeList[T]:
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
        elif isinstance(self[0], Contour):
            self.data = [
                shape.data
                for shape in sorted(
                    self,
                    key=lambda shape: _KEY_FUNCTIONS[by](self.center_of_bitmap, shape),
                )
            ]

        return self
