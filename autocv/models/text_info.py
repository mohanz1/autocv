"""This module defines the TextInfo data class for representing text information detected in an image.

It includes attributes for the text itself, its confidence level,and its bounding box coordinates as defined by the
Rectangle class.
"""

from __future__ import annotations

__all__ = ("TextInfo",)

from dataclasses import dataclass
from typing import Any

from autocv.models.rectangle import Rectangle

SLOTS_DATACLASS = {"slots": True} if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(frozen=True, **SLOTS_DATACLASS)
class TextInfo:
    """A dataclass representing information about text detected in an image.

    Inherits from Rectangle class to store the bounding box coordinates of the text.
    """

    text: str
    confidence: float
    rectangle: Rectangle

    @classmethod
    def from_row(cls: type[TextInfo], row: dict[str, Any]) -> TextInfo:
        """Creates a TextInfo object from a dictionary-like object.

        Args:
        ----
            row (Dict[str, Any]): A dictionary-like object with the following keys:
                              "left", "top", "width", "height", "confidence", "text".

        Returns:
        -------
            TextInfo: A TextInfo object with the specified attributes.
        """
        rectangle = Rectangle.from_row(row)
        return cls(text=row["text"], confidence=row["confidence"], rectangle=rectangle)
