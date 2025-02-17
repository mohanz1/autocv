"""This module provides utility functions for searching sequences and iterables."""

from __future__ import annotations

__all__ = ("find_first", "get_first")

from operator import attrgetter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

T = TypeVar("T")


def find_first(predicate: Callable[[T], bool], seq: Sequence[T]) -> T | None:
    """Find the first element in a sequence that satisfies the given predicate.

    Args:
        predicate (Callable[[T], bool]): A function that takes an element of the sequence as input and returns a bool
            indicating whether the element satisfies some condition.
        seq (Sequence[T]): A sequence of elements to search.

    Returns:
        T | None: The first element of the sequence that satisfies the predicate, or None if no such element is found.
    """
    for element in seq:
        if predicate(element):
            return element
    return None


def get_first(iterable: Iterable[T], **kwargs: str | float) -> T | None:
    """Find the first element in an iterable that has the specified attribute(s) with the specified value(s).

    Args:
        iterable (Iterable[T]): An iterable of elements to search.
        **kwargs (str | float): One or more keyword arguments specifying the name and value of an attribute to match.
            The attribute name can use "__" to indicate nested attributes, e.g. "foo__bar" would match the "bar"
            attribute of an object in the "foo" attribute.

    Returns:
        T | None: The first element of the iterable that has the specified attribute(s) with the specified value(s), or
            None if no such element is found.
    """
    all_ = all
    attrget = attrgetter

    # Special case the single element call
    if len(kwargs) == 1:
        k, v = kwargs.popitem()
        pred = attrget(k.replace("__", "."))
        for elem in iterable:
            if pred(elem) == v:
                return elem
        return None

    converted = [(attrget(attr.replace("__", ".")), value) for attr, value in kwargs.items()]

    for elem in iterable:
        if all_(pred(elem) == value for pred, value in converted):
            return elem
    return None
