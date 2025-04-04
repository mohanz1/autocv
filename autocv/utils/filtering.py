"""Utility functions for searching sequences and iterables.

This module provides helper functions to find the first matching element in sequences and iterables
based on a predicate or specific attribute criteria.
"""

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
        predicate: A function that takes an element of the sequence as input and returns a boolean
            indicating whether the element satisfies a certain condition.
        seq: A sequence of elements to search.

    Returns:
        The first element of the sequence that satisfies the predicate, or None if no such element is found.
    """
    for element in seq:
        if predicate(element):
            return element
    return None


def get_first(iterable: Iterable[T], **kwargs: str | float) -> T | None:
    """Find the first element in an iterable with attributes matching the specified criteria.

    The function accepts keyword arguments where each key is the name of an attribute (using "__" to
    denote nested attributes) and the corresponding value is the value to match. For example,
    a keyword argument of foo__bar=10 will match objects where the attribute "foo.bar" equals 10.

    Args:
        iterable: An iterable of elements to search.
        **kwargs: One or more attribute names and values to match against each element.
            The attribute names can use "__" to indicate nested attributes.

    Returns:
        The first element in the iterable that has all the specified attributes with matching values,
        or None if no such element is found.
    """
    # Special case for a single attribute to match
    if len(kwargs) == 1:
        key, value = next(iter(kwargs.items()))
        getter = attrgetter(key.replace("__", "."))
        for elem in iterable:
            if getter(elem) == value:
                return elem
        return None

    # Prepare a list of (getter, expected_value) tuples for each attribute criterion
    converters = [(attrgetter(attr.replace("__", ".")), value) for attr, value in kwargs.items()]

    for elem in iterable:
        if all(getter(elem) == expected for getter, expected in converters):
            return elem
    return None
