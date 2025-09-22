"""Sequence filtering helpers used across AutoCV utilities."""

from __future__ import annotations

__all__ = ("find_first", "get_first")

from operator import attrgetter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

T = TypeVar("T")


def find_first(predicate: Callable[[T], bool], seq: Sequence[T]) -> T | None:
    """Return the first element that satisfies ``predicate``.

    Args:
        predicate (Callable[[T], bool]): Predicate evaluated against each item.
        seq (Sequence[T]): Sequence to scan in order.

    Returns:
        T | None: First matching element, or ``None`` if nothing matches.
    """
    for element in seq:
        if predicate(element):
            return element
    return None


def get_first(iterable: Iterable[T], **kwargs: object) -> T | None:
    """Return the first element whose attributes match ``kwargs``.

    Attribute lookups support Django-style ``__`` separators so ``foo__bar`` maps
    to ``foo.bar``.

    Args:
        iterable (Iterable[T]): Iterable of candidates to evaluate.
        **kwargs (object): Attribute/value pairs that must all match.

    Returns:
        T | None: First element satisfying every attribute constraint, otherwise ``None``.
    """
    if len(kwargs) == 1:
        key, value = next(iter(kwargs.items()))
        getter = attrgetter(key.replace("__", "."))
        for elem in iterable:
            if getter(elem) == value:
                return elem
        return None

    converters = [(attrgetter(attr.replace("__", ".")), value) for attr, value in kwargs.items()]

    for elem in iterable:
        if all(getter(elem) == expected for getter, expected in converters):
            return elem
    return None
