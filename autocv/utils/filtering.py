"""Sequence filtering helpers used across AutoCV utilities."""

from __future__ import annotations

__all__ = ("find_first", "get_first")

from operator import attrgetter
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

T = TypeVar("T")


def find_first(predicate: Callable[[T], bool], seq: Sequence[T]) -> T | None:
    """Return the first element that satisfies ``predicate``.

    Args:
        predicate: Predicate evaluated against each item.
        seq: Sequence to scan in order.

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
        iterable: Iterable of candidates to evaluate.
        **kwargs: Attribute/value pairs that must all match.

    Returns:
        T | None: First element satisfying every attribute constraint, otherwise ``None``.
    """
    if not kwargs:
        return None

    if len(kwargs) == 1:
        key, expected = next(iter(kwargs.items()))
        getter = attrgetter(key.replace("__", "."))
        for elem in iterable:
            if getter(elem) == expected:
                return elem
        return None

    matchers: list[tuple[Callable[[Any], Any], object]] = [
        (attrgetter(attr.replace("__", ".")), expected) for attr, expected in kwargs.items()
    ]

    for elem in iterable:
        if all(getter(elem) == expected for getter, expected in matchers):
            return elem
    return None
