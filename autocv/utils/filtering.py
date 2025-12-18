"""Sequence filtering helpers.

AutoCV frequently needs to pick a single "best" item from collections (OCR hits,
template matches, shapes, etc.). This module provides lightweight helpers for
common selection patterns without introducing additional dependencies.
"""

from __future__ import annotations

__all__ = ("find_first", "get_first")

import functools
from collections.abc import Callable, Iterable, Sequence
from operator import attrgetter
from typing import Any, TypeAlias, TypeVar

T = TypeVar("T")

_Getter: TypeAlias = Callable[[Any], Any]


@functools.lru_cache(maxsize=128)
def _make_attr_getter(path: str) -> _Getter:
    """Return an attribute getter supporting Django-style separators."""
    return attrgetter(path.replace("__", "."))


def find_first(predicate: Callable[[T], bool], seq: Sequence[T]) -> T | None:
    """Return the first element that satisfies ``predicate``.

    Args:
        predicate: Predicate evaluated against each item.
        seq: Sequence to scan in order.

    Returns:
        The first matching element, or ``None`` if nothing matches.
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
        First element satisfying every attribute constraint, otherwise ``None``.
    """
    if not kwargs:
        return None

    if len(kwargs) == 1:
        key, expected = next(iter(kwargs.items()))
        getter = _make_attr_getter(key)
        for elem in iterable:
            if getter(elem) == expected:
                return elem
        return None

    matchers: list[tuple[_Getter, object]] = [(_make_attr_getter(attr), expected) for attr, expected in kwargs.items()]

    for elem in iterable:
        if all(getter(elem) == expected for getter, expected in matchers):
            return elem
    return None
