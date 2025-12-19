"""Shared constants.

The :mod:`autocv.constants` module centralizes small tuning values that are
used throughout the codebase (UI refresh cadence, pixel sampling parameters,
and input motion defaults).
"""

from __future__ import annotations

from typing import Final

__all__ = (
    "FALLBACK_COLOR",
    "MOTION_GRAVITY_JITTER",
    "MOTION_GRAVITY_MIN",
    "MOTION_SPEED_BASE",
    "MOTION_TIMEOUT_SECONDS",
    "MOTION_WIND_JITTER",
    "MOTION_WIND_MIN",
    "PIXEL_RADIUS",
    "PIXEL_ZOOM",
    "REFRESH_DELAY_MS",
    "VK_LEFT_BUTTON",
)

REFRESH_DELAY_MS: Final[int] = 5
PIXEL_ZOOM: Final[int] = 40
PIXEL_RADIUS: Final[int] = 1
FALLBACK_COLOR: Final[int] = 217

# Input motion defaults
MOTION_SPEED_BASE: Final[int] = 16
MOTION_GRAVITY_MIN: Final[float] = 8.0
MOTION_GRAVITY_JITTER: Final[float] = 0.5
MOTION_WIND_MIN: Final[float] = 4.0
MOTION_WIND_JITTER: Final[float] = 0.5
MOTION_TIMEOUT_SECONDS: Final[float] = 15.0

# Win32 key codes
VK_LEFT_BUTTON: Final[int] = 0x01
