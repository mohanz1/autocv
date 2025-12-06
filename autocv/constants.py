"""Shared constants for UI refresh, Win32 codes, and default tuning values."""

from __future__ import annotations

REFRESH_DELAY_MS: int = 5
PIXEL_ZOOM: int = 40
PIXEL_RADIUS: int = 1
FALLBACK_COLOR: int = 217

# Input motion defaults
MOTION_SPEED_BASE: int = 16
MOTION_GRAVITY_MIN: float = 8.0
MOTION_GRAVITY_JITTER: float = 0.5
MOTION_WIND_MIN: float = 4.0
MOTION_WIND_JITTER: float = 0.5
MOTION_TIMEOUT_SECONDS: float = 15.0

# Win32 key codes
VK_LEFT_BUTTON: int = 0x01
