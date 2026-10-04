"""Reusable UI widgets and styling helpers for DiarizeFlow frontend."""

import re
from typing import Tuple

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)

SPEAKER_COLORS = [
    "#38bdf8",  # Speaker 1: Sky Blue
    "#c084fc",  # Speaker 2: Purple
    "#34d399",  # Speaker 3: Emerald
    "#f472b6",  # Speaker 4: Pink
    "#fbbf24",  # Speaker 5: Amber
    "#60a5fa",  # Speaker 6: Blue
    "#a78bfa",  # Speaker 7: Violet
    "#f87171",  # Speaker 8: Rose
]


def get_speaker_color(speaker_label: str) -> str:
    """Return a deterministic color for the given speaker name."""
    try:
        # Extract number if present
        nums = re.findall(r"\d+", speaker_label)
        if nums:
            idx = (int(nums[0]) - 1) % len(SPEAKER_COLORS)
            return SPEAKER_COLORS[idx]
    except Exception:
        pass
    return SPEAKER_COLORS[hash(speaker_label) % len(SPEAKER_COLORS)]


def create_app_icon() -> QIcon:
    """Create a clean vector application icon for HUD window and system tray."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Background rounded badge
    p.setBrush(QBrush(QColor("#0f172a")))
    p.setPen(QPen(QColor("#38bdf8"), 2))
    p.drawRoundedRect(QRectF(4, 4, 56, 56), 16, 16)

    # Microphone capsule
    p.setBrush(QBrush(QColor("#38bdf8")))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(26, 15, 12, 22), 6, 6)

    # Microphone arc / stand
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor("#38bdf8"), 2.5))
    p.drawArc(QRectF(20, 22, 24, 22), 0, -180 * 16)
    p.drawLine(32, 44, 32, 50)
    p.drawLine(24, 50, 40, 50)

    # Sound wave indicators
    p.setPen(QPen(QColor("#34d399"), 2))
    p.drawArc(QRectF(14, 18, 36, 30), 45 * 16, 90 * 16)
    p.drawArc(QRectF(14, 18, 36, 30), -135 * 16, 90 * 16)

    p.end()
    return QIcon(pix)


def format_vu_level(rms: float) -> Tuple[str, str]:
    """Return (status_text, color_hex) based on real-time RMS audio volume level."""
    if rms > 0.04:
        return ("● 收音中 ▂▄▆█", "#22c55e")
    elif rms > 0.015:
        return ("● 收音中 ▂▄▆", "#10b981")
    elif rms > 0.005:
        return ("● 收音中 ▂▄", "#34d399")
    elif rms > 0.001:
        return ("● 監聽中 ▂", "#38bdf8")
    else:
        return ("● 監聽中 ·", "#64748b")


__all__ = [
    "SPEAKER_COLORS",
    "get_speaker_color",
    "create_app_icon",
    "format_vu_level",
]
