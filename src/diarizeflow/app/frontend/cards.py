"""Subtitle card and speaker badge components with independent lifecycles and animations."""

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from diarizeflow.app.config import AppConfig
from diarizeflow.app.frontend.widgets import get_speaker_color


class SpeakerBadge(QLabel):
    """Interactive speaker badge that triggers rename on click or double-click."""

    clicked = Signal()

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("點擊以自訂講者名稱並釘選聲紋")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class SubtitleCardWidget(QFrame):
    """Individual subtitle card representing a single speaker utterance with an independent lifecycle."""

    dismissed = Signal(object)  # Emits self when fade out finishes or dismissed
    rename_requested = Signal(str)  # Emits current speaker name on badge click

    def __init__(
        self,
        speaker: str,
        original: str,
        translated: str,
        confidence: float = 1.0,
        config: Optional[AppConfig] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.speaker = speaker
        self.original = original
        self.translated = translated
        self.confidence = confidence
        self.config = config or AppConfig()

        self.setObjectName("SubtitleCardItem")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._build_ui()
        self._init_timer()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(3)

        # Header: Interactive Speaker Badge
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        color = get_speaker_color(self.speaker)
        self.badge = SpeakerBadge(self.speaker)
        self.badge.setObjectName("SpeakerBadge")
        self.badge.setStyleSheet(f"""
            QLabel#SpeakerBadge {{
                background-color: {color}28;
                color: {color};
                border: 1px solid {color}88;
                border-radius: 4px;
                padding: 1px 7px;
                font-size: 11px;
                font-weight: 700;
                min-height: 18px;
                max-height: 18px;
            }}
        """)
        self.badge.clicked.connect(self._on_badge_clicked)
        header_layout.addWidget(self.badge)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Translated Text
        self.translated_label = QLabel(self.translated)
        self.translated_label.setWordWrap(True)
        self.translated_label.setTextFormat(Qt.TextFormat.PlainText)
        self.translated_label.setStyleSheet(f"""
            QLabel {{
                color: #ffffff;
                font-size: {self.config.ui.font_size}px;
                font-weight: 700;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
                background: transparent;
                padding: 0px;
            }}
        """)
        layout.addWidget(self.translated_label)

        # Original Text
        self.original_label = QLabel(self.original)
        self.original_label.setWordWrap(True)
        self.original_label.setTextFormat(Qt.TextFormat.PlainText)
        self.original_label.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                font-size: 12px;
                font-style: italic;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
                background: transparent;
                padding: 0px;
            }
        """)
        if self.config.ui.show_original and self.original:
            self.original_label.show()
        else:
            self.original_label.hide()
        layout.addWidget(self.original_label)

        self._update_style()

    def _update_style(self):
        self.setStyleSheet("""
            #SubtitleCardItem {
                background-color: rgba(30, 41, 59, 0.45);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
            }
        """)

    def _init_timer(self):
        self.fade_timer = QTimer(self)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(self.start_fade_out)
        fade_ms = max(500, int(self.config.ui.fade_out_seconds * 1000))
        self.fade_timer.start(fade_ms)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(350)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.anim.finished.connect(self._on_fade_finished)

    def start_fade_out(self):
        if self.anim.state() != QPropertyAnimation.State.Running:
            self.anim.start()

    def _on_fade_finished(self):
        self.dismissed.emit(self)

    def dismiss_immediately(self):
        if self.fade_timer.isActive():
            self.fade_timer.stop()
        if self.anim.state() == QPropertyAnimation.State.Running:
            self.anim.stop()
        self.dismissed.emit(self)

    def update_config(self, new_cfg: AppConfig):
        self.config = new_cfg
        self.translated_label.setStyleSheet(f"""
            QLabel {{
                color: #ffffff;
                font-size: {self.config.ui.font_size}px;
                font-weight: 700;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
                background: transparent;
                padding: 0px;
            }}
        """)
        if self.config.ui.show_original and self.original:
            self.original_label.show()
        else:
            self.original_label.hide()

    def _on_badge_clicked(self):
        self.rename_requested.emit(self.speaker)

    def update_speaker(self, new_speaker: str, color: Optional[str] = None):
        """Dynamically update card speaker label and badge styling."""
        self.speaker = new_speaker
        if color is None:
            color = get_speaker_color(new_speaker)
        self.badge.setText(new_speaker)
        self.badge.setStyleSheet(f"""
            QLabel#SpeakerBadge {{
                background-color: {color}28;
                color: {color};
                border: 1px solid {color}88;
                border-radius: 4px;
                padding: 1px 7px;
                font-size: 11px;
                font-weight: 700;
                min-height: 18px;
                max-height: 18px;
            }}
        """)


__all__ = [
    "SpeakerBadge",
    "SubtitleCardWidget",
]
