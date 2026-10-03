"""Native Transparent Floating Overlay UI for Windows and Linux (PySide6).

Features:
- Transparent, frameless, always-on-top floating HUD
- Draggable and resizable
- Click-through mode toggle
- Auto-dismiss fade-out animation (avoids results lingering on screen)
- Vibrant speaker badges with unique color identities
- Integrated audio capture controller (Microphone + System Audio Loopback)
- Settings dialog for audio devices, target languages, LLM backend, and UI tuning
"""

import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional, Union

from PySide6.QtCore import (
    QByteArray,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
import html
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
    QTextDocument,
)
import queue
from websockets.sync.client import connect as ws_connect
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from diarizeflow.app.audio.capture import AudioCaptureStream
from diarizeflow.app.audio.devices import list_audio_devices
from diarizeflow.app.config import AppConfig, resolve_app_path


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
        import re
        nums = re.findall(r"\d+", speaker_label)
        if nums:
            idx = (int(nums[0]) - 1) % len(SPEAKER_COLORS)
            return SPEAKER_COLORS[idx]
    except Exception:
        pass
    return SPEAKER_COLORS[hash(speaker_label) % len(SPEAKER_COLORS)]


class SettingsDialog(QDialog):
    """Configuration dialog for audio devices, translation, and UI appearance."""

    settings_saved = Signal(AppConfig)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("DiarizeFlow - 即時字幕與音訊設定")
        self.resize(520, 520)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", "Segoe UI Emoji", system-ui, sans-serif;
            }
            QLabel {
                color: #94a3b8;
                font-size: 13px;
                font-weight: 500;
                padding: 2px 0;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
                min-height: 24px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #f8fafc;
                selection-background-color: #38bdf8;
                selection-color: #0f172a;
                padding: 4px;
            }
            QPushButton {
                background-color: #38bdf8;
                color: #0f172a;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 600;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #7dd3fc;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 13px;
                spacing: 8px;
                padding: 2px 0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Hardware Calibration Status Card
        calib_file = resolve_app_path("models/.calibrated")
        is_calibrated = getattr(self.config, "hardware_calibrated", False) or calib_file.exists()
        diar_prec = "FP16 (Tensor Core)" if getattr(self.config.diarization, "use_fp16", True) else "INT8/FP32"
        status_box = QLabel(
            f"⚡ <b>硬體加速與量化狀態</b>: "
            f"{'✓ 已完成硬體適配與量化校準 (秒開模式, ' + diar_prec + ')' if is_calibrated else '⚙️ 尚未執行自適應量化'}"
        )
        status_box.setStyleSheet("""
            QLabel {
                background-color: rgba(16, 185, 129, 0.12);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.35);
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 12px;
            }
        """)
        layout.addWidget(status_box)

        form = QFormLayout()
        form.setSpacing(12)

        # 1. Microphone device
        self.mic_combo = QComboBox()
        self.loopback_combo = QComboBox()
        self._populate_audio_devices()
        form.addRow("麥克風裝置:", self.mic_combo)
        form.addRow("系統音訊 (遊戲/瀏覽器):", self.loopback_combo)

        # Audio gain control
        self.agc_check = QCheckBox("啟用輸入音量智慧調節 (AGC)")
        self.agc_check.setChecked(getattr(self.config.audio, "agc_enabled", True))
        self.agc_check.setToolTip("自動放大過小聲音、壓限過大聲音，確保 ASR 語音特徵處於最佳動態範圍")
        form.addRow("音量動態調節:", self.agc_check)

        # Dual-track audio routing mode
        self.routing_combo = QComboBox()
        self.routing_combo.addItem("智慧活動動態分流 (抗迴音/無失真, 推薦)", "smart")
        self.routing_combo.addItem("傳統相加混音 (Legacy Additive Mix)", "mix")
        self.routing_combo.addItem("僅收本機麥克風", "mic_only")
        self.routing_combo.addItem("僅收電腦系統聲音", "loopback_only")
        cur_routing = getattr(self.config.audio, "routing_mode", "smart")
        idx = self.routing_combo.findData(cur_routing)
        if idx >= 0:
            self.routing_combo.setCurrentIndex(idx)
        form.addRow("雙軌音訊路由模式:", self.routing_combo)

        # 2. ASR Engine & Faster-Whisper Configuration
        self.asr_combo = QComboBox()
        self.asr_combo.addItem("SenseVoiceSmall (超低延遲 60~80ms)", "sensevoice")
        self.asr_combo.addItem("Faster-Whisper (高精準度 150~250ms)", "faster-whisper")
        cur_engine = getattr(self.config.asr, "engine", "sensevoice")
        self.asr_combo.setCurrentIndex(1 if "whisper" in cur_engine.lower() else 0)
        form.addRow("語音辨識 ASR 引擎:", self.asr_combo)

        self.whisper_model_edit = QLineEdit(getattr(self.config.asr, "whisper_model", "models/faster-whisper-large-v2"))
        form.addRow("Whisper 模型路徑:", self.whisper_model_edit)

        self.prec_combo = QComboBox()
        self.prec_combo.addItems(["float16", "int8", "float32", "fp8", "w4a16", "nvfp4", "mxfp4"])
        cur_prec = getattr(self.config.asr, "whisper_precision", "float16")
        self.prec_combo.setCurrentText(cur_prec)
        form.addRow("Whisper 量化精度:", self.prec_combo)

        # 3. Target language
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["繁體中文", "English", "日本語", "한국어", "簡體中文", "Français", "Deutsch", "Español"])
        self.lang_combo.setCurrentText(self.config.llm.target_language)
        form.addRow("翻譯目標語言:", self.lang_combo)

        # 3. LLM Provider
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["vllm", "llama.cpp", "openai", "claude", "ollama", "bypass"])
        self.provider_combo.setCurrentText(self.config.llm.provider)
        form.addRow("LLM 翻譯引擎:", self.provider_combo)

        # 4. LLM Base URL
        self.url_edit = QLineEdit(self.config.llm.base_url)
        form.addRow("API 端點 (Base URL):", self.url_edit)

        # 5. LLM Model Name
        self.model_edit = QLineEdit(self.config.llm.model_name)
        form.addRow("模型名稱 (Model):", self.model_edit)

        # 6. Auto-dismiss delay
        self.fade_spin = QDoubleSpinBox()
        self.fade_spin.setRange(1.0, 30.0)
        self.fade_spin.setSingleStep(0.5)
        self.fade_spin.setValue(self.config.ui.fade_out_seconds)
        self.fade_spin.setSuffix(" 秒後淡出消失")
        form.addRow("字幕保留時間:", self.fade_spin)

        # 7. Font size
        self.font_spin = QSpinBox()
        self.font_spin.setRange(14, 48)
        self.font_spin.setValue(self.config.ui.font_size)
        form.addRow("字幕字型大小:", self.font_spin)

        # 7b. Max cards queue limit
        self.max_cards_spin = QSpinBox()
        self.max_cards_spin.setRange(1, 10)
        self.max_cards_spin.setValue(getattr(self.config.ui, "max_cards", 3))
        self.max_cards_spin.setSuffix(" 則對話卡片")
        self.max_cards_spin.setToolTip("多卡片佇列上限：同時並存的講者發言卡片數量 (建議 2 ~ 4 則)")
        form.addRow("多卡片佇列上限:", self.max_cards_spin)

        # 8. Speaker Separation Mode & Threshold
        self.streaming_mode_combo = QComboBox()
        self.streaming_mode_combo.addItem("Low Latency (官方預設 1.04s, 換句即分)", "low_latency")
        self.streaming_mode_combo.addItem("Very Low Latency (超低延遲 0.64s)", "very_low_latency")
        self.streaming_mode_combo.addItem("Ultra-Low Latency (極限延遲 0.32s)", "ultra_low_latency")
        self.streaming_mode_combo.addItem("Offline (批次基準 30.4s)", "offline")
        cur_mode = getattr(self.config.diarization, "streaming_mode", "low_latency")
        idx = self.streaming_mode_combo.findData(cur_mode)
        if idx >= 0:
            self.streaming_mode_combo.setCurrentIndex(idx)
        else:
            self.streaming_mode_combo.setCurrentIndex(0)
        self.streaming_mode_combo.setToolTip("NVIDIA Nemotron-3 官方串流架構延遲設定：以 1.04 秒小區間滑動分析，多人對話輪替時精準分離講者")
        form.addRow("Nemotron 串流分離延遲:", self.streaming_mode_combo)

        self.spk_thresh_spin = QDoubleSpinBox()
        self.spk_thresh_spin.setRange(0.50, 0.98)
        self.spk_thresh_spin.setSingleStep(0.02)
        cur_thresh = getattr(self.config.diarization, "speaker_threshold", 0.82)
        self.spk_thresh_spin.setValue(cur_thresh if cur_thresh >= 0.5 else 0.82)
        self.spk_thresh_spin.setToolTip("聲線相似度門檻（越高越容易判定為新講者，建議 0.78 ~ 0.86）")
        form.addRow("語者分離敏感度:", self.spk_thresh_spin)

        # 9. VAD Voice Activity Sensitivity
        self.vad_thresh_spin = QDoubleSpinBox()
        self.vad_thresh_spin.setRange(0.001, 0.15)
        self.vad_thresh_spin.setSingleStep(0.002)
        self.vad_thresh_spin.setDecimals(4)
        self.vad_thresh_spin.setValue(self.config.vad.energy_threshold)
        self.vad_thresh_spin.setToolTip("語音活動檢測門檻（數值越小越靈敏，微弱聲音如 0.006 也能快速捕捉，建議 0.004 ~ 0.015）")
        form.addRow("VAD 人聲檢測門檻:", self.vad_thresh_spin)

        # 10. Max Speech Accumulation Cap (Duration)
        self.max_speech_spin = QDoubleSpinBox()
        self.max_speech_spin.setRange(2.0, 10.0)
        self.max_speech_spin.setSingleStep(0.5)
        self.max_speech_spin.setDecimals(1)
        self.max_speech_spin.setSuffix(" 秒")
        cur_max_s = getattr(self.config.vad, "max_speech_s", 3.5)
        self.max_speech_spin.setValue(cur_max_s if cur_max_s <= 10.0 else 3.5)
        self.max_speech_spin.setToolTip("單句最大音訊累積上限（長話持續不間斷時的最長切片時間，建議 3.0 ~ 4.5 秒，越短字幕出現越快）")
        form.addRow("單句音訊累積上限:", self.max_speech_spin)

        # 11. Silence Timeout
        self.silence_timeout_spin = QSpinBox()
        self.silence_timeout_spin.setRange(150, 800)
        self.silence_timeout_spin.setSingleStep(50)
        self.silence_timeout_spin.setSuffix(" 毫秒 (ms)")
        cur_silence = getattr(self.config.vad, "silence_timeout_ms", 300)
        self.silence_timeout_spin.setValue(cur_silence)
        self.silence_timeout_spin.setToolTip("靜音截斷換句間隔（一句話講完後停頓多少時間即判定句子結束，建議 250 ~ 350 ms）")
        form.addRow("斷句停頓間隔 (Silence):", self.silence_timeout_spin)

        # 10. UI Opacity (Transparency)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.15, 0.95)
        self.opacity_spin.setSingleStep(0.05)
        cur_op = getattr(self.config.ui, "opacity", 0.55)
        self.opacity_spin.setValue(cur_op if cur_op <= 0.85 else 0.55)
        self.opacity_spin.setToolTip("視窗半透明度（數值越小越透明，建議 0.35 ~ 0.60）")
        form.addRow("視窗透明度 (Opacity):", self.opacity_spin)

        # 11. Show original text
        self.show_orig_check = QCheckBox("同時顯示原始辨識語音")
        self.show_orig_check.setChecked(self.config.ui.show_original)
        form.addRow("", self.show_orig_check)

        # 10. View & Clear Logs
        log_box = QHBoxLayout()
        log_btn = QPushButton("📋 打開日誌")
        log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        log_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #38bdf8;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #0284c7; color: #ffffff; }
        """)
        log_btn.clicked.connect(self._open_log_file)
        log_box.addWidget(log_btn)

        clear_log_btn = QPushButton("🗑️ 清空日誌")
        clear_log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #f87171;
                border: 1px solid #ef4444;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #b91c1c; color: #ffffff; }
        """)
        clear_log_btn.clicked.connect(self._clear_log_file)
        log_box.addWidget(clear_log_btn)

        form.addRow("系統運作日誌:", log_box)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("background-color: #334155; color: #f8fafc;")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("儲存設定")
        save_btn.clicked.connect(self._save_and_apply)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _open_log_file(self):
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "diarizeflow.log"
        else:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / "diarizeflow.log"
        if not log_path.exists():
            try:
                log_path.touch()
            except Exception:
                pass
        try:
            if sys.platform == "win32":
                os.startfile(str(log_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as e:
            print(f"[!] 無法打開日誌: {e}")

    def _clear_log_file(self):
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "diarizeflow.log"
        else:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / "diarizeflow.log"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 日誌已由使用者清空\n")
            QMessageBox.information(self, "成功", "已成功清空 diarizeflow.log 日誌！")
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"清空日誌失敗: {e}")

    def _populate_audio_devices(self):
        mics, loopbacks = list_audio_devices()
        self.mic_combo.addItem("【未選取/關閉麥克風】", -1)
        for mic in mics:
            self.mic_combo.addItem(mic.display_name(), mic.id)

        self.loopback_combo.addItem("【未選取/關閉系統聲音】", -1)
        for loop in loopbacks:
            self.loopback_combo.addItem(loop.display_name(), loop.id)

        # Set current selection
        mic_dev = self.config.audio.mic_device
        if mic_dev is not None and mic_dev != -1 and mic_dev != "-1":
            idx = self.mic_combo.findData(mic_dev)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)
        else:
            self.mic_combo.setCurrentIndex(0)

        loop_dev = self.config.audio.loopback_device
        if loop_dev is not None and loop_dev != -1 and loop_dev != "-1":
            idx = self.loopback_combo.findData(loop_dev)
            if idx >= 0:
                self.loopback_combo.setCurrentIndex(idx)
            else:
                for i in range(self.loopback_combo.count()):
                    if str(loop_dev) in self.loopback_combo.itemText(i):
                        self.loopback_combo.setCurrentIndex(i)
                        break
        elif loopbacks:
            # Default to first loopback if available
            self.loopback_combo.setCurrentIndex(1)

    def _save_and_apply(self):
        self.config.audio.mic_device = self.mic_combo.currentData()
        self.config.audio.loopback_device = self.loopback_combo.currentData()
        self.config.audio.agc_enabled = self.agc_check.isChecked()
        self.config.audio.routing_mode = self.routing_combo.currentData()
        self.config.asr.engine = self.asr_combo.currentData()
        self.config.asr.whisper_model = self.whisper_model_edit.text().strip()
        self.config.asr.whisper_precision = self.prec_combo.currentText()
        self.config.llm.target_language = self.lang_combo.currentText()
        self.config.llm.provider = self.provider_combo.currentText()
        self.config.llm.base_url = self.url_edit.text().strip()
        self.config.llm.model_name = self.model_edit.text().strip()
        self.config.ui.fade_out_seconds = float(self.fade_spin.value())
        self.config.ui.font_size = int(self.font_spin.value())
        self.config.ui.max_cards = int(self.max_cards_spin.value())
        self.config.ui.opacity = float(self.opacity_spin.value())
        self.config.ui.show_original = self.show_orig_check.isChecked()
        self.config.diarization.streaming_mode = self.streaming_mode_combo.currentData()
        self.config.diarization.speaker_threshold = float(self.spk_thresh_spin.value())
        self.config.vad.energy_threshold = float(self.vad_thresh_spin.value())
        self.config.vad.max_speech_s = float(self.max_speech_spin.value())
        self.config.vad.silence_timeout_ms = int(self.silence_timeout_spin.value())

        self.config.save()
        self.settings_saved.emit(self.config)
        self.accept()


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


class TransparentSubtitleOverlay(QWidget):
    """Floating transparent HUD for live translated subtitles."""

    connection_status_signal = Signal(str, str)
    audio_chunk_signal = Signal(bytes)
    audio_level_signal = Signal(float)
    subtitle_received_signal = Signal(object)
    config_synced_signal = Signal(object)

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        pipeline=None,
        parent=None,
        enable_network: bool = True,
        auto_start_capture: bool = True,
    ):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.pipeline = pipeline
        self.enable_network = enable_network
        self.auto_start_capture = auto_start_capture
        if self.pipeline and not self.pipeline.is_running:
            print("[*] 正在啟動 DiarizeFlow Pipeline 背景處理核心...")
            self.pipeline.start()
        self.is_running = True
        self.audio_queue = queue.Queue(maxsize=100)
        self._current_speaker = "講者 1"
        self._current_translated_text = ""
        self._current_original_text = ""

        # Window attributes for floating transparent UI across Windows and Linux
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.drag_position = QPoint()
        self.is_dragging = False
        self.is_clickthrough = False
        self.anchor_bottom_y: Optional[int] = None
        self._cards: list[SubtitleCardWidget] = []
        self._header_speaker_label = QLabel("講者 1")
        self._header_speaker_label.hide()
        self._legacy_original_label = QLabel("")
        self._legacy_original_label.hide()

        # Audio capture stream
        self.audio_stream: Optional[AudioCaptureStream] = None
        self.is_capturing = False

        # Auto-dismiss timer (applied only to subtitle container, so control bar stays visible!)
        self.fade_timer = QTimer(self)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(self._start_fade_out)

        self._build_ui()
        self._init_system_tray()
        self._init_shortcuts()
        self._init_network()
        self._connect_signals()

        # Position at bottom-center of primary screen with flexible stable bounds
        self.setMinimumSize(480, 140)
        self.resize(self.config.ui.window_width, self.config.ui.window_height)
        self._center_at_bottom()

        # Initial welcome display (customized for first-run calibration or normal launch)
        if getattr(self.config, "_just_calibrated", False):
            gpu = getattr(self.config, "_calib_gpu_name", "NVIDIA GPU")
            prec = getattr(self.config, "_calib_precision", "FP16")
            self.show_subtitle(
                speaker="⚡ 首次啟動：硬體適配校準完成",
                original=f"檢測到運算硬體: {gpu} | 最優量化模式: {prec}",
                translated=f"🚀 首次啟動已自動配置 {prec} Tensor Core 加速！未來啟動將直接秒開。請對著麥克風說話測試...",
                confidence=1.0,
            )
        else:
            self.show_subtitle(
                speaker="DiarizeFlow",
                original="即時語音辨識與翻譯系統已啟動",
                translated="🎙️ 已自動啟動即時收音，請對著麥克風說話測試...",
                confidence=1.0,
            )

        # Auto-start capture immediately so user doesn't have to search for the button
        if self.auto_start_capture:
            QTimer.singleShot(300, self._start_capture)

    @property
    def cards(self) -> list[SubtitleCardWidget]:
        """Return list of active subtitle card widgets."""
        return list(self._cards)

    @property
    def active_card_count(self) -> int:
        """Return number of currently active cards in the queue."""
        return len(self._cards)

    @property
    def max_cards(self) -> int:
        """Return configured maximum number of cards."""
        return getattr(self.config.ui, "max_cards", 3)

    @property
    def translated_text(self) -> QLabel:
        """Return latest card's translated label, or idle label if no cards exist."""
        if self._cards:
            return self._cards[-1].translated_label
        return self.idle_label

    @property
    def speaker_label(self) -> QLabel:
        """Return latest card's speaker badge, or header speaker label if empty."""
        if self._cards:
            return self._cards[-1].badge
        return self._header_speaker_label

    @property
    def original_text(self) -> QLabel:
        """Return latest card's original label, or legacy label if empty."""
        if self._cards:
            return self._cards[-1].original_label
        return self._legacy_original_label

    def _update_card_style(self):
        """Update card glassmorphism style based on configured opacity."""
        op = getattr(self.config.ui, "opacity", 0.60)
        if op < 0.40:
            op = 0.60
        if hasattr(self, "card"):
            self.card.setStyleSheet(f"""
                #SubtitleCard {{
                    background-color: rgba(15, 23, 42, {op:.2f});
                    border: 1px solid rgba(255, 255, 255, 0.20);
                    border-radius: 14px;
                    font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", "Segoe UI Emoji", system-ui, sans-serif;
                }}
            """)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        # Translucent Card Frame with dynamic glassmorphism opacity
        # Note: Do NOT use QGraphicsDropShadowEffect here as it breaks DirectWrite alpha compositing on WA_TranslucentBackground
        self.card = QFrame(self)
        self.card.setObjectName("SubtitleCard")
        self._update_card_style()

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 12, 18, 14)
        card_layout.setSpacing(8)

        # 1. Header Bar: Brand + Speaker Badge + Dynamic VU Wave + Controls
        self.header_widget = QWidget(self.card)
        self.header_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # App Brand Title / Drag handle
        self.app_title = QLabel("🎙️ DiarizeFlow")
        self.app_title.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; padding: 2px 0;")
        header_layout.addWidget(self.app_title)

        # Header Speaker Tag (hidden fallback)
        self._header_speaker_label = QLabel("講者 1")
        self._header_speaker_label.setObjectName("SpeakerLabel")
        self._header_speaker_label.setStyleSheet("""
            #SpeakerLabel {
                background-color: rgba(56, 189, 248, 0.20);
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.40);
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 700;
                min-height: 22px;
                max-height: 22px;
            }
        """)
        self._header_speaker_label.hide()
        header_layout.addWidget(self._header_speaker_label)

        # Status / Dynamic Real-Time VU Meter Wave
        self.vu_indicator = QLabel("● 待機")
        self.vu_indicator.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600; padding: 2px 0;")
        header_layout.addWidget(self.vu_indicator)

        header_layout.addStretch()

        # Capture Toggle Button
        self.btn_capture = QPushButton("▶ 開始收音")
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #0f172a;
                border: none;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
                min-height: 20px;
            }
            QPushButton:hover { background-color: #34d399; }
        """)
        self.btn_capture.clicked.connect(self._toggle_capture)
        header_layout.addWidget(self.btn_capture)

        # Click-through toggle button
        self.btn_clickthrough = QPushButton("🛡️ 穿透")
        self.btn_clickthrough.setCheckable(True)
        self.btn_clickthrough.setToolTip("開啟/關閉滑鼠穿透（快捷鍵: Alt+Shift+H）")
        self.btn_clickthrough.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:checked {
                background-color: #38bdf8;
                color: #0f172a;
                font-weight: 700;
            }
        """)
        self.btn_clickthrough.toggled.connect(self._toggle_clickthrough)
        header_layout.addWidget(self.btn_clickthrough)

        # Settings Button
        self.btn_settings = QPushButton("⚙ 設定")
        self.btn_settings.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
                min-height: 20px;
            }
            QPushButton:hover { color: #f8fafc; background-color: #334155; }
        """)
        self.btn_settings.clicked.connect(self._open_settings)
        header_layout.addWidget(self.btn_settings)

        # Logs Button
        self.btn_logs = QPushButton("📋 日誌")
        self.btn_logs.setToolTip("查看即時除錯日誌 (diarizeflow.log)")
        self.btn_logs.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #0284c7;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
                min-height: 20px;
            }
            QPushButton:hover { background-color: #0369a1; color: #ffffff; }
        """)
        self.btn_logs.clicked.connect(self._open_log_file)
        header_layout.addWidget(self.btn_logs)

        # Close Button
        self.btn_close = QPushButton("✕")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #64748b;
                border: none;
                font-size: 13px;
                padding: 2px 6px;
                min-height: 20px;
            }
            QPushButton:hover { color: #ef4444; }
        """)
        self.btn_close.clicked.connect(self.close)
        header_layout.addWidget(self.btn_close)

        card_layout.addWidget(self.header_widget)

        # 2. Subtitle Content Container (Multi-Card Message Stack, Bottom-Anchored)
        self.subtitle_container = QWidget(self.card)
        self.subtitle_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cards_layout = QVBoxLayout(self.subtitle_container)
        self.cards_layout.setContentsMargins(0, 4, 0, 4)
        self.cards_layout.setSpacing(6)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        # Idle placeholder label
        self.idle_label = QLabel("● 待機中，等待說話聲音...")
        self.idle_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.idle_label.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 13px;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
                background: transparent;
                padding: 4px 2px;
            }
        """)
        self.cards_layout.addWidget(self.idle_label)
        self.idle_label.show()

        card_layout.addWidget(self.subtitle_container, 1)
        root_layout.addWidget(self.card)

    def _center_at_bottom(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = geo.height() - self.height() - 60
            self.move(x, y)
            self.anchor_bottom_y = y + self.height()

    def _create_app_icon(self) -> QIcon:
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

    def _init_system_tray(self):
        """Initialize system tray icon and context menu."""
        app = QApplication.instance()
        if app:
            app.setQuitOnLastWindowClosed(False)

        self.app_icon = self._create_app_icon()
        self.setWindowIcon(self.app_icon)

        self.tray_icon = QSystemTrayIcon(self.app_icon, self)
        self.tray_icon.setToolTip("DiarizeFlow - 即時語音辨識與翻譯")

        # Tray Context Menu
        self.tray_menu = QMenu(self)
        self.tray_menu.setStyleSheet("""
            QMenu {
                background-color: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
                font-family: "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans TC", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QMenu::item {
                padding: 6px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #1e293b;
                color: #38bdf8;
            }
            QMenu::separator {
                height: 1px;
                background-color: #334155;
                margin: 4px 8px;
            }
        """)

        # Action: Toggle Click-through
        self.action_tray_clickthrough = QAction("🛡️ 切換滑鼠穿透 (Alt+Shift+H)", self)
        self.action_tray_clickthrough.setCheckable(True)
        self.action_tray_clickthrough.setChecked(self.is_clickthrough)
        self.action_tray_clickthrough.triggered.connect(lambda: self._toggle_clickthrough())
        self.tray_menu.addAction(self.action_tray_clickthrough)

        # Action: Toggle Audio Capture
        self.action_tray_capture = QAction("▶ 開始收音", self)
        self.action_tray_capture.triggered.connect(self._toggle_capture)
        self.tray_menu.addAction(self.action_tray_capture)

        self.tray_menu.addSeparator()

        # Action: Settings
        self.action_tray_settings = QAction("⚙ 開啟設定視窗", self)
        self.action_tray_settings.triggered.connect(self._open_settings)
        self.tray_menu.addAction(self.action_tray_settings)

        # Action: Logs
        self.action_tray_logs = QAction("📋 查看除錯日誌", self)
        self.action_tray_logs.triggered.connect(self._open_log_file)
        self.tray_menu.addAction(self.action_tray_logs)

        self.tray_menu.addSeparator()

        # Action: Toggle Window Visibility
        self.action_tray_toggle_win = QAction("👁️ 顯示/隱藏 HUD 視窗", self)
        self.action_tray_toggle_win.triggered.connect(self._toggle_window_visibility)
        self.tray_menu.addAction(self.action_tray_toggle_win)

        self.tray_menu.addSeparator()

        # Action: Quit
        self.action_tray_quit = QAction("✕ 退出 DiarizeFlow", self)
        self.action_tray_quit.triggered.connect(self.close)
        self.tray_menu.addAction(self.action_tray_quit)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)

        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon.show()

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts for HUD interaction."""
        self.shortcut_h = QShortcut(QKeySequence("Alt+Shift+H"), self)
        self.shortcut_h.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_h.activated.connect(lambda: self._toggle_clickthrough())

        self.shortcut_t = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
        self.shortcut_t.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_t.activated.connect(lambda: self._toggle_clickthrough())

        if sys.platform == "win32":
            self._register_windows_global_hotkeys()

    def _register_windows_global_hotkeys(self):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = int(self.winId())
            MOD_ALT = 0x0001
            MOD_CONTROL = 0x0002
            MOD_SHIFT = 0x0004
            MOD_NOREPEAT = 0x4000
            user32.RegisterHotKey(hwnd, 0xDF01, MOD_ALT | MOD_SHIFT | MOD_NOREPEAT, 0x48)  # Alt+Shift+H
            user32.RegisterHotKey(hwnd, 0xDF02, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, 0x54)  # Ctrl+Shift+T
            self._registered_win32_hotkeys = True
        except Exception as e:
            print(f"[!] 無法註冊 Windows 全域快捷鍵: {e}")
            self._registered_win32_hotkeys = False

    def _unregister_windows_global_hotkeys(self):
        if sys.platform == "win32" and getattr(self, "_registered_win32_hotkeys", False):
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = int(self.winId())
                user32.UnregisterHotKey(hwnd, 0xDF01)
                user32.UnregisterHotKey(hwnd, 0xDF02)
                self._registered_win32_hotkeys = False
            except Exception:
                pass

    def _toggle_window_visibility(self):
        """Toggle HUD window between visible and hidden."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self._toggle_window_visibility()

    def _init_network(self):
        """Initialize background network listeners or direct pipeline integration."""
        if not getattr(self, "enable_network", True):
            return

        if self.pipeline:
            # Direct in-process pipeline: connect callbacks directly without network overhead
            self.pipeline.on_subtitle(lambda evt: self.subtitle_received_signal.emit(evt))
            self.connection_status_signal.emit("● 本地直連", "#10b981")
            return

        # Standalone frontend mode: connect via standard WebSocket bridge
        self.sub_thread = threading.Thread(target=self._subtitles_worker, daemon=True)
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.sub_thread.start()
        self.audio_thread.start()

    def _subtitles_worker(self):
        """Background worker listening for translated subtitle events."""
        host = getattr(self.config.server, "host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = self.config.server.port
        url = f"ws://{host}:{port}/ws/subtitles"

        while self.is_running:
            try:
                with ws_connect(url, open_timeout=2.0) as ws:
                    self.connection_status_signal.emit("● 已連線", "#38bdf8")
                    # Fetch and align remote backend configuration upon connection
                    self._fetch_remote_backend_config()
                    while self.is_running:
                        try:
                            msg = ws.recv(timeout=1.0)
                            data = json.loads(msg)
                            if data.get("type") != "connection":
                                self.subtitle_received_signal.emit(data)
                        except TimeoutError:
                            continue
            except Exception:
                if self.is_running:
                    self.connection_status_signal.emit("● 等待後端", "#64748b")
                    time.sleep(1.5)

    def _audio_worker(self):
        """Background worker streaming audio chunks to backend."""
        host = getattr(self.config.server, "host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = self.config.server.port
        url = f"ws://{host}:{port}/ws/audio"

        while self.is_running:
            try:
                with ws_connect(url, open_timeout=2.0) as ws:
                    while self.is_running:
                        try:
                            data = self.audio_queue.get(timeout=0.5)
                            ws.send(data)
                        except queue.Empty:
                            continue
            except Exception:
                if self.is_running:
                    time.sleep(1.5)

    def _connect_signals(self):
        self.audio_chunk_signal.connect(self._send_audio_bytes)
        self.audio_level_signal.connect(self._update_audio_level)
        self.subtitle_received_signal.connect(self._handle_subtitle_event)
        self.connection_status_signal.connect(self._update_status_indicator)
        self.config_synced_signal.connect(self._on_remote_config_synced)

    @Slot(str, str)
    def _update_status_indicator(self, text: str, color: str):
        if not self.is_capturing:
            self.vu_indicator.setText(text)
            self.vu_indicator.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")

    @Slot(float)
    def _update_audio_level(self, rms: float):
        """Real-time sound wave animation on the HUD header."""
        if not self.is_capturing:
            return
        if rms > 0.04:
            self.vu_indicator.setText("● 收音中 ▂▄▆█")
            self.vu_indicator.setStyleSheet("color: #22c55e; font-size: 11px; font-weight: 700;")
        elif rms > 0.015:
            self.vu_indicator.setText("● 收音中 ▂▄▆")
            self.vu_indicator.setStyleSheet("color: #10b981; font-size: 11px; font-weight: 700;")
        elif rms > 0.005:
            self.vu_indicator.setText("● 收音中 ▂▄")
            self.vu_indicator.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 600;")
        elif rms > 0.001:
            self.vu_indicator.setText("● 監聽中 ▂")
            self.vu_indicator.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        else:
            self.vu_indicator.setText("● 監聽中 ·")
            self.vu_indicator.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")

    @Slot(bytes)
    def _send_audio_bytes(self, data: bytes):
        try:
            self.audio_queue.put_nowait(data)
        except queue.Full:
            pass

    @Slot(object)
    def _handle_subtitle_event(self, event):
        if not event:
            return
        if hasattr(event, "to_dict"):
            event = event.to_dict()
        elif not isinstance(event, dict):
            return

        evt_type = event.get("type")
        if evt_type == "speaker_renamed":
            self._handle_speaker_renamed_event(event)
            return
        elif evt_type == "speaker_deleted":
            self._handle_speaker_deleted_event(event)
            return
        elif evt_type in ("connection", "ping", "pong"):
            return
        elif evt_type is not None and evt_type != "subtitle":
            print(f"[UI] 忽略非字幕控制事件: {evt_type}")
            return

        orig = event.get("original_text", "")
        trans = event.get("translated_text", "")
        if not (trans and trans.strip()) and not (orig and orig.strip()):
            return

        speaker = event.get("speaker", "講者 1")
        conf = event.get("confidence", 1.0)
        print(f"[UI] 收到即時字幕: [{speaker}] 譯文='{trans}'")
        latency = event.get("latency", {})
        if latency:
            asr = latency.get("asr_ms", 0)
            diar = latency.get("diar_ms", 0)
            llm = latency.get("trans_ms", 0)
            tot = latency.get("total_ms", 0)
            dur = latency.get("duration_s", 0)
            asr_eng = getattr(self.config.asr, "engine", "sensevoice")
            asr_prec = getattr(self.config.asr, "whisper_precision", "FP16").upper() if "whisper" in asr_eng else ("FP16" if getattr(self.config.diarization, "use_fp16", True) else "INT8")
            diar_prec = "FP16 (Tensor Core)" if getattr(self.config.diarization, "use_fp16", True) else "INT8/FP32"
            stream_mode = getattr(self.config.diarization, "streaming_mode", "low_latency")
            tip = (
                f"⏱️ 即時延遲與硬體精度診斷:\n"
                f"- 音訊段長: {dur}s\n"
                f"- ASR語音辨識 [{asr_prec}]: {asr:.0f}ms\n"
                f"- 語者分離 [{diar_prec} | {stream_mode}]: {diar:.0f}ms\n"
                f"- LLM模型翻譯: {llm:.0f}ms\n"
                f"- 總處理耗時: {tot:.0f}ms"
            )
            self.vu_indicator.setToolTip(tip)
            if self.is_capturing:
                self.vu_indicator.setText(f"● 延遲 {int(tot)}ms")
        self.show_subtitle(speaker, orig, trans, conf)

    def _handle_speaker_renamed_event(self, event: dict):
        """Handle speaker rename event by updating active cards and header badge."""
        old_spk = event.get("old_speaker")
        new_spk = event.get("new_speaker")
        col = event.get("color")
        if old_spk and new_spk:
            for c in self._cards:
                if c.speaker == old_spk:
                    c.update_speaker(new_spk, col)
            if self._current_speaker == old_spk:
                self._current_speaker = new_spk
            if self._header_speaker_label.text() == old_spk:
                self._header_speaker_label.setText(new_spk)
            print(f"[*] [HUD] 講者「{old_spk}」已重新命名為「{new_spk}」")

    def _handle_speaker_deleted_event(self, event: dict):
        """Handle speaker profile deletion event by updating active cards and tracking."""
        spk_id = event.get("speaker_id") or event.get("speaker") or event.get("name")
        if not spk_id:
            return

        for card in self._cards:
            if card.speaker == spk_id:
                card.update_speaker("未知講者")

        if self._current_speaker == spk_id:
            self._current_speaker = "未知講者"
        if self._header_speaker_label.text() == spk_id:
            self._header_speaker_label.setText("未知講者")

        print(f"[*] [HUD] 收到講者刪除事件: 講者「{spk_id}」已被移除，活躍卡片已標記為未知講者")

    def _format_translated_text(self, speaker: str, text: str) -> str:
        """Format translated text with clean 'Speaker: text' prefix."""
        return f"{speaker}: {text}" if speaker else text

    def _format_original_text(self, speaker: str, text: str) -> str:
        """Format original text with clean 'Speaker: text' prefix."""
        return f"{speaker}: {text}" if speaker else text

    def _format_translated_html(self, speaker: str, text: str) -> str:
        return self._format_translated_text(speaker, text)

    def _format_original_html(self, speaker: str, text: str) -> str:
        return self._format_original_text(speaker, text)

    def _maintain_bottom_anchor(self):
        """Keep the window bottom edge strictly fixed to anchor_bottom_y."""
        if self.anchor_bottom_y is not None and not self.is_dragging:
            expected_y = self.anchor_bottom_y - self.height()
            if self.y() != expected_y:
                self.move(self.x(), expected_y)

    def _update_window_height(self):
        """Keep stable window height and maintain bottom-anchoring to prevent window jumping."""
        self._maintain_bottom_anchor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._maintain_bottom_anchor()

    def show_subtitle(self, speaker: str, original: str, translated: str, confidence: float = 1.0):
        """Display translated subtitle in the bottom-anchored multi-card queue."""
        # Empty text defense: ignore and do not render ghost empty cards
        if not (translated and translated.strip()) and not (original and original.strip()):
            return

        self.idle_label.hide()

        self._current_speaker = speaker
        self._current_translated_text = translated
        self._current_original_text = original

        card = SubtitleCardWidget(
            speaker=speaker,
            original=original,
            translated=translated,
            confidence=confidence,
            config=self.config,
            parent=self.subtitle_container,
        )
        card.dismissed.connect(self._on_card_dismissed)
        card.rename_requested.connect(self._on_rename_requested)

        self._cards.append(card)
        if getattr(self, "is_clickthrough", False):
            card.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.cards_layout.addWidget(card)
        card.show()

        # Enforce max_cards queue cap
        max_cards = getattr(self.config.ui, "max_cards", 3)
        while len(self._cards) > max_cards:
            oldest = self._cards.pop(0)
            oldest.dismiss_immediately()

        self.subtitle_container.show()
        self._maintain_bottom_anchor()
        self.card.update()
        self.update()

    def _on_card_dismissed(self, card: SubtitleCardWidget):
        """Handle individual card dismissal upon timeout or manual ejection."""
        if card in self._cards:
            self._cards.remove(card)
        self.cards_layout.removeWidget(card)
        card.deleteLater()

        if not self._cards:
            self.idle_label.show()

        self._maintain_bottom_anchor()
        self.card.update()
        self.update()

    def _on_rename_requested(self, speaker: str):
        """Prompt user with dialog to rename speaker and pin their voiceprint profile."""
        new_name, ok = QInputDialog.getText(
            self,
            "自訂講者名稱與聲紋釘選",
            f"將講者「{speaker}」重新命名為:",
            QLineEdit.EchoMode.Normal,
            speaker,
        )
        if ok and new_name and new_name.strip() and new_name.strip() != speaker:
            self.rename_speaker(speaker, new_name.strip())

    def rename_speaker(self, old_name: str, new_name: str, color: Optional[str] = None):
        """Rename speaker across active HUD cards, pipeline diarizer, and persistent storage."""
        # 1. Update all currently visible subtitle cards
        for card in self._cards:
            if card.speaker == old_name:
                card.update_speaker(new_name, color)

        # 2. Update current speaker tracking
        if self._current_speaker == old_name:
            self._current_speaker = new_name
        if self._header_speaker_label.text() == old_name:
            self._header_speaker_label.setText(new_name)

        # 3. Update pipeline diarizer if running in-process
        if self.pipeline and hasattr(self.pipeline, "rename_speaker"):
            self.pipeline.rename_speaker(old_name, new_name, color)
        elif getattr(self, "enable_network", True):
            # Remote background sync via REST API
            threading.Thread(
                target=self._send_rename_api,
                args=(old_name, new_name, color),
                daemon=True,
            ).start()

        print(f"[*] [HUD] 講者「{old_name}」已重新命名為「{new_name}」並已釘選至永久聲紋資料庫")

    def _send_rename_api(self, old_name: str, new_name: str, color: Optional[str] = None):
        try:
            import urllib.request
            import urllib.parse
            port = self.config.server.port
            url = f"http://127.0.0.1:{port}/api/speakers/{urllib.parse.quote(old_name)}/rename"
            payload = json.dumps({"name": new_name, "color": color}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except Exception as e:
            print(f"[!] 無法透過 REST API 同步講者重命名: {e}")

    def delete_speaker(self, speaker_id: str):
        """Delete speaker across active HUD cards, pipeline diarizer, and persistent storage."""
        self._handle_speaker_deleted_event({"speaker_id": speaker_id})
        if self.pipeline and hasattr(self.pipeline, "delete_speaker"):
            self.pipeline.delete_speaker(speaker_id)
        elif getattr(self, "enable_network", True):
            threading.Thread(
                target=self._send_delete_api,
                args=(speaker_id,),
                daemon=True,
            ).start()

    def _send_delete_api(self, speaker_id: str):
        try:
            import urllib.request
            import urllib.parse
            port = self.config.server.port
            url = f"http://127.0.0.1:{port}/api/speakers/{urllib.parse.quote(speaker_id)}"
            req = urllib.request.Request(
                url,
                headers={"Content-Type": "application/json"},
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except Exception as e:
            print(f"[!] 無法透過 REST API 同步講者刪除: {e}")

    def _start_fade_out(self):
        """Dismiss all active cards and display the idle placeholder."""
        self._current_translated_text = ""
        self._current_original_text = ""
        for card in list(self._cards):
            card.dismiss_immediately()
        self._cards.clear()
        self.idle_label.show()
        self._maintain_bottom_anchor()
        self.card.update()
        self.update()

    def _on_fade_finished(self):
        pass

    def _toggle_capture(self):
        if not self.is_capturing:
            self._start_capture()
        else:
            self._stop_capture()

    def _open_log_file(self):
        """Open diarizeflow.log in default text editor."""
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "diarizeflow.log"
        else:
            log_path = Path(__file__).resolve().parent.parent.parent.parent / "diarizeflow.log"
        if not log_path.exists():
            try:
                log_path.touch()
            except Exception:
                pass
        try:
            if sys.platform == "win32":
                os.startfile(str(log_path))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as e:
            print(f"[!] 無法打開日誌: {e}")

    def _start_capture(self):
        mic = self.config.audio.mic_device
        loop = self.config.audio.loopback_device

        mics, loopbacks = list_audio_devices()

        # If user explicitly disabled (marked as -1 or "-1"), keep as None
        use_mic = None if (mic is None or mic == -1 or mic == "-1") else mic
        use_loop = None if (loop is None or loop == -1 or loop == "-1") else loop

        # Initial cold start (both unconfigured): auto-detect available devices
        if mic is None and loop is None:
            try:
                import sounddevice as sd
                default_in = sd.default.device[0]
                if default_in is not None and default_in >= 0:
                    use_mic = int(default_in)
                    self.config.audio.mic_device = use_mic
            except Exception:
                pass

            if loopbacks:
                use_loop = loopbacks[0].id
                self.config.audio.loopback_device = use_loop

        print(f"[*] 啟動音訊監聽: 麥克風={use_mic}, 系統聲音(Loopback)={use_loop}")
        try:
            routing_mode = getattr(self.config.audio, "routing_mode", "smart")
            mic_thresh = getattr(self.config.audio, "mic_activity_threshold", 0.008)
            loop_thresh = getattr(self.config.audio, "loopback_activity_threshold", 0.008)
            bleed_supp = getattr(self.config.audio, "bleed_suppression", True)
            bleed_ratio = getattr(self.config.audio, "bleed_ratio", 0.40)

            self.audio_stream = AudioCaptureStream(
                target_sample_rate=self.config.audio.sample_rate,
                chunk_ms=self.config.audio.chunk_ms,
                gain=self.config.audio.gain,
                on_audio_chunk=self._on_audio_chunk,
                routing_mode=routing_mode,
                mic_activity_threshold=mic_thresh,
                loopback_activity_threshold=loop_thresh,
                bleed_suppression=bleed_supp,
                bleed_ratio=bleed_ratio,
            )
            self.audio_stream.start(mic_device=use_mic, loopback_device=use_loop)
            self.is_capturing = True
            self.btn_capture.setText("⏹ 停止收音")
            if hasattr(self, "action_tray_capture") and self.action_tray_capture:
                self.action_tray_capture.setText("⏹ 停止收音")
            self.btn_capture.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: #ffffff;
                    border: none;
                    border-radius: 5px;
                    padding: 4px 10px;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton:hover { background-color: #f87171; }
            """)
            self.vu_indicator.setText("● 監聽中 ·")
            self.vu_indicator.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        except Exception as e:
            print(f"[!] Error starting capture stream: {e}")
            self.is_capturing = False
            self.vu_indicator.setText("● 啟動失敗")
            self.vu_indicator.setStyleSheet("color: #ef4444; font-size: 11px; font-weight: 600;")

    def _stop_capture(self):
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream = None
        self.is_capturing = False
        self.btn_capture.setText("▶ 開始收音")
        if hasattr(self, "action_tray_capture") and self.action_tray_capture:
            self.action_tray_capture.setText("▶ 開始收音")
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #0f172a;
                border: none;
                border-radius: 5px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #34d399; }
        """)
        self.vu_indicator.setText("● 待機")
        self.vu_indicator.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")

    def _on_audio_chunk(self, chunk, rms: float):
        # Update real-time VU indicator
        self.audio_level_signal.emit(rms)
        # If direct in-process pipeline is available, pass audio directly for lowest latency
        if self.pipeline:
            self.pipeline.push_audio(chunk, rms)
        else:
            self.audio_chunk_signal.emit(chunk.tobytes())

    def _toggle_clickthrough(self, checked: Optional[bool] = None):
        """Toggle mouse click-through mode with multi-layer safety protections."""
        if checked is None:
            checked = not getattr(self, "is_clickthrough", False)

        self.is_clickthrough = bool(checked)

        # 1. Update UI button state without signal loops
        if hasattr(self, "btn_clickthrough") and self.btn_clickthrough:
            if self.btn_clickthrough.isChecked() != self.is_clickthrough:
                self.btn_clickthrough.blockSignals(True)
                self.btn_clickthrough.setChecked(self.is_clickthrough)
                self.btn_clickthrough.blockSignals(False)

            if self.is_clickthrough:
                self.btn_clickthrough.setText("🛡️ 穿透中")
                self.btn_clickthrough.setToolTip("滑鼠穿透已開啟（按 Alt+Shift+H 或托盤關閉）")
            else:
                self.btn_clickthrough.setText("🛡️ 穿透")
                self.btn_clickthrough.setToolTip("開啟/關閉滑鼠穿透（快捷鍵: Alt+Shift+H）")

        # 2. Update Tray Menu Action state
        if hasattr(self, "action_tray_clickthrough") and self.action_tray_clickthrough:
            self.action_tray_clickthrough.blockSignals(True)
            self.action_tray_clickthrough.setChecked(self.is_clickthrough)
            self.action_tray_clickthrough.blockSignals(False)

        # 3. Apply mouse transparency
        if sys.platform == "win32":
            # On Windows, keep window WA_TransparentForMouseEvents False so WM_NCHITTEST
            # is processed in nativeEvent, leaving header_widget interactive!
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        else:
            # On Linux/X11, apply transparent attribute; system tray & hotkey provide safety exit
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.is_clickthrough)

        # Subtitle cards & container
        if hasattr(self, "subtitle_container") and self.subtitle_container:
            self.subtitle_container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.is_clickthrough)
        if hasattr(self, "_cards"):
            for card in self._cards:
                card.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.is_clickthrough)
        if hasattr(self, "idle_label") and self.idle_label:
            self.idle_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.is_clickthrough)
        if hasattr(self, "header_widget") and self.header_widget:
            self.header_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # 4. Notify user via system tray message balloon if enabled
        if self.is_clickthrough and hasattr(self, "tray_icon") and self.tray_icon and QSystemTrayIcon.isSystemTrayAvailable():
            try:
                self.tray_icon.showMessage(
                    "DiarizeFlow 懸浮字幕",
                    "🛡️ 已開啟滑鼠穿透模式！\n隨時按 Alt+Shift+H 或右鍵點擊右下角系統托盤即可解除穿透。",
                    QSystemTrayIcon.MessageIcon.Information,
                    3500,
                )
            except Exception:
                pass

        print(f"[*] [HUD] 滑鼠穿透模式已{'開啟' if self.is_clickthrough else '關閉'} (快捷鍵: Alt+Shift+H / Ctrl+Shift+T)")

    def nativeEvent(self, eventType, message):
        if sys.platform == "win32":
            try:
                if eventType in (b"windows_generic_MSG", "windows_generic_MSG"):
                    import ctypes
                    import ctypes.wintypes
                    msg = ctypes.wintypes.MSG.from_address(int(message))
                    if msg.message == 0x0312:  # WM_HOTKEY
                        if msg.wParam in (0xDF01, 0xDF02):
                            self._toggle_clickthrough()
                            return True, 0
                    elif msg.message == 0x0084 and getattr(self, "is_clickthrough", False):  # WM_NCHITTEST
                        x = ctypes.c_short(msg.lParam & 0xFFFF).value
                        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                        pt = self.mapFromGlobal(QPoint(x, y))
                        if hasattr(self, "header_widget") and self.header_widget:
                            header_rect = self.header_widget.rect()
                            header_top_left = self.header_widget.mapTo(self, QPoint(0, 0))
                            header_in_self = header_rect.translated(header_top_left)
                            if not header_in_self.contains(pt):
                                # Point is outside header bar -> return HTTRANSPARENT (-1)
                                return True, -1
            except Exception:
                pass
        return super().nativeEvent(eventType, message)

    def _open_settings(self):
        dlg = SettingsDialog(self.config, self)
        dlg.settings_saved.connect(self._apply_updated_config)
        dlg.exec()

    def _apply_updated_config(self, new_cfg: AppConfig, sync_to_remote: bool = True):
        self.config = new_cfg
        if self.pipeline:
            self.pipeline.update_config(new_cfg)
        elif sync_to_remote and getattr(self, "enable_network", True):
            self._sync_config_to_remote_backend(new_cfg)

        for card in list(self._cards):
            card.update_config(new_cfg)

        max_cards = getattr(self.config.ui, "max_cards", 3)
        while len(self._cards) > max_cards:
            oldest = self._cards.pop(0)
            oldest.dismiss_immediately()

        self._update_card_style()
        if self.width() != new_cfg.ui.window_width or self.height() != new_cfg.ui.window_height:
            self.resize(new_cfg.ui.window_width, new_cfg.ui.window_height)
        self._maintain_bottom_anchor()

        # Restart capture if running
        if self.is_capturing:
            self._stop_capture()
            self._start_capture()

    def _sync_config_to_remote_backend(self, new_cfg: AppConfig):
        """Asynchronously push updated configuration to remote backend via POST /api/config."""
        def _send():
            try:
                import urllib.request
                host = getattr(self.config.server, "host", "127.0.0.1")
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                port = getattr(self.config.server, "port", 8000)
                url = f"http://{host}:{port}/api/config"
                payload = json.dumps(new_cfg.to_dict()).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        print("[✓] 前端設定已成功同步至後端伺服器 (POST /api/config)")
            except Exception as e:
                print(f"[!] 同步設定至後端伺服器失敗: {e}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()
        return t

    def _fetch_remote_backend_config(self):
        """Fetch remote backend configuration via GET /api/config and align frontend state."""
        def _fetch():
            try:
                import urllib.request
                host = getattr(self.config.server, "host", "127.0.0.1")
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                port = getattr(self.config.server, "port", 8000)
                url = f"http://{host}:{port}/api/config"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8")
                        remote_dict = json.loads(raw)
                        remote_cfg = AppConfig.from_dict(remote_dict)
                        print("[✓] 成功從後端伺服器拉取最新配置並對齊前端 (GET /api/config)")
                        self.config_synced_signal.emit(remote_cfg)
            except Exception as e:
                print(f"[!] 從後端拉取最新配置失敗: {e}")

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()
        return t

    @Slot(object)
    def _on_remote_config_synced(self, remote_cfg: AppConfig):
        """Apply remote backend configuration without echoing back to backend."""
        self._apply_updated_config(remote_cfg, sync_to_remote=False)

    # --- Mouse Drag Support for Floating HUD ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.anchor_bottom_y = self.y() + self.height()

    def closeEvent(self, event):
        self.is_running = False
        self._stop_capture()
        self._unregister_windows_global_hotkeys()
        if hasattr(self, "tray_icon") and self.tray_icon:
            self.tray_icon.hide()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        event.accept()
        if getattr(self, "_exit_on_close", False):
            app = QApplication.instance()
            if app:
                app.quit()
            import os
            import threading
            def _force_exit():
                time.sleep(0.15)
                os._exit(0)
            threading.Thread(target=_force_exit, daemon=True).start()


def run_overlay_app(config: Optional[AppConfig] = None, pipeline=None) -> int:
    """Launch PySide6 Desktop Floating Subtitle Application."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication.instance() or QApplication(sys.argv)

    # Wake up Python interpreter periodically so Ctrl-C cleanly exits without crashing
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(250)

    overlay = TransparentSubtitleOverlay(config, pipeline=pipeline)
    overlay._exit_on_close = True
    overlay.show()
    return app.exec()


def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="DiarizeFlow 桌面透明飄浮字幕前端")
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    cfg = AppConfig.load(args.config)
    sys.exit(run_overlay_app(cfg))


if __name__ == "__main__":
    run_cli()
