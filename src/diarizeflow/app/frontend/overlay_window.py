"""Transparent, floating, bottom-anchored HUD subtitle window (PySide6)."""

import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional

from PySide6.QtCore import (
    QPoint,
    QRect,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from diarizeflow.app.audio.capture import AudioCaptureStream
from diarizeflow.app.audio.devices import list_audio_devices
from diarizeflow.app.audio.protocol import pack_audio_frame
from diarizeflow.app.config import AppConfig
from diarizeflow.app.frontend.cards import SubtitleCardWidget
from diarizeflow.app.frontend.network import (
    fetch_remote_backend_config,
    run_audio_client,
    run_subtitles_client,
    send_remote_delete_api,
    send_remote_rename_api,
    sync_config_to_remote_backend,
)
from diarizeflow.app.frontend.settings_dialog import SettingsDialog
from diarizeflow.app.frontend.widgets import create_app_icon, format_vu_level


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
        enable_persistence: bool = True,
    ):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.pipeline = pipeline
        self.enable_network = enable_network
        self.auto_start_capture = auto_start_capture
        self.enable_persistence = enable_persistence
        self._geometry_modified = False
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

        # Debounce timer for saving window geometry changes
        self._save_geometry_timer = QTimer(self)
        self._save_geometry_timer.setSingleShot(True)
        self._save_geometry_timer.setInterval(800)
        self._save_geometry_timer.timeout.connect(self._save_window_geometry)

        self._build_ui()
        self._init_system_tray()
        self._init_shortcuts()
        self._init_network()
        self._connect_signals()

        # Position with boundary validation or fallback to bottom-center of primary screen
        self.setMinimumSize(480, 140)
        self.resize(self.config.ui.window_width, self.config.ui.window_height)
        self._restore_or_center_geometry()

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

    def _is_geometry_valid_on_any_screen(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if a window rectangle has sufficient visible and grab-able area on any active display."""
        screens = QApplication.screens()
        if not screens:
            return False

        win_rect = QRect(x, y, max(width, 50), max(height, 50))
        for scr in screens:
            avail_geo = scr.availableGeometry()
            intersection = avail_geo.intersected(win_rect)
            # Require at least 80px width and 40px height to be visible and grab-able on an active display
            if intersection.width() >= min(80, win_rect.width()) and intersection.height() >= min(40, win_rect.height()):
                return True
        return False

    def _restore_or_center_geometry(self):
        """Restore window position from config if valid on an active screen, otherwise center at bottom."""
        win_x = self.config.ui.window_x
        win_y = self.config.ui.window_y

        if win_x is not None and win_y is not None:
            if self._is_geometry_valid_on_any_screen(win_x, win_y, self.width(), self.height()):
                self.move(win_x, win_y)
                self.anchor_bottom_y = win_y + self.height()
                return

        # Fallback to bottom-center of primary screen
        self._center_at_bottom()

    def _center_at_bottom(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = geo.height() - self.height() - 60
            self.move(x, y)
            self.anchor_bottom_y = y + self.height()

    def _save_window_geometry(self):
        """Persist current window position and size to configuration."""
        if not getattr(self, "_geometry_modified", False):
            return
        self.config.ui.window_x = self.x()
        self.config.ui.window_y = self.y()
        self.config.ui.window_width = self.width()
        self.config.ui.window_height = self.height()
        if not getattr(self, "enable_persistence", True):
            return
        if os.environ.get("PYTEST_CURRENT_TEST") and not getattr(self, "_force_save_in_test", False):
            return
        try:
            self.config.save()
        except Exception as e:
            print(f"[!] 保存 HUD 視窗幾何配置失敗: {e}")

    def _reset_window_position(self):
        """Reset window position to primary screen bottom-center and persist."""
        self._center_at_bottom()
        self._geometry_modified = True
        self._save_window_geometry()

    def _create_app_icon(self) -> QIcon:
        """Create a clean vector application icon for HUD window and system tray."""
        return create_app_icon()

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

        # Action: Reset Window Position
        self.action_tray_reset_pos = QAction("🎯 重設視窗位置至主螢幕中央", self)
        self.action_tray_reset_pos.triggered.connect(self._reset_window_position)
        self.tray_menu.addAction(self.action_tray_reset_pos)

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
        port = self.config.server.port
        run_subtitles_client(
            host=host,
            port=port,
            is_running=lambda: self.is_running,
            on_status=lambda text, color: self.connection_status_signal.emit(text, color),
            on_message=lambda data: self.subtitle_received_signal.emit(data),
            on_connected=lambda: self._fetch_remote_backend_config(),
        )

    def _audio_worker(self):
        """Background worker streaming audio chunks to backend."""
        host = getattr(self.config.server, "host", "127.0.0.1")
        port = self.config.server.port
        run_audio_client(
            host=host,
            port=port,
            audio_queue=self.audio_queue,
            is_running=lambda: self.is_running,
        )

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
        text, color = format_vu_level(rms)
        self.vu_indicator.setText(text)
        weight = "700" if rms > 0.015 else "600"
        self.vu_indicator.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: {weight};")

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
        send_remote_rename_api(self.config.server.port, old_name, new_name, color)

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
        send_remote_delete_api(self.config.server.port, speaker_id)

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

    def _on_audio_chunk(self, chunk, rms: float, source: str = "mixed"):
        # Update real-time VU indicator
        self.audio_level_signal.emit(rms)
        # If direct in-process pipeline is available, pass audio directly for lowest latency
        if self.pipeline:
            self.pipeline.push_audio(chunk, rms, source=source)
        else:
            sr = getattr(self.config.audio, "sample_rate", 16000)
            framed = pack_audio_frame(chunk, track=source, sample_rate=sr)
            self.audio_chunk_signal.emit(framed)

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
        host = getattr(self.config.server, "host", "127.0.0.1")
        port = getattr(self.config.server, "port", 8000)
        return sync_config_to_remote_backend(host, port, new_cfg)

    def _fetch_remote_backend_config(self):
        """Fetch remote backend configuration via GET /api/config and align frontend state."""
        host = getattr(self.config.server, "host", "127.0.0.1")
        port = getattr(self.config.server, "port", 8000)
        return fetch_remote_backend_config(
            host,
            port,
            on_success=lambda remote_cfg: self.config_synced_signal.emit(remote_cfg),
        )

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
        self.config.ui.window_x = self.x()
        self.config.ui.window_y = self.y()
        self.config.ui.window_width = self.width()
        self.config.ui.window_height = self.height()
        self._geometry_modified = True
        if hasattr(self, "_save_geometry_timer"):
            self._save_geometry_timer.start()

    def closeEvent(self, event):
        self.is_running = False
        if hasattr(self, "_save_geometry_timer") and self._save_geometry_timer.isActive():
            self._save_geometry_timer.stop()
            self._geometry_modified = True
        if getattr(self, "_geometry_modified", False):
            self._save_window_geometry()
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


__all__ = [
    "TransparentSubtitleOverlay",
    "run_overlay_app",
    "run_cli",
]
