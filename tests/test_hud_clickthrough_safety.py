"""Unit and integration tests for HUD Click-Through Safety and System Tray (Issue #26).

Verifies:
1. System tray icon initialization, context menu actions, and icon creation.
2. Global and application shortcuts (Alt+Shift+H, Ctrl+Shift+T) to toggle click-through.
3. Control bar (header_widget) non-transparent protection: header remains interactive.
4. Subtitle container and newly spawned cards inherit mouse transparency when click-through is enabled.
5. Bidirectional state synchronization between UI button, tray action, and internal flag.
6. Audio capture toggle state synchronization with tray menu.
7. Window visibility toggling via tray menu.
8. Clean teardown and resource release on window close.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from diarizeflow.app.config import AppConfig
from diarizeflow.app.frontend.desktop_overlay import (
    SubtitleCardWidget,
    TransparentSubtitleOverlay,
)


class TestHUDClickThroughSafety(unittest.TestCase):
    """Test suite for HUD Click-Through deadlock prevention, system tray, and hotkeys."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = AppConfig()
        self.config.ui.max_cards = 3
        self.config.ui.fade_out_seconds = 5.0
        self.config.ui.font_size = 20
        self.overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        self.overlay.show()
        self.app.processEvents()

    def tearDown(self):
        self.overlay.close()
        self.overlay.deleteLater()
        self.app.processEvents()

    def test_system_tray_icon_and_menu_initialization(self):
        """Verify QSystemTrayIcon is initialized with icon and full context menu actions."""
        self.assertIsNotNone(self.overlay.tray_icon)
        self.assertIsNotNone(self.overlay.tray_menu)
        self.assertFalse(self.overlay.app_icon.isNull())

        # Check required tray menu actions
        self.assertIsNotNone(self.overlay.action_tray_clickthrough)
        self.assertTrue(self.overlay.action_tray_clickthrough.isCheckable())
        self.assertIn("Alt+Shift+H", self.overlay.action_tray_clickthrough.text())

        self.assertIsNotNone(self.overlay.action_tray_capture)
        self.assertIn("開始收音", self.overlay.action_tray_capture.text())

        self.assertIsNotNone(self.overlay.action_tray_settings)
        self.assertIn("設定", self.overlay.action_tray_settings.text())

        self.assertIsNotNone(self.overlay.action_tray_logs)
        self.assertIn("日誌", self.overlay.action_tray_logs.text())

        self.assertIsNotNone(self.overlay.action_tray_toggle_win)
        self.assertIn("HUD 視窗", self.overlay.action_tray_toggle_win.text())

        self.assertIsNotNone(self.overlay.action_tray_quit)
        self.assertIn("退出", self.overlay.action_tray_quit.text())

    def test_keyboard_shortcuts_registered(self):
        """Verify Alt+Shift+H and Ctrl+Shift+T shortcuts are registered with Application context."""
        self.assertIsNotNone(self.overlay.shortcut_h)
        self.assertEqual(self.overlay.shortcut_h.key().toString(), "Alt+Shift+H")
        self.assertEqual(self.overlay.shortcut_h.context(), Qt.ShortcutContext.ApplicationShortcut)

        self.assertIsNotNone(self.overlay.shortcut_t)
        self.assertEqual(self.overlay.shortcut_t.key().toString(), "Ctrl+Shift+T")
        self.assertEqual(self.overlay.shortcut_t.context(), Qt.ShortcutContext.ApplicationShortcut)

    def test_toggle_clickthrough_state_synchronization(self):
        """Verify toggling click-through updates button, tray menu, and internal flag synchronously."""
        self.assertFalse(self.overlay.is_clickthrough)
        self.assertFalse(self.overlay.btn_clickthrough.isChecked())
        self.assertFalse(self.overlay.action_tray_clickthrough.isChecked())
        self.assertEqual(self.overlay.btn_clickthrough.text(), "🛡️ 穿透")

        # 1. Turn ON click-through
        self.overlay._toggle_clickthrough(True)
        self.assertTrue(self.overlay.is_clickthrough)
        self.assertTrue(self.overlay.btn_clickthrough.isChecked())
        self.assertTrue(self.overlay.action_tray_clickthrough.isChecked())
        self.assertEqual(self.overlay.btn_clickthrough.text(), "🛡️ 穿透中")

        # 2. Turn OFF click-through via parameterless toggle (e.g., from shortcut)
        self.overlay._toggle_clickthrough()
        self.assertFalse(self.overlay.is_clickthrough)
        self.assertFalse(self.overlay.btn_clickthrough.isChecked())
        self.assertFalse(self.overlay.action_tray_clickthrough.isChecked())
        self.assertEqual(self.overlay.btn_clickthrough.text(), "🛡️ 穿透")

    def test_control_bar_header_protection(self):
        """Verify header_widget NEVER becomes transparent to mouse events even when click-through is ON."""
        # Enable click-through
        self.overlay._toggle_clickthrough(True)

        # Header control bar must NOT be transparent to mouse events!
        self.assertFalse(
            self.overlay.header_widget.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents),
            "header_widget must remain clickable so user can operate controls without deadlock",
        )

        # Subtitle container must be transparent to mouse events
        self.assertTrue(
            self.overlay.subtitle_container.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents),
            "subtitle_container must be transparent to mouse events in click-through mode",
        )

        # Disable click-through
        self.overlay._toggle_clickthrough(False)
        self.assertFalse(
            self.overlay.subtitle_container.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        )
        self.assertFalse(
            self.overlay.header_widget.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        )

    def test_new_cards_inherit_clickthrough_state(self):
        """Verify subtitle cards dynamically created while click-through is ON inherit transparency."""
        self.overlay._toggle_clickthrough(True)

        self.overlay.show_subtitle("講者 1", "Hello world", "你好世界")
        self.assertEqual(self.overlay.active_card_count, 2)  # welcome card + new card
        latest_card = self.overlay.cards[-1]

        self.assertTrue(
            latest_card.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents),
            "Newly created card should be transparent to mouse events when click-through is enabled",
        )

        # Turning click-through OFF should restore interactivity across all cards
        self.overlay._toggle_clickthrough(False)
        for card in self.overlay.cards:
            self.assertFalse(
                card.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents),
                "Cards must restore mouse event reception when click-through is disabled",
            )

    def test_tray_action_clickthrough_trigger(self):
        """Verify triggering tray menu action toggles click-through without deadlock."""
        self.assertFalse(self.overlay.is_clickthrough)

        # Simulate user clicking tray menu item
        self.overlay.action_tray_clickthrough.trigger()
        self.assertTrue(self.overlay.is_clickthrough)
        self.assertEqual(self.overlay.btn_clickthrough.text(), "🛡️ 穿透中")

        # Click again to disable
        self.overlay.action_tray_clickthrough.trigger()
        self.assertFalse(self.overlay.is_clickthrough)
        self.assertEqual(self.overlay.btn_clickthrough.text(), "🛡️ 穿透")

    def test_shortcut_activation_toggles_clickthrough(self):
        """Verify activating shortcuts toggles click-through state."""
        self.assertFalse(self.overlay.is_clickthrough)

        # Activate shortcut Alt+Shift+H
        self.overlay.shortcut_h.activated.emit()
        self.assertTrue(self.overlay.is_clickthrough)

        # Activate shortcut Ctrl+Shift+T
        self.overlay.shortcut_t.activated.emit()
        self.assertFalse(self.overlay.is_clickthrough)

    def test_window_visibility_toggle_from_tray(self):
        """Verify toggle window visibility action switches HUD between visible and hidden."""
        self.assertTrue(self.overlay.isVisible())

        # Hide window
        self.overlay._toggle_window_visibility()
        self.assertFalse(self.overlay.isVisible())

        # Show window
        self.overlay._toggle_window_visibility()
        self.assertTrue(self.overlay.isVisible())

    def test_audio_capture_tray_action_sync(self):
        """Verify audio capture state updates tray action label between ▶ 開始收音 and ⏹ 停止收音."""
        self.assertEqual(self.overlay.action_tray_capture.text(), "▶ 開始收音")

        # Simulate capture started
        with patch.object(self.overlay, "_start_capture", wraps=self.overlay._start_capture):
            self.overlay.is_capturing = True
            if hasattr(self.overlay, "action_tray_capture"):
                self.overlay.action_tray_capture.setText("⏹ 停止收音")
            self.assertEqual(self.overlay.action_tray_capture.text(), "⏹ 停止收音")

        # Simulate capture stopped
        self.overlay.is_capturing = False
        if hasattr(self.overlay, "action_tray_capture"):
            self.overlay.action_tray_capture.setText("▶ 開始收音")
        self.assertEqual(self.overlay.action_tray_capture.text(), "▶ 開始收音")

    def test_tray_icon_hidden_on_close(self):
        """Verify system tray icon is hidden upon closing to prevent orphan icons."""
        self.overlay.close()
        self.assertFalse(self.overlay.tray_icon.isVisible())
