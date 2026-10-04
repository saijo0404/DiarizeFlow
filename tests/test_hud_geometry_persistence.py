"""Unit and integration tests for HUD Window Position and Geometry Persistence (Issue #34).

Verifies:
1. UIConfig and AppConfig serialization/deserialization for window_x and window_y.
2. Safe handling of invalid, None, or string coordinate values.
3. Startup restoration to configured (window_x, window_y) when within active screen geometry.
4. Out-of-bounds screen boundary safety: fallback to bottom-center when coordinates are outside all displays.
5. Window drag-and-release updates config.ui coordinates and triggers debounce save.
6. Immediate flush of pending geometry persistence on closeEvent.
7. System tray "Reset Window Position" action centers the window and persists new coordinates.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from diarizeflow.config import AppConfig, UIConfig
from diarizeflow.ui.desktop_overlay import TransparentSubtitleOverlay


class TestHUDGeometryPersistence(unittest.TestCase):
    """Test suite for HUD geometry memory, boundary clamping, and persistence."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = AppConfig()
        self.config.ui.max_cards = 3
        self.config.ui.window_width = 600
        self.config.ui.window_height = 200
        self.mock_save = MagicMock()
        self.config.save = self.mock_save

    def test_uiconfig_serialization_and_deserialization(self):
        """Verify window_x and window_y are serialized to dict and deserialized safely."""
        ui_cfg = UIConfig(window_x=150, window_y=300)
        cfg = AppConfig(ui=ui_cfg)
        d = cfg.to_dict()

        self.assertEqual(d["ui"]["window_x"], 150)
        self.assertEqual(d["ui"]["window_y"], 300)

        restored = AppConfig.from_dict(d)
        self.assertEqual(restored.ui.window_x, 150)
        self.assertEqual(restored.ui.window_y, 300)

    def test_uiconfig_invalid_coordinates_resilience(self):
        """Verify invalid or non-numeric window coordinates are safely handled without crash."""
        raw_data = {
            "ui": {
                "window_x": "not-an-int",
                "window_y": None,
                "window_width": "800",
                "window_height": "invalid",
            }
        }
        cfg = AppConfig.from_dict(raw_data)
        self.assertIsNone(cfg.ui.window_x)
        self.assertIsNone(cfg.ui.window_y)
        self.assertEqual(cfg.ui.window_width, 800)
        # Invalid window_height falls back to default 260
        self.assertEqual(cfg.ui.window_height, 260)

    def test_startup_restores_valid_geometry(self):
        """Verify overlay restores configured (window_x, window_y) on startup when within screen."""
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)

        target_x = avail.left() + 50
        target_y = avail.top() + 80

        self.config.ui.window_x = target_x
        self.config.ui.window_y = target_y

        overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        overlay.show()
        self.app.processEvents()

        try:
            self.assertEqual(overlay.x(), target_x)
            self.assertEqual(overlay.y(), target_y)
            self.assertEqual(overlay.anchor_bottom_y, target_y + overlay.height())
        finally:
            overlay.close()
            overlay.deleteLater()
            self.app.processEvents()

    def test_startup_out_of_bounds_falls_back_to_center_at_bottom(self):
        """Verify overlay falls back to bottom-center when coordinates are outside all displays."""
        # Unplugged secondary monitor coordinates (far offscreen)
        self.config.ui.window_x = 99999
        self.config.ui.window_y = 99999

        overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        overlay.show()
        self.app.processEvents()

        try:
            screen = QApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                expected_x = (geo.width() - overlay.width()) // 2
                expected_y = geo.height() - overlay.height() - 60
                self.assertEqual(overlay.x(), expected_x)
                self.assertEqual(overlay.y(), expected_y)
        finally:
            overlay.close()
            overlay.deleteLater()
            self.app.processEvents()

    def test_mouse_drag_and_release_persists_geometry(self):
        """Verify dragging the overlay updates config.ui coordinates and triggers debounce save."""
        overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        overlay.show()
        self.app.processEvents()

        try:
            self.assertTrue(hasattr(overlay, "_save_geometry_timer"))

            # Move window to a new position
            new_pos = QPoint(180, 220)
            overlay.move(new_pos)

            # Trigger mouseReleaseEvent
            release_event = QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(50, 50),
                QPointF(230, 270),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            overlay.mouseReleaseEvent(release_event)

            self.assertEqual(overlay.config.ui.window_x, 180)
            self.assertEqual(overlay.config.ui.window_y, 220)
            self.assertEqual(overlay.anchor_bottom_y, 220 + overlay.height())
            self.assertTrue(overlay._save_geometry_timer.isActive())
        finally:
            overlay.close()
            overlay.deleteLater()
            self.app.processEvents()

    def test_close_event_flushes_pending_geometry_save(self):
        """Verify closeEvent immediately flushes and saves pending geometry changes."""
        overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        overlay.show()
        self.app.processEvents()

        overlay.move(250, 350)
        overlay._force_save_in_test = True
        overlay._save_geometry_timer.start()

        with patch.object(overlay.config, "save") as mock_save:
            overlay.close()
            mock_save.assert_called_once()
            self.assertEqual(overlay.config.ui.window_x, 250)
            self.assertEqual(overlay.config.ui.window_y, 350)

        overlay.deleteLater()
        self.app.processEvents()

    def test_tray_reset_window_position_action(self):
        """Verify tray action resets window position to bottom-center and persists."""
        overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        overlay._force_save_in_test = True
        overlay.show()
        self.app.processEvents()

        try:
            self.assertTrue(hasattr(overlay, "action_tray_reset_pos"))
            # Move off somewhere
            overlay.move(300, 100)

            with patch.object(overlay.config, "save") as mock_save:
                overlay.action_tray_reset_pos.trigger()
                mock_save.assert_called()

                screen = QApplication.primaryScreen()
                if screen:
                    geo = screen.availableGeometry()
                    expected_x = (geo.width() - overlay.width()) // 2
                    expected_y = geo.height() - overlay.height() - 60
                    self.assertEqual(overlay.x(), expected_x)
                    self.assertEqual(overlay.y(), expected_y)
                    self.assertEqual(overlay.config.ui.window_x, expected_x)
                    self.assertEqual(overlay.config.ui.window_y, expected_y)
        finally:
            overlay.close()
            overlay.deleteLater()
            self.app.processEvents()
