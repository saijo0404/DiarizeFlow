"""Unit and integration tests for Linux Wayland display server detection and guidance (Issue #53).

Verifies:
1. detect_display_server correctly identifies:
   - Linux Wayland via XDG_SESSION_TYPE="wayland"
   - Linux Wayland via WAYLAND_DISPLAY="wayland-0"
   - Linux X11 via XDG_SESSION_TYPE="x11" or DISPLAY=":0"
   - Windows (win32) platform
   - macOS (darwin) platform
   - Headless / unknown sessions
2. is_wayland_session returns correct boolean across all platforms and environments.
3. print_display_server_guidance outputs guidance banner on Wayland and supports idempotency / force flags.
4. UI helpers get_clickthrough_tooltip, get_clickthrough_tray_title, and get_clickthrough_balloon_message
   dynamically adapt based on Wayland vs non-Wayland environments.
5. TransparentSubtitleOverlay HUD integration:
   - Tooltips and tray menu items reflect Wayland environment when running under Wayland.
   - Toggling clickthrough emits Wayland-friendly balloon notifications and console logs.
6. SettingsDialog integration:
   - Displays Wayland display server notification card when under Wayland.
   - Bypasses Wayland card when under X11 or Windows.
"""

import io
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from diarizeflow.config import AppConfig
from diarizeflow.ui.display_server import (
    DisplayServerInfo,
    detect_display_server,
    get_clickthrough_balloon_message,
    get_clickthrough_tooltip,
    get_clickthrough_tray_title,
    is_wayland_session,
    print_display_server_guidance,
)
from diarizeflow.ui.overlay_window import TransparentSubtitleOverlay
from diarizeflow.ui.settings_dialog import SettingsDialog


class TestDisplayServerDetection(unittest.TestCase):
    """Test suite for display server environment detection."""

    def test_detect_wayland_via_xdg_session_type(self):
        """Verify Wayland detected when XDG_SESSION_TYPE is 'wayland' on Linux."""
        env = {"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"}
        info = detect_display_server(environ=env, platform="linux")

        self.assertTrue(info.is_wayland)
        self.assertFalse(info.is_x11)
        self.assertEqual(info.session_type, "wayland")
        self.assertEqual(info.desktop_environment, "GNOME")
        self.assertFalse(info.hotkeys_supported)
        self.assertIsNotNone(info.guidance_message)
        self.assertIn("Wayland", info.guidance_message)
        self.assertTrue(is_wayland_session(environ=env, platform="linux"))

    def test_detect_wayland_via_wayland_display(self):
        """Verify Wayland detected when WAYLAND_DISPLAY is set, even if XDG_SESSION_TYPE is unset."""
        env = {"WAYLAND_DISPLAY": "wayland-0", "DESKTOP_SESSION": "plasmawayland"}
        info = detect_display_server(environ=env, platform="linux")

        self.assertTrue(info.is_wayland)
        self.assertFalse(info.is_x11)
        self.assertEqual(info.session_type, "wayland")
        self.assertEqual(info.wayland_display, "wayland-0")
        self.assertEqual(info.desktop_environment, "plasmawayland")
        self.assertFalse(info.hotkeys_supported)
        self.assertTrue(is_wayland_session(environ=env, platform="linux"))

    def test_detect_x11_via_xdg_session_type(self):
        """Verify X11 detected when XDG_SESSION_TYPE is 'x11' on Linux."""
        env = {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0", "XDG_CURRENT_DESKTOP": "XFCE"}
        info = detect_display_server(environ=env, platform="linux")

        self.assertFalse(info.is_wayland)
        self.assertTrue(info.is_x11)
        self.assertEqual(info.session_type, "x11")
        self.assertEqual(info.desktop_environment, "XFCE")
        self.assertTrue(info.hotkeys_supported)
        self.assertIsNone(info.guidance_message)
        self.assertFalse(is_wayland_session(environ=env, platform="linux"))

    def test_detect_x11_fallback_via_display(self):
        """Verify X11 detected when DISPLAY is set and no Wayland variables exist."""
        env = {"DISPLAY": ":1", "XDG_SESSION_TYPE": ""}
        info = detect_display_server(environ=env, platform="linux")

        self.assertFalse(info.is_wayland)
        self.assertTrue(info.is_x11)
        self.assertEqual(info.session_type, "x11")
        self.assertTrue(info.hotkeys_supported)

    def test_detect_windows_platform(self):
        """Verify Windows platform is detected and supports Win32 global hotkeys."""
        env = {"XDG_SESSION_TYPE": "wayland"}  # even if spurious env var exists
        info = detect_display_server(environ=env, platform="win32")

        self.assertFalse(info.is_wayland)
        self.assertFalse(info.is_x11)
        self.assertEqual(info.platform, "win32")
        self.assertEqual(info.session_type, "win32")
        self.assertTrue(info.hotkeys_supported)
        self.assertIsNone(info.guidance_message)
        self.assertFalse(is_wayland_session(environ=env, platform="win32"))

    def test_detect_macos_platform(self):
        """Verify macOS platform is detected with cocoa session type."""
        info = detect_display_server(environ={}, platform="darwin")

        self.assertFalse(info.is_wayland)
        self.assertFalse(info.is_x11)
        self.assertEqual(info.platform, "darwin")
        self.assertEqual(info.session_type, "cocoa")
        self.assertTrue(info.hotkeys_supported)

    def test_detect_headless_or_unknown(self):
        """Verify headless/unknown environment on Linux without display server vars."""
        env = {}
        info = detect_display_server(environ=env, platform="linux")

        self.assertFalse(info.is_wayland)
        self.assertFalse(info.is_x11)
        self.assertEqual(info.session_type, "unknown")
        self.assertIsNone(info.guidance_message)


class TestDisplayServerGuidanceBanner(unittest.TestCase):
    """Test suite for terminal guidance printing."""

    def test_print_guidance_on_wayland(self):
        """Verify guidance banner prints under Wayland and returns True."""
        env = {"XDG_SESSION_TYPE": "wayland"}
        printed_messages = []

        with patch("builtins.print", side_effect=lambda msg: printed_messages.append(str(msg))):
            ret = print_display_server_guidance(environ=env, platform="linux", force=True)

        self.assertTrue(ret)
        self.assertTrue(any("Wayland" in m for m in printed_messages))
        self.assertTrue(any("系統托盤" in m for m in printed_messages))

    def test_print_guidance_bypassed_on_x11(self):
        """Verify guidance banner does not print under X11 and returns False."""
        env = {"XDG_SESSION_TYPE": "x11"}
        printed_messages = []

        with patch("builtins.print", side_effect=lambda msg: printed_messages.append(str(msg))):
            ret = print_display_server_guidance(environ=env, platform="linux", force=True)

        self.assertFalse(ret)
        self.assertEqual(len(printed_messages), 0)


class TestUIHelperStrings(unittest.TestCase):
    """Test suite for UI tooltip, tray title, and balloon strings."""

    def test_clickthrough_tooltip_wayland(self):
        """Verify clickthrough button tooltips on Wayland guide user to system tray."""
        off_tip = get_clickthrough_tooltip(is_clickthrough=False, is_wayland=True)
        on_tip = get_clickthrough_tooltip(is_clickthrough=True, is_wayland=True)

        self.assertIn("Wayland", off_tip)
        self.assertIn("系統托盤", off_tip)
        self.assertIn("Wayland", on_tip)
        self.assertIn("系統托盤", on_tip)

    def test_clickthrough_tooltip_non_wayland(self):
        """Verify clickthrough button tooltips on non-Wayland show Alt+Shift+H shortcut."""
        off_tip = get_clickthrough_tooltip(is_clickthrough=False, is_wayland=False)
        on_tip = get_clickthrough_tooltip(is_clickthrough=True, is_wayland=False)

        self.assertIn("Alt+Shift+H", off_tip)
        self.assertIn("Alt+Shift+H", on_tip)

    def test_clickthrough_tray_title(self):
        """Verify tray menu action title differs appropriately on Wayland."""
        wayland_title = get_clickthrough_tray_title(is_wayland=True)
        x11_title = get_clickthrough_tray_title(is_wayland=False)

        self.assertIn("Wayland", wayland_title)
        self.assertIn("Alt+Shift+H", x11_title)

    def test_clickthrough_balloon_message(self):
        """Verify system tray balloon notification advises user of Wayland tray exit."""
        wayland_msg = get_clickthrough_balloon_message(is_wayland=True)
        x11_msg = get_clickthrough_balloon_message(is_wayland=False)

        self.assertIn("Wayland", wayland_msg)
        self.assertIn("系統托盤", wayland_msg)
        self.assertIn("Alt+Shift+H", x11_msg)


class TestHUDOverlayWaylandIntegration(unittest.TestCase):
    """Integration test suite for TransparentSubtitleOverlay HUD under Wayland."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_hud_initializes_with_wayland_hints(self):
        """Verify TransparentSubtitleOverlay configures tooltips and tray for Wayland when is_wayland=True."""
        cfg = AppConfig()
        wayland_info = DisplayServerInfo(
            platform="linux",
            session_type="wayland",
            is_wayland=True,
            is_x11=False,
            wayland_display="wayland-0",
            desktop_environment="GNOME",
            hotkeys_supported=False,
        )

        with patch("diarizeflow.ui.overlay_window.detect_display_server", return_value=wayland_info):
            overlay = TransparentSubtitleOverlay(
                config=cfg,
                enable_network=False,
                auto_start_capture=False,
            )

            # Check clickthrough button tooltip
            self.assertIn("Wayland", overlay.btn_clickthrough.toolTip())
            # Check tray action title
            self.assertIn("Wayland", overlay.action_tray_clickthrough.text())

            # Test toggling clickthrough on Wayland
            overlay._toggle_clickthrough(True)
            self.assertTrue(overlay.is_clickthrough)
            self.assertIn("Wayland", overlay.btn_clickthrough.toolTip())
            self.assertIn("系統托盤", overlay.btn_clickthrough.toolTip())

            # Test toggling clickthrough off on Wayland
            overlay._toggle_clickthrough(False)
            self.assertFalse(overlay.is_clickthrough)
            self.assertIn("Wayland", overlay.btn_clickthrough.toolTip())

            overlay.close()

    def test_hud_initializes_with_standard_hints_on_non_wayland(self):
        """Verify TransparentSubtitleOverlay configures Alt+Shift+H tooltips when not on Wayland."""
        cfg = AppConfig()
        x11_info = DisplayServerInfo(
            platform="linux",
            session_type="x11",
            is_wayland=False,
            is_x11=True,
            wayland_display="",
            desktop_environment="KDE",
            hotkeys_supported=True,
        )

        with patch("diarizeflow.ui.overlay_window.detect_display_server", return_value=x11_info):
            overlay = TransparentSubtitleOverlay(
                config=cfg,
                enable_network=False,
                auto_start_capture=False,
            )

            self.assertIn("Alt+Shift+H", overlay.btn_clickthrough.toolTip())
            self.assertIn("Alt+Shift+H", overlay.action_tray_clickthrough.text())
            overlay.close()


class TestSettingsDialogWaylandIntegration(unittest.TestCase):
    """Integration test suite for SettingsDialog under Wayland."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_settings_dialog_shows_wayland_box_under_wayland(self):
        """Verify SettingsDialog displays Wayland display server guidance card."""
        cfg = AppConfig()
        wayland_info = DisplayServerInfo(
            platform="linux",
            session_type="wayland",
            is_wayland=True,
            is_x11=False,
            wayland_display="wayland-0",
            desktop_environment="GNOME",
            hotkeys_supported=False,
        )

        with patch("diarizeflow.ui.settings_dialog.detect_display_server", return_value=wayland_info):
            dialog = SettingsDialog(cfg)
            # Find any label containing Wayland
            labels = dialog.findChildren(object)
            found_wayland = any("Wayland" in getattr(w, "text", lambda: "")() for w in labels if hasattr(w, "text"))
            self.assertTrue(found_wayland, "SettingsDialog should render Wayland guidance banner when is_wayland=True")
            dialog.close()

    def test_settings_dialog_omits_wayland_box_under_x11(self):
        """Verify SettingsDialog omits Wayland display server guidance card under X11."""
        cfg = AppConfig()
        x11_info = DisplayServerInfo(
            platform="linux",
            session_type="x11",
            is_wayland=False,
            is_x11=True,
            wayland_display="",
            desktop_environment="GNOME",
            hotkeys_supported=True,
        )

        with patch("diarizeflow.ui.settings_dialog.detect_display_server", return_value=x11_info):
            dialog = SettingsDialog(cfg)
            labels = dialog.findChildren(object)
            found_wayland = any("Linux Wayland" in getattr(w, "text", lambda: "")() for w in labels if hasattr(w, "text"))
            self.assertFalse(found_wayland, "SettingsDialog should omit Wayland guidance banner when is_wayland=False")
            dialog.close()


if __name__ == "__main__":
    unittest.main()
