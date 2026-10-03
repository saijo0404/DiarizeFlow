"""Unit tests for frontend modularization and backward-compatibility (Issue #36).

Verifies:
1. Decomposed module imports (widgets, cards, settings_dialog, network, overlay_window).
2. Backward-compatibility of all symbols re-exported from desktop_overlay.
3. Functional verification of widgets (get_speaker_color, create_app_icon, format_vu_level).
4. Functional verification of cards (SpeakerBadge, SubtitleCardWidget lifecycle).
5. Functional verification of settings_dialog (SettingsDialog initialization and saving).
6. Functional verification of network helpers (REST API helpers and config sync).
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QApplication

from diarizeflow.app.config import AppConfig

# 1. Direct submodule imports
import diarizeflow.app.frontend as frontend_pkg
from diarizeflow.app.frontend.cards import SpeakerBadge, SubtitleCardWidget
from diarizeflow.app.frontend.network import (
    fetch_remote_backend_config,
    send_remote_delete_api,
    send_remote_rename_api,
    sync_config_to_remote_backend,
)
from diarizeflow.app.frontend.overlay_window import (
    TransparentSubtitleOverlay,
    run_cli,
    run_overlay_app,
)
from diarizeflow.app.frontend.settings_dialog import SettingsDialog
from diarizeflow.app.frontend.widgets import (
    SPEAKER_COLORS,
    create_app_icon,
    format_vu_level,
    get_speaker_color,
)

# 2. Backward-compatible desktop_overlay imports
import diarizeflow.app.frontend.desktop_overlay as legacy_overlay


class TestFrontendModularizationExports(unittest.TestCase):
    """Verify clean module separation and 100% backward-compatibility exports."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_package_exports(self):
        """Verify __init__.py exports expected symbols."""
        self.assertIs(frontend_pkg.TransparentSubtitleOverlay, TransparentSubtitleOverlay)
        self.assertIs(frontend_pkg.run_overlay_app, run_overlay_app)
        self.assertIs(frontend_pkg.run_cli, run_cli)
        self.assertIs(frontend_pkg.SubtitleCardWidget, SubtitleCardWidget)
        self.assertIs(frontend_pkg.SpeakerBadge, SpeakerBadge)
        self.assertIs(frontend_pkg.SettingsDialog, SettingsDialog)
        self.assertIs(frontend_pkg.get_speaker_color, get_speaker_color)
        self.assertIs(frontend_pkg.create_app_icon, create_app_icon)

    def test_legacy_desktop_overlay_backward_compatibility(self):
        """Verify desktop_overlay re-exports all symbols with exact object identity."""
        self.assertIs(legacy_overlay.TransparentSubtitleOverlay, TransparentSubtitleOverlay)
        self.assertIs(legacy_overlay.SubtitleCardWidget, SubtitleCardWidget)
        self.assertIs(legacy_overlay.SpeakerBadge, SpeakerBadge)
        self.assertIs(legacy_overlay.SettingsDialog, SettingsDialog)
        self.assertIs(legacy_overlay.get_speaker_color, get_speaker_color)
        self.assertIs(legacy_overlay.create_app_icon, create_app_icon)
        self.assertIs(legacy_overlay.format_vu_level, format_vu_level)
        self.assertIs(legacy_overlay.run_overlay_app, run_overlay_app)
        self.assertIs(legacy_overlay.run_cli, run_cli)
        self.assertEqual(legacy_overlay.SPEAKER_COLORS, SPEAKER_COLORS)


class TestWidgetsModule(unittest.TestCase):
    """Verify widgets and styling helpers."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_get_speaker_color_deterministic(self):
        """Verify get_speaker_color deterministically maps speaker names."""
        color1 = get_speaker_color("Speaker 1")
        color2 = get_speaker_color("Speaker 2")
        color1_again = get_speaker_color("Speaker 1")

        self.assertEqual(color1, color1_again)
        self.assertNotEqual(color1, color2)
        self.assertIn(color1, SPEAKER_COLORS)

    def test_create_app_icon_valid(self):
        """Verify create_app_icon generates a valid, non-null QIcon."""
        icon = create_app_icon()
        self.assertIsInstance(icon, QIcon)
        self.assertFalse(icon.isNull())

    def test_format_vu_level_thresholds(self):
        """Verify format_vu_level maps RMS values to correct display text and colors."""
        text_high, color_high = format_vu_level(0.05)
        self.assertIn("收音中", text_high)
        self.assertEqual(color_high, "#22c55e")

        text_med, color_med = format_vu_level(0.02)
        self.assertIn("收音中", text_med)
        self.assertEqual(color_med, "#10b981")

        text_low, color_low = format_vu_level(0.008)
        self.assertIn("收音中", text_low)
        self.assertEqual(color_low, "#34d399")

        text_idle, color_idle = format_vu_level(0.0001)
        self.assertIn("監聽中", text_idle)
        self.assertEqual(color_idle, "#64748b")


class TestCardsModule(unittest.TestCase):
    """Verify SubtitleCardWidget and SpeakerBadge lifecycle."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_speaker_badge_click_signal(self):
        """Verify SpeakerBadge emits clicked signal on mouse press."""
        badge = SpeakerBadge("Alice")
        emitted = []
        badge.clicked.connect(lambda: emitted.append(True))

        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            badge.rect().center().toPointF(),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        badge.mousePressEvent(event)
        self.assertEqual(len(emitted), 1)

    def test_subtitle_card_widget_lifecycle(self):
        """Verify SubtitleCardWidget updates text and emits dismissed."""
        cfg = AppConfig()
        cfg.ui.font_size = 20
        cfg.ui.fade_out_seconds = 0.1

        card = SubtitleCardWidget(
            speaker="Speaker 1",
            original="Good morning",
            translated="早安",
            confidence=0.98,
            config=cfg,
        )
        self.assertEqual(card.speaker, "Speaker 1")
        self.assertEqual(card.translated_label.text(), "早安")
        self.assertEqual(card.original_label.text(), "Good morning")

        # Update speaker
        card.update_speaker("Speaker 2", "#c084fc")
        self.assertEqual(card.speaker, "Speaker 2")
        self.assertEqual(card.badge.text(), "Speaker 2")

        # Test dismissal signal
        dismissed_cards = []
        card.dismissed.connect(lambda c: dismissed_cards.append(c))
        card.dismiss_immediately()
        self.assertEqual(len(dismissed_cards), 1)
        self.assertIs(dismissed_cards[0], card)


class TestSettingsDialogModule(unittest.TestCase):
    """Verify SettingsDialog configuration loading and saving."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_settings_dialog_save_and_apply(self):
        """Verify SettingsDialog modifies config and emits settings_saved signal."""
        cfg = AppConfig()
        cfg.save = MagicMock()

        dialog = SettingsDialog(cfg)
        saved_events = []
        dialog.settings_saved.connect(lambda c: saved_events.append(c))

        dialog.lang_combo.setCurrentText("日本語")
        dialog.fade_spin.setValue(12.0)
        dialog._save_and_apply()

        self.assertEqual(cfg.llm.target_language, "日本語")
        self.assertEqual(cfg.ui.fade_out_seconds, 12.0)
        self.assertEqual(len(saved_events), 1)
        cfg.save.assert_called_once()
        dialog.close()


class TestNetworkModule(unittest.TestCase):
    """Verify network helper functions."""

    def test_sync_config_to_remote_backend(self):
        """Verify sync_config_to_remote_backend sends POST /api/config."""
        sent_requests = []

        class MockResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=None):
            sent_requests.append(req)
            return MockResponse()

        cfg = AppConfig()
        cfg.llm.target_language = "Français"

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            t = sync_config_to_remote_backend("127.0.0.1", 8000, cfg)
            t.join(timeout=2.0)

        self.assertEqual(len(sent_requests), 1)
        req = sent_requests[0]
        self.assertEqual(req.method, "POST")
        self.assertTrue(req.full_url.endswith("/api/config"))
        data = json.loads(req.data.decode("utf-8"))
        self.assertEqual(data["llm"]["target_language"], "Français")

    def test_fetch_remote_backend_config(self):
        """Verify fetch_remote_backend_config sends GET /api/config and executes callback."""
        remote_cfg = AppConfig()
        remote_cfg.llm.target_language = "Italiano"
        raw_bytes = json.dumps(remote_cfg.to_dict()).encode("utf-8")

        class MockGetResponse:
            status = 200
            def read(self):
                return raw_bytes
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        received_cfg = []

        with patch("urllib.request.urlopen", return_value=MockGetResponse()):
            t = fetch_remote_backend_config(
                "127.0.0.1",
                8000,
                on_success=lambda c: received_cfg.append(c),
            )
            t.join(timeout=2.0)

        self.assertEqual(len(received_cfg), 1)
        self.assertEqual(received_cfg[0].llm.target_language, "Italiano")

    def test_send_remote_rename_and_delete_api(self):
        """Verify rename and delete REST helper calls."""
        sent_requests = []

        class MockResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=None):
            sent_requests.append(req)
            return MockResponse()

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            send_remote_rename_api(8000, "Old Speaker", "New Speaker", "#ff0000")
            send_remote_delete_api(8000, "Old Speaker")

        self.assertEqual(len(sent_requests), 2)
        rename_req, delete_req = sent_requests
        self.assertEqual(rename_req.method, "POST")
        self.assertIn("Old%20Speaker/rename", rename_req.full_url)
        self.assertEqual(delete_req.method, "DELETE")
        self.assertIn("Old%20Speaker", delete_req.full_url)


if __name__ == "__main__":
    unittest.main()
