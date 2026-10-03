"""Unit and integration tests for HUD WebSocket event dispatching and empty-card defense (Issue #28).

Verifies:
1. "speaker_deleted" event does NOT create a blank subtitle card.
2. "speaker_deleted" updates active matching cards to "未知講者" without adding cards.
3. "speaker_deleted" with non-matching speaker leaves existing cards unaffected.
4. "speaker_renamed" correctly updates matching card badges and colors.
5. Heartbeat ("ping", "pong") and system ("connection") packets are safely ignored.
6. Unknown control packets are safely ignored without falling through to subtitle rendering.
7. Empty text defense in show_subtitle and _handle_subtitle_event prevents ghost cards.
8. Valid subtitle events still render properly.
9. TransparentSubtitleOverlay.delete_speaker properly delegates to pipeline and updates HUD state.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from diarizeflow.app.config import AppConfig
from diarizeflow.app.frontend.desktop_overlay import (
    SubtitleCardWidget,
    TransparentSubtitleOverlay,
)


class TestHUDWebSocketEvents(unittest.TestCase):
    """Test suite for HUD WebSocket event handling and speaker deletion."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = AppConfig()
        self.config.ui.max_cards = 5
        self.config.ui.fade_out_seconds = 10.0
        self.config.ui.font_size = 18
        self.overlay = TransparentSubtitleOverlay(
            config=self.config,
            enable_network=False,
            auto_start_capture=False,
        )
        self.overlay.show()
        # Clear default startup welcome card to start with clean slate
        self.overlay._start_fade_out()
        self.app.processEvents()

    def tearDown(self):
        self.overlay.close()
        self.overlay.deleteLater()
        self.app.processEvents()

    def test_speaker_deleted_does_not_create_empty_card(self):
        """Verify receiving speaker_deleted event does NOT produce a ghost blank card."""
        initial_count = self.overlay.active_card_count
        self.assertEqual(initial_count, 0)

        # Simulate backend broadcasting speaker_deleted event
        delete_event = {
            "type": "speaker_deleted",
            "speaker_id": "Alice",
        }
        self.overlay._handle_subtitle_event(delete_event)
        self.app.processEvents()

        # No empty card should have been added
        self.assertEqual(self.overlay.active_card_count, 0)
        self.assertTrue(self.overlay.idle_label.isVisible())

    def test_speaker_deleted_resets_matching_active_card_to_unknown(self):
        """Verify speaker_deleted modifies active card belonging to deleted speaker to 未知講者."""
        # 1. Show a valid subtitle card for Alice
        self.overlay.show_subtitle("Alice", "Hello world", "你好世界", 0.95)
        self.app.processEvents()
        self.assertEqual(self.overlay.active_card_count, 1)
        card = self.overlay.cards[0]
        self.assertEqual(card.speaker, "Alice")
        self.assertEqual(card.badge.text(), "Alice")

        # 2. Simulate speaker_deleted for Alice
        delete_event = {
            "type": "speaker_deleted",
            "speaker_id": "Alice",
        }
        self.overlay._handle_subtitle_event(delete_event)
        self.app.processEvents()

        # Card count must remain 1 (no new card created)
        self.assertEqual(self.overlay.active_card_count, 1)
        # Existing card's speaker badge should be updated to 未知講者
        self.assertEqual(card.speaker, "未知講者")
        self.assertEqual(card.badge.text(), "未知講者")
        self.assertEqual(self.overlay._current_speaker, "未知講者")

    def test_speaker_deleted_leaves_other_speakers_unaffected(self):
        """Verify deleting Alice does not affect Bob's active card."""
        self.overlay.show_subtitle("Bob", "Good day", "祝你有美好的一天", 0.99)
        self.app.processEvents()
        self.assertEqual(self.overlay.active_card_count, 1)

        delete_event = {
            "type": "speaker_deleted",
            "speaker_id": "Alice",
        }
        self.overlay._handle_subtitle_event(delete_event)
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 1)
        self.assertEqual(self.overlay.cards[0].speaker, "Bob")
        self.assertEqual(self.overlay.cards[0].badge.text(), "Bob")

    def test_speaker_renamed_event(self):
        """Verify speaker_renamed updates speaker badge and color properly."""
        self.overlay.show_subtitle("講者 1", "Sample text", "範例文本", 0.90)
        self.app.processEvents()
        card = self.overlay.cards[0]

        rename_event = {
            "type": "speaker_renamed",
            "old_speaker": "講者 1",
            "new_speaker": "Charlie",
            "color": "#10b981",
        }
        self.overlay._handle_subtitle_event(rename_event)
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 1)
        self.assertEqual(card.speaker, "Charlie")
        self.assertEqual(card.badge.text(), "Charlie")
        self.assertEqual(self.overlay._current_speaker, "Charlie")

    def test_heartbeat_and_system_events_ignored(self):
        """Verify connection, ping, pong and unknown control events are ignored without creating cards."""
        events_to_ignore = [
            {"type": "connection"},
            {"type": "ping"},
            {"type": "pong"},
            {"type": "unknown_future_control", "payload": {}},
        ]
        for evt in events_to_ignore:
            self.overlay._handle_subtitle_event(evt)
            self.app.processEvents()
            self.assertEqual(self.overlay.active_card_count, 0)

    def test_show_subtitle_empty_text_defense(self):
        """Verify show_subtitle silently drops empty/whitespace strings."""
        self.assertEqual(self.overlay.active_card_count, 0)

        # Empty strings
        self.overlay.show_subtitle("講者 1", "", "")
        self.app.processEvents()
        self.assertEqual(self.overlay.active_card_count, 0)

        # Whitespace strings
        self.overlay.show_subtitle("講者 1", "   \t", " \n  ")
        self.app.processEvents()
        self.assertEqual(self.overlay.active_card_count, 0)

        # None values
        self.overlay.show_subtitle("講者 1", None, None)
        self.app.processEvents()
        self.assertEqual(self.overlay.active_card_count, 0)

        self.assertTrue(self.overlay.idle_label.isVisible())

    def test_handle_subtitle_event_empty_text_defense(self):
        """Verify _handle_subtitle_event drops packets with empty text."""
        empty_payloads = [
            {"original_text": "", "translated_text": ""},
            {"speaker": "講者 2", "original_text": "   ", "translated_text": ""},
        ]
        for payload in empty_payloads:
            self.overlay._handle_subtitle_event(payload)
            self.app.processEvents()
            self.assertEqual(self.overlay.active_card_count, 0)

    def test_valid_subtitle_event_creates_card(self):
        """Verify normal subtitle events create cards with correct content."""
        valid_evt = {
            "speaker": "David",
            "original_text": "Good morning",
            "translated_text": "早安",
            "confidence": 0.98,
        }
        self.overlay._handle_subtitle_event(valid_evt)
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 1)
        card = self.overlay.cards[0]
        self.assertEqual(card.speaker, "David")
        self.assertEqual(card.original, "Good morning")
        self.assertEqual(card.translated, "早安")
        self.assertFalse(self.overlay.idle_label.isVisible())

    def test_overlay_delete_speaker_method(self):
        """Verify delete_speaker helper updates HUD and notifies backend pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline.delete_speaker.return_value = True
        self.overlay.pipeline = mock_pipeline

        self.overlay.show_subtitle("Emily", "Testing pipeline delete", "測試管線刪除")
        self.app.processEvents()
        self.assertEqual(self.overlay.cards[0].speaker, "Emily")

        self.overlay.delete_speaker("Emily")
        self.app.processEvents()

        mock_pipeline.delete_speaker.assert_called_once_with("Emily")
        self.assertEqual(self.overlay.cards[0].speaker, "未知講者")
        self.assertEqual(self.overlay.cards[0].badge.text(), "未知講者")


if __name__ == "__main__":
    unittest.main()
