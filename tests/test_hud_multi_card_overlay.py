"""Unit and integration tests for HUD Multi-Card Queue and Bottom-Anchoring (Issue #14).

Verifies:
1. Multi-card stack creation and FIFO eviction when exceeding max_cards.
2. Independent per-card fade-out lifecycle without mutually clearing other cards.
3. Vertical bottom-anchoring layout alignment (newest card at bottom, older pushed up).
4. Window geometry anti-jitter anchoring (bottom edge remains fixed to anchor_bottom_y on resize).
5. Dynamic configuration updates (max_cards, font_size, show_original).
6. Backward compatibility with existing properties (translated_text, speaker_label, original_text).
"""

import os
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

from diarizeflow.app.config import AppConfig
from diarizeflow.app.frontend.desktop_overlay import (
    SubtitleCardWidget,
    TransparentSubtitleOverlay,
    get_speaker_color,
)


class TestHUDMultiCardOverlay(unittest.TestCase):
    """Test suite for TransparentSubtitleOverlay multi-card queue and bottom-anchoring."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = AppConfig()
        self.config.ui.max_cards = 3
        self.config.ui.fade_out_seconds = 5.0
        self.config.ui.font_size = 20
        self.config.ui.show_original = True
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

    def test_initial_welcome_card_and_active_count(self):
        """Verify startup creates an initial welcome card in the queue."""
        self.assertEqual(self.overlay.active_card_count, 1)
        self.assertEqual(len(self.overlay.cards), 1)
        self.assertFalse(self.overlay.idle_label.isVisible())

    def test_multi_card_creation_and_stacking(self):
        """Verify multiple subtitles create stacked cards in FIFO order."""
        # Clear welcome card for clean test
        self.overlay._start_fade_out()
        self.assertEqual(self.overlay.active_card_count, 0)
        self.assertTrue(self.overlay.idle_label.isVisible())

        # Post first utterance
        self.overlay.show_subtitle("講者 1", "Hello everyone", "各位好")
        self.assertEqual(self.overlay.active_card_count, 1)
        self.assertEqual(self.overlay.cards[0].speaker, "講者 1")
        self.assertEqual(self.overlay.cards[0].translated, "各位好")
        self.assertEqual(self.overlay.cards[0].original, "Hello everyone")
        self.assertFalse(self.overlay.idle_label.isVisible())

        # Post second utterance
        self.overlay.show_subtitle("講者 2", "Can you hear me?", "聽得到我的聲音嗎？")
        self.assertEqual(self.overlay.active_card_count, 2)
        self.assertEqual(self.overlay.cards[0].speaker, "講者 1")
        self.assertEqual(self.overlay.cards[1].speaker, "講者 2")

        # Post third utterance
        self.overlay.show_subtitle("講者 1", "Yes, loud and clear.", "可以，非常清楚。")
        self.assertEqual(self.overlay.active_card_count, 3)
        self.assertEqual(self.overlay.cards[0].speaker, "講者 1")
        self.assertEqual(self.overlay.cards[1].speaker, "講者 2")
        self.assertEqual(self.overlay.cards[2].speaker, "講者 1")

    def test_max_cards_queue_eviction(self):
        """Verify that exceeding max_cards evicts the oldest card (FIFO)."""
        self.overlay._start_fade_out()
        self.overlay.config.ui.max_cards = 2

        self.overlay.show_subtitle("講者 1", "Utterance 1", "話語一")
        self.overlay.show_subtitle("講者 2", "Utterance 2", "話語二")
        self.assertEqual(self.overlay.active_card_count, 2)
        self.assertEqual(self.overlay.cards[0].translated, "話語一")
        self.assertEqual(self.overlay.cards[1].translated, "話語二")

        # Push 3rd utterance: should evict Utterance 1
        self.overlay.show_subtitle("講者 3", "Utterance 3", "話語三")
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 2)
        self.assertEqual(self.overlay.cards[0].translated, "話語二")
        self.assertEqual(self.overlay.cards[1].translated, "話語三")

    def test_independent_card_fade_out_lifecycle(self):
        """Verify that each card has an independent fade timer and does not clear other cards."""
        self.overlay._start_fade_out()

        self.overlay.show_subtitle("講者 1", "Speaker 1 says hello", "講者一說你好")
        self.overlay.show_subtitle("講者 2", "Speaker 2 answers", "講者二回應")
        self.assertEqual(self.overlay.active_card_count, 2)

        card1 = self.overlay.cards[0]
        card2 = self.overlay.cards[1]

        # Dismiss card 1 only (simulating timer expiry for speaker 1)
        card1.dismiss_immediately()
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 1)
        self.assertIn(card2, self.overlay.cards)
        self.assertNotIn(card1, self.overlay.cards)
        self.assertEqual(self.overlay.cards[0].speaker, "講者 2")

        # Dismiss card 2
        card2.dismiss_immediately()
        self.app.processEvents()

        self.assertEqual(self.overlay.active_card_count, 0)
        self.assertTrue(self.overlay.idle_label.isVisible())

    def test_cards_layout_bottom_alignment(self):
        """Verify cards stack vertically from bottom to top (newest at bottom, older above)."""
        self.overlay._start_fade_out()

        self.overlay.show_subtitle("講者 1", "First", "第一句")
        self.overlay.show_subtitle("講者 2", "Second", "第二句")
        self.overlay.show_subtitle("講者 3", "Third", "第三句")
        self.app.processEvents()

        self.assertEqual(len(self.overlay.cards), 3)
        y0 = self.overlay.cards[0].y()
        y1 = self.overlay.cards[1].y()
        y2 = self.overlay.cards[2].y()

        # Older card (index 0) must be positioned above newer card (index 1 and index 2)
        self.assertLess(y0, y1, f"Card 0 (y={y0}) should be above Card 1 (y={y1})")
        self.assertLess(y1, y2, f"Card 1 (y={y1}) should be above Card 2 (y={y2})")

    def test_window_bottom_anchoring_anti_jitter(self):
        """Verify that resizing window maintains strictly fixed bottom edge (anchor_bottom_y)."""
        initial_bottom = self.overlay.anchor_bottom_y
        self.assertIsNotNone(initial_bottom)
        self.assertEqual(self.overlay.y() + self.overlay.height(), initial_bottom)

        # Expand height by 80px
        orig_y = self.overlay.y()
        new_height = self.overlay.height() + 80
        self.overlay.resize(self.overlay.width(), new_height)
        self.app.processEvents()

        # The bottom position MUST remain strictly identical
        self.assertEqual(
            self.overlay.y() + self.overlay.height(),
            initial_bottom,
            "Bottom anchor jittered after window resize!",
        )
        self.assertEqual(self.overlay.y(), orig_y - 80)

    def test_mouse_drag_updates_anchor_bottom(self):
        """Verify that dragging and releasing the HUD updates anchor_bottom_y."""
        # Simulate moving window
        self.overlay.move(200, 400)
        # Simulate mouse release
        class MockEvent:
            pass

        self.overlay.mouseReleaseEvent(MockEvent())

        expected_bottom = 400 + self.overlay.height()
        self.assertEqual(self.overlay.anchor_bottom_y, expected_bottom)

    def test_apply_updated_config_updates_cards_and_anchor(self):
        """Verify dynamic config updates refresh cards and respect max_cards reduction."""
        self.overlay._start_fade_out()
        self.overlay.show_subtitle("講者 1", "One", "一")
        self.overlay.show_subtitle("講者 2", "Two", "二")
        self.overlay.show_subtitle("講者 3", "Three", "三")
        self.assertEqual(self.overlay.active_card_count, 3)

        # Update config: reduce max_cards to 2 and change font size
        new_config = AppConfig()
        new_config.ui.max_cards = 2
        new_config.ui.font_size = 28
        new_config.ui.show_original = False
        new_config.ui.window_height = 320

        old_bottom = self.overlay.anchor_bottom_y
        self.overlay._apply_updated_config(new_config)
        self.app.processEvents()

        # Queue should now have only 2 cards
        self.assertEqual(self.overlay.active_card_count, 2)
        self.assertEqual(self.overlay.cards[0].translated, "二")
        self.assertEqual(self.overlay.cards[1].translated, "三")

        # Original text should be hidden as per config
        self.assertFalse(self.overlay.cards[0].original_label.isVisible())

        # Bottom anchor should be maintained
        self.assertEqual(self.overlay.y() + self.overlay.height(), old_bottom)

    def test_speaker_badge_color_identity(self):
        """Verify that speaker badges use deterministic color matching get_speaker_color."""
        self.overlay._start_fade_out()
        self.overlay.show_subtitle("講者 1", "Hi", "嗨")
        card = self.overlay.cards[0]
        color1 = get_speaker_color("講者 1")
        self.assertIn(color1, card.badge.styleSheet())

        self.overlay.show_subtitle("講者 2", "Hi 2", "嗨二")
        card2 = self.overlay.cards[1]
        color2 = get_speaker_color("講者 2")
        self.assertIn(color2, card2.badge.styleSheet())
        self.assertNotEqual(color1, color2)

    def test_backward_compatible_properties(self):
        """Verify that translated_text, speaker_label, and original_text properties work properly."""
        self.overlay._start_fade_out()

        # When empty, translated_text returns idle_label
        self.assertEqual(self.overlay.translated_text.text(), "● 待機中，等待說話聲音...")

        # Add subtitle
        self.overlay.show_subtitle("講者 5", "Testing compatibility", "相容性測試")
        self.assertEqual(self.overlay.translated_text.text(), "相容性測試")
        self.assertEqual(self.overlay.speaker_label.text(), "講者 5")
        self.assertEqual(self.overlay.original_text.text(), "Testing compatibility")

        # Add another subtitle
        self.overlay.show_subtitle("講者 6", "Second message", "第二則訊息")
        self.assertEqual(self.overlay.translated_text.text(), "第二則訊息")
        self.assertEqual(self.overlay.speaker_label.text(), "講者 6")
        self.assertEqual(self.overlay.original_text.text(), "Second message")


if __name__ == "__main__":
    unittest.main()
