"""Comprehensive tests for Persistent Voiceprint Profiles and Custom Speaker Identification (Issue #11).

Tests:
1. VoiceprintDatabase persistence, atomic save, and JSON roundtrip.
2. Zero-shot bootstrapping: pre-enrolled speaker is immediately identified by name.
3. Anchor protection & anti-drift engine:
   - Golden anchor vector protected when duration < 1.5s or similarity < 0.86.
   - Gentle assimilation (0.95 anchor + 0.05 new) when duration >= 1.5s and similarity >= 0.86.
4. Renaming promotes temporary speakers to pinned persistent profiles and saves to disk.
5. REST API endpoints: GET /api/speakers, POST /api/speakers/{id}/rename, DELETE /api/speakers/{id}.
6. HUD SubtitleCardWidget SpeakerBadge click and rename signal propagation.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np

from diarizeflow.app.config import AppConfig, DiarizationConfig
from diarizeflow.app.backend.voiceprint import SpeakerProfile, VoiceprintDatabase
from diarizeflow.app.backend.diarizer import NemotronDiarizer
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline
from diarizeflow.app.backend.server import create_app
from fastapi.testclient import TestClient
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from diarizeflow.app.frontend.desktop_overlay import (
    SpeakerBadge,
    SubtitleCardWidget,
    TransparentSubtitleOverlay,
    get_speaker_color,
)


class TestVoiceprintDatabase(unittest.TestCase):
    """Test VoiceprintDatabase disk operations, serialization, and profile management."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "speakers" / "profiles.json"
        self.db = VoiceprintDatabase(storage_path=self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_and_load_profiles(self):
        # Create a mock 512-dim normalized vector
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        profile = SpeakerProfile(
            id="spk_aimi_001",
            name="Aimi",
            color="#f472b6",
            is_pinned=True,
            sample_count=5,
            anchor_emb=vec,
        )
        self.db.add_or_update(profile)

        # File should exist on disk
        self.assertTrue(self.db_path.exists())

        # Load fresh database instance from the same path
        new_db = VoiceprintDatabase(storage_path=self.db_path)
        loaded = new_db.find_profile("spk_aimi_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Aimi")
        self.assertEqual(loaded.color, "#f472b6")
        self.assertTrue(loaded.is_pinned)
        self.assertEqual(loaded.sample_count, 5)
        self.assertEqual(len(loaded.anchor_emb), 512)
        # Cosine similarity should be ~1.0
        sim = float(np.dot(vec, loaded.anchor_emb))
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_unpinned_profiles_not_persisted_to_disk(self):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        profile = SpeakerProfile(
            id="spk_temp_001",
            name="講者 1",
            is_pinned=False,
            anchor_emb=vec,
        )
        self.db.add_or_update(profile)

        # Since is_pinned=False, file on disk shouldn't have it
        new_db = VoiceprintDatabase(storage_path=self.db_path)
        self.assertIsNone(new_db.find_profile("spk_temp_001"))

    def test_delete_profile(self):
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)

        profile = SpeakerProfile(
            id="spk_john_001",
            name="John",
            is_pinned=True,
            anchor_emb=vec,
        )
        self.db.add_or_update(profile)
        self.assertIsNotNone(self.db.find_profile("John"))

        deleted = self.db.delete("John")
        self.assertTrue(deleted)
        self.assertIsNone(self.db.find_profile("John"))

        # Re-check loaded from disk
        new_db = VoiceprintDatabase(storage_path=self.db_path)
        self.assertIsNone(new_db.find_profile("John"))


class TestNemotronZeroShotAndAntiDrift(unittest.TestCase):
    """Test NemotronDiarizer zero-shot bootstrapping and anti-drift anchor protection."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "profiles.json"
        self.config = DiarizationConfig(
            profiles_path=str(self.db_path),
            speaker_threshold=0.82,
            anti_drift_min_duration=1.5,
            anti_drift_min_similarity=0.86,
            anti_drift_alpha=0.05,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_zero_shot_bootstrapping(self):
        """When an enrolled speaker exists in profiles.json, they are identified immediately."""
        # Pre-enroll Aimi with a known vector
        np.random.seed(42)
        aimi_vec = np.random.randn(512).astype(np.float32)
        aimi_vec /= np.linalg.norm(aimi_vec)

        db = VoiceprintDatabase(storage_path=self.db_path)
        db.add_or_update(SpeakerProfile(
            id="spk_aimi",
            name="Aimi",
            is_pinned=True,
            sample_count=1,
            anchor_emb=aimi_vec,
        ))

        # Start diarizer with empty ONNX session (acoustic/direct embedding matching)
        diarizer = NemotronDiarizer(self.config)

        # Create audio vector identical to Aimi's embedding
        query_vec = aimi_vec.copy()
        label, sim = diarizer._match_or_register_speaker(query_vec, engine_tag="Test", duration_s=1.0)

        # Should immediately identify as Aimi, NOT "講者 1"
        self.assertEqual(label, "Aimi")
        self.assertGreaterEqual(sim, 0.99)

        # Now test an unknown speaker
        unknown_vec = np.random.randn(512).astype(np.float32)
        unknown_vec /= np.linalg.norm(unknown_vec)
        # Ensure orthogonal/low similarity
        while np.dot(aimi_vec, unknown_vec) > 0.3:
            unknown_vec = np.random.randn(512).astype(np.float32)
            unknown_vec /= np.linalg.norm(unknown_vec)

        label2, sim2 = diarizer._match_or_register_speaker(unknown_vec, engine_tag="Test", duration_s=1.0)
        self.assertEqual(label2, "講者 1")
        self.assertEqual(sim2, 1.0)

    def test_anchor_protection_against_short_or_low_conf_drift(self):
        """Short utterances (<1.5s) or lower similarity (<0.86) do NOT drift the golden anchor vector."""
        np.random.seed(100)
        anchor_vec = np.random.randn(512).astype(np.float32)
        anchor_vec /= np.linalg.norm(anchor_vec)

        db = VoiceprintDatabase(storage_path=self.db_path)
        db.add_or_update(SpeakerProfile(
            id="spk_pinned_01",
            name="Host",
            is_pinned=True,
            sample_count=1,
            anchor_emb=anchor_vec.copy(),
        ))

        diarizer = NemotronDiarizer(self.config)
        orig_anchor = diarizer.voiceprint_db.find_profile("Host").anchor_emb.copy()

        # Case 1: High similarity (0.95), but short duration (0.8s < 1.5s)
        ortho1 = np.random.randn(512).astype(np.float32)
        ortho1 -= np.dot(ortho1, anchor_vec) * anchor_vec
        ortho1 /= np.linalg.norm(ortho1)
        target_sim1 = 0.95
        noisy_vec = target_sim1 * anchor_vec + np.sqrt(1.0 - target_sim1**2) * ortho1
        noisy_vec /= np.linalg.norm(noisy_vec)
        label, sim = diarizer._match_or_register_speaker(noisy_vec, duration_s=0.8)

        self.assertEqual(label, "Host")
        cur_anchor = diarizer.voiceprint_db.find_profile("Host").anchor_emb
        # Vector must remain completely unchanged (sim == 1.0 with original)
        self.assertAlmostEqual(float(np.dot(orig_anchor, cur_anchor)), 1.0, places=5)

        # Case 2: Sufficient duration (2.0s >= 1.5s), but lower confidence (0.83 < 0.86)
        # Construct vector with cosine similarity ~0.83
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= np.dot(ortho, anchor_vec) * anchor_vec
        ortho /= np.linalg.norm(ortho)
        target_sim = 0.83
        vec_83 = target_sim * anchor_vec + np.sqrt(1.0 - target_sim**2) * ortho
        vec_83 /= np.linalg.norm(vec_83)

        label, sim = diarizer._match_or_register_speaker(vec_83, duration_s=2.0)
        self.assertEqual(label, "Host")
        cur_anchor2 = diarizer.voiceprint_db.find_profile("Host").anchor_emb
        # Must still be protected!
        self.assertAlmostEqual(float(np.dot(orig_anchor, cur_anchor2)), 1.0, places=5)

    def test_anti_drift_gentle_assimilation(self):
        """When duration >= 1.5s AND similarity >= 0.86, anchor vector gently updates."""
        np.random.seed(200)
        anchor_vec = np.random.randn(512).astype(np.float32)
        anchor_vec /= np.linalg.norm(anchor_vec)

        db = VoiceprintDatabase(storage_path=self.db_path)
        db.add_or_update(SpeakerProfile(
            id="spk_pinned_02",
            name="Alice",
            is_pinned=True,
            sample_count=1,
            anchor_emb=anchor_vec.copy(),
        ))

        diarizer = NemotronDiarizer(self.config)
        orig_anchor = diarizer.voiceprint_db.find_profile("Alice").anchor_emb.copy()

        # Construct vector with high similarity 0.90
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= np.dot(ortho, anchor_vec) * anchor_vec
        ortho /= np.linalg.norm(ortho)
        target_sim = 0.90
        vec_90 = target_sim * anchor_vec + np.sqrt(1.0 - target_sim**2) * ortho
        vec_90 /= np.linalg.norm(vec_90)

        # Duration 2.5s >= 1.5s, Sim ~0.90 >= 0.86
        label, sim = diarizer._match_or_register_speaker(vec_90, duration_s=2.5)
        self.assertEqual(label, "Alice")
        self.assertGreaterEqual(sim, 0.89)

        updated_anchor = diarizer.voiceprint_db.find_profile("Alice").anchor_emb
        # Expected new vector: 0.95 * orig + 0.05 * vec_90, normalized
        expected = 0.95 * orig_anchor + 0.05 * vec_90
        expected /= np.linalg.norm(expected)

        self.assertAlmostEqual(float(np.dot(expected, updated_anchor)), 1.0, places=5)

        # Verify disk was updated
        disk_db = VoiceprintDatabase(storage_path=self.db_path)
        disk_prof = disk_db.find_profile("Alice")
        self.assertAlmostEqual(float(np.dot(expected, disk_prof.anchor_emb)), 1.0, places=4)

    def test_renaming_promotes_temporary_speaker_to_pinned_profile(self):
        """Renaming '講者 1' to 'Charlie' pins profile and saves to profiles.json."""
        diarizer = NemotronDiarizer(self.config)

        # Register temporary speaker
        np.random.seed(300)
        vec1 = np.random.randn(512).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)
        label, _ = diarizer._match_or_register_speaker(vec1, duration_s=1.0)
        self.assertEqual(label, "講者 1")

        # Now rename "講者 1" to "Charlie"
        success = diarizer.rename_speaker("講者 1", "Charlie", color="#38bdf8")
        self.assertTrue(success)

        # Profile should be pinned in memory and on disk
        prof = diarizer.voiceprint_db.find_profile("Charlie")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.name, "Charlie")
        self.assertEqual(prof.color, "#38bdf8")
        self.assertTrue(prof.is_pinned)

        # Verify persistence on disk
        disk_db = VoiceprintDatabase(storage_path=self.db_path)
        disk_prof = disk_db.find_profile("Charlie")
        self.assertIsNotNone(disk_prof)
        self.assertEqual(disk_prof.name, "Charlie")

        # Diarizer reset should keep Charlie
        diarizer.reset()
        label_after_reset, _ = diarizer._match_or_register_speaker(vec1, duration_s=1.0)
        self.assertEqual(label_after_reset, "Charlie")


class TestSpeakerRESTAPI(unittest.TestCase):
    """Test FastAPI /api/speakers REST endpoints."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "profiles.json"

        cfg = AppConfig()
        cfg.diarization.profiles_path = str(self.db_path)
        self.pipeline = DiarizeFlowPipeline(cfg)

        # Seed an enrolled speaker
        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        self.pipeline.diarizer.voiceprint_db.add_or_update(
            SpeakerProfile(
                id="spk_david",
                name="David",
                color="#c084fc",
                is_pinned=True,
                anchor_emb=vec,
            )
        )

        app = create_app(cfg, self.pipeline)
        self.client = TestClient(app)

    def tearDown(self):
        self.pipeline.stop()
        self.tmp_dir.cleanup()

    def test_get_speakers(self):
        res = self.client.get("/api/speakers")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("profiles", data)
        self.assertGreaterEqual(data["count"], 1)
        names = [p["name"] for p in data["profiles"]]
        self.assertIn("David", names)

    def test_rename_speaker(self):
        # Rename David to Dave
        res = self.client.post("/api/speakers/spk_david/rename", json={"name": "Dave", "color": "#10b981"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["profile"]["name"], "Dave")
        self.assertEqual(data["profile"]["color"], "#10b981")

        # Verify renaming reflected in GET /api/speakers
        get_res = self.client.get("/api/speakers")
        names = [p["name"] for p in get_res.json()["profiles"]]
        self.assertIn("Dave", names)
        self.assertNotIn("David", names)

    def test_rename_nonexistent_speaker_returns_404(self):
        res = self.client.post("/api/speakers/nonexistent_id/rename", json={"name": "Nobody"})
        self.assertEqual(res.status_code, 404)

    def test_delete_speaker(self):
        res = self.client.delete("/api/speakers/spk_david")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "deleted")

        get_res = self.client.get("/api/speakers")
        self.assertEqual(get_res.json()["count"], 0)

    def test_delete_nonexistent_speaker_returns_404(self):
        res = self.client.delete("/api/speakers/nonexistent_id")
        self.assertEqual(res.status_code, 404)


class TestHUDOverlaySpeakerRename(unittest.TestCase):
    """Test PySide6 HUD SpeakerBadge interaction and overlay renaming propagation."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_speaker_badge_click_and_rename_signal(self):
        card = SubtitleCardWidget(
            speaker="講者 1",
            original="Hello",
            translated="你好",
            confidence=0.95,
        )
        self.assertIsInstance(card.badge, SpeakerBadge)

        # Test rename_requested signal emission
        emitted_speakers = []
        card.rename_requested.connect(lambda spk: emitted_speakers.append(spk))

        # Simulate left-click on badge
        card.badge.clicked.emit()
        self.assertEqual(len(emitted_speakers), 1)
        self.assertEqual(emitted_speakers[0], "講者 1")

        # Test updating speaker label on card
        card.update_speaker("Aimi", "#f472b6")
        self.assertEqual(card.speaker, "Aimi")
        self.assertEqual(card.badge.text(), "Aimi")
        self.assertIn("#f472b6", card.badge.styleSheet())

    def test_overlay_rename_speaker_updates_all_cards(self):
        overlay = TransparentSubtitleOverlay(
            enable_network=False,
            auto_start_capture=False,
        )

        # Show multiple subtitle cards
        overlay.show_subtitle("講者 1", "Utterance 1", "發言 1")
        overlay.show_subtitle("講者 2", "Utterance 2", "發言 2")
        overlay.show_subtitle("講者 1", "Utterance 3", "發言 3")

        self.assertEqual(len(overlay.cards), 3)

        # Rename "講者 1" to "Alice"
        overlay.rename_speaker("講者 1", "Alice", "#34d399")

        # Both cards with "講者 1" should now be "Alice"
        speakers = [c.speaker for c in overlay.cards]
        self.assertEqual(speakers, ["Alice", "講者 2", "Alice"])
        self.assertEqual(overlay.cards[0].badge.text(), "Alice")
        self.assertEqual(overlay.cards[2].badge.text(), "Alice")

        overlay.close()


if __name__ == "__main__":
    unittest.main()
