"""Unit tests for AppConfig serialization and deserialization (Issue #2)."""

import json
import tempfile
import unittest
from pathlib import Path

from diarizeflow.app.config import AppConfig


class TestAppConfigSerialization(unittest.TestCase):
    """Test that AppConfig properly serializes and deserializes all configuration fields."""

    def test_from_dict_preserves_hardware_calibrated_true(self):
        """Verify AppConfig.from_dict restores hardware_calibrated when True."""
        data = {
            "hardware_calibrated": True,
        }
        cfg = AppConfig.from_dict(data)
        self.assertTrue(
            cfg.hardware_calibrated,
            "AppConfig.from_dict dropped hardware_calibrated=True!",
        )

    def test_from_dict_defaults_hardware_calibrated_false(self):
        """Verify AppConfig.from_dict defaults hardware_calibrated to False when absent."""
        data = {}
        cfg = AppConfig.from_dict(data)
        self.assertFalse(cfg.hardware_calibrated)

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verify round-trip serialization preserves hardware_calibrated state."""
        original = AppConfig(hardware_calibrated=True)
        as_dict = original.to_dict()
        self.assertIn("hardware_calibrated", as_dict)
        self.assertTrue(as_dict["hardware_calibrated"])

        reconstructed = AppConfig.from_dict(as_dict)
        self.assertEqual(
            reconstructed.hardware_calibrated,
            original.hardware_calibrated,
            "Roundtrip serialization failed to preserve hardware_calibrated",
        )

    def test_disk_save_and_load_roundtrip(self):
        """Verify saving to disk and loading preserves hardware_calibrated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.json"
            cfg = AppConfig(hardware_calibrated=True)
            cfg.save(filepath=str(config_file))

            self.assertTrue(config_file.exists())
            with open(config_file, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            self.assertTrue(raw_json.get("hardware_calibrated"))

            loaded = AppConfig.load(filepath=str(config_file))
            self.assertTrue(
                loaded.hardware_calibrated,
                "Loading config from disk dropped hardware_calibrated flag!",
            )

    def test_diarization_sad_threshold_roundtrip(self):
        """Verify sad_threshold in DiarizationConfig is preserved in roundtrip."""
        cfg = AppConfig()
        cfg.diarization.sad_threshold = 0.42
        d = cfg.to_dict()
        loaded = AppConfig.from_dict(d)
        self.assertEqual(loaded.diarization.sad_threshold, 0.42)

    def test_vad_padding_roundtrip(self):
        """Verify pre_pad_ms and post_pad_ms in VADConfig are preserved in roundtrip."""
        cfg = AppConfig()
        self.assertEqual(cfg.vad.pre_pad_ms, 150)
        self.assertEqual(cfg.vad.post_pad_ms, 150)
        cfg.vad.pre_pad_ms = 220
        cfg.vad.post_pad_ms = 180
        d = cfg.to_dict()
        loaded = AppConfig.from_dict(d)
        self.assertEqual(loaded.vad.pre_pad_ms, 220)
        self.assertEqual(loaded.vad.post_pad_ms, 180)

    def test_default_config_json_contents(self):
        """Verify config.json includes sad_threshold, pre_pad_ms, post_pad_ms, and concurrency_limit."""
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("sad_threshold", data.get("diarization", {}))
        self.assertEqual(data["diarization"]["sad_threshold"], 0.50)
        self.assertIn("pre_pad_ms", data.get("vad", {}))
        self.assertEqual(data["vad"]["pre_pad_ms"], 150)
        self.assertIn("post_pad_ms", data.get("vad", {}))
        self.assertEqual(data["vad"]["post_pad_ms"], 150)
        self.assertIn("concurrency_limit", data.get("llm", {}))
        self.assertEqual(data["llm"]["concurrency_limit"], 3)

    def test_llm_concurrency_limit_roundtrip(self):
        """Verify concurrency_limit in LLMConfig is preserved in roundtrip."""
        cfg = AppConfig()
        self.assertEqual(cfg.llm.concurrency_limit, 3)
        cfg.llm.concurrency_limit = 5
        d = cfg.to_dict()
        loaded = AppConfig.from_dict(d)
        self.assertEqual(loaded.llm.concurrency_limit, 5)


if __name__ == "__main__":
    unittest.main()


