"""Unit tests verifying AppConfig deserialization robustness and corrupted file recovery (Issue #29).

Verifies:
1. filter_dataclass_kwargs safely removes unknown/legacy fields from dictionaries.
2. AppConfig.from_dict safely parses payloads with unknown keys across all sub-configs.
3. AppConfig.from_dict gracefully handles non-dict root and sub-config payloads.
4. AppConfig.from_dict gracefully handles invalid threshold/opacity values without throwing.
5. AppConfig.load automatically backs up corrupted/unparseable config files and falls back to defaults.
6. AppConfig.load handles non-object JSON payloads (e.g. arrays or primitives).
"""

import json
from pathlib import Path
import tempfile
import unittest

from diarizeflow.config import (
    AppConfig,
    AudioConfig,
    DiarizationConfig,
    LLMConfig,
    ServerConfig,
    UIConfig,
    VADConfig,
    filter_dataclass_kwargs,
)


class TestConfigDeserializationRobustness(unittest.TestCase):
    """Test suite for safe field filtering and resilient deserialization."""

    def test_filter_dataclass_kwargs_removes_unexpected_keys(self):
        """Verify filter_dataclass_kwargs discards extra keys and keeps valid dataclass fields."""
        payload = {
            "sample_rate": 48000,
            "channels": 2,
            "legacy_buffer_size": 1024,
            "random_custom_key": "val",
        }
        filtered = filter_dataclass_kwargs(AudioConfig, payload)
        self.assertEqual(filtered.get("sample_rate"), 48000)
        self.assertEqual(filtered.get("channels"), 2)
        self.assertNotIn("legacy_buffer_size", filtered)
        self.assertNotIn("random_custom_key", filtered)

    def test_filter_dataclass_kwargs_non_dict_input(self):
        """Verify filter_dataclass_kwargs returns empty dict when given non-dict inputs."""
        self.assertEqual(filter_dataclass_kwargs(AudioConfig, None), {})
        self.assertEqual(filter_dataclass_kwargs(AudioConfig, "string"), {})
        self.assertEqual(filter_dataclass_kwargs(AudioConfig, [1, 2, 3]), {})

    def test_from_dict_with_unknown_fields_across_all_subconfigs(self):
        """Verify from_dict succeeds without TypeError when unknown fields exist in any section."""
        polluted_data = {
            "audio": {
                "sample_rate": 16000,
                "unknown_audio_flag": True,
                "deprecated_alsa_buffer": 2048,
            },
            "vad": {
                "energy_threshold": 0.012,
                "silence_timeout_ms": 350,
                "removed_webrtc_mode": 3,
            },
            "diarization": {
                "streaming_mode": "very_low_latency",
                "speaker_threshold": 0.85,
                "obsolete_pyannote_checkpoint": "/tmp/ckpt",
            },
            "asr": {
                "engine": "sensevoice",
                "beam_size": 2,
                "experimental_temperature": 0.2,
            },
            "llm": {
                "provider": "ollama",
                "target_language": "English",
                "max_tokens": 256,
                "unknown_context_window": 8192,
            },
            "ui": {
                "max_cards": 4,
                "opacity": 0.70,
                "legacy_web_hud": False,
            },
            "server": {
                "port": 9000,
                "cors_origins": ["*"],
            },
            "root_level_extra_info": "should_be_ignored",
            "hardware_calibrated": True,
        }

        cfg = AppConfig.from_dict(polluted_data)

        # Check values were correctly parsed despite unknown keys
        self.assertEqual(cfg.audio.sample_rate, 16000)
        self.assertEqual(cfg.vad.energy_threshold, 0.012)
        self.assertEqual(cfg.diarization.streaming_mode, "very_low_latency")
        self.assertEqual(cfg.asr.beam_size, 2)
        self.assertEqual(cfg.llm.provider, "ollama")
        self.assertEqual(cfg.llm.target_language, "English")
        self.assertEqual(cfg.ui.max_cards, 4)
        self.assertEqual(cfg.server.port, 9000)
        self.assertTrue(cfg.hardware_calibrated)

    def test_from_dict_non_dict_payloads(self):
        """Verify from_dict returns default configuration on non-dict inputs."""
        cfg_none = AppConfig.from_dict(None)
        self.assertIsInstance(cfg_none, AppConfig)
        self.assertEqual(cfg_none.audio.sample_rate, 16000)

        cfg_str = AppConfig.from_dict("invalid string")
        self.assertIsInstance(cfg_str, AppConfig)

        cfg_list = AppConfig.from_dict([{"audio": {}}])
        self.assertIsInstance(cfg_list, AppConfig)

    def test_from_dict_malformed_subconfig_values(self):
        """Verify from_dict does not crash if sub-config entries are non-dicts or corrupt."""
        corrupt_subconfigs = {
            "audio": "not a dict",
            "vad": None,
            "diarization": 12345,
            "ui": {"opacity": "invalid_float_string"},
            "diarization": {"speaker_threshold": "bad_threshold"},
        }
        cfg = AppConfig.from_dict(corrupt_subconfigs)
        self.assertIsInstance(cfg.audio, AudioConfig)
        self.assertIsInstance(cfg.vad, VADConfig)
        self.assertIsInstance(cfg.diarization, DiarizationConfig)
        self.assertEqual(cfg.ui.opacity, 0.50)
        self.assertEqual(cfg.diarization.speaker_threshold, 0.82)

    def test_corrupted_config_json_file_automatic_backup_and_recovery(self):
        """Verify load() backs up corrupted JSON file and returns healthy defaults without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_file = Path(tmpdir) / "broken_config.json"
            corrupt_content = '{"audio": {"sample_rate": 16000, broken syntax here!!!'
            corrupt_file.write_text(corrupt_content, encoding="utf-8")

            # Loading corrupted config should NOT raise Exception
            loaded_cfg = AppConfig.load(filepath=str(corrupt_file))
            self.assertIsInstance(loaded_cfg, AppConfig)
            self.assertEqual(loaded_cfg.audio.sample_rate, 16000)

            # Check that a backup file was created
            backup_files = list(Path(tmpdir).glob("broken_config.corrupt_*"))
            self.assertGreaterEqual(len(backup_files), 1, "Backup file was not created for corrupted config!")
            self.assertEqual(backup_files[0].read_text(encoding="utf-8"), corrupt_content)

            # Check that the config file was restored to valid JSON
            restored_data = json.loads(corrupt_file.read_text(encoding="utf-8"))
            self.assertIn("audio", restored_data)
            self.assertEqual(restored_data["audio"]["sample_rate"], 16000)

    def test_non_object_json_file_handled_gracefully(self):
        """Verify load() handles JSON file containing an array or primitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            array_file = Path(tmpdir) / "array_config.json"
            array_file.write_text('[1, 2, 3, "unexpected"]', encoding="utf-8")

            loaded_cfg = AppConfig.load(filepath=str(array_file))
            self.assertIsInstance(loaded_cfg, AppConfig)

            backup_files = list(Path(tmpdir).glob("array_config.corrupt_*"))
            self.assertGreaterEqual(len(backup_files), 1)


if __name__ == "__main__":
    unittest.main()
