"""Unit and integration tests for remote configuration synchronization via REST API (Issue #32).

Verifies:
1. FastAPI backend POST /api/config receives and applies new configuration to pipeline and components.
2. FastAPI backend GET /api/config reflects the active configuration.
3. Error handling in POST /api/config for invalid payloads.
4. TransparentSubtitleOverlay in standalone mode (_apply_updated_config with pipeline=None)
   pushes configuration to remote backend via POST /api/config.
5. In-process mode directly updates pipeline without network HTTP calls.
6. TransparentSubtitleOverlay pulls and aligns remote configuration via GET /api/config
   upon connection without triggering circular echo updates.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from diarizeflow.app.config import AppConfig
from diarizeflow.app.backend.server import create_app
from diarizeflow.app.frontend.desktop_overlay import TransparentSubtitleOverlay


def get_or_create_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestBackendConfigAPI(unittest.TestCase):
    """Test suite verifying backend FastAPI /api/config REST endpoints."""

    def setUp(self):
        self.config = AppConfig()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.config = self.config
        self.mock_pipeline.is_running = True
        self.app = create_app(self.config, self.mock_pipeline)
        self.client = TestClient(self.app)

    def test_get_config_returns_active_configuration(self):
        """Verify GET /api/config returns full dictionary representation of active configuration."""
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("audio", data)
        self.assertIn("asr", data)
        self.assertIn("llm", data)
        self.assertIn("ui", data)
        self.assertIn("vad", data)
        self.assertIn("diarization", data)
        self.assertEqual(data["llm"]["target_language"], self.config.llm.target_language)

    def test_post_config_updates_pipeline_configuration(self):
        """Verify POST /api/config updates pipeline and returns updated configuration."""
        update_payload = {
            "llm": {
                "provider": "vllm",
                "target_language": "日文",
                "base_url": "http://192.168.1.100:8000/v1",
                "model_name": "qwen2.5:7b",
            },
            "asr": {
                "engine": "whisper",
                "whisper_precision": "int8",
            },
            "diarization": {
                "speaker_threshold": 0.88,
            },
        }

        resp = self.client.post("/api/config", json=update_payload)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertEqual(res_data["status"], "updated")
        self.assertEqual(res_data["config"]["llm"]["target_language"], "日文")
        self.assertEqual(res_data["config"]["asr"]["engine"], "whisper")
        self.assertEqual(res_data["config"]["diarization"]["speaker_threshold"], 0.88)

        # Assert pipeline.update_config was invoked with the parsed AppConfig
        self.mock_pipeline.update_config.assert_called_once()
        passed_cfg = self.mock_pipeline.update_config.call_args[0][0]
        self.assertIsInstance(passed_cfg, AppConfig)
        self.assertEqual(passed_cfg.llm.target_language, "日文")
        self.assertEqual(passed_cfg.asr.engine, "whisper")
        self.assertEqual(passed_cfg.diarization.speaker_threshold, 0.88)

    def test_post_config_handles_malformed_payload(self):
        """Verify POST /api/config handles malformed payloads gracefully with 400 Bad Request."""
        with patch("diarizeflow.app.config.AppConfig.from_dict", side_effect=ValueError("Corrupted data")):
            resp = self.client.post("/api/config", json={"audio": "not-a-dict"})
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Invalid configuration payload", resp.json()["detail"])


class TestFrontendRemoteConfigSync(unittest.TestCase):
    """Test suite verifying TransparentSubtitleOverlay remote config push and pull behavior."""

    @classmethod
    def setUpClass(cls):
        cls.qapp = get_or_create_qapp()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.config = AppConfig()
        self.config.server.port = 8765

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_apply_updated_config_in_standalone_mode_pushes_post_api(self):
        """Verify _apply_updated_config in standalone mode pushes new configuration via POST /api/config."""
        hud = TransparentSubtitleOverlay(
            config=self.config,
            pipeline=None,  # Standalone remote frontend mode
            enable_network=True,
            auto_start_capture=False,
        )

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

        new_cfg = AppConfig()
        new_cfg.llm.target_language = "德文"
        new_cfg.diarization.speaker_threshold = 0.92

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            hud._apply_updated_config(new_cfg)

            # Wait for background thread to complete
            import time
            for _ in range(20):
                if len(sent_requests) >= 1:
                    break
                time.sleep(0.05)

        self.assertEqual(len(sent_requests), 1)
        req = sent_requests[0]
        self.assertEqual(req.method, "POST")
        self.assertTrue(req.full_url.endswith("/api/config"))
        self.assertEqual(req.headers.get("Content-type"), "application/json")

        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["llm"]["target_language"], "德文")
        self.assertEqual(payload["diarization"]["speaker_threshold"], 0.92)

        hud.close()

    def test_apply_updated_config_in_process_mode_bypasses_network_post(self):
        """Verify in-process mode directly updates pipeline without making network HTTP calls."""
        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True

        hud = TransparentSubtitleOverlay(
            config=self.config,
            pipeline=mock_pipeline,  # In-process unified mode
            enable_network=True,
            auto_start_capture=False,
        )

        sent_requests = []

        def mock_urlopen(req, timeout=None):
            sent_requests.append(req)
            return None

        new_cfg = AppConfig()
        new_cfg.llm.target_language = "西班牙文"

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            hud._apply_updated_config(new_cfg)
            import time
            time.sleep(0.1)

        # In-process mode directly updates pipeline
        mock_pipeline.update_config.assert_called_once_with(new_cfg)
        # MUST NOT send remote HTTP requests
        self.assertEqual(len(sent_requests), 0)

        hud.close()

    def test_fetch_remote_backend_config_aligns_frontend_without_echo(self):
        """Verify _fetch_remote_backend_config fetches GET /api/config and aligns UI without echoing."""
        hud = TransparentSubtitleOverlay(
            config=self.config,
            pipeline=None,
            enable_network=True,
            auto_start_capture=False,
        )

        remote_cfg = AppConfig()
        remote_cfg.llm.target_language = "韓文"
        remote_cfg.asr.engine = "whisper"
        remote_json = json.dumps(remote_cfg.to_dict()).encode("utf-8")

        class MockGetResponse:
            status = 200
            def read(self):
                return remote_json
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        sent_posts = []

        def mock_urlopen(req, timeout=None):
            if req.method == "GET":
                return MockGetResponse()
            else:
                sent_posts.append(req)
                return MockGetResponse()

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            t = hud._fetch_remote_backend_config()
            t.join(timeout=2.0)

            # Process Qt event loop to handle config_synced_signal
            QCoreApplication.processEvents()

        # Assert frontend local config is now updated to remote backend state
        self.assertEqual(hud.config.llm.target_language, "韓文")
        self.assertEqual(hud.config.asr.engine, "whisper")

        # CRITICAL ASSERTION: Pulling remote config must NOT echo back with a POST request!
        self.assertEqual(len(sent_posts), 0)

        hud.close()


if __name__ == "__main__":
    unittest.main()
