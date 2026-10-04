"""Unit tests for models catalog, auto-downloading, and verification (Issue #51).

Verifies:
1. Model catalog specifications and URL resolution across Hugging Face, ModelScope, and mirrors.
2. Hardware precision detection and model filtering (auto, fp16, int8, fp32, all).
3. Checksum calculation and verification.
4. Resumable download handling (HTTP Range headers, 206 Partial Content).
5. Existing file skip and force overwrite logic.
6. check_missing_models detection logic for launcher startup guidance.
7. Dry-run mode and CLI argument parsing.
"""

import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from diarizeflow.cli.downloader import (
    MODEL_CATALOG,
    ModelFileSpec,
    calculate_sha256,
    check_missing_models,
    detect_recommended_precision,
    download_file,
    download_models,
    format_size,
    get_target_model_specs,
    render_progress_bar,
    resolve_model_url,
)


class TestDownloadModelsCatalogAndResolution(unittest.TestCase):
    """Test suite verifying model catalog structure, URLs, and mirror resolution."""

    def test_model_catalog_structure(self):
        """Verify MODEL_CATALOG entries are properly formed and contain required keys."""
        self.assertGreater(len(MODEL_CATALOG), 0)
        for spec in MODEL_CATALOG:
            self.assertIsInstance(spec.rel_path, str)
            self.assertIn(spec.category, ("sensevoice", "nemotron"))
            self.assertIn(spec.precision, ("all", "fp32", "fp16", "int8"))
            self.assertIn("huggingface", spec.urls)
            self.assertTrue(spec.urls["huggingface"].startswith("http"))

    def test_resolve_model_url_sources(self):
        """Verify URL resolution for Hugging Face and ModelScope sources."""
        spec = MODEL_CATALOG[0]  # am.mvn

        hf_url = resolve_model_url(spec, source="huggingface")
        self.assertIn("huggingface.co", hf_url)

        ms_url = resolve_model_url(spec, source="modelscope")
        self.assertIn("modelscope.cn", ms_url)

    def test_resolve_model_url_hf_mirror(self):
        """Verify Hugging Face mirror URL rewriting via parameter or HF_ENDPOINT env."""
        spec = MODEL_CATALOG[0]

        # 1. Via hf_mirror argument
        mirrored_url = resolve_model_url(spec, source="huggingface", hf_mirror="https://hf-mirror.com")
        self.assertTrue(mirrored_url.startswith("https://hf-mirror.com/"))
        self.assertNotIn("huggingface.co", mirrored_url)

        # 2. Via HF_ENDPOINT environment variable
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://custom-mirror.example.com"}):
            env_url = resolve_model_url(spec, source="huggingface")
            self.assertTrue(env_url.startswith("https://custom-mirror.example.com/"))

    def test_format_size(self):
        """Verify format_size correctly handles bytes, KB, MB, and GB."""
        self.assertEqual(format_size(500), "500 B")
        self.assertEqual(format_size(2048), "2.0 KB")
        self.assertEqual(format_size(10 * 1024 * 1024), "10.0 MB")
        self.assertEqual(format_size(2 * 1024 * 1024 * 1024), "2.00 GB")

    def test_render_progress_bar(self):
        """Verify progress bar rendering does not crash and formats percentages."""
        bar = render_progress_bar("test.onnx", 50, 100, 1.0)
        self.assertIn("50.0%", bar)
        self.assertIn("test.onnx", bar or "test.onnx")

        finished_bar = render_progress_bar("test.onnx", 100, 100, 1.0, finished=True)
        self.assertIn("100.0%", finished_bar)
        self.assertIn("完成", finished_bar)


class TestModelSelectionAndFiltering(unittest.TestCase):
    """Test suite verifying precision and model category filtering."""

    def test_get_target_model_specs_precision_filtering(self):
        """Verify filtering by precision includes precision-specific and 'all' precision files."""
        int8_specs = get_target_model_specs(models="all", precision="int8")
        int8_paths = [s.rel_path for s in int8_specs]

        # Should include common assets
        self.assertIn("sensevoice_small/am.mvn", int8_paths)
        self.assertIn("sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model", int8_paths)
        # Should include INT8 models
        self.assertIn("sensevoice_small/SenseVoiceSmall_int8.onnx", int8_paths)
        self.assertIn("nemotron_diarization/Nemotron-3-Diarization_int8.onnx", int8_paths)
        # Should NOT include FP16 models
        self.assertNotIn("sensevoice_small/SenseVoiceSmall_fp16.onnx", int8_paths)
        self.assertNotIn("nemotron_diarization/Nemotron-3-Diarization_fp16.onnx", int8_paths)

        # FP16 filtering
        fp16_specs = get_target_model_specs(models="all", precision="fp16")
        fp16_paths = [s.rel_path for s in fp16_specs]
        self.assertIn("sensevoice_small/SenseVoiceSmall_fp16.onnx", fp16_paths)
        self.assertIn("nemotron_diarization/Nemotron-3-Diarization_fp16.onnx", fp16_paths)
        self.assertNotIn("sensevoice_small/SenseVoiceSmall_int8.onnx", fp16_paths)

    def test_get_target_model_specs_category_filtering(self):
        """Verify filtering by category (sensevoice vs nemotron)."""
        sv_specs = get_target_model_specs(models="sensevoice", precision="all")
        for s in sv_specs:
            self.assertEqual(s.category, "sensevoice")

        nemo_specs = get_target_model_specs(models="nemotron", precision="all")
        for s in nemo_specs:
            self.assertEqual(s.category, "nemotron")

    def test_detect_recommended_precision(self):
        """Verify detect_recommended_precision returns fp16 or int8."""
        prec = detect_recommended_precision()
        self.assertIn(prec, ("fp16", "int8"))


class TestMissingModelsDetection(unittest.TestCase):
    """Test suite verifying check_missing_models in empty and populated directories."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.target_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_check_missing_models_in_empty_dir(self):
        """Empty directory should report missing SenseVoice and Nemotron models."""
        missing = check_missing_models(target_dir=self.target_dir)
        self.assertGreater(len(missing), 0)
        # Should mention CMVN, tokenizer, SenseVoice model, Nemotron model
        missing_text = " ".join(missing)
        self.assertIn("CMVN", missing_text)
        self.assertIn("分詞模型", missing_text)
        self.assertIn("SenseVoiceSmall", missing_text)
        self.assertIn("Nemotron", missing_text)

    def test_check_missing_models_when_all_present(self):
        """When all required model files are present, check_missing_models returns empty list."""
        sv_dir = self.target_dir / "sensevoice_small"
        nemo_dir = self.target_dir / "nemotron_diarization"
        sv_dir.mkdir(parents=True, exist_ok=True)
        nemo_dir.mkdir(parents=True, exist_ok=True)

        (sv_dir / "am.mvn").write_text("dummy cmvn")
        (sv_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model").write_text("dummy bpe")
        (sv_dir / "SenseVoiceSmall_int8.onnx").write_bytes(b"dummy onnx")
        (nemo_dir / "Nemotron-3-Diarization_int8.onnx").write_bytes(b"dummy nemo")

        missing = check_missing_models(target_dir=self.target_dir)
        self.assertEqual(missing, [])


class TestDownloadFileLogic(unittest.TestCase):
    """Test suite verifying resumable downloading, SHA256 integrity, and file writing."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.target_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_calculate_sha256(self):
        """Verify calculate_sha256 accurately computes SHA-256."""
        test_file = self.target_dir / "sample.txt"
        content = b"DiarizeFlow Model Verification Test String"
        test_file.write_bytes(content)

        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(calculate_sha256(test_file), expected_hash)

    def test_download_file_skip_existing(self):
        """Existing file without --force should be skipped without calling network."""
        target = self.target_dir / "existing.bin"
        content = b"Existing file data"
        target.write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()

        mock_session = MagicMock()
        # Should not make any GET calls
        ok = download_file(
            url="http://example.com/existing.bin",
            target_path=target,
            expected_sha256=expected_hash,
            force=False,
            session=mock_session,
            quiet=True,
        )
        self.assertTrue(ok)
        mock_session.get.assert_not_called()

    def test_download_file_mock_network_success(self):
        """Verify successful download writes atomic file and validates checksum."""
        target = self.target_dir / "model.onnx"
        content = b"Simulated ONNX Model Binary Payload" * 100
        content_hash = hashlib.sha256(content).hexdigest()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": str(len(content))}
        mock_response.iter_content = MagicMock(return_value=[content[:100], content[100:]])

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        ok = download_file(
            url="http://example.com/model.onnx",
            target_path=target,
            expected_sha256=content_hash,
            force=True,
            session=mock_session,
            quiet=True,
        )

        self.assertTrue(ok)
        self.assertTrue(target.exists())
        self.assertFalse(target.with_name("model.onnx.part").exists())
        self.assertEqual(target.read_bytes(), content)

    def test_download_file_resumable_range(self):
        """Verify partial .part file triggers Range request and appends remainder."""
        target = self.target_dir / "resumable.onnx"
        part_file = target.with_name("resumable.onnx.part")

        initial_chunk = b"Initial 100 bytes of partial download. " * 3  # len = 117
        remainder_chunk = b"Remainder bytes after connection drop." * 2

        part_file.write_bytes(initial_chunk)
        initial_len = len(initial_chunk)

        mock_response = MagicMock()
        mock_response.status_code = 206  # Partial Content
        mock_response.headers = {
            "Content-Range": f"bytes {initial_len}-{initial_len + len(remainder_chunk) - 1}/{initial_len + len(remainder_chunk)}",
            "Content-Length": str(len(remainder_chunk)),
        }
        mock_response.iter_content = MagicMock(return_value=[remainder_chunk])

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        ok = download_file(
            url="http://example.com/resumable.onnx",
            target_path=target,
            session=mock_session,
            quiet=True,
        )

        self.assertTrue(ok)
        self.assertTrue(target.exists())
        # Range header must have been passed with initial_len
        call_kwargs = mock_session.get.call_args[1]
        self.assertEqual(call_kwargs["headers"]["Range"], f"bytes={initial_len}-")
        self.assertEqual(target.read_bytes(), initial_chunk + remainder_chunk)

    def test_download_file_sha256_mismatch_fails_and_cleans_up(self):
        """Verify sha256 mismatch deletes .part file and returns False."""
        target = self.target_dir / "corrupted.onnx"
        content = b"Corrupted bytes"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": str(len(content))}
        mock_response.iter_content = MagicMock(return_value=[content])

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        ok = download_file(
            url="http://example.com/corrupted.onnx",
            target_path=target,
            expected_sha256=wrong_hash,
            session=mock_session,
            quiet=True,
        )

        self.assertFalse(ok)
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name("corrupted.onnx.part").exists())

    def test_download_models_dry_run(self):
        """Verify download_models in dry_run mode does not create files and returns True."""
        ok = download_models(
            models="sensevoice",
            precision="int8",
            target_dir=self.target_dir,
            dry_run=True,
            quiet=True,
        )
        self.assertTrue(ok)
        # Target dir should not have been created or modified
        self.assertEqual(list(self.target_dir.glob("**/*")), [])

    def test_launcher_displays_missing_models_prompt_when_models_absent(self):
        """Verify launcher.run_cli outputs friendly guidance when models are missing."""
        import sys
        from diarizeflow.cli.launcher import run_cli

        test_cfg_path = str(self.target_dir / "test_config.json")
        test_args = ["diarizeflow-app", "--config", test_cfg_path]
        with patch.object(sys, "argv", test_args), \
             patch("diarizeflow.cli.launcher.setup_global_logging"), \
             patch("diarizeflow.cli.launcher.check_missing_models", return_value=["SenseVoiceSmall 語音辨識 ONNX 模型"]), \
             patch("diarizeflow.core.calibration.ensure_calibrated_models"), \
             patch("diarizeflow.cli.launcher.DiarizeFlowPipeline"), \
             patch("diarizeflow.cli.launcher.create_app"), \
             patch("diarizeflow.cli.launcher.start_server_in_thread", return_value=(MagicMock(), MagicMock())), \
             patch("diarizeflow.ui.desktop_overlay.run_overlay_app", return_value=0), \
             patch("builtins.print") as mock_print:

            code = run_cli()
            self.assertEqual(code, 0)

            # Check that the guidance message was printed
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
            has_guidance = any("scripts/download_models.py" in msg for msg in printed_messages)
            self.assertTrue(has_guidance, f"Expected download_models.py prompt in printed messages: {printed_messages}")


if __name__ == "__main__":
    unittest.main()
