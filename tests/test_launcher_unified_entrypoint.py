"""Unit tests verifying unified launcher entrypoint logic (Issue #27).

Covers:
- DualLogger writing, flushing, clearing, and 5MB rotation.
- get_app_log_path resolution (standard vs frozen PyInstaller).
- attach_console_on_windows and setup_global_logging exception hooking.
- find_available_port conflict handling.
- ensure_calibrated_models invocation during launcher startup.
- scripts/run_app.py thin wrapper parity with diarizeflow.app.launcher.
"""

import io
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from diarizeflow.app.launcher import (
    DualLogger,
    attach_console_on_windows,
    find_available_port,
    get_app_log_path,
    main as launcher_main,
    run_cli,
    setup_global_logging,
    start_server_in_thread,
)


class TestDualLogger(unittest.TestCase):
    """Test DualLogger stream functionality, file rotation, and error safety."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "test_app.log"
        self.fallback = io.StringIO()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dual_logger_write_and_flush(self):
        logger = DualLogger(self.log_path, fallback_stream=self.fallback)
        test_msg = "Hello DiarizeFlow Unified Launcher!\n"
        written = logger.write(test_msg)
        logger.flush()

        self.assertEqual(written, len(test_msg))
        self.assertIn("Hello DiarizeFlow", self.fallback.getvalue())

        self.assertTrue(self.log_path.exists())
        log_content = self.log_path.read_text(encoding="utf-8")
        self.assertIn("Hello DiarizeFlow", log_content)
        self.assertFalse(logger.isatty())
        self.assertEqual(logger.encoding, "utf-8")

    def test_dual_logger_clear(self):
        logger = DualLogger(self.log_path, fallback_stream=self.fallback)
        logger.write("Old log entry that should be cleared.\n")
        logger.flush()
        self.assertIn("Old log entry", self.log_path.read_text(encoding="utf-8"))

        success = logger.clear()
        self.assertTrue(success)
        cleared_content = self.log_path.read_text(encoding="utf-8")
        self.assertNotIn("Old log entry", cleared_content)
        self.assertIn("日誌已清空", cleared_content)

    def test_dual_logger_rotation_over_max_bytes(self):
        # Set max_bytes to 50 bytes for fast test
        logger = DualLogger(self.log_path, fallback_stream=self.fallback, max_bytes=50)
        chunk = "A" * 60 + "\n"
        logger.write(chunk)
        logger.flush()

        # At this point, the file has > 50 bytes. Next write should trigger rotation
        next_chunk = "B" * 20 + "\n"
        logger.write(next_chunk)
        logger.flush()

        backup_path = self.log_path.with_suffix(".log.old")
        self.assertTrue(backup_path.exists(), "Old rotated log file was not created!")
        self.assertIn("AAAA", backup_path.read_text(encoding="utf-8"))
        self.assertIn("BBBB", self.log_path.read_text(encoding="utf-8"))

    def test_dual_logger_resilient_to_broken_fallback(self):
        broken_fallback = MagicMock()
        broken_fallback.write.side_effect = IOError("Simulated broken console")
        broken_fallback.flush.side_effect = IOError("Simulated broken console flush")

        logger = DualLogger(self.log_path, fallback_stream=broken_fallback)
        # Should not raise exception
        written = logger.write("Resilient write test\n")
        logger.flush()
        self.assertEqual(written, len("Resilient write test\n"))
        self.assertIn("Resilient write test", self.log_path.read_text(encoding="utf-8"))


class TestLauncherEnvironmentAndLogging(unittest.TestCase):
    """Test environment resolution, Windows console attachment, and global logging setup."""

    def test_get_app_log_path_frozen(self):
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", "/opt/diarizeflow/DiarizeFlow"):
            log_p = get_app_log_path()
            self.assertEqual(log_p, Path("/opt/diarizeflow/diarizeflow.log"))

    def test_get_app_log_path_development(self):
        with patch.object(sys, "frozen", False, create=True):
            log_p = get_app_log_path()
            self.assertEqual(log_p.name, "diarizeflow.log")
            self.assertTrue(log_p.parent.exists())

    def test_attach_console_on_windows_mocked(self):
        with patch.object(sys, "platform", "win32"):
            mock_ctypes = MagicMock()
            mock_ctypes.windll.kernel32.AttachConsole.return_value = 1
            with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
                with patch("builtins.open", return_value=io.StringIO()) as mock_open:
                    stream = attach_console_on_windows()
                    mock_ctypes.windll.kernel32.AttachConsole.assert_called_once_with(0xFFFFFFFF)
                    mock_open.assert_called_once_with("CONOUT$", "w", encoding="utf-8", errors="replace")

    def test_setup_global_logging_and_excepthook(self):
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        orig_hook = sys.excepthook

        temp_dir = tempfile.TemporaryDirectory()
        try:
            test_log = Path(temp_dir.name) / "global_test.log"
            logger_instance = setup_global_logging(log_path=test_log)

            self.assertIsInstance(sys.stdout, DualLogger)
            self.assertIsInstance(sys.stderr, DualLogger)
            self.assertEqual(sys.stdout, logger_instance)

            # Test exception hook captures unhandled exceptions into log
            try:
                raise ValueError("Test Unhandled Crash")
            except ValueError:
                exc_type, exc_val, exc_tb = sys.exc_info()
                sys.excepthook(exc_type, exc_val, exc_tb)

            logger_instance.flush()
            log_text = test_log.read_text(encoding="utf-8")
            self.assertIn("Unhandled Exception", log_text)
            self.assertIn("Test Unhandled Crash", log_text)

            # Test KeyboardInterrupt bypasses custom hook
            mock_orig_hook = MagicMock()
            with patch.object(sys, "__excepthook__", mock_orig_hook):
                kb_exc = KeyboardInterrupt()
                sys.excepthook(KeyboardInterrupt, kb_exc, None)
                mock_orig_hook.assert_called_once_with(KeyboardInterrupt, kb_exc, None)

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            sys.excepthook = orig_hook
            temp_dir.cleanup()


class TestPortDiscoveryAndServer(unittest.TestCase):
    """Test port conflict detection and server launching."""

    def test_find_available_port_free(self):
        port = find_available_port(start_port=29876)
        self.assertGreaterEqual(port, 29876)

    def test_find_available_port_when_occupied(self):
        # Occupy a port with an active listener
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 29880))
            s.listen(1)
            # Searching starting at 29880 should find 29881 or higher
            next_port = find_available_port(start_port=29880)
            self.assertGreater(next_port, 29880)

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    def test_start_server_in_thread(self, mock_config, mock_server):
        mock_app = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.run.side_effect = lambda: time.sleep(0.2)
        mock_server.return_value = mock_server_instance

        server, thread = start_server_in_thread(mock_app, "127.0.0.1", 8765)
        self.assertIsNotNone(server)
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        mock_config.assert_called_once_with(
            mock_app, host="127.0.0.1", port=8765, log_level="warning", log_config=None
        )

    def test_find_available_port_fallback(self):
        # Test fallback when all attempts are exhausted
        with patch("socket.socket") as mock_sock:
            mock_inst = MagicMock()
            mock_inst.__enter__.return_value = mock_inst
            mock_inst.bind.side_effect = OSError("Port occupied")
            mock_sock.return_value = mock_inst
            port = find_available_port(start_port=8000, max_attempts=5)
            self.assertEqual(port, 8000)


class TestUnifiedLauncherExecution(unittest.TestCase):
    """Test run_cli execution, hardware auto-calibration invocation, and run_app.py thin wrapper."""

    @patch("diarizeflow.calibration.ensure_calibrated_models")
    @patch("diarizeflow.app.launcher.setup_global_logging")
    @patch("diarizeflow.app.launcher.start_server_in_thread", return_value=(MagicMock(), MagicMock()))
    @patch("diarizeflow.app.launcher.DiarizeFlowPipeline")
    @patch("diarizeflow.app.frontend.desktop_overlay.run_overlay_app", return_value=0)
    def test_run_cli_triggers_ensure_calibrated_models(
        self, mock_overlay, mock_pipeline, mock_server, mock_logging, mock_calib
    ):
        mock_pipeline_inst = MagicMock()
        mock_pipeline.return_value = mock_pipeline_inst
        mock_calib.side_effect = lambda cfg: cfg

        test_args = ["launcher.py", "--port", "18765", "--target-lang", "English"]
        with patch.object(sys, "argv", test_args):
            exit_code = run_cli()

        self.assertEqual(exit_code, 0)
        mock_logging.assert_called_once()
        mock_calib.assert_called_once()
        mock_pipeline_inst.start.assert_called_once()
        mock_overlay.assert_called_once()

    def test_scripts_run_app_wrapper_parity(self):
        """Verify scripts/run_app.py exports unified launcher symbols and main delegates to run_cli."""
        import importlib.util

        project_root = Path(__file__).resolve().parent.parent
        run_app_path = project_root / "scripts" / "run_app.py"

        spec = importlib.util.spec_from_file_location("scripts_run_app", run_app_path)
        run_app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_app_mod)

        self.assertTrue(hasattr(run_app_mod, "run_cli"))
        self.assertTrue(hasattr(run_app_mod, "main"))
        self.assertTrue(hasattr(run_app_mod, "DualLogger"))
        self.assertTrue(hasattr(run_app_mod, "setup_global_logging"))
        self.assertIs(run_app_mod.run_cli, run_cli)
        self.assertIs(run_app_mod.main, launcher_main)


if __name__ == "__main__":
    unittest.main()
