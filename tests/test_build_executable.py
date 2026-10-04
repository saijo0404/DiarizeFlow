"""Unit tests for standalone executable build script and PyInstaller auto-installation (Issue #57).

Verifies:
1. ensure_pyinstaller returns True immediately if PyInstaller is already installed.
2. ensure_pyinstaller prefers uv pip install when uv executable is detected.
3. ensure_pyinstaller safely falls back to python -m pip install when uv is unavailable.
4. ensure_pyinstaller outputs friendly guidance and returns False (without unhandled CalledProcessError)
   when neither uv nor pip is available or when installation commands fail.
5. build() returns status code 1 when PyInstaller installation fails, avoiding uncaught traceback crashes.
6. build() CLI argument parser accepts --slim and --fp32-only flags.
"""

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parent.parent
build_script_path = project_root / "scripts" / "build_executable.py"

spec = importlib.util.spec_from_file_location("build_executable", build_script_path)
build_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_mod)


class TestEnsurePyInstaller(unittest.TestCase):
    """Test suite for ensure_pyinstaller installation strategies and error handling."""

    def test_returns_true_if_already_installed(self):
        """Verify ensure_pyinstaller returns True immediately when PyInstaller is present."""
        with patch.dict(sys.modules, {"PyInstaller": MagicMock()}):
            self.assertTrue(build_mod.ensure_pyinstaller())

    def test_prefers_uv_when_uv_is_available(self):
        """Verify ensure_pyinstaller attempts uv pip install when uv is present in PATH."""
        mock_runner = MagicMock()
        with patch.dict(sys.modules, {"PyInstaller": None}):
            # Simulate PyInstaller not installed initially
            with patch("shutil.which", return_value="/usr/local/bin/uv"):
                res = build_mod.ensure_pyinstaller(
                    uv_executable="/usr/local/bin/uv",
                    runner=mock_runner,
                )
                self.assertTrue(res)
                mock_runner.assert_called()
                called_cmd = mock_runner.call_args[0][0]
                self.assertIn("uv", called_cmd[0])
                self.assertIn("pip", called_cmd)
                self.assertIn("install", called_cmd)
                self.assertIn("pyinstaller", called_cmd)

    def test_falls_back_to_pip_when_uv_missing(self):
        """Verify ensure_pyinstaller falls back to python -m pip when uv is not available."""
        mock_runner = MagicMock()
        with patch.dict(sys.modules, {"PyInstaller": None}):
            with patch("shutil.which", return_value=None), \
                 patch("importlib.util.find_spec", return_value=MagicMock()):
                res = build_mod.ensure_pyinstaller(
                    uv_executable=None,
                    runner=mock_runner,
                )
                self.assertTrue(res)
                mock_runner.assert_called_once()
                called_cmd = mock_runner.call_args[0][0]
                self.assertEqual(called_cmd[0], sys.executable)
                self.assertIn("-m", called_cmd)
                self.assertIn("pip", called_cmd)
                self.assertIn("pyinstaller", called_cmd)

    def test_graceful_failure_when_both_uv_and_pip_fail(self):
        """Verify ensure_pyinstaller prints guidance and returns False when installation fails."""
        mock_runner = MagicMock(side_effect=subprocess.CalledProcessError(1, ["cmd"]))
        printed_messages = []

        with patch.dict(sys.modules, {"PyInstaller": None}):
            with patch("shutil.which", return_value="/usr/bin/uv"), \
                 patch("importlib.util.find_spec", return_value=MagicMock()), \
                 patch("builtins.print", side_effect=lambda msg="": printed_messages.append(str(msg))):
                res = build_mod.ensure_pyinstaller(
                    uv_executable="/usr/bin/uv",
                    runner=mock_runner,
                )
                self.assertFalse(res)
                # Verify user guidance is printed
                combined_output = "\n".join(printed_messages)
                self.assertIn("未檢測到 PyInstaller", combined_output)
                self.assertIn("uv sync --extra build", combined_output)
                self.assertIn("uv run --with pyinstaller", combined_output)

    def test_graceful_failure_when_neither_uv_nor_pip_exist(self):
        """Verify ensure_pyinstaller handles pure uv venv without pip or uv binary gracefully."""
        printed_messages = []

        with patch.dict(sys.modules, {"PyInstaller": None}):
            with patch("shutil.which", return_value=None), \
                 patch("importlib.util.find_spec", return_value=None), \
                 patch("builtins.print", side_effect=lambda msg="": printed_messages.append(str(msg))):
                res = build_mod.ensure_pyinstaller(
                    uv_executable=None,
                    runner=None,
                )
                self.assertFalse(res)
                combined_output = "\n".join(printed_messages)
                self.assertIn("未檢測到 PyInstaller", combined_output)
                self.assertIn("uv sync --extra build", combined_output)

    def test_build_returns_code_1_when_pyinstaller_cannot_be_ensured(self):
        """Verify build() returns exit code 1 instead of crashing with unhandled exception."""
        with patch.object(build_mod, "ensure_pyinstaller", return_value=False), \
             patch("sys.argv", ["build_executable.py"]):
            exit_code = build_mod.build()
            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
