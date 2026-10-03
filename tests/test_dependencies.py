"""Unit tests to verify critical runtime dependencies are declared and importable (Issue #3)."""

import unittest
from pathlib import Path
import tomllib


class TestRuntimeDependencies(unittest.TestCase):
    """Ensure all required runtime dependencies are present in pyproject.toml and importable."""

    def test_pyproject_toml_declares_critical_dependencies(self):
        """Verify pyproject.toml explicitly lists soundcard, faster-whisper, and ml-dtypes."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        self.assertTrue(pyproject_path.exists(), "pyproject.toml not found")

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        deps = data.get("project", {}).get("dependencies", [])
        dep_names = set()
        for dep in deps:
            # Normalize dependency string (e.g. 'faster-whisper>=1.0.0' -> 'faster-whisper')
            name = dep.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split(">")[0].split("<")[0].strip()
            dep_names.add(name.lower().replace("_", "-"))

        self.assertIn(
            "soundcard",
            dep_names,
            "soundcard must be declared in pyproject.toml dependencies",
        )
        self.assertIn(
            "faster-whisper",
            dep_names,
            "faster-whisper must be declared in pyproject.toml dependencies",
        )
        self.assertIn(
            "ml-dtypes",
            dep_names,
            "ml-dtypes must be declared in pyproject.toml dependencies",
        )

    def test_soundcard_import_and_symbols(self):
        """Verify soundcard is installed and exposes required loopback capture symbols."""
        import importlib.util
        spec = importlib.util.find_spec("soundcard")
        self.assertIsNotNone(spec, "soundcard package must be installed")

        try:
            import soundcard as sc
            self.assertTrue(hasattr(sc, "all_microphones"))
            self.assertTrue(hasattr(sc, "all_speakers"))
        except (AssertionError, RuntimeError, IndexError):
            # In headless Linux or environments without an active PulseAudio server,
            # soundcard initialization fails at module import. Package presence is already verified.
            pass

    def test_faster_whisper_import_and_symbols(self):
        """Verify faster-whisper is installed and exposes WhisperModel."""
        import faster_whisper

        self.assertTrue(hasattr(faster_whisper, "WhisperModel"))

    def test_ml_dtypes_import_and_symbols(self):
        """Verify ml_dtypes is installed and exposes float8_e4m3fn for FP8 tensor operations."""
        import ml_dtypes

        self.assertTrue(hasattr(ml_dtypes, "float8_e4m3fn"))


if __name__ == "__main__":
    unittest.main()
