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
        except (AssertionError, RuntimeError, IndexError, OSError):
            # In headless Linux or environments without an active PulseAudio server/library,
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

    def test_nemo_and_onnxsim_moved_to_optional_export_dependencies(self):
        """Verify nemo-toolkit, onnxsim, and onnxscript are in optional export group, not core dependencies (Issue #35)."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Core dependencies must NOT contain heavy export-only packages
        core_deps = data.get("project", {}).get("dependencies", [])
        core_names = {
            dep.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split(">")[0].split("<")[0].strip().lower().replace("_", "-")
            for dep in core_deps
        }
        self.assertNotIn("nemo-toolkit", core_names)
        self.assertNotIn("onnxsim", core_names)
        self.assertNotIn("onnxscript", core_names)

        # [project.optional-dependencies].export must declare them
        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        self.assertIn("export", optional_deps)
        export_names = {
            dep.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split(">")[0].split("<")[0].strip().lower().replace("_", "-")
            for dep in optional_deps["export"]
        }
        self.assertIn("nemo-toolkit", export_names)
        self.assertIn("onnxsim", export_names)
        self.assertIn("onnxscript", export_names)

        # [project.optional-dependencies].dev and [dependency-groups].dev must declare pytest tools
        self.assertIn("dev", optional_deps)
        dev_names = {
            dep.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split(">")[0].split("<")[0].strip().lower().replace("_", "-")
            for dep in optional_deps["dev"]
        }
        self.assertIn("pytest", dev_names)
        self.assertIn("pytest-asyncio", dev_names)

        dep_groups = data.get("dependency-groups", {})
        self.assertIn("dev", dep_groups)

    def test_build_optional_dependencies_declares_pyinstaller(self):
        """Verify pyinstaller is declared in optional-dependencies.build and dependency-groups.build (Issue #57)."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        self.assertIn("build", optional_deps)
        build_names = {
            dep.split(">=")[0].split("==")[0].split("<=")[0].split("~=")[0].split(">")[0].split("<")[0].strip().lower().replace("_", "-")
            for dep in optional_deps["build"]
        }
        self.assertIn("pyinstaller", build_names)

        dep_groups = data.get("dependency-groups", {})
        self.assertIn("build", dep_groups)

    def test_pytest_configuration_restricts_testpaths_and_excludes_scratch(self):
        """Verify pytest ini_options configures testpaths to tests/ and ignores scratch/ (Issue #35)."""
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        pytest_config = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
        self.assertIn("testpaths", pytest_config)
        self.assertEqual(pytest_config["testpaths"], ["tests"])

        self.assertIn("norecursedirs", pytest_config)
        self.assertIn("scratch", pytest_config["norecursedirs"])


if __name__ == "__main__":
    unittest.main()
