"""Tests for package layering (Issue #64).

Verifies the flattened ``src/diarizeflow`` layout (no redundant ``app/`` layer)
and that inter-package dependencies form a strictly downward DAG::

    cli -> ui / engine -> audio -> config
    cli -> core -> config
"""

import ast
import importlib
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
PKG_ROOT = project_root / "src" / "diarizeflow"

LAYERS = {"config", "core", "audio", "engine", "ui", "cli"}

# package -> set of sibling layers it is allowed to import from
ALLOWED = {
    "config": set(),
    "core": {"config"},
    "audio": {"config"},
    "engine": {"config", "audio"},
    "ui": {"config", "audio"},
    "cli": {"config", "core", "audio", "engine", "ui"},
}


def _layer_of(path: Path) -> str:
    rel = path.relative_to(PKG_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _collect_imports():
    """Yield (layer, file, lineno, imported_layer) for each diarizeflow import."""
    for f in sorted(PKG_ROOT.rglob("*.py")):
        layer = _layer_of(f)
        if layer == "__init__":
            continue  # top-level package facade may re-export from any layer
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                parts = m.split(".")
                if parts[0] == "diarizeflow" and len(parts) > 1:
                    yield layer, f, node.lineno, parts[1]


class TestPackageLayout(unittest.TestCase):
    def test_app_layer_removed(self):
        self.assertFalse((PKG_ROOT / "app").exists(), "redundant app/ layer must be removed")

    def test_expected_subpackages_exist(self):
        for name in ("core", "audio", "engine", "ui", "cli"):
            self.assertTrue((PKG_ROOT / name / "__init__.py").is_file(), f"{name}/ missing")
        self.assertTrue((PKG_ROOT / "config.py").is_file())

    def test_legacy_toplevel_modules_moved(self):
        for legacy in ("calibration", "export_onnx", "hardware", "models", "patches", "quantize"):
            self.assertFalse((PKG_ROOT / f"{legacy}.py").exists(), f"{legacy}.py still at top level")

    def test_expected_module_locations(self):
        expected = [
            "core/hardware.py", "core/quantize.py", "core/export_onnx.py",
            "core/patches.py", "core/calibration.py",
            "audio/capture.py", "audio/devices.py", "audio/protocol.py",
            "audio/segmenter.py", "audio/tse.py", "audio/agc.py", "audio/vad.py",
            "engine/asr.py", "engine/diarizer.py", "engine/translator.py",
            "engine/voiceprint.py", "engine/pipeline.py", "engine/server.py",
            "ui/overlay_window.py", "ui/desktop_overlay.py", "ui/cards.py",
            "ui/settings_dialog.py", "ui/widgets.py", "ui/network.py",
            "cli/launcher.py", "cli/downloader.py",
        ]
        for rel in expected:
            self.assertTrue((PKG_ROOT / rel).is_file(), f"{rel} missing")


class TestDependencyDirection(unittest.TestCase):
    def test_no_reference_to_removed_app_package(self):
        for f in sorted(PKG_ROOT.rglob("*.py")):
            self.assertNotIn("diarizeflow.app", f.read_text(encoding="utf-8"), str(f))

    def test_layer_dependencies_follow_allowed_direction(self):
        violations = []
        for layer, f, lineno, target in _collect_imports():
            if target not in LAYERS and target not in {"cli"}:
                continue
            if target == layer:
                continue
            if target not in ALLOWED.get(layer, set()):
                violations.append(f"{f.relative_to(project_root)}:{lineno} {layer} -> {target}")
        self.assertEqual(violations, [], "Layer violations:\n" + "\n".join(violations))

    def test_no_cyclic_layer_dependencies(self):
        graph = {layer: set() for layer in LAYERS}
        for layer, _f, _ln, target in _collect_imports():
            if layer in graph and target in graph and target != layer:
                graph[layer].add(target)

        visiting, done = set(), set()

        def dfs(node, stack):
            if node in done:
                return
            self.assertNotIn(node, visiting, f"cycle: {' -> '.join(stack + [node])}")
            visiting.add(node)
            for nxt in graph[node]:
                dfs(nxt, stack + [node])
            visiting.discard(node)
            done.add(node)

        for n in graph:
            dfs(n, [])

    def test_calibration_launcher_not_bidirectional(self):
        """Regression for the original calibration <-> launcher inversion."""
        calib = (PKG_ROOT / "core" / "calibration.py").read_text(encoding="utf-8")
        self.assertNotIn("diarizeflow.cli", calib)
        self.assertNotIn("launcher", calib)


class TestImportsResolve(unittest.TestCase):
    def test_new_import_paths(self):
        for mod in (
            "diarizeflow.config",
            "diarizeflow.core.hardware",
            "diarizeflow.core.calibration",
            "diarizeflow.audio.protocol",
            "diarizeflow.audio.devices",
            "diarizeflow.engine",
            "diarizeflow.cli.launcher",
            "diarizeflow.cli.downloader",
        ):
            importlib.import_module(mod)

    def test_engine_facade_exports_pipeline(self):
        from diarizeflow.engine import DiarizeFlowPipeline, create_app

        self.assertTrue(callable(DiarizeFlowPipeline))
        self.assertTrue(callable(create_app))

    def test_toplevel_reexports_appconfig(self):
        import diarizeflow
        from diarizeflow.config import AppConfig

        self.assertIs(diarizeflow.AppConfig, AppConfig)

    def test_old_import_paths_gone(self):
        for mod in ("diarizeflow.app", "diarizeflow.app.config", "diarizeflow.models", "diarizeflow.hardware"):
            with self.assertRaises(ImportError, msg=mod):
                importlib.import_module(mod)

    def test_pyproject_entrypoints_resolve(self):
        import tomllib

        data = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = data["project"]["scripts"]
        for name, target in scripts.items():
            self.assertNotIn(".app.", target, name)
            if name in ("diarizeflow-export", "diarizeflow-quantize"):
                continue  # require optional heavy deps (torch/onnx)
            module, func = target.split(":")
            mod = importlib.import_module(module)
            self.assertTrue(hasattr(mod, func), f"{target} not found")


if __name__ == "__main__":
    unittest.main()
