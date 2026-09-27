"""Unit tests verifying deprecation and removal of the Web frontend (Issue #16)."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock

from diarizeflow.app.config import AppConfig
from diarizeflow.app.backend.server import create_app


class TestWebFrontendRemoval(unittest.TestCase):
    """Ensure all traces of the legacy web frontend are cleanly removed."""

    def test_web_static_directory_removed(self):
        """Verify that src/diarizeflow/app/frontend/web no longer exists."""
        project_root = Path(__file__).resolve().parent.parent
        web_dir = project_root / "src" / "diarizeflow" / "app" / "frontend" / "web"
        self.assertFalse(
            web_dir.exists(),
            f"Web frontend directory {web_dir} still exists and should be deleted!",
        )

    def test_server_does_not_mount_overlay_or_static(self):
        """Verify FastAPI backend server no longer mounts /overlay or /static."""
        cfg = AppConfig()
        pipeline = MagicMock()
        app = create_app(cfg, pipeline)

        route_paths = [getattr(r, "path", "") for r in app.routes]
        self.assertNotIn(
            "/overlay",
            route_paths,
            "FastAPI app still exposes legacy /overlay endpoint",
        )
        self.assertNotIn(
            "/static",
            route_paths,
            "FastAPI app still mounts /static directory",
        )

    def test_build_executable_excludes_web_assets(self):
        """Verify build_executable.py and DiarizeFlow.spec do not package frontend/web."""
        project_root = Path(__file__).resolve().parent.parent
        build_script = (project_root / "scripts" / "build_executable.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "frontend/web",
            build_script,
            "build_executable.py still packages frontend/web assets!",
        )

        spec_file = (project_root / "DiarizeFlow.spec").read_text(encoding="utf-8")
        self.assertNotIn(
            "frontend/web",
            spec_file,
            "DiarizeFlow.spec still includes frontend/web in datas!",
        )

    def test_root_endpoint_returns_json_status(self):
        """Verify FastAPI root endpoint / returns JSON status instead of HTML."""
        from fastapi.testclient import TestClient

        cfg = AppConfig()
        pipeline = MagicMock()
        app = create_app(cfg, pipeline)

        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "running")
        self.assertIn("DiarizeFlow", data.get("service", ""))

    def test_launchers_do_not_contain_web_mode_or_fallback(self):
        """Verify launcher and run scripts do not have web fallback or web mode."""
        project_root = Path(__file__).resolve().parent.parent
        for rel_path in [
            "src/diarizeflow/app/launcher.py",
            "scripts/run_app.py",
            "scripts/run_frontend.py",
        ]:
            content = (project_root / rel_path).read_text(encoding="utf-8")
            self.assertNotIn(
                '"web"',
                content,
                f"{rel_path} still contains 'web' mode choice!",
            )
            self.assertNotIn(
                "webbrowser.open",
                content,
                f"{rel_path} still attempts to open browser fallback!",
            )


if __name__ == "__main__":
    unittest.main()
