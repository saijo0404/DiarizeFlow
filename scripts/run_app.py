#!/usr/bin/env python3
"""All-in-One Launcher for DiarizeFlow (Backend + Frontend Floating HUD).

Thin wrapper delegating directly to diarizeflow.app.launcher.run_cli.
"""

import sys
from pathlib import Path

# Add src to python path for development and PyInstaller
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.app.launcher import (  # noqa: F401
    DualLogger,
    attach_console_on_windows,
    find_available_port,
    get_app_log_path,
    main,
    run_cli,
    setup_global_logging,
    start_server_in_thread,
)


if __name__ == "__main__":
    sys.exit(main())
