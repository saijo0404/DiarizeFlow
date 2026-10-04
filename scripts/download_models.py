#!/usr/bin/env python3
"""CLI script for one-click downloading and initial guidance of pre-trained speech models.

Usage:
    uv run python scripts/download_models.py
    python scripts/download_models.py --precision auto
    python scripts/download_models.py --source modelscope
    python scripts/download_models.py --dry-run
"""

from pathlib import Path
import sys

# Ensure src/ is on python search path for direct repository execution
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from diarizeflow.cli.downloader import main

if __name__ == "__main__":
    main()
