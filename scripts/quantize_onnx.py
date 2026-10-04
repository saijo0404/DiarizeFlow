#!/usr/bin/env python3
"""CLI script to automatically detect hardware, quantize Nemotron-3 Diarization ONNX model, and verify similarity."""

import sys
from pathlib import Path

# Add src to python path for direct execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.core.quantize import main

if __name__ == "__main__":
    main()
