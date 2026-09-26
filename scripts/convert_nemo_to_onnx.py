#!/usr/bin/env python3
"""CLI script to convert Nemotron-3-Diarization.nemo to ONNX and apply onnxsim."""

import sys
from pathlib import Path

# Add src to python path for direct execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.export_onnx import main

if __name__ == "__main__":
    main()
