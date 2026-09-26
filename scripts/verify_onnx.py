#!/usr/bin/env python3
"""Verification script for Nemotron-3 Diarization ONNX model."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def inspect_and_verify(onnx_path: str) -> None:
    path = Path(os.path.expanduser(onnx_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    print(f"[*] 載入並檢查 ONNX 模型結構: {path}")
    print(f"    檔案大小: {path.stat().st_size / (1024 * 1024):.2f} MB")

    # Inspect with onnx proto
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    print("    [✓] onnx.checker 格式驗證通過！")
    print(f"    ONNX IR 版本: {model.ir_version}, 產生者: {model.producer_name}")

    # Inspect with onnxruntime
    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    print(f"[*] 使用 Execution Providers: {providers}")

    session = ort.InferenceSession(str(path), providers=providers)

    print("\n[*] 模型輸入規格 (Inputs):")
    inputs = session.get_inputs()
    feed_dict = {}
    for inp in inputs:
        print(f"    - {inp.name:<26}: shape={inp.shape}, type={inp.type}")
        # Generate dummy array based on shape and type
        dtype = np.float32 if "float" in inp.type else np.int64
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if dtype == np.float32:
            feed_dict[inp.name] = np.random.randn(*shape).astype(dtype)
        else:
            feed_dict[inp.name] = np.ones(shape, dtype=dtype)

    print("\n[*] 模型輸出規格 (Outputs):")
    outputs = session.get_outputs()
    for out in outputs:
        print(f"    - {out.name:<26}: shape={out.shape}, type={out.type}")

    # Run inference test
    print("\n[*] 執行測試推論 (Inference Test)...")
    t0 = time.time()
    res = session.run(None, feed_dict)
    elapsed = time.time() - t0
    print(f"    [✓] 推論成功，耗時: {elapsed * 1000:.2f} ms")

    for out, arr in zip(outputs, res):
        print(f"      - {out.name:<26}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    print("\n[✓] 模型檢驗全部完成！可正常使用 ONNX Runtime 進行推論。")


def main() -> None:
    parser = argparse.ArgumentParser(description="驗證 Nemotron-3 Diarization ONNX 模型")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/nemotron_diarization/Nemotron-3-Diarization.onnx",
        help="ONNX 模型檔案路徑 (預設: models/nemotron_diarization/Nemotron-3-Diarization.onnx)",
    )
    args = parser.parse_args()
    inspect_and_verify(args.model)


if __name__ == "__main__":
    main()
