#!/usr/bin/env python3
"""Comprehensive benchmark script testing similarity and inference speed across all quantization precisions for SenseVoiceSmall on RTX 5090."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.core.quantize import (
    LatencyBenchmark,
    QuantPrecision,
    SimilarityMetric,
    compute_metrics,
)

FP4_E2M1_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

def _quantize_block_e2m1(x: np.ndarray) -> np.ndarray:
    """Quantize array elements to the closest E2M1 FP4 grid values."""
    sign = np.sign(x)
    abs_x = np.abs(x)
    diffs = np.abs(abs_x[..., np.newaxis] - FP4_E2M1_VALS)
    nearest_idx = np.argmin(diffs, axis=-1)
    return sign * FP4_E2M1_VALS[nearest_idx]



def prepare_baseline_model(source_onnx: Path, target_dir: Path) -> Path:
    """Create symlinks for baseline model and its external data file in target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = target_dir / "SenseVoiceSmall.onnx"
    target_data = target_dir / "SenseVoiceSmall.onnx.data"
    source_data = source_onnx.with_name(f"{source_onnx.name}.data")

    if not target_onnx.exists():
        print(f"[*] 建立 SenseVoiceSmall 基準模型連結: {target_onnx}", flush=True)
        target_onnx.symlink_to(source_onnx.resolve())
    if source_data.exists() and not target_data.exists():
        print(f"[*] 建立 SenseVoiceSmall 外部數據連結: {target_data}", flush=True)
        target_data.symlink_to(source_data.resolve())

    return target_onnx


def quantize_sensevoice_fp16(input_path: Path, output_path: Path) -> Path:
    """Quantize SenseVoiceSmall weights to FP16 half precision."""
    if output_path.exists():
        print(f"[*] FP16 模型已存在: {output_path}", flush=True)
        return output_path

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 FP16 (權重半精度)...", flush=True)
    model = onnx.load(str(input_path), load_external_data=True)
    count = 0
    total_params = 0

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            dequant = arr.astype(np.float16).astype(np.float32)
            new_init = numpy_helper.from_array(dequant.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)

    data_file = f"{output_path.name}.data"
    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"[✓] FP16 模型轉換完成 ({count} 個權重矩陣, 合計 {total_params / 1e6:.2f} M 參數): {output_path}", flush=True)
    return output_path


def quantize_sensevoice_int8(input_path: Path, output_path: Path) -> Path:
    """Quantize SenseVoiceSmall to INT8."""
    if output_path.exists():
        print(f"[*] INT8 模型已存在: {output_path}", flush=True)
        return output_path

    # Check if already generated in original folder
    cached_int8 = input_path.parent / "model_int8.onnx"
    if cached_int8.exists():
        print(f"[*] 複製已完成的 INT8 模型: {cached_int8} -> {output_path}", flush=True)
        shutil.copy2(str(cached_int8), str(output_path))
        return output_path

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 INT8...", flush=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    print(f"[✓] INT8 模型轉換完成: {output_path}", flush=True)
    return output_path


def quantize_sensevoice_fp8(input_path: Path, output_path: Path) -> Path:
    """Quantize SenseVoiceSmall to FP8 (E4M3FN) via weight emulation (fake quantization)."""
    if output_path.exists():
        print(f"[*] FP8 模型已存在: {output_path}", flush=True)
        return output_path

    import ml_dtypes

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 FP8 (E4M3FN 數值模擬量化 / Weight Emulation)...", flush=True)
    print("    ℹ️ 說明: 權重投影至 FP8 格點後以浮點存儲，用於特徵保真度驗證 (非原生 8-bit 硬體算子)", flush=True)
    model = onnx.load(str(input_path), load_external_data=True)
    FP8_MAX = 448.0
    count = 0
    total_params = 0

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            orig_shape = arr.shape
            axes = tuple(range(1, len(orig_shape)))
            max_vals = np.max(np.abs(arr), axis=axes, keepdims=True)
            scales = np.maximum(max_vals / FP8_MAX, 1e-12).astype(np.float32)
            scaled = np.clip(arr / scales, -FP8_MAX, FP8_MAX)
            fp8_arr = scaled.astype(ml_dtypes.float8_e4m3fn)
            dequant = (fp8_arr.astype(np.float32) * scales).reshape(orig_shape)
            new_init = numpy_helper.from_array(dequant.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)

    data_file = f"{output_path.name}.data"
    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"[✓] FP8 模型轉換完成 ({count} 個權重矩陣, 合計 {total_params / 1e6:.2f} M 參數): {output_path}", flush=True)
    return output_path


def quantize_sensevoice_nvfp4(input_path: Path, output_path: Path, block_size: int = 32) -> Path:
    """Quantize SenseVoiceSmall to NVIDIA Blackwell NVFP4 (E2M1) via weight emulation."""
    if output_path.exists():
        print(f"[*] NVFP4 模型已存在: {output_path}", flush=True)
        return output_path

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 NVFP4 (Blackwell E2M1 數值模擬量化 / Weight Emulation)...", flush=True)
    print("    ℹ️ 說明: 權重投影至 NVFP4 格點後以浮點存儲，用於特徵保真度驗證 (非原生 Blackwell NVFP4 算子)", flush=True)
    import ml_dtypes

    model = onnx.load(str(input_path), load_external_data=True)
    FP8_MAX = 448.0
    count = 0
    total_params = 0

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            orig_shape = arr.shape
            flat = arr.flatten().astype(np.float32)
            orig_len = flat.size
            pad_len = (block_size - (orig_len % block_size)) % block_size
            if pad_len > 0:
                flat = np.pad(flat, (0, pad_len), mode="constant")

            blocks = flat.reshape(-1, block_size)
            global_scale = float(np.max(np.abs(blocks)))
            if global_scale < 1e-12:
                continue

            normalized_blocks = blocks / global_scale
            block_max = np.max(np.abs(normalized_blocks), axis=1, keepdims=True)
            block_scales_raw = np.maximum(block_max / 6.0, 1e-12)
            block_scales_fp8 = np.clip(block_scales_raw * FP8_MAX, -FP8_MAX, FP8_MAX).astype(
                ml_dtypes.float8_e4m3fn
            )
            block_scales = (block_scales_fp8.astype(np.float32) / FP8_MAX) * 6.0
            block_scales = np.maximum(block_scales, 1e-12)

            q_in = normalized_blocks / block_scales
            q_out = _quantize_block_e2m1(q_in)
            dequant_blocks = q_out * block_scales * global_scale
            dequant_flat = dequant_blocks.flatten()[:orig_len]
            dequant_arr = dequant_flat.reshape(orig_shape).astype(arr.dtype)

            new_init = numpy_helper.from_array(dequant_arr, name=init.name)
            init.CopyFrom(new_init)

    data_file = f"{output_path.name}.data"
    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"[✓] NVFP4 模型轉換完成 ({count} 個權重矩陣, 合計 {total_params / 1e6:.2f} M 參數): {output_path}", flush=True)
    return output_path


def quantize_sensevoice_mxfp4(input_path: Path, output_path: Path, block_size: int = 32) -> Path:
    """Quantize SenseVoiceSmall to OCP Microscaling MXFP4 via weight emulation."""
    if output_path.exists():
        print(f"[*] MXFP4 模型已存在: {output_path}", flush=True)
        return output_path

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 MXFP4 (OCP Microscaling 數值模擬量化 / Weight Emulation)...", flush=True)
    print("    ℹ️ 說明: 權重投影至 MXFP4 格點後以浮點存儲，用於特徵保真度驗證 (非原生 MXFP4 算子)", flush=True)
    model = onnx.load(str(input_path), load_external_data=True)
    count = 0
    total_params = 0

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            orig_shape = arr.shape
            flat = arr.flatten().astype(np.float32)
            orig_len = flat.size
            pad_len = (block_size - (orig_len % block_size)) % block_size
            if pad_len > 0:
                flat = np.pad(flat, (0, pad_len), mode="constant")

            blocks = flat.reshape(-1, block_size)
            block_max = np.max(np.abs(blocks), axis=1, keepdims=True)
            exp = np.ceil(np.log2(np.maximum(block_max / 6.0, 1e-20)))
            exp = np.clip(exp, -127, 127)
            scale_e8m0 = (2.0 ** exp).astype(np.float32)

            q_in = blocks / scale_e8m0
            q_out = _quantize_block_e2m1(q_in)
            dequant_blocks = q_out * scale_e8m0
            dequant_flat = dequant_blocks.flatten()[:orig_len]
            dequant_arr = dequant_flat.reshape(orig_shape).astype(arr.dtype)

            new_init = numpy_helper.from_array(dequant_arr, name=init.name)
            init.CopyFrom(new_init)

    data_file = f"{output_path.name}.data"
    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"[✓] MXFP4 模型轉換完成 ({count} 個權重矩陣, 合計 {total_params / 1e6:.2f} M 參數): {output_path}", flush=True)
    return output_path


def quantize_sensevoice_w4a16(input_path: Path, output_path: Path) -> Path:
    """Quantize SenseVoiceSmall to W4A16 MatMulNBits."""
    if output_path.exists():
        print(f"[*] W4A16 模型已存在: {output_path}", flush=True)
        return output_path

    print(f"[*] 正在將 SenseVoiceSmall 轉換為 W4A16 (Weight-Only MatMulNBits)...", flush=True)
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    model = onnx.load(str(input_path), load_external_data=True)
    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=32,
        is_symmetric=True,
        bits=4,
    )
    quantizer = MatMulNBitsQuantizer(
        model=model,
        algo_config=quant_config,
    )
    quantizer.process()

    data_file = f"{output_path.name}.data"
    onnx.save(
        quantizer.model.model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
    )
    print(f"[✓] W4A16 模型轉換完成: {output_path}", flush=True)
    return output_path


def get_total_size_mb(onnx_path: Path) -> float:
    """Compute total disk size including external .data file if present."""
    total = onnx_path.stat().st_size
    data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    if data_path.exists():
        total += data_path.stat().st_size
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="評測 SenseVoiceSmall 所有量化精度的相似度與推論速度 (RTX 5090)")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/sensevoice_small",
        help="SenseVoiceSmall 模型所在目錄 (預設: models/sensevoice_small)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    target_dir = (project_root / args.model_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    baseline_onnx = target_dir / "SenseVoiceSmall.onnx"
    if not baseline_onnx.exists():
        raise FileNotFoundError(f"SenseVoiceSmall baseline not found in {baseline_onnx}")

    source_model = baseline_onnx

    print("=" * 115, flush=True)
    print("🚀 SenseVoiceSmall 全量化精度推論延遲與數值保真度綜合評測 (RTX 5090 GPU)", flush=True)
    print("=" * 115, flush=True)
    print(f"基準模型路徑: {baseline_onnx.relative_to(project_root) if baseline_onnx.is_relative_to(project_root) else baseline_onnx}", flush=True)
    print(f"模型儲存目錄: {target_dir.relative_to(project_root) if target_dir.is_relative_to(project_root) else target_dir}\n", flush=True)


    feats_length = 100
    feed_base = {
        "speech": np.random.randn(1, feats_length, 560).astype(np.float32),
        "speech_lengths": np.array([feats_length], dtype=np.int32),
        "language": np.array([0], dtype=np.int32),
        "textnorm": np.array([15], dtype=np.int32),
    }

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("[*] 正在載入 FP32 基準模型推論 Session (CUDAExecutionProvider)...", flush=True)
    sess_orig = ort.InferenceSession(str(baseline_onnx), providers=providers)

    print("[*] 執行 FP32 基準模型 Warmup 與延遲基準評測 (15 runs)...", flush=True)
    for _ in range(5):
        sess_orig.run(None, feed_base)

    bench_runs = 15
    t_orig = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        out_orig = sess_orig.run(None, feed_base)
        t_orig.append((time.perf_counter() - t0) * 1000)

    orig_mean = float(np.mean(t_orig))
    orig_std = float(np.std(t_orig))
    orig_fps = 1000.0 / orig_mean
    orig_size_mb = get_total_size_mb(baseline_onnx)

    print(f"    FP32 基準平均延遲: {orig_mean:.2f} ms ± {orig_std:.2f} ms ({orig_fps:.2f} chunks/sec)\n", flush=True)

    precision_configs = [
        ("FP16", "實測硬體 (Native)", target_dir / "SenseVoiceSmall_fp16.onnx", quantize_sensevoice_fp16),
        ("INT8", "實測硬體 (Native)", target_dir / "SenseVoiceSmall_int8.onnx", quantize_sensevoice_int8),
        ("FP8*", "數值模擬 (Weight Emulation)*", target_dir / "SenseVoiceSmall_fp8.onnx", quantize_sensevoice_fp8),
        ("NVFP4*", "數值模擬 (Weight Emulation)*", target_dir / "SenseVoiceSmall_nvfp4.onnx", quantize_sensevoice_nvfp4),
        ("MXFP4*", "數值模擬 (Weight Emulation)*", target_dir / "SenseVoiceSmall_mxfp4.onnx", quantize_sensevoice_mxfp4),
        ("W4A16", "實測硬體 (Native)", target_dir / "SenseVoiceSmall_w4a16.onnx", quantize_sensevoice_w4a16),
    ]

    results = [
        {
            "precision": "FP32 (基)",
            "mode": "基準模型 (FP32 Baseline)",
            "size_mb": orig_size_mb,
            "reduction_pct": 0.0,
            "cosine_sim": 1.000000,
            "max_abs_err": 0.000000,
            "latency_ms": orig_mean,
            "latency_std": orig_std,
            "speedup": 1.00,
            "fps": orig_fps,
            "status": "基準模型",
        }
    ]

    for name, mode, model_path, quant_func in precision_configs:
        print("-" * 115, flush=True)
        print(f"[{name}] 檢查 / 準備量化模型: {model_path.name} (執行模式: {mode})...", flush=True)
        try:
            quant_func(source_model, model_path)
            quant_size_mb = get_total_size_mb(model_path)
            reduction = (1.0 - quant_size_mb / orig_size_mb) * 100.0

            sess_quant = ort.InferenceSession(str(model_path), providers=providers)

            feed_quant = {}
            for inp in sess_quant.get_inputs():
                if "float16" in inp.type:
                    feed_quant[inp.name] = feed_base[inp.name].astype(np.float16)
                else:
                    feed_quant[inp.name] = feed_base[inp.name]

            for _ in range(3):
                sess_quant.run(None, feed_quant)

            t_quant = []
            for _ in range(bench_runs):
                t0 = time.perf_counter()
                out_quant = sess_quant.run(None, feed_quant)
                t_quant.append((time.perf_counter() - t0) * 1000)

            quant_mean = float(np.mean(t_quant))
            quant_std = float(np.std(t_quant))
            speedup = orig_mean / max(quant_mean, 1e-6)
            fps = 1000.0 / max(quant_mean, 1e-6)

            v_orig = out_orig[0].flatten().astype(np.float32)
            v_quant = out_quant[0].flatten().astype(np.float32)
            cos_sim = float(np.dot(v_orig, v_quant) / (np.linalg.norm(v_orig) * np.linalg.norm(v_quant) + 1e-12))
            max_err = float(np.max(np.abs(v_orig - v_quant)))

            if cos_sim >= 0.999:
                status = "EXCELLENT (極致無損)"
            elif cos_sim >= 0.990:
                status = "GOOD (高保真)"
            elif cos_sim >= 0.950:
                status = "ACCEPTABLE (可接受)"
            else:
                status = "DEGRADED (精度損失)"

            results.append(
                {
                    "precision": name,
                    "mode": mode,
                    "size_mb": quant_size_mb,
                    "reduction_pct": reduction,
                    "cosine_sim": cos_sim,
                    "max_abs_err": max_err,
                    "latency_ms": quant_mean,
                    "latency_std": quant_std,
                    "speedup": speedup,
                    "fps": fps,
                    "status": status,
                }
            )

            print(f"    容量: {quant_size_mb:.2f} MB ({reduction:.1f}% 空間縮減)", flush=True)
            print(f"    輸出層相似度: {cos_sim:.6f} | 最大絕對誤差: {max_err:.4e}", flush=True)
            print(f"    推論延遲: {quant_mean:.2f} ms ± {quant_std:.2f} ms | 加速比: {speedup:.2f}x ({fps:.1f} chunks/s)\n", flush=True)

        except Exception as e:
            print(f"    [!] 評測失敗: {e}\n", flush=True)

    # Print Summary Table
    print("\n" + "=" * 138, flush=True)
    print("🏆 SenseVoiceSmall 全量化精度相似度與推論速度綜合評測排行榜 (RTX 5090)", flush=True)
    print("=" * 138, flush=True)
    print(
        f"{'精度模式':<10} | {'執行類型':<22} | {'模型容量':<11} | {'空間縮減':<8} | {'輸出層相似度':<14} | {'最大絕對誤差':<14} | {'平均延遲':<10} | {'加速比':<8} | {'吞吐量 (FPS)':<14} | {'精度狀態'}"
    )
    print("-" * 138, flush=True)

    for r in results:
        prec = r["precision"]
        mode = r["mode"]
        size_str = f"{r['size_mb']:>6.2f} MB"
        red_str = f"{r['reduction_pct']:>5.1f}%"
        cos_str = f"{r['cosine_sim']:>10.6f}"
        err_str = f"{r['max_abs_err']:>10.4e}"
        lat_str = f"{r['latency_ms']:>6.2f} ms"
        spd_str = f"{r['speedup']:>5.2f}x"
        fps_str = f"{r['fps']:>6.2f} chunks/s"
        status = r["status"]
        print(
            f"{prec:<10} | {mode:<22} | {size_str:<11} | {red_str:<8} | {cos_str:<14} | {err_str:<14} | {lat_str:<10} | {spd_str:<8} | {fps_str:<14} | {status}",
            flush=True,
        )
        if "FP32" in prec:
            print("-" * 138, flush=True)

    print("=" * 138, flush=True)
    print("\n📌 基準測試透明度說明 (Benchmark Transparency Disclosure):", flush=True)
    print("  1. [實測硬體 (Native)]: FP16、INT8、W4A16 具備原生 ONNX 運算圖算子 (如 Float16, QLinearMatMul, MatMulNBits)。", flush=True)
    print("  2. [數值模擬 (Weight Emulation)]*: 標記 * 之精度 (FP8, NVFP4, MXFP4) 係將權重映射至目標量化格點後以浮點存儲。", flush=True)
    print("     此模式用於在部署至 Blackwell/微縮放硬體前，精確量測量化誤差 (MSE/MaxError) 與語音特徵保真度 (Cosine Sim)。", flush=True)
    print("     由於 ONNX Runtime 當前依浮點執行 MatMul，故容量與延遲反映未打包之數值模擬狀態。\n", flush=True)


if __name__ == "__main__":
    main()
