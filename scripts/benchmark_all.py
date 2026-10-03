#!/usr/bin/env python3
"""Comprehensive benchmark script testing similarity and inference speed across all quantization precisions."""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.quantize import (
    LatencyBenchmark,
    QuantPrecision,
    SimilarityMetric,
    benchmark_model_latency,
    compare_model_similarity,
    compute_metrics,
    get_quantization_execution_mode,
    is_simulated_quantization,
    quantize_fp16,
    quantize_fp8,
    quantize_int8,
    quantize_mxfp4,
    quantize_nvfp4,
    quantize_w4a16,
)


def run_full_benchmark(
    base_model_path: str,
    bench_runs: int = 15,
    warmup_runs: int = 5,
):
    base_path = Path(base_model_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_path}")

    base_size_mb = base_path.stat().st_size / (1024 * 1024)

    precisions = [
        ("FP16", QuantPrecision.FP16, quantize_fp16, "_fp16.onnx"),
        ("INT8", QuantPrecision.INT8, quantize_int8, "_int8.onnx"),
        ("FP8*", QuantPrecision.FP8, quantize_fp8, "_fp8.onnx"),
        ("NVFP4*", QuantPrecision.NVFP4, quantize_nvfp4, "_nvfp4.onnx"),
        ("MXFP4*", QuantPrecision.MXFP4, quantize_mxfp4, "_mxfp4.onnx"),
        ("W4A16", QuantPrecision.W4A16, quantize_w4a16, "_w4a16.onnx"),
    ]

    providers = ort.get_available_providers()
    active_ep = "CUDAExecutionProvider (NVIDIA GPU Tensor Core 加速)" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider (純 CPU 模式，無 GPU 硬體加速)"

    print("=" * 105)
    print(f"📊 Nemotron-3 Diarization 全量化精度綜合評測 (基準模型: {base_path.name}, 大小: {base_size_mb:.2f} MB)")
    print(f"   執行後端引擎: {active_ep}")
    print(f"   評測設定: 每種精度進行 {warmup_runs} 次預熱 + {bench_runs} 次獨立微秒級統計計時")
    print("=" * 105)

    results = []

    for name, prec_enum, quant_fn, suffix in precisions:
        out_model = base_path.with_name(f"{base_path.stem}{suffix}")
        mode_str = get_quantization_execution_mode(prec_enum)
        print(f"\n[{name}] 檢查 / 準備量化模型: {out_model.name} (執行模式: {mode_str})...")

        if not out_model.exists():
            print(f"    模型不存在，正在執行 {name} 量化...")
            if prec_enum == QuantPrecision.FP16:
                quant_fn(str(base_path), str(out_model), keep_io_types=False)
            else:
                quant_fn(str(base_path), str(out_model))

        model_size_mb = out_model.stat().st_size / (1024 * 1024)
        size_reduction = (1.0 - model_size_mb / base_size_mb) * 100.0

        print(f"    大小: {model_size_mb:.2f} MB (節省 {size_reduction:.1f}%)")
        if is_simulated_quantization(prec_enum):
            print("    ℹ️ 說明: 該精度為權重數值誤差模擬量化 (Weight Emulation)，輸出以浮點存儲，用於特徵保真度驗證。")
        print(f"    執行相似度比對與延遲基準測試 (runs={bench_runs})...")

        metrics, bench = compare_model_similarity(
            original_model_path=str(base_path),
            quantized_model_path=str(out_model),
            bench_runs=bench_runs,
            warmup_runs=warmup_runs,
        )

        spk_metric = next((m for m in metrics if "spkcache_fifo_chunk_preds" in m.tensor_name), metrics[0])
        emb_metric = next((m for m in metrics if "chunk_pre_encode_embs" in m.tensor_name), metrics[1] if len(metrics) > 1 else metrics[0])

        results.append({
            "name": name,
            "mode": mode_str,
            "size_mb": model_size_mb,
            "reduction": size_reduction,
            "spk_cos": spk_metric.cosine_similarity,
            "spk_max_err": spk_metric.max_abs_error,
            "emb_cos": emb_metric.cosine_similarity,
            "status": spk_metric.status,
            "bench": bench,
        })

    # Print final summary table
    print("\n\n" + "=" * 138)
    print("🏆 全量化精度相似度與推論速度綜合評測排行榜 (Benchmark Summary Table)")
    print("=" * 138)
    header = (
        f"{'精度模式':<8} | {'執行類型':<22} | {'模型容量':<9} | {'空間縮減':<8} | {'語者預測相似度':<12} | "
        f"{'最大絕對誤差':<12} | {'平均延遲':<10} | {'加速比':<8} | {'吞吐量 (FPS)':<12} | {'精度狀態'}"
    )
    print(header)
    print("-" * 138)

    # First print baseline FP32
    if results:
        base_bench = results[0]["bench"]
        print(
            f"{'FP32 (基)':<8} | {'基準模型 (FP32 Baseline)':<22} | {base_size_mb:>6.2f} MB | {'0.0%':<8} | {'1.000000':<14} | "
            f"{'0.000000e+00':<12} | {base_bench.orig_mean_ms:>6.2f} ms  | {'1.00x':<8} | "
            f"{base_bench.orig_fps:>6.2f} chunks/s | 基準模型"
        )
        print("-" * 138)

    for r in results:
        b = r["bench"]
        print(
            f"{r['name']:<8} | {r['mode']:<22} | {r['size_mb']:>6.2f} MB | {r['reduction']:>5.1f}%  | "
            f"{r['spk_cos']:>12.6f}   | {r['spk_max_err']:<12.4e} | {b.quant_mean_ms:>6.2f} ms  | "
            f"{b.speedup:>5.2f}x   | {b.quant_fps:>6.2f} chunks/s | {r['status']}"
        )

    print("=" * 138)
    print("\n📌 基準測試透明度說明 (Benchmark Transparency Disclosure):")
    print("  1. [實測硬體 (Native Graph Operators)]: FP16、INT8、W4A16 具備原生 ONNX 運算圖算子 (如 Float16, QLinearMatMul, MatMulNBits)。")
    print("  2. [數值模擬 (Weight Emulation)]*: 標記 * 之精度 (FP8, NVFP4, MXFP4) 係將權重映射至目標量化格點後以浮點存儲。")
    print("     此模式用於在部署至 Blackwell/微縮放硬體前，精確量測量化誤差 (MSE/MaxError) 與語者特徵保真度 (Cosine Sim)。")
    print("     由於 ONNX Runtime 當前依浮點執行 MatMul，故容量與延遲反映未打包之數值模擬狀態。\n")


def main():
    parser = argparse.ArgumentParser(description="評測 Nemotron-3 Diarization 所有量化精度的相似度與推論速度")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/nemotron_diarization/Nemotron-3-Diarization.onnx",
        help="基準 FP32 ONNX 模型路徑 (預設: models/nemotron_diarization/Nemotron-3-Diarization.onnx)",
    )
    parser.add_argument(
        "--bench-runs",
        type=int,
        default=15,
        help="每種精度推論速度測試次數 (預設: 15)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="預熱推論次數 (預設: 5)",
    )
    args = parser.parse_args()
    run_full_benchmark(args.model, bench_runs=args.bench_runs, warmup_runs=args.warmup_runs)


if __name__ == "__main__":
    main()
