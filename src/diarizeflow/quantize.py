"""Quantization and accuracy/similarity comparison pipeline for Nemotron-3 Diarization ONNX models."""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from diarizeflow.hardware import DeviceInfo, QuantPrecision, get_system_device_info


@dataclass
class SimilarityMetric:
    """Metrics comparing original FP32 output and quantized model output."""
    tensor_name: str
    cosine_similarity: float
    pearson_correlation: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    snr_db: float
    passed: bool
    status: str


@dataclass
class LatencyBenchmark:
    """Benchmark results comparing inference speed, throughput and latency."""
    orig_mean_ms: float
    orig_std_ms: float
    quant_mean_ms: float
    quant_std_ms: float
    speedup: float               # e.g. 2.02x
    latency_reduction_pct: float # e.g. 50.5%
    orig_fps: float              # chunks per second
    quant_fps: float             # chunks per second
    num_iterations: int
    warmup_iterations: int

    def __str__(self) -> str:
        status_symbol = "⚡" if self.speedup >= 1.0 else "⏳"
        return (
            f"基準模型 (FP32) : {self.orig_mean_ms:.2f} ms ± {self.orig_std_ms:.2f} ms ({self.orig_fps:.2f} chunks/s)\n"
            f"量化模型       : {self.quant_mean_ms:.2f} ms ± {self.quant_std_ms:.2f} ms ({self.quant_fps:.2f} chunks/s)\n"
            f"{'-'*70}\n"
            f"{status_symbol} 推理加速倍率 (Speedup)   : {self.speedup:.2f}x\n"
            f"📉 延遲降低幅度 (Reduction) : {self.latency_reduction_pct:.1f}%"
        )


def compute_metrics(
    orig: np.ndarray,
    quant: np.ndarray,
    tensor_name: str,
    min_cosine_threshold: float = 0.95,
) -> SimilarityMetric:
    """Compute mathematical similarity metrics between original and quantized outputs."""
    orig_flat = orig.flatten().astype(np.float64)
    quant_flat = quant.flatten().astype(np.float64)

    # 1. Max & Mean Absolute Error
    diff = np.abs(orig_flat - quant_flat)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    # 2. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # 3. Cosine Similarity
    norm_orig = np.linalg.norm(orig_flat)
    norm_quant = np.linalg.norm(quant_flat)
    if norm_orig > 1e-12 and norm_quant > 1e-12:
        cosine_sim = float(np.dot(orig_flat, quant_flat) / (norm_orig * norm_quant))
    else:
        cosine_sim = 1.0 if np.allclose(orig_flat, quant_flat) else 0.0

    # 4. Pearson Correlation
    orig_centered = orig_flat - np.mean(orig_flat)
    quant_centered = quant_flat - np.mean(quant_flat)
    std_orig = np.std(orig_flat)
    std_quant = np.std(quant_flat)
    if std_orig > 1e-12 and std_quant > 1e-12:
        pearson = float(np.mean(orig_centered * quant_centered) / (std_orig * std_quant))
    else:
        pearson = 1.0 if cosine_sim > 0.999 else 0.0

    # 5. Signal-to-Noise Ratio (SNR) in dB
    noise_power = np.mean(diff ** 2)
    signal_power = np.mean(orig_flat ** 2)
    if noise_power > 1e-15 and signal_power > 1e-15:
        snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_db = 99.99  # essentially lossless

    passed = cosine_sim >= min_cosine_threshold
    if cosine_sim >= 0.999:
        status = "EXCELLENT (極致無損)"
    elif cosine_sim >= 0.98:
        status = "GOOD (高保真)"
    elif cosine_sim >= min_cosine_threshold:
        status = "ACCEPTABLE (可接受)"
    else:
        status = "DEGRADED (精度損失警告)"

    return SimilarityMetric(
        tensor_name=tensor_name,
        cosine_similarity=cosine_sim,
        pearson_correlation=pearson,
        max_abs_error=max_err,
        mean_abs_error=mean_err,
        rmse=rmse,
        snr_db=snr_db,
        passed=passed,
        status=status,
    )


def quantize_fp16(
    input_model_path: str,
    output_model_path: str,
    keep_io_types: bool = False,
) -> Path:
    """Convert an ONNX model to FP16 half precision.

    Args:
        input_model_path: Path to the input FP32 ONNX model.
        output_model_path: Path to save the converted FP16 ONNX model.
        keep_io_types: If False (recommended), converts graph I/O to native FP16
                       to avoid Reshape/View node type conflicts.
    """
    from onnxconverter_common import float16

    print(f"[*] 載入模型進行 FP16 量化: {input_model_path}")
    model = onnx.load(input_model_path)

    print(f"[*] 轉換算子與權重至 Float16 (keep_io_types={keep_io_types})...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=False,
    )

    # Fix: convert any residual Cast nodes with to=FLOAT (1) to to=FLOAT16 (10)
    # When keep_io_types=False, internal/mask Cast operations must output float16
    # to match downstream fp16 operators and prevent ONNX Runtime type mismatch.
    for node in model_fp16.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == int(onnx.TensorProto.FLOAT):
                    attr.i = int(onnx.TensorProto.FLOAT16)

    onnx.checker.check_model(model_fp16)

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(out_path))
    return out_path


def quantize_int8(
    input_model_path: str,
    output_model_path: str,
    per_channel: bool = True,
) -> Path:
    """Quantize an ONNX model to INT8 using dynamic quantization.

    Args:
        input_model_path: Path to the input FP32 ONNX model.
        output_model_path: Path to save the quantized INT8 ONNX model.
        per_channel: Whether to use per-channel weight quantization for higher accuracy.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[*] 執行 Dynamic INT8 量化 (per_channel={per_channel}): {input_model_path}")
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(out_path),
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Linear"],
        extra_options={"EnableSubgraph": True, "ForceQuantizeNoInputCheck": True},
    )
    return out_path


def is_simulated_quantization(precision: Union[str, QuantPrecision]) -> bool:
    """Return True if precision represents weight-only numerical emulation (fake quantization)."""
    p_str = precision.value if isinstance(precision, QuantPrecision) else str(precision).lower()
    return p_str in ("fp8", "nvfp4", "mxfp4")


def is_native_quantization(precision: Union[str, QuantPrecision]) -> bool:
    """Return True if precision produces native hardware-accelerated ONNX graph operators."""
    p_str = precision.value if isinstance(precision, QuantPrecision) else str(precision).lower()
    return p_str in ("fp16", "int8", "w4a16")


def get_quantization_execution_mode(precision: Union[str, QuantPrecision]) -> str:
    """Return human-readable execution mode for the precision."""
    if is_simulated_quantization(precision):
        return "數值模擬 (Weight Emulation / Fake Quant)"
    elif is_native_quantization(precision):
        return "實測硬體 (Native Graph Operators)"
    return "基準模型 (FP32 Baseline)"


def quantize_fp8(
    input_model_path: str,
    output_model_path: str,
    per_channel: bool = True,
) -> Path:
    """Quantize ONNX model weights to Float8 E4M3FN precision via weight-only emulation (fake quantization).

    NOTE (Transparency Notice):
    Weights are clamped and projected to the Float8 E4M3FN grid, then stored as floating-point
    initializers. This allows mathematical evaluation of accuracy loss, SNR, and cosine similarity
    without requiring native Float8 Tensor Core runtime execution operators in ONNX Runtime.

    Args:
        input_model_path: Path to the input FP32/FP16 ONNX model.
        output_model_path: Path to save the quantized FP8 ONNX model.
        per_channel: Whether to scale weights per-channel or per-tensor.
    """
    import ml_dtypes

    print(f"[*] 執行 FP8 (E4M3FN) 數值誤差模擬量化 (Weight Emulation / Fake Quantization, per_channel={per_channel}): {input_model_path}")
    print("    ℹ️ 說明: 權重投影至 FP8 格點後以浮點存儲，用於評估數值特徵相似度 (非原生硬體 8-bit MatMul 算子)")
    model = onnx.load(str(input_model_path))
    FP8_MAX = 448.0
    count = 0
    total_params = 0

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            orig_shape = arr.shape
            if per_channel:
                axes = tuple(range(1, len(orig_shape)))
                max_vals = np.max(np.abs(arr), axis=axes, keepdims=True)
            else:
                max_vals = np.max(np.abs(arr))
            scales = np.maximum(max_vals / FP8_MAX, 1e-12).astype(np.float32)
            scaled = np.clip(arr / scales, -FP8_MAX, FP8_MAX)
            fp8_arr = scaled.astype(ml_dtypes.float8_e4m3fn)
            dequant = (fp8_arr.astype(np.float32) * scales).reshape(orig_shape)
            new_init = numpy_helper.from_array(dequant.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)

    print(f"    共完成 {count} 個矩陣權重之 FP8 E4M3FN 模擬量化 (合計 {total_params / 1e6:.2f} M 參數)")
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def quantize_nvfp4(
    input_model_path: str,
    output_model_path: str,
    block_size: int = 32,
) -> Path:
    """Quantize ONNX model weights to NVIDIA Blackwell FP4 (NVFP4, E2M1) format via weight-only emulation.

    NOTE (Transparency Notice):
    Uses NVIDIA Blackwell two-level scaling:
    tensor = global_scale * (block_scale_fp8 * q_fp4)
    where each block (size 32) is quantized to the E2M1 FP4 grid with FP8 scale factor.
    Dequantized float initializers are stored for numerical verification and accuracy evaluation.
    """
    import ml_dtypes

    print(f"[*] 執行 NVIDIA NVFP4 (E2M1 + Block {block_size} FP8 Scale) 數值誤差模擬量化: {input_model_path}")
    print("    ℹ️ 說明: 權重投影至 NVFP4 格點後以浮點存儲，用於評估數值特徵相似度 (非原生 Blackwell NVFP4 算子)")
    FP4_E2M1_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    FP4_MAX = 6.0

    def _quant_nvfp4(tensor: np.ndarray) -> np.ndarray:
        orig_shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        max_global = float(np.max(np.abs(flat)))
        if max_global < 1e-12:
            return tensor.copy()

        pad_len = (block_size - (len(flat) % block_size)) % block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len))

        blocks = flat.reshape(-1, block_size)
        block_max = np.max(np.abs(blocks), axis=1, keepdims=True)

        global_scale = max_global / 448.0
        block_scales = block_max / (global_scale * FP4_MAX)
        block_scales_fp8 = np.clip(block_scales, 0.0, 448.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
        effective_scale = np.maximum(global_scale * block_scales_fp8, 1e-12)

        norm_blocks = blocks / effective_scale
        sign = np.sign(norm_blocks)
        abs_x = np.clip(np.abs(norm_blocks), 0.0, FP4_MAX)
        diffs = np.abs(abs_x[..., np.newaxis] - FP4_E2M1_VALS)
        nearest_idx = np.argmin(diffs, axis=-1)
        q_blocks = sign * FP4_E2M1_VALS[nearest_idx]

        dequant = (q_blocks * effective_scale).flatten()
        if pad_len > 0:
            dequant = dequant[:-pad_len]
        return dequant.reshape(orig_shape)

    model = onnx.load(str(input_model_path))
    count = 0
    total_params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            q_arr = _quant_nvfp4(arr)
            new_init = numpy_helper.from_array(q_arr.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)

    print(f"    共完成 {count} 個矩陣權重之 NVFP4 (E2M1) 模擬量化 (合計 {total_params / 1e6:.2f} M 參數)")
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def quantize_mxfp4(
    input_model_path: str,
    output_model_path: str,
    block_size: int = 32,
) -> Path:
    """Quantize ONNX model weights to OCP Microscaling FP4 (MXFP4, E2M1 + E8M0 scale) format via weight-only emulation.

    NOTE (Transparency Notice):
    Follows Open Compute Project (OCP) MX Specification with block size 32
    and shared power-of-2 (2^k) E8M0 scale factor. Dequantized float initializers
    are stored for numerical verification and accuracy evaluation.
    """
    print(f"[*] 執行 OCP Microscaling MXFP4 (E2M1 + E8M0 2^k Scale, Block {block_size}) 數值誤差模擬量化: {input_model_path}")
    print("    ℹ️ 說明: 權重投影至 MXFP4 格點後以浮點存儲，用於評估數值特徵相似度 (非原生 MXFP4 算子)")
    FP4_E2M1_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    FP4_MAX = 6.0

    def _quant_mxfp4(tensor: np.ndarray) -> np.ndarray:
        orig_shape = tensor.shape
        flat = tensor.flatten().astype(np.float32)
        pad_len = (block_size - (len(flat) % block_size)) % block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len))

        blocks = flat.reshape(-1, block_size)
        max_vals = np.max(np.abs(blocks), axis=1, keepdims=True)
        ratios = np.maximum(max_vals / FP4_MAX, 1e-12)
        exp = np.ceil(np.log2(ratios))
        scales_e8m0 = np.power(2.0, exp).astype(np.float32)

        norm_blocks = blocks / scales_e8m0
        sign = np.sign(norm_blocks)
        abs_x = np.clip(np.abs(norm_blocks), 0.0, FP4_MAX)
        diffs = np.abs(abs_x[..., np.newaxis] - FP4_E2M1_VALS)
        nearest_idx = np.argmin(diffs, axis=-1)
        q_blocks = sign * FP4_E2M1_VALS[nearest_idx]

        dequant = (q_blocks * scales_e8m0).flatten()
        if pad_len > 0:
            dequant = dequant[:-pad_len]
        return dequant.reshape(orig_shape)

    model = onnx.load(str(input_model_path))
    count = 0
    total_params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim >= 2 and np.issubdtype(arr.dtype, np.floating):
            count += 1
            total_params += arr.size
            q_arr = _quant_mxfp4(arr)
            new_init = numpy_helper.from_array(q_arr.astype(arr.dtype), name=init.name)
            init.CopyFrom(new_init)

    print(f"    共完成 {count} 個矩陣權重之 MXFP4 (E2M1 + E8M0) 模擬量化 (合計 {total_params / 1e6:.2f} M 參數)")
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path


def quantize_w4a16(
    input_model_path: str,
    output_model_path: str,
    block_size: int = 128,
    is_symmetric: bool = True,
) -> Path:
    """Quantize ONNX model weights to 4-bit integer with 16-bit/32-bit activations (W4A16).

    Uses ONNX Runtime MatMulNBitsQuantizer to produce optimized MatMulNBits operators.
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    print(f"[*] 執行 W4A16 權重量化 (block_size={block_size}, symmetric={is_symmetric}): {input_model_path}")
    model = onnx.load(str(input_model_path))
    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=is_symmetric,
        bits=4,
    )
    quantizer = MatMulNBitsQuantizer(
        model=model,
        algo_config=quant_config,
    )
    quantizer.process()
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(quantizer.model.model, str(out_path))
    return out_path


def _prepare_session_inputs(sess: ort.InferenceSession, base_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Adapt test inputs to match the expected data types and shapes of the ONNX session."""
    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        if name in base_data:
            val = base_data[name]
            if "float16" in inp.type:
                feed[name] = val.astype(np.float16)
            elif "float" in inp.type:
                feed[name] = val.astype(np.float32)
            elif "int64" in inp.type:
                feed[name] = val.astype(np.int64)
            elif "int32" in inp.type:
                feed[name] = val.astype(np.int32)
            else:
                feed[name] = val
        else:
            # Generate fallback array if input name doesn't match
            dtype = np.float16 if "float16" in inp.type else (np.float32 if "float" in inp.type else np.int64)
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            if np.issubdtype(dtype, np.floating):
                feed[name] = np.random.randn(*shape).astype(dtype)
            else:
                feed[name] = np.ones(shape, dtype=dtype)
    return feed


def benchmark_model_latency(
    sess_orig: ort.InferenceSession,
    sess_quant: ort.InferenceSession,
    feed_orig: Dict[str, np.ndarray],
    feed_quant: Dict[str, np.ndarray],
    warmup_runs: int = 5,
    bench_runs: int = 15,
) -> LatencyBenchmark:
    """Run warm-up and multi-iteration benchmark to accurately measure inference latency and speedup."""
    # 1. Warm-up
    for _ in range(warmup_runs):
        sess_orig.run(None, feed_orig)
        sess_quant.run(None, feed_quant)

    # 2. Benchmark baseline model
    orig_times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        sess_orig.run(None, feed_orig)
        orig_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. Benchmark quantized model
    quant_times = []
    for _ in range(bench_runs):
        t0 = time.perf_counter()
        sess_quant.run(None, feed_quant)
        quant_times.append((time.perf_counter() - t0) * 1000.0)

    orig_mean = float(np.mean(orig_times))
    orig_std = float(np.std(orig_times))
    quant_mean = float(np.mean(quant_times))
    quant_std = float(np.std(quant_times))

    speedup = orig_mean / max(quant_mean, 1e-6)
    latency_reduction = (1.0 - quant_mean / max(orig_mean, 1e-6)) * 100.0
    orig_fps = 1000.0 / max(orig_mean, 1e-6)
    quant_fps = 1000.0 / max(quant_mean, 1e-6)

    bench = LatencyBenchmark(
        orig_mean_ms=orig_mean,
        orig_std_ms=orig_std,
        quant_mean_ms=quant_mean,
        quant_std_ms=quant_std,
        speedup=speedup,
        latency_reduction_pct=latency_reduction,
        orig_fps=orig_fps,
        quant_fps=quant_fps,
        num_iterations=bench_runs,
        warmup_iterations=warmup_runs,
    )

    print("\n" + "=" * 98)
    print(f"🚀 推理速度與延遲基準評測 (Latency & Speedup Benchmark, {bench_runs} 次測試平均)")
    print("=" * 98)
    print(f"基準模型 (FP32) 平均延遲: {orig_mean:>7.2f} ms ± {orig_std:>5.2f} ms ({orig_fps:>6.2f} chunks/sec)")
    print(f"量化模型       平均延遲: {quant_mean:>7.2f} ms ± {quant_std:>5.2f} ms ({quant_fps:>6.2f} chunks/sec)")
    print("-" * 98)
    status_icon = "⚡" if speedup >= 1.0 else "⏳"
    trend = "提升" if speedup >= 1.0 else "降低"
    reduction_trend = "耗時縮減" if latency_reduction >= 0 else "耗時增加"
    print(f"{status_icon} 推理加速倍率 (Speedup)   : {speedup:>6.2f}x ({trend})")
    print(f"📉 延遲降低幅度 (Reduction) : {latency_reduction:>6.1f}% ({reduction_trend})")
    print("=" * 98 + "\n")

    return bench


def compare_model_similarity(
    original_model_path: str,
    quantized_model_path: str,
    min_cosine_threshold: float = 0.95,
    bench_runs: int = 15,
    warmup_runs: int = 5,
) -> Tuple[List[SimilarityMetric], LatencyBenchmark]:
    """Execute both models with identical inputs, compare numerical similarity, and benchmark latency."""
    print(f"\n[*] 執行量化前後數值相似度比對與推論速度評測...")
    print(f"    原始基準模型: {original_model_path}")
    print(f"    目標量化模型: {quantized_model_path}")

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_orig = ort.InferenceSession(original_model_path, providers=providers)
    sess_quant = ort.InferenceSession(quantized_model_path, providers=providers)

    # Generate reference inputs
    inputs_meta = sess_orig.get_inputs()
    base_data = {}
    for inp in inputs_meta:
        dtype = np.float32 if "float" in inp.type else np.int64
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if dtype == np.float32:
            base_data[inp.name] = np.random.randn(*shape).astype(dtype)
        else:
            base_data[inp.name] = np.ones(shape, dtype=dtype)

    feed_orig = _prepare_session_inputs(sess_orig, base_data)
    feed_quant = _prepare_session_inputs(sess_quant, base_data)

    # 1. Numerical output comparison run
    orig_outs = sess_orig.run(None, feed_orig)
    quant_outs = sess_quant.run(None, feed_quant)

    output_names = [o.name for o in sess_orig.get_outputs()]
    metrics_list = []

    print("\n" + "=" * 98)
    print(f"{'輸出張量名稱':<28} | {'餘弦相似度':<10} | {'皮爾森相關':<10} | {'最大絕對誤差':<12} | {'SNR (dB)':<8} | {'狀態'}")
    print("=" * 98)

    all_passed = True
    for name, o_out, q_out in zip(output_names, orig_outs, quant_outs):
        o_np = o_out.astype(np.float32)
        q_np = q_out.astype(np.float32)
        metric = compute_metrics(o_np, q_np, name, min_cosine_threshold=min_cosine_threshold)
        metrics_list.append(metric)
        if not metric.passed:
            all_passed = False

        print(
            f"{metric.tensor_name:<28} | "
            f"{metric.cosine_similarity:<10.6f} | "
            f"{metric.pearson_correlation:<10.6f} | "
            f"{metric.max_abs_error:<12.6e} | "
            f"{metric.snr_db:<8.2f} | "
            f"{metric.status}"
        )

    print("=" * 98)
    if all_passed:
        print("[✓] 相似度檢驗通過！量化模型精確度優異，無顯著效能與精度損失。\n")
    else:
        print("[!] 警告：部分輸出張量餘弦相似度低於門檻，建議評估是否改用 FP16 或增加校正資料。\n")

    # 2. Multi-iteration latency & throughput benchmark
    benchmark = benchmark_model_latency(
        sess_orig=sess_orig,
        sess_quant=sess_quant,
        feed_orig=feed_orig,
        feed_quant=feed_quant,
        warmup_runs=warmup_runs,
        bench_runs=bench_runs,
    )

    return metrics_list, benchmark


def auto_quantize_and_verify(
    input_model: str,
    output_model: Optional[str] = None,
    precision: str = "auto",
    min_cosine_threshold: float = 0.95,
    bench_runs: int = 15,
) -> Tuple[Path, DeviceInfo, List[SimilarityMetric], LatencyBenchmark]:
    """Automatically detects hardware, selects optimal precision, quantizes, and verifies similarity & speedup."""
    input_path = Path(os.path.expanduser(input_model)).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    # 1. Detect hardware across Linux/Windows
    print("[*] 正在檢測系統硬體規格與運算裝置能力 (Linux / Windows)...")
    dev_info = get_system_device_info()
    print(f"\n{'-'*60}\n{dev_info}\n{'-'*60}\n")

    # 2. Determine target precision
    if precision.lower() == "auto":
        target_precision = dev_info.recommended_precision
        print(f"[*] 自動硬體判定策略已啟用：選定量化精度 -> {target_precision.value.upper()}")
    else:
        try:
            target_precision = QuantPrecision(precision.lower())
            print(f"[*] 使用者手動指定量化精度: {target_precision.value.upper()}")
        except ValueError:
            raise ValueError(
                f"不支援的量化精度: {precision}。支援選項: auto, fp16, int8, fp8, nvfp4, mxfp4, w4a16"
            )

    # 3. Determine output path
    if output_model:
        out_path = Path(os.path.expanduser(output_model)).resolve()
    else:
        out_path = input_path.with_name(f"{input_path.stem}_{target_precision.value}{input_path.suffix}")

    # 4. Perform quantization
    t_start = time.time()
    if target_precision == QuantPrecision.FP16:
        result_path = quantize_fp16(str(input_path), str(out_path), keep_io_types=False)
    elif target_precision == QuantPrecision.INT8:
        result_path = quantize_int8(str(input_path), str(out_path))
    elif target_precision == QuantPrecision.FP8:
        result_path = quantize_fp8(str(input_path), str(out_path))
    elif target_precision == QuantPrecision.NVFP4:
        result_path = quantize_nvfp4(str(input_path), str(out_path))
    elif target_precision == QuantPrecision.MXFP4:
        result_path = quantize_mxfp4(str(input_path), str(out_path))
    elif target_precision == QuantPrecision.W4A16:
        result_path = quantize_w4a16(str(input_path), str(out_path))
    else:
        print(f"[*] 精確度為 {target_precision.value}，保持原始 FP32 模型。")
        result_path = input_path

    elapsed = time.time() - t_start
    orig_mb = input_path.stat().st_size / (1024 * 1024)
    quant_mb = result_path.stat().st_size / (1024 * 1024)
    ratio = (1 - quant_mb / orig_mb) * 100

    exec_mode = get_quantization_execution_mode(target_precision)
    print(f"\n[✓] 量化處理完成！耗時 {elapsed:.2f} 秒 (執行模式: {exec_mode})")
    print(f"    原始模型大小: {orig_mb:.2f} MB")
    print(f"    量化模型大小: {quant_mb:.2f} MB (節省 {ratio:.1f}% 空間)")
    if is_simulated_quantization(target_precision):
        print("    ℹ️ 備註: 該精度為權重數值模擬量化 (Weight Emulation)，輸出模型以浮點儲存投影權重以評估特徵保真度。")
    print(f"    量化模型儲存至: {result_path}")

    # 5. Similarity comparison and latency benchmark against original model
    metrics, benchmark = compare_model_similarity(
        original_model_path=str(input_path),
        quantized_model_path=str(result_path),
        min_cosine_threshold=min_cosine_threshold,
        bench_runs=bench_runs,
    )

    return result_path, dev_info, metrics, benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nemotron-3 Diarization 硬體自動檢測、量化、相似度比對與推論加速評測管線"
    )
    default_model = "models/nemotron_diarization/Nemotron-3-Diarization.onnx"
    if not os.path.exists(default_model) and os.path.exists("models/Nemotron-3-Diarization.onnx"):
        default_model = "models/Nemotron-3-Diarization.onnx"

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_model,
        help="輸入的原始 FP32 ONNX 模型路徑 (預設: models/nemotron_diarization/Nemotron-3-Diarization.onnx)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="輸出的量化 ONNX 模型路徑 (預設: 自動命名，例如 _fp16.onnx, _fp8.onnx, _nvfp4.onnx 等)",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        choices=["auto", "fp16", "int8", "fp8", "nvfp4", "mxfp4", "w4a16"],
        default="auto",
        help="量化精度模式 (支援: auto, fp16, int8, fp8, nvfp4, mxfp4, w4a16)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="餘弦相似度容許最低門檻 (預設: 0.95)",
    )
    parser.add_argument(
        "--bench-runs",
        type=int,
        default=15,
        help="推論速度測試重複次數 (預設: 15)",
    )

    args = parser.parse_args()
    auto_quantize_and_verify(
        input_model=args.model,
        output_model=args.output,
        precision=args.precision,
        min_cosine_threshold=args.threshold,
        bench_runs=args.bench_runs,
    )


if __name__ == "__main__":
    main()
