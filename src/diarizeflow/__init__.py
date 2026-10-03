"""DiarizeFlow package."""

try:
    from diarizeflow.export_onnx import export_nemo_to_onnx, StreamingSortformerOnnxWrapper
except ImportError:
    export_nemo_to_onnx = None
    StreamingSortformerOnnxWrapper = None

try:
    from diarizeflow.hardware import (
        DeviceInfo,
        QuantPrecision,
        detect_nvidia_gpu,
        get_system_device_info,
        recommend_precision,
    )
except ImportError:
    pass

try:
    from diarizeflow.patches import apply_onnx_export_patches
except ImportError:
    apply_onnx_export_patches = None

try:
    from diarizeflow.quantize import (
        LatencyBenchmark,
        SimilarityMetric,
        auto_quantize_and_verify,
        benchmark_model_latency,
        compare_model_similarity,
        compute_metrics,
        quantize_fp16,
        quantize_int8,
        quantize_fp8,
        quantize_nvfp4,
        quantize_mxfp4,
        quantize_w4a16,
        is_simulated_quantization,
        is_native_quantization,
        get_quantization_execution_mode,
    )
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = [
    "export_nemo_to_onnx",
    "StreamingSortformerOnnxWrapper",
    "apply_onnx_export_patches",
    "detect_nvidia_gpu",
    "get_system_device_info",
    "recommend_precision",
    "QuantPrecision",
    "DeviceInfo",
    "quantize_fp16",
    "quantize_int8",
    "quantize_fp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
    "quantize_w4a16",
    "is_simulated_quantization",
    "is_native_quantization",
    "get_quantization_execution_mode",
    "auto_quantize_and_verify",
    "compare_model_similarity",
    "benchmark_model_latency",
    "compute_metrics",
    "SimilarityMetric",
    "LatencyBenchmark",
]
