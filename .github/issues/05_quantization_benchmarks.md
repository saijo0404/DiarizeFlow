# [Architecture/Benchmark]: Simulated weight-only quantizations (FP8/NVFP4/MXFP4) lack native ONNX operators and benchmark transparency

**Labels**: `enhancement`, `documentation`

## Description
There is a fundamental discrepancy between the advertised quantization acceleration capabilities in the documentation/benchmarks and the actual ONNX graphs produced:

### 1. Simulated Weight Emulation (Fake Quantization) in `quantize.py`
In [`src/diarizeflow/quantize.py:229-232`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/quantize.py#L229-L232), the FP8, NVFP4, and MXFP4 quantization functions quantize weights to their respective grid and then *immediately dequantize back to Float32* before storing them in ONNX initializers:
```python
dequant = (fp8_arr.astype(np.float32) * scales).reshape(orig_shape)
new_init = numpy_helper.from_array(dequant.astype(arr.dtype), name=init.name)
init.CopyFrom(new_init)
```
- The output ONNX files (`_fp8.onnx`, `_nvfp4.onnx`, `_mxfp4.onnx`) are identical in byte size to the FP32 model (~398MB).
- The graph uses standard FP32 `MatMul` nodes.
- When run in ONNX Runtime, no hardware Tensor Core acceleration (such as Ada Lovelace FP8 or Blackwell NVFP4) is utilized. This is numerical error emulation, not a natively accelerated quantized model graph.

### 2. Static / Hardcoded Benchmark Outputs in `benchmark_whisper.py`
In [`scripts/benchmark_whisper.py:238-275`](file:///home/yijun/Project/DiarizeFlow/scripts/benchmark_whisper.py#L238-L275), similarity metrics (e.g. `cos_sim = 0.992680`, `text_sim = 97.8%`) are hardcoded constants. Faster-Whisper's underlying CTranslate2 engine does not support `float8_e4m3fn`, `nvfp4`, or `mxfp4` execution; the script catches the failure and computes theoretical latencies by scaling FP16 timing by a multiplier (`lat_factor / 0.45`), presenting them as empirical benchmark measurements.

## Recommendations
1. **Documentation Transparency**: Clearly document in `README.md` that FP8, NVFP4, and MXFP4 currently represent *mathematical weight-quantization emulation* to evaluate accuracy loss and cosine similarity prior to deployment, rather than hardware-accelerated 4-bit ONNX runtime execution.
2. **Benchmark Reporting**: Clearly delineate between measured hardware runtime numbers and theoretical/simulated estimations.
