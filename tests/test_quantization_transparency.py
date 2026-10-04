"""Unit and integration tests for quantization transparency and weight emulation.

Tests verify:
1. Transparency query APIs (is_simulated_quantization, is_native_quantization, get_quantization_execution_mode).
2. Weight emulation behavior for FP8, NVFP4, and MXFP4 (verifying weights are projected and stored as valid float initializers).
3. ONNX model inference execution on emulated models.
"""

import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import pytest

from diarizeflow.core.quantize import (
    QuantPrecision,
    is_simulated_quantization,
    is_native_quantization,
    get_quantization_execution_mode,
    quantize_fp8,
    quantize_nvfp4,
    quantize_mxfp4,
)
from diarizeflow import (
    is_simulated_quantization as api_is_sim,
    is_native_quantization as api_is_native,
    get_quantization_execution_mode as api_get_mode,
)


def create_simple_onnx_model(model_path: Path, in_features: int = 32, out_features: int = 16):
    """Create a minimal ONNX model with a single MatMul and Add node for testing."""
    # Inputs & Outputs
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_features])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_features])

    # Initializers (weights and bias)
    np.random.seed(42)
    w_vals = (np.random.randn(in_features, out_features) * 2.0).astype(np.float32)
    b_vals = np.zeros((1, out_features), dtype=np.float32)

    W = numpy_helper.from_array(w_vals, name="W")
    B = numpy_helper.from_array(b_vals, name="B")

    matmul_node = helper.make_node("MatMul", ["X", "W"], ["matmul_out"], name="matmul1")
    add_node = helper.make_node("Add", ["matmul_out", "B"], ["Y"], name="add1")

    graph = helper.make_graph(
        [matmul_node, add_node],
        "SimpleModel",
        [X],
        [Y],
        [W, B],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, str(model_path))
    return model_path


class TestQuantizationTransparencyAPIs:
    """Test transparency helper functions exported at top-level."""

    def test_exported_in_init(self):
        assert api_is_sim is is_simulated_quantization
        assert api_is_native is is_native_quantization
        assert api_get_mode is get_quantization_execution_mode

    def test_simulated_quantization_flags(self):
        # Simulated precisions
        assert is_simulated_quantization("fp8") is True
        assert is_simulated_quantization("FP8") is True
        assert is_simulated_quantization(QuantPrecision.FP8) is True
        assert is_simulated_quantization("nvfp4") is True
        assert is_simulated_quantization(QuantPrecision.NVFP4) is True
        assert is_simulated_quantization("mxfp4") is True
        assert is_simulated_quantization(QuantPrecision.MXFP4) is True

        # Native precisions
        assert is_simulated_quantization("fp16") is False
        assert is_simulated_quantization(QuantPrecision.FP16) is False
        assert is_simulated_quantization("int8") is False
        assert is_simulated_quantization(QuantPrecision.INT8) is False
        assert is_simulated_quantization("w4a16") is False
        assert is_simulated_quantization(QuantPrecision.W4A16) is False
        assert is_simulated_quantization("fp32") is False

    def test_native_quantization_flags(self):
        # Native precisions
        assert is_native_quantization("fp16") is True
        assert is_native_quantization("FP16") is True
        assert is_native_quantization(QuantPrecision.FP16) is True
        assert is_native_quantization("int8") is True
        assert is_native_quantization(QuantPrecision.INT8) is True
        assert is_native_quantization("w4a16") is True
        assert is_native_quantization(QuantPrecision.W4A16) is True

        # Simulated precisions
        assert is_native_quantization("fp8") is False
        assert is_native_quantization(QuantPrecision.FP8) is False
        assert is_native_quantization("nvfp4") is False
        assert is_native_quantization(QuantPrecision.NVFP4) is False
        assert is_native_quantization("mxfp4") is False
        assert is_native_quantization(QuantPrecision.MXFP4) is False

    def test_get_execution_mode_descriptions(self):
        mode_fp8 = get_quantization_execution_mode("fp8")
        assert "數值模擬" in mode_fp8
        assert "Weight Emulation" in mode_fp8

        mode_nvfp4 = get_quantization_execution_mode(QuantPrecision.NVFP4)
        assert "數值模擬" in mode_nvfp4

        mode_mxfp4 = get_quantization_execution_mode("mxfp4")
        assert "數值模擬" in mode_mxfp4

        mode_fp16 = get_quantization_execution_mode("fp16")
        assert "實測硬體" in mode_fp16
        assert "Native" in mode_fp16

        mode_int8 = get_quantization_execution_mode(QuantPrecision.INT8)
        assert "實測硬體" in mode_int8

        mode_fp32 = get_quantization_execution_mode("fp32")
        assert "基準模型" in mode_fp32


class TestWeightEmulationExecution:
    """Verify that simulated weight quantizations produce valid floating-point models."""

    @pytest.fixture
    def test_model(self, tmp_path):
        model_path = tmp_path / "test_model.onnx"
        create_simple_onnx_model(model_path, in_features=32, out_features=16)
        return model_path

    def test_fp8_emulation_generates_valid_model(self, test_model, tmp_path):
        out_path = tmp_path / "model_fp8.onnx"
        res = quantize_fp8(str(test_model), str(out_path))
        assert Path(res).exists()

        loaded = onnx.load(str(out_path))
        # Check that weights are stored as floating point (weight emulation)
        for init in loaded.graph.initializer:
            if init.name == "W":
                assert init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
                arr = numpy_helper.to_array(init)
                assert arr.shape == (32, 16)
                assert arr.dtype == np.float32

        # Verify inference succeeds with ONNX Runtime
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        inp = np.ones((1, 32), dtype=np.float32)
        out = sess.run(None, {"X": inp})
        assert len(out) == 1
        assert out[0].shape == (1, 16)

    def test_nvfp4_emulation_generates_valid_model(self, test_model, tmp_path):
        out_path = tmp_path / "model_nvfp4.onnx"
        res = quantize_nvfp4(str(test_model), str(out_path), block_size=32)
        assert Path(res).exists()

        loaded = onnx.load(str(out_path))
        for init in loaded.graph.initializer:
            if init.name == "W":
                assert init.data_type == TensorProto.FLOAT
                arr = numpy_helper.to_array(init)
                assert arr.shape == (32, 16)

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        inp = np.ones((1, 32), dtype=np.float32)
        out = sess.run(None, {"X": inp})
        assert len(out) == 1
        assert out[0].shape == (1, 16)

    def test_mxfp4_emulation_generates_valid_model(self, test_model, tmp_path):
        out_path = tmp_path / "model_mxfp4.onnx"
        res = quantize_mxfp4(str(test_model), str(out_path), block_size=32)
        assert Path(res).exists()

        loaded = onnx.load(str(out_path))
        for init in loaded.graph.initializer:
            if init.name == "W":
                assert init.data_type == TensorProto.FLOAT
                arr = numpy_helper.to_array(init)
                assert arr.shape == (32, 16)

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        inp = np.ones((1, 32), dtype=np.float32)
        out = sess.run(None, {"X": inp})
        assert len(out) == 1
        assert out[0].shape == (1, 16)
