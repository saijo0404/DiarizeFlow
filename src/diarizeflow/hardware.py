"""Cross-platform hardware detection and quantization precision selection for Linux and Windows.

This module determines whether a GPU is present, extracts NVIDIA GPU architecture
and Compute Capability via the CUDA Driver API (nvcuda.dll on Windows, libcuda.so on Linux),
and recommends the optimal quantization precision.
"""

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class QuantPrecision(str, Enum):
    """Supported quantization precision targets."""
    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"
    W4A16 = "w4a16"
    NONE = "none"  # Keep FP32


@dataclass
class DeviceInfo:
    """Device hardware information and capability details."""
    platform: str
    has_gpu: bool
    gpu_name: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None
    vram_gb: Optional[float] = None
    arch_name: Optional[str] = None
    recommended_precision: QuantPrecision = QuantPrecision.INT8
    precision_reason: str = ""

    def __str__(self) -> str:
        if not self.has_gpu:
            return (
                f"平台: {self.platform} | 運算裝置: CPU (無偵測到 NVIDIA GPU)\n"
                f"建議量化精度: {self.recommended_precision.value.upper()} ({self.precision_reason})"
            )
        sm_str = f"SM {self.compute_capability[0]}.{self.compute_capability[1]}" if self.compute_capability else "未知"
        vram_str = f"{self.vram_gb:.2f} GB" if self.vram_gb else "未知"
        return (
            f"平台: {self.platform} | GPU: {self.gpu_name} ({self.arch_name}, {sm_str}, VRAM: {vram_str})\n"
            f"建議量化精度: {self.recommended_precision.value.upper()} ({self.precision_reason})"
        )


def _get_cuda_driver_lib():
    """Load the CUDA driver shared library across Windows and Linux."""
    sys_name = platform.system()
    if sys_name == "Windows":
        candidates = ["nvcuda.dll"]
        for lib in candidates:
            try:
                return ctypes.WinDLL(lib)
            except OSError:
                pass
    else:  # Linux / Unix
        candidates = [
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",  # WSL2 support
        ]
        for lib in candidates:
            try:
                return ctypes.CDLL(lib)
            except OSError:
                pass
    return None


def _get_arch_name(major: int, minor: int) -> str:
    """Map compute capability major and minor numbers to microarchitecture names."""
    if major >= 10:
        return "Blackwell (RTX 50 系列 / B100 / B200)"
    elif major == 9:
        return "Hopper (H100 / H200 / H800)"
    elif major == 8 and minor == 9:
        return "Ada Lovelace (RTX 40 系列 / L40)"
    elif major == 8:
        return "Ampere (RTX 30 系列 / A100 / A10)"
    elif major == 7 and minor == 5:
        return "Turing (RTX 20 系列 / T4 / GTX 16 系列)"
    elif major == 7:
        return "Volta (V100 / Titan V)"
    elif major == 6:
        return "Pascal (GTX 10 系列 / P100)"
    elif major == 5:
        return "Maxwell (GTX 9 系列)"
    return f"NVIDIA Architecture (SM {major}.{minor})"


def detect_nvidia_gpu() -> Optional[DeviceInfo]:
    """Detect NVIDIA GPU hardware using ctypes CUDA Driver API, with torch/smi fallback."""
    sys_name = platform.system()

    # Method 1: Direct ctypes via CUDA Driver API
    cuda = _get_cuda_driver_lib()
    if cuda is not None:
        try:
            # cuInit(0)
            if cuda.cuInit(0) == 0:
                count = ctypes.c_int()
                if cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0:
                    device = ctypes.c_int()
                    if cuda.cuDeviceGet(ctypes.byref(device), 0) == 0:
                        major = ctypes.c_int()
                        minor = ctypes.c_int()
                        # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                        # 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
                        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)

                        # GPU Name
                        name_buf = ctypes.create_string_buffer(256)
                        gpu_name = "NVIDIA GPU"
                        if cuda.cuDeviceGetName(name_buf, 256, device) == 0:
                            gpu_name = name_buf.value.decode("utf-8", errors="ignore").strip()

                        # Total Memory (prefer 64-bit v2 function to support GPUs with >4GB VRAM)
                        total_mem = ctypes.c_size_t()
                        vram_gb = None
                        if hasattr(cuda, "cuDeviceTotalMem_v2"):
                            if cuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device) == 0:
                                vram_gb = total_mem.value / (1024 ** 3)
                        elif hasattr(cuda, "cuDeviceTotalMem"):
                            if cuda.cuDeviceTotalMem(ctypes.byref(total_mem), device) == 0:
                                vram_gb = total_mem.value / (1024 ** 3)

                        cc = (major.value, minor.value)
                        arch = _get_arch_name(cc[0], cc[1])
                        rec_prec, reason = recommend_precision(has_gpu=True, compute_capability=cc, vram_gb=vram_gb)
                        return DeviceInfo(
                            platform=sys_name,
                            has_gpu=True,
                            gpu_name=gpu_name,
                            compute_capability=cc,
                            vram_gb=vram_gb,
                            arch_name=arch,
                            recommended_precision=rec_prec,
                            precision_reason=reason,
                        )
        except Exception:
            pass

    # Method 2: PyTorch fallback if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            arch = _get_arch_name(cc[0], cc[1])
            rec_prec, reason = recommend_precision(has_gpu=True, compute_capability=cc, vram_gb=vram_gb)
            return DeviceInfo(
                platform=sys_name,
                has_gpu=True,
                gpu_name=gpu_name,
                compute_capability=cc,
                vram_gb=vram_gb,
                arch_name=arch,
                recommended_precision=rec_prec,
                precision_reason=reason,
            )
    except Exception:
        pass

    # Method 3: nvidia-smi CLI fallback
    smi_cmd = shutil.which("nvidia-smi")
    if smi_cmd:
        try:
            res = subprocess.run(
                [smi_cmd, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                if lines:
                    parts = [p.strip() for p in lines[0].split(",")]
                    name = parts[0]
                    vram_gb = float(parts[1]) / 1024.0 if len(parts) > 1 else None
                    # Estimate compute capability from known naming
                    cc = None
                    if "5090" in name or "5080" in name or "5070" in name:
                        cc = (10, 0)
                    elif "4090" in name or "4080" in name or "4070" in name:
                        cc = (8, 9)
                    elif "3090" in name or "3080" in name or "3070" in name or "A100" in name:
                        cc = (8, 6)
                    elif "2080" in name or "T4" in name:
                        cc = (7, 5)
                    arch = _get_arch_name(cc[0], cc[1]) if cc else "NVIDIA GPU"
                    rec_prec, reason = recommend_precision(has_gpu=True, compute_capability=cc, vram_gb=vram_gb)
                    return DeviceInfo(
                        platform=sys_name,
                        has_gpu=True,
                        gpu_name=name,
                        compute_capability=cc,
                        vram_gb=vram_gb,
                        arch_name=arch,
                        recommended_precision=rec_prec,
                        precision_reason=reason,
                    )
        except Exception:
            pass

    return None


def recommend_precision(
    has_gpu: bool,
    compute_capability: Optional[Tuple[int, int]] = None,
    vram_gb: Optional[float] = None,
) -> Tuple[QuantPrecision, str]:
    """Determine the optimal quantization precision based on hardware characteristics."""
    if not has_gpu:
        return (
            QuantPrecision.INT8,
            "純 CPU 環境，推薦使用 Dynamic INT8 或 W4A16 獲得 AVX2/AVX-512/VNNI 加速與 70%+ 記憶體節省"
        )

    if compute_capability is not None:
        major, minor = compute_capability

        # Blackwell (RTX 50 系列, SM 10.x / SM 12.x)
        if major >= 10:
            return (
                QuantPrecision.FP16,
                f"Blackwell 架構 (SM {major}.{minor})：具備第 5 代 Tensor Core，完整支援 FP16、FP8、NVFP4、MXFP4、W4A16！推薦首選 FP16 (極致無損)，或 NVFP4 / FP8 (極限吞吐)"
            )

        # Hopper (SM 9.0) / Ada Lovelace (RTX 40 系列, SM 8.9)
        if major == 9 or (major == 8 and minor == 9):
            return (
                QuantPrecision.FP16,
                "Ada Lovelace / Hopper 架構 (SM 8.9+)：第 4 代 Tensor Core，支援 FP16、FP8 與 W4A16 高保真加速"
            )

        # Ampere (RTX 30 系列, SM 8.x) / Turing (RTX 20 系列, SM 7.5) / Volta (SM 7.0)
        if major >= 7:
            return (
                QuantPrecision.FP16,
                f"Tensor Core GPU (SM {major}.{minor})：支援硬體 FP16 與 W4A16 運算加速"
            )

        # Older Pascal (SM 6.x)
        return (
            QuantPrecision.INT8,
            f"舊代架構 (SM {major}.{minor})：缺乏專用 FP16 Tensor Core，推薦使用 INT8 (DP4A) 進行整數運算加速"
        )

    # Has GPU but unknown SM
    if vram_gb is not None and vram_gb < 4.0:
        return (
            QuantPrecision.INT8,
            "GPU 顯存小於 4GB，推薦使用 INT8 量化降低記憶體佔用"
        )

    return (
        QuantPrecision.FP16,
        "偵測到 GPU 裝置，預設推薦使用相容性最高、無損精度的 FP16 量化"
    )


def get_system_device_info() -> DeviceInfo:
    """Retrieve full system device information with auto-detected hardware and recommendation."""
    gpu_info = detect_nvidia_gpu()
    if gpu_info is not None:
        return gpu_info

    # CPU-only fallback
    sys_name = platform.system()
    rec_prec, reason = recommend_precision(has_gpu=False)
    return DeviceInfo(
        platform=sys_name,
        has_gpu=False,
        recommended_precision=rec_prec,
        precision_reason=reason,
    )
