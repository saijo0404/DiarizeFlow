"""First-Run Hardware Detection and Auto-Quantization Calibration for DiarizeFlow.

When DiarizeFlow is packaged with only baseline FP32 models, this module runs
on the first launch to:
1. Detect local GPU architecture, compute capability, and VRAM.
2. Automatically generate and cache the optimal quantized model (FP16 / INT8)
   directly from the baseline FP32 ONNX weights.
3. Update config.json and mark hardware_calibrated = True.
4. Future runs skip this entirely with 0 overhead and direct high-speed startup!
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

from diarizeflow.app.config import AppConfig, resolve_app_path
from diarizeflow.hardware import QuantPrecision, get_system_device_info


def ensure_calibrated_models(config: AppConfig, force: bool = False) -> AppConfig:
    """Check hardware calibration on startup; auto-quantize if first run."""
    # Check if calibration already completed
    calib_marker = resolve_app_path("models/.calibrated")
    if getattr(config, "hardware_calibrated", False) and calib_marker.exists() and not force:
        print("[✓] 硬體加速與量化配置已完成校準，直接啟動最優推論模式（無等待秒開）")
        return config

    print("\n" + "=" * 75)
    print("⚙️  [DiarizeFlow 首次啟動硬體自適應校準]")
    print("   檢測到首次執行或環境變更，正在為當前硬體自動尋找並生成最佳量化模型...")
    print("   (此校準僅需執行一次，完成後未來每次啟動將直接秒開！)")
    print("=" * 75)

    t0 = time.perf_counter()

    # 1. Hardware Detection
    dev_info = get_system_device_info()
    print(f"[*] 運算平台: {dev_info.platform}")
    if dev_info.has_gpu:
        sm_str = f"SM {dev_info.compute_capability[0]}.{dev_info.compute_capability[1]}" if dev_info.compute_capability else ""
        vram_str = f"{dev_info.vram_gb:.1f} GB" if dev_info.vram_gb else ""
        print(f"[*] 檢測到 GPU: {dev_info.gpu_name} ({dev_info.arch_name}, {sm_str}, VRAM: {vram_str})")
    else:
        print(f"[*] 未檢測到獨立 NVIDIA GPU，採用 CPU 加速模式")

    rec_prec = dev_info.recommended_precision
    print(f"[*] 硬體最優量化策略: {rec_prec.value.upper()} ({dev_info.precision_reason})")

    # 2. Nemotron-3 Diarization Model Optimization
    try:
        from diarizeflow.quantize import quantize_fp16, quantize_int8

        nemo_base = resolve_app_path("models/nemotron_diarization/Nemotron-3-Diarization.onnx")
        nemo_fp16 = resolve_app_path("models/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx")
        nemo_int8 = resolve_app_path("models/nemotron_diarization/Nemotron-3-Diarization_int8.onnx")

        if nemo_base.exists():
            if dev_info.has_gpu:
                # GPU mode: generate FP16 if not exists
                target_fp16 = nemo_base.parent / "Nemotron-3-Diarization_fp16.onnx"
                if not target_fp16.exists():
                    print(f"[*] 正在為 GPU 生成 Nemotron FP16 加速模型: {target_fp16}...")
                    quantize_fp16(str(nemo_base), str(target_fp16))
                config.diarization.fp16_model_path = "models/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx"
                config.diarization.use_fp16 = True
                print(f"[✓] Nemotron 語者分離: 已啟用 FP16 GPU Tensor Core 加速 ({target_fp16.name})")
            else:
                # CPU mode: generate INT8 if not exists
                target_int8 = nemo_base.parent / "Nemotron-3-Diarization_int8.onnx"
                if not target_int8.exists():
                    print(f"[*] 正在為 CPU 生成 Nemotron INT8 輕量化模型: {target_int8}...")
                    quantize_int8(str(nemo_base), str(target_int8))
                config.diarization.model_path = "models/nemotron_diarization/Nemotron-3-Diarization_int8.onnx"
                config.diarization.use_fp16 = False
                print(f"[✓] Nemotron 語者分離: 已啟用 CPU INT8 輕量化推論 ({target_int8.name})")
    except Exception as e:
        print(f"[!] Nemotron 自動量化跳過或發生非致命異常: {e}，維持預設模型配置")

    # 3. SenseVoiceSmall ASR Model Optimization
    try:
        sense_base = resolve_app_path("models/sensevoice_small/SenseVoiceSmall.onnx")
        sense_int8 = resolve_app_path("models/sensevoice_small/SenseVoiceSmall_int8.onnx")

        if sense_base.exists():
            # If INT8 already exists or can be linked/used
            if sense_int8.exists():
                config.asr.model_path = "models/sensevoice_small/SenseVoiceSmall_int8.onnx"
                print(f"[✓] SenseVoiceSmall 語音辨識: 採用最速 INT8 最佳化模型")
            else:
                config.asr.model_path = "models/sensevoice_small/SenseVoiceSmall.onnx"
                print(f"[✓] SenseVoiceSmall 語音辨識: 採用原始高精度模型")
    except Exception as e:
        print(f"[!] SenseVoice 自動配置異常: {e}")

    # 4. Mark calibration completed and persist
    elapsed = time.perf_counter() - t0
    config.hardware_calibrated = True
    config._just_calibrated = True
    config._calib_gpu_name = dev_info.gpu_name or "CPU"
    config._calib_precision = rec_prec.value.upper()

    try:
        config.save()
        calib_marker.parent.mkdir(parents=True, exist_ok=True)
        with open(calib_marker, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "platform": dev_info.platform,
                "gpu": dev_info.gpu_name,
                "recommended_precision": rec_prec.value,
            }, indent=2))
        print(f"[✓] 硬體最優化校準完成 (耗時: {elapsed:.2f}s)！")
        print("    已寫入本地快取標記，未來啟動將直接秒開，永不需再次校準。\n")
    except Exception as e:
        print(f"[!] 儲存校準狀態警告: {e}")

    return config
