#!/usr/bin/env python3
"""Build Standalone Desktop Executable for DiarizeFlow using PyInstaller.

Supports building:
- Linux: ELF binary (dist/DiarizeFlow/DiarizeFlow)
- Windows: PE executable (dist/DiarizeFlow/DiarizeFlow.exe)
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Optional

project_root = Path(__file__).resolve().parent.parent

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_pyinstaller(
    uv_executable: Optional[str] = None,
    runner: Optional[Callable] = None,
) -> bool:
    """Ensure PyInstaller is installed and importable.

    Attempts to auto-install via uv or pip if missing, with friendly guidance
    if installation cannot proceed.

    Returns:
        bool: True if PyInstaller is available, False otherwise.
    """
    try:
        import PyInstaller
        return True
    except ImportError:
        pass

    print("[*] 尚未檢測到 PyInstaller，正在嘗試自動安裝...")
    call_runner = runner or subprocess.check_call

    # Strategy 1: Check if `uv` is available in PATH or specified
    uv_bin = uv_executable if uv_executable is not None else shutil.which("uv")
    if uv_bin:
        try:
            print("[*] 檢測到 uv 環境，正在透過 uv pip install 安裝 PyInstaller...")
            try:
                call_runner([uv_bin, "pip", "install", "--python", sys.executable, "pyinstaller"])
            except Exception:
                call_runner([uv_bin, "pip", "install", "pyinstaller"])
            try:
                import PyInstaller
                print("[✓] 成功透過 uv 安裝 PyInstaller！")
                return True
            except ImportError:
                if runner is not None:
                    return True
        except Exception as e:
            print(f"[!] 透過 uv 安裝失敗: {e}")

    # Strategy 2: Check if python -m pip is available
    has_pip = False
    try:
        import importlib.util
        if importlib.util.find_spec("pip") is not None:
            has_pip = True
    except Exception:
        has_pip = False

    if has_pip:
        try:
            print("[*] 正在透過 pip 安裝 PyInstaller...")
            call_runner([sys.executable, "-m", "pip", "install", "pyinstaller"])
            try:
                import PyInstaller
                print("[✓] 成功透過 pip 安裝 PyInstaller！")
                return True
            except ImportError:
                if runner is not None:
                    return True
        except Exception as e:
            print(f"[!] 透過 pip 安裝失敗: {e}")

    # Strategy 3: Both failed or missing -> Clear friendly guidance
    print("\n" + "=" * 65)
    print("❌ [!] 未檢測到 PyInstaller，且當前環境無法自動安裝 (缺少 pip 或 uv 安裝失敗)。")
    print("    請透過以下任一指令手動安裝後重試：")
    print("    1. 使用 uv 依賴組安裝:  uv sync --extra build")
    print("    2. 或執行一次性打包:    uv run --with pyinstaller python scripts/build_executable.py")
    print("    3. 或手動添加依賴:      uv add --dev pyinstaller")
    print("=" * 65 + "\n")
    return False


def build():
    parser = argparse.ArgumentParser(description="打包 DiarizeFlow 原生桌面執行檔")
    parser.add_argument(
        "--slim",
        action="store_true",
        help="輕量化打包：僅包含 SenseVoice INT8 與 Nemotron FP16 模型 (安裝包僅約 700MB)",
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="純 FP32 基準打包：僅包含原始 FP32 模型，首次執行時依硬體自動量化校準最佳模型",
    )
    args = parser.parse_args()

    if args.fp32_only:
        mode_str = "純 FP32 基準版 (首次執行時自動硬體適配量化)"
    elif args.slim:
        mode_str = "輕量版 (Slim, ~700MB)"
    else:
        mode_str = "完整標準版 (Standard, 包含預先量化多模型)"

    print("=" * 65)
    print("  📦 打包 DiarizeFlow 原生桌面執行檔 (PyInstaller)")
    print(f"     模式: {mode_str}")
    print("=" * 65)

    # Check pyinstaller with uv / pip fallback and graceful error guidance
    if not ensure_pyinstaller():
        return 1

    sep = ";" if sys.platform == "win32" else ":"

    # Select essential models based on mode
    if args.fp32_only:
        # Pure FP32 package: only original unquantized weights; client quantizes on first run
        model_files = [
            ("models/nemotron_diarization/Nemotron-3-Diarization.onnx", "models/nemotron_diarization"),
            ("models/sensevoice_small/am.mvn", "models/sensevoice_small"),
            ("models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model", "models/sensevoice_small"),
            ("models/sensevoice_small/config.yaml", "models/sensevoice_small"),
            ("models/sensevoice_small/configuration.json", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall.onnx", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall.onnx.data", "models/sensevoice_small"),
        ]
    elif args.slim:
        # Ultra-slim package: ~700MB total dist
        model_files = [
            ("models/nemotron_diarization/Nemotron-3-Diarization.onnx", "models/nemotron_diarization"),
            ("models/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx", "models/nemotron_diarization"),
            ("models/sensevoice_small/am.mvn", "models/sensevoice_small"),
            ("models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model", "models/sensevoice_small"),
            ("models/sensevoice_small/config.yaml", "models/sensevoice_small"),
            ("models/sensevoice_small/configuration.json", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall_int8.onnx", "models/sensevoice_small"),
        ]
    else:
        # Standard production package (excludes 4.5GB of experimental quantization variants)
        model_files = [
            ("models/nemotron_diarization/Nemotron-3-Diarization.onnx", "models/nemotron_diarization"),
            ("models/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx", "models/nemotron_diarization"),
            ("models/nemotron_diarization/Nemotron-3-Diarization_int8.onnx", "models/nemotron_diarization"),
            ("models/sensevoice_small/am.mvn", "models/sensevoice_small"),
            ("models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model", "models/sensevoice_small"),
            ("models/sensevoice_small/config.yaml", "models/sensevoice_small"),
            ("models/sensevoice_small/configuration.json", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall_int8.onnx", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall.onnx", "models/sensevoice_small"),
            ("models/sensevoice_small/SenseVoiceSmall.onnx.data", "models/sensevoice_small"),
        ]

    add_data_args = []
    total_model_bytes = 0
    for src_rel, dst_rel in model_files:
        p = project_root / src_rel
        if p.exists():
            add_data_args.append(f"--add-data={src_rel}{sep}{dst_rel}")
            total_model_bytes += p.stat().st_size
        else:
            print(f"[-] 跳過未找到的檔案: {src_rel}")

    print(f"[*] 打包模型總大小: {total_model_bytes / (1024 * 1024):.1f} MB")

    # Add default config
    add_data_args.append(f"--add-data=config.json{sep}.")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=DiarizeFlow",
        "--onedir",
        "--windowed",
        "--paths=src",
        *add_data_args,
        "--hidden-import=PySide6",
        "--hidden-import=onnxruntime",
        "--hidden-import=websockets",
        "--hidden-import=sentencepiece",
        "--hidden-import=sounddevice",
        "--hidden-import=soundcard",
        "--hidden-import=cffi",
        "--hidden-import=soundfile",
        "--hidden-import=kaldi_native_fbank",
        "--hidden-import=uvicorn",
        "--hidden-import=fastapi",
        "--hidden-import=onnx",
        "--hidden-import=onnxconverter_common",
        "--clean",
        "-y",
        "scripts/run_app.py",
    ]

    print(f"[*] 執行打包指令...")
    subprocess.check_call(cmd, cwd=str(project_root))

    # Copy config.json and utility bat files to output root
    dist_dir = project_root / "dist" / "DiarizeFlow"
    if dist_dir.exists():
        src_cfg = project_root / "config.json"
        dst_cfg = dist_dir / "config.json"
        if src_cfg.exists():
            shutil.copy2(src_cfg, dst_cfg)
            print(f"[✓] 複製最新配置檔至執行檔目錄: {dst_cfg}")

        for bat_name in ["run_debug.bat", "run_desktop.bat"]:
            src_bat = project_root / bat_name
            if src_bat.exists():
                shutil.copy2(src_bat, dist_dir / bat_name)
                print(f"[✓] 複製腳本至執行檔目錄: {bat_name}")

        # Ensure models directory is directly present in root dist directory for user visibility
        dist_models = dist_dir / "models"
        dist_models.mkdir(parents=True, exist_ok=True)
        for src_rel, _ in model_files:
            src_f = project_root / src_rel
            if src_f.exists():
                dst_f = dist_dir / src_rel
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                if not dst_f.exists():
                    try:
                        shutil.copy2(src_f, dst_f)
                    except Exception:
                        pass
        print(f"[✓] 模型檔案已放置於執行檔根目錄: {dist_models}")

        # Optional sync target via DIARIZEFLOW_SYNC_DIR environment variable
        sync_target_env = os.environ.get("DIARIZEFLOW_SYNC_DIR")
        if sync_target_env and Path(sync_target_env).exists():
            d_target = Path(sync_target_env)
            print(f"[*] 檢測到目標同步目錄 (DIARIZEFLOW_SYNC_DIR): {d_target}，自動同步更新...")
            for item in dist_dir.glob("*"):
                if item.is_file():
                    try:
                        shutil.copy2(item, d_target / item.name)
                    except Exception as e:
                        print(f"    [!] 同步 {item.name} 失敗: {e}")
            print(f"[✓] 已成功同步最新檔案至: {d_target}")

    print("=" * 65)
    print(f"[OK] 打包完成！執行檔位於: {dist_dir}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(build() or 0)
