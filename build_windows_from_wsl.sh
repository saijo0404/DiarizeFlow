#!/usr/bin/env bash
# ==============================================================================
# DiarizeFlow - Build Windows .exe from WSL using Windows Host Interop
# 在 WSL 環境下直接呼叫 Windows 宿主環境打包 Windows 專用執行檔 (.exe)
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  📦 在 WSL 中呼叫 Windows 宿主環境打包 DiarizeFlow (.exe)"
echo "=================================================================="

# Ensure UTF-8 is passed to Windows Python via WSLENV
export WSLENV="PYTHONUTF8:PYTHONIOENCODING"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Check if running in WSL
if [ ! -d "/mnt/c" ]; then
    echo "[!] 錯誤: 此腳本需在 WSL 環境下執行（未檢測到 /mnt/c）。"
    exit 1
fi

# 1. 優先使用已建立的 Windows 虛擬環境 (.venv-win)
if [ -f ".venv-win/Scripts/python.exe" ]; then
    echo "[*] 檢測到 Windows 虛擬環境 (.venv-win)，確保編譯依賴完整..."
    .venv-win/Scripts/python.exe -m pip install onnx onnxconverter-common 2>/dev/null || true
    .venv-win/Scripts/python.exe scripts/build_executable.py "$@"
    if [ -n "$DIARIZEFLOW_SYNC_DIR" ] && [ -d "$DIARIZEFLOW_SYNC_DIR" ]; then
        echo "[*] 自動同步最新打包產物至 $DIARIZEFLOW_SYNC_DIR ..."
        cp -r dist/DiarizeFlow/* "$DIARIZEFLOW_SYNC_DIR/" 2>/dev/null || true
        echo "[✓] 已自動同步更新至 $DIARIZEFLOW_SYNC_DIR！"
    fi
    exit 0
fi

# 2. 檢測 Windows uv
WIN_UV=""
if [ -f "/mnt/c/Users/$USER/.local/bin/uv.exe" ]; then
    WIN_UV="/mnt/c/Users/$USER/.local/bin/uv.exe"
fi
if [ -z "$WIN_UV" ]; then
    # Try finding uv.exe across Windows user directories
    FOUND=$(find /mnt/c/Users -maxdepth 4 -name "uv.exe" 2>/dev/null | head -n 1 || true)
    if [ -n "$FOUND" ]; then
        WIN_UV="$FOUND"
    fi
fi

if [ -n "$WIN_UV" ]; then
    echo "[*] 使用 Windows uv ($WIN_UV) 建立 Windows 環境並打包..."
    "$WIN_UV" venv .venv-win
    "$WIN_UV" pip install --python .venv-win/Scripts/python.exe pyside6 onnxruntime onnx onnxconverter-common fastapi uvicorn websockets sounddevice soundcard soundfile sentencepiece kaldi-native-fbank requests aiohttp scipy librosa pyinstaller
    chmod +x .venv-win/Scripts/*.exe 2>/dev/null || true
    exec .venv-win/Scripts/python.exe scripts/build_executable.py "$@"
fi

# 3. 透過 Windows PowerShell 安裝 uv 並打包
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
if [ -f "$POWERSHELL" ]; then
    echo "[*] 正在透過 Windows PowerShell 進行一鍵安裝與打包..."
    WIN_PATH="$(wslpath -w "$SCRIPT_DIR")"
    "$POWERSHELL" -ExecutionPolicy Bypass -NoProfile -Command "
        irm https://astral.sh/uv/install.ps1 | iex
        & \"\$env:USERPROFILE\.local\bin\uv.exe\" venv '$WIN_PATH\.venv-win'
        & \"\$env:USERPROFILE\.local\bin\uv.exe\" pip install --python '$WIN_PATH\.venv-win\Scripts\python.exe' pyside6 onnxruntime fastapi uvicorn websockets sounddevice soundfile sentencepiece kaldi-native-fbank requests aiohttp scipy librosa pyinstaller
        & '$WIN_PATH\.venv-win\Scripts\python.exe' '$WIN_PATH\scripts\build_executable.py'
    "
    exit 0
fi

echo "[!] 未找到合適的 Windows 執行工具。請在 Windows 宿主端執行 build_windows.bat。"
exit 1
