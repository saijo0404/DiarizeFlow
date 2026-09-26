#!/usr/bin/env bash
# ==============================================================================
# DiarizeFlow - Linux Native Qt Transparent Overlay Launcher
# 跨平台 (Linux) 原生 Qt 桌面透明飄浮字幕啟動腳本
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  🚀 啟動 DiarizeFlow 原生桌面飄浮視窗 (Linux Qt Overlay)"
echo "=================================================================="

# Ensure DISPLAY or WAYLAND_DISPLAY is set
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "[!] 警告: 未檢測到圖形介面環境變數 (DISPLAY 或 WAYLAND_DISPLAY)。"
    echo "    若在 SSH 連線中，請使用 ssh -X / -Y 或在桌面環境的終端機中執行。"
fi

# Prefer uv if available, fallback to .venv or system python
if command -v uv &> /dev/null; then
    exec uv run python scripts/run_app.py --mode desktop "$@"
elif [ -f ".venv/bin/python" ]; then
    exec .venv/bin/python scripts/run_app.py --mode desktop "$@"
else
    exec python3 scripts/run_app.py --mode desktop "$@"
fi
