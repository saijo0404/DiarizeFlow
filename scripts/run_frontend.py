#!/usr/bin/env python3
"""Run the DiarizeFlow Transparent Floating Overlay Frontend."""

import argparse
from pathlib import Path
# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="DiarizeFlow 透明飄浮即時字幕前端 UI")
    parser.add_argument("--config", type=str, default="config.json", help="設定檔路徑 (預設: config.json)")
    parser.add_argument("--mode", type=str, default="desktop", choices=["desktop"], help="前端模式 (預設: desktop 原生桌面 HUD)")
    parser.add_argument("--port", type=int, default=8765, help="後端伺服器連接埠 (預設: 8765)")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.server.port = args.port

    try:
        from diarizeflow.ui.desktop_overlay import run_overlay_app
        print("[*] 啟動 PySide6 原生透明飄浮字幕視窗 (Windows / Linux)...")
        sys.exit(run_overlay_app(cfg))
    except Exception as e:
        print(f"[!] 無法啟動原生桌面視窗: {e}")
        print("[!] 請確認系統具備桌面圖形環境 (如 X11/Wayland/Windows 桌面)。")
        sys.exit(1)


if __name__ == "__main__":
    main()
