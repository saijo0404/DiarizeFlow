#!/usr/bin/env python3
"""Run the DiarizeFlow Transparent Floating Overlay Frontend."""

import argparse
from pathlib import Path
import sys
import webbrowser

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.app.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="DiarizeFlow 透明飄浮即時字幕前端 UI")
    parser.add_argument("--config", type=str, default="config.json", help="設定檔路徑 (預設: config.json)")
    parser.add_argument("--mode", type=str, default="desktop", choices=["desktop", "web"], help="前端模式: desktop (PySide6 原生懸浮視窗) 或 web (瀏覽器/OBS 懸浮 HUD)")
    parser.add_argument("--port", type=int, default=8765, help="後端伺服器連接埠 (預設: 8765)")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.server.port = args.port

    if args.mode == "web":
        url = f"http://localhost:{args.port}/overlay"
        print(f"[*] 開啟 Web 透明浮動 HUD: {url}")
        webbrowser.open(url)
    else:
        try:
            from diarizeflow.app.frontend.desktop_overlay import run_overlay_app
            print(f"[*] 啟動 PySide6 原生透明飄浮字幕視窗 (Windows / Linux)...")
            sys.exit(run_overlay_app(cfg))
        except Exception as e:
            print(f"[!] 無法啟動原生桌面視窗 ({e})，自動切換至 Web 透明 HUD 模式...")
            url = f"http://localhost:{args.port}/overlay"
            webbrowser.open(url)


if __name__ == "__main__":
    main()
