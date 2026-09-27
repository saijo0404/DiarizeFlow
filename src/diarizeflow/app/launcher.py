"""Launcher module for DiarizeFlow Application."""

import argparse
from pathlib import Path
import socket
import sys
import threading
import time
import uvicorn

from diarizeflow.app.config import AppConfig
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline
from diarizeflow.app.backend.server import create_app


def find_available_port(host: str = "127.0.0.1", start_port: int = 8765, max_attempts: int = 50) -> int:
    """Find the first available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start_port


def start_server_in_thread(app, host: str, port: int):
    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        return server, thread
    except Exception as e:
        print(f"[!] Warning: Cannot start web server on {host}:{port}: {e}")
        return None, None


def run_cli():
    parser = argparse.ArgumentParser(description="DiarizeFlow 一鍵啟動後端與透明飄浮前端")
    parser.add_argument("--mode", type=str, default="desktop", choices=["desktop"], help="前端模式 (預設: desktop 原生桌面 HUD)")
    parser.add_argument("--port", type=int, default=8765, help="連接埠 (預設: 8765)")
    parser.add_argument("--target-lang", type=str, default="繁體中文", help="目標翻譯語言 (預設: 繁體中文)")
    parser.add_argument("--llm-provider", type=str, default="vllm", help="LLM 供應商 (vllm, llama.cpp, openai, claude, bypass)")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:8000/v1", help="LLM Base URL")
    args = parser.parse_args()

    cfg = AppConfig.load("config.json")
    
    # Auto-detect available port if requested port is occupied
    actual_port = find_available_port("127.0.0.1", args.port)
    if actual_port != args.port:
        print(f"[*] 注意: 預設連接埠 {args.port} 已被佔用，自動切換至可用連接埠: {actual_port}")
    cfg.server.port = actual_port

    cfg.llm.target_language = args.target_lang
    cfg.llm.provider = args.llm_provider
    cfg.llm.base_url = args.llm_url

    pipeline = DiarizeFlowPipeline(cfg)
    pipeline.start()
    app = create_app(cfg, pipeline)

    stream_mode = getattr(cfg.diarization, "streaming_mode", "low_latency")
    asr_eng = getattr(cfg.asr, "engine", "sensevoice")
    print("=" * 75)
    print("🚀 啟動 DiarizeFlow 一體化即時語者分離翻譯應用")
    print(f"   後端服務: http://127.0.0.1:{actual_port}")
    print(f"   語者分離: NVIDIA Nemotron-3 Diarization 【串流架構: {stream_mode}】")
    print(f"   語音辨識: {asr_eng.upper()} ASR 引擎")
    print(f"   翻譯引擎: {cfg.llm.provider} -> 【{cfg.llm.target_language}】")
    print("=" * 75)

    server, thread = start_server_in_thread(app, "127.0.0.1", actual_port)
    time.sleep(0.5)

    try:
        from diarizeflow.app.frontend.desktop_overlay import run_overlay_app
        print("[*] 正在啟動 PySide6 原生透明飄浮字幕視窗 (Windows & Linux 相容)...")
        code = run_overlay_app(cfg, pipeline=pipeline)
        import os
        os._exit(code if isinstance(code, int) else 0)
    except Exception as e:
        print(f"[!] 無法載入桌面圖形視窗: {e}")
        print("[!] 請確認系統具備桌面圖形介面 (如 X11/Wayland/Windows 桌面)。")
        import os
        os._exit(1)


if __name__ == "__main__":
    run_cli()
