#!/usr/bin/env python3
"""Run the DiarizeFlow Backend Server."""

import argparse
from pathlib import Path
import sys
import uvicorn

# Add src to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from diarizeflow.config import AppConfig
from diarizeflow.engine.pipeline import DiarizeFlowPipeline
from diarizeflow.engine.server import create_app


def main():
    parser = argparse.ArgumentParser(description="DiarizeFlow 即時語者分離翻譯後端伺服器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="監聽位址 (預設: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="監聽連接埠 (預設: 8765)")
    parser.add_argument("--config", type=str, default="config.json", help="設定檔路徑 (預設: config.json)")
    parser.add_argument("--target-lang", type=str, default=None, help="目標翻譯語言 (例如: 繁體中文, English)")
    parser.add_argument("--llm-provider", type=str, default=None, help="LLM 供應商 (vllm, llama.cpp, openai, claude, bypass)")
    parser.add_argument("--llm-url", type=str, default=None, help="LLM Base URL (例如: http://127.0.0.1:8000/v1)")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.server.host = args.host
    cfg.server.port = args.port

    if args.target_lang:
        cfg.llm.target_language = args.target_lang
    if args.llm_provider:
        cfg.llm.provider = args.llm_provider
    if args.llm_url:
        cfg.llm.base_url = args.llm_url

    pipeline = DiarizeFlowPipeline(cfg)
    app = create_app(cfg, pipeline)

    print("=" * 70)
    print(f"🚀 DiarizeFlow 後端伺服器啟動於 http://{args.host}:{args.port}")
    print(f"   - 語者分離引擎: Nemotron-3-Diarization")
    print(f"   - 語音辨識 (ASR): SenseVoiceSmall")
    print(f"   - 翻譯 API: {cfg.llm.provider} -> 目標語言: 【{cfg.llm.target_language}】")
    print(f"   - API 文件與互動端點: http://localhost:{args.port}/docs")
    print("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
