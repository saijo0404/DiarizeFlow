"""Launcher module for DiarizeFlow Application.

Single source of truth for:
1. Dual logging to terminal and diarizeflow.log with 5MB file rotation.
2. Windows parent console attachment in GUI mode.
3. Unhandled exception capture and logging via sys.excepthook.
4. First-run hardware calibration & auto-quantization (ensure_calibrated_models).
5. Dynamic port conflict detection and backend pipeline startup.
6. Launching PySide6 desktop floating HUD.
"""

import argparse
import io
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Optional, Tuple
import uvicorn

from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline
from diarizeflow.app.backend.server import create_app
from diarizeflow.app.config import AppConfig


class DualLogger(io.TextIOBase):
    """Logs to both console (if available) and disk log file with 5MB rotation."""

    def __init__(self, log_path: Path, fallback_stream=None, max_bytes: int = 5 * 1024 * 1024):
        self.fallback = fallback_stream
        self.log_path = log_path
        self.max_bytes = max_bytes
        self.log_file = None
        self._open_log()

    def _open_log(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            self.log_file = None

    def rotate_if_needed(self):
        if not self.log_file or not self.log_path.exists():
            return
        try:
            if self.log_path.stat().st_size > self.max_bytes:
                self.log_file.close()
                backup_path = self.log_path.with_suffix(".log.old")
                if backup_path.exists():
                    backup_path.unlink()
                self.log_path.rename(backup_path)
                self._open_log()
        except Exception:
            pass

    def clear(self) -> bool:
        """Truncate and reset the log file."""
        try:
            if self.log_file:
                self.log_file.close()
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 日誌已清空\n")
            self._open_log()
            return True
        except Exception as e:
            if self.fallback and hasattr(self.fallback, "write"):
                self.fallback.write(f"[!] 清空日誌失敗: {e}\n")
            return False

    def write(self, s: str) -> int:
        if not s:
            return 0
        if self.fallback and hasattr(self.fallback, "write"):
            try:
                self.fallback.write(s)
                self.fallback.flush()
            except Exception:
                pass
        if self.log_file:
            try:
                self.rotate_if_needed()
                self.log_file.write(s)
                self.log_file.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        if self.fallback and hasattr(self.fallback, "flush"):
            try:
                self.fallback.flush()
            except Exception:
                pass
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


def get_app_log_path() -> Path:
    """Return path to diarizeflow.log located next to executable or in project root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "diarizeflow.log"
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if (project_root / "pyproject.toml").exists():
        return project_root / "diarizeflow.log"
    return Path.cwd() / "diarizeflow.log"


def attach_console_on_windows():
    """On Windows, attach to parent console in GUI mode so logs appear in Command Prompt or PowerShell."""
    console_stream = sys.stdout
    if sys.platform == "win32":
        try:
            import ctypes
            # ATTACH_PARENT_PROCESS = 0xFFFFFFFF
            if ctypes.windll.kernel32.AttachConsole(0xFFFFFFFF):
                console_stream = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except Exception:
            pass
    return console_stream


def setup_global_logging(log_path: Optional[Path] = None) -> DualLogger:
    """Configure DualLogger on sys.stdout and sys.stderr with unhandled exception hooks."""
    if log_path is None:
        log_path = get_app_log_path()

    console_stream = attach_console_on_windows()
    logger_stream = DualLogger(log_path, console_stream)
    sys.stdout = logger_stream
    sys.stderr = logger_stream
    if getattr(sys, "__stdout__", None) is None:
        sys.__stdout__ = sys.stdout
    if getattr(sys, "__stderr__", None) is None:
        sys.__stderr__ = sys.stderr
    if sys.stdin is None or getattr(sys, "__stdin__", None) is None:
        sys.stdin = io.StringIO("")
        sys.__stdin__ = sys.stdin

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        import traceback
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"❌ [{time.strftime('%Y-%m-%d %H:%M:%S')}] Unhandled Exception:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

    sys.excepthook = handle_exception
    return logger_stream


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


def start_server_in_thread(app, host: str, port: int) -> Tuple[Optional[uvicorn.Server], Optional[threading.Thread]]:
    """Start uvicorn server in a daemon background thread."""
    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", log_config=None)
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        return server, thread
    except Exception as e:
        print(f"[!] Warning: Cannot start web server on {host}:{port}: {e}")
        return None, None


def run_cli() -> int:
    """Main CLI entrypoint for DiarizeFlow."""
    parser = argparse.ArgumentParser(description="DiarizeFlow 一鍵啟動後端與透明飄浮前端")
    parser.add_argument("--mode", type=str, default="desktop", choices=["desktop"], help="前端模式 (預設: desktop 原生桌面 HUD)")
    parser.add_argument("--port", type=int, default=8765, help="連接埠 (預設: 8765)")
    parser.add_argument("--target-lang", type=str, default="繁體中文", help="目標翻譯語言 (預設: 繁體中文)")
    parser.add_argument("--llm-provider", type=str, default="vllm", help="LLM 供應商 (vllm, llama.cpp, openai, claude, bypass)")
    parser.add_argument("--llm-url", type=str, default="http://127.0.0.1:8000/v1", help="LLM Base URL")
    parser.add_argument("--config", type=str, default="config.json", help="配置檔路徑 (預設: config.json)")
    args = parser.parse_args()

    # 1. Setup Dual Logging & Exception Hook
    setup_global_logging()

    # 2. Load Configuration
    cfg = AppConfig.load(args.config)

    # 3. First-run hardware calibration & auto-quantization
    try:
        from diarizeflow.calibration import ensure_calibrated_models
        cfg = ensure_calibrated_models(cfg)
    except Exception as calib_err:
        print(f"[!] 硬體校準檢查注意: {calib_err}")

    # 4. Port Conflict Detection & Resolution
    actual_port = find_available_port("127.0.0.1", args.port)
    if actual_port != args.port:
        print(f"[*] 注意: 預設連接埠 {args.port} 已被佔用，自動切換至可用連接埠: {actual_port}")
    cfg.server.port = actual_port

    cfg.llm.target_language = args.target_lang
    cfg.llm.provider = args.llm_provider
    cfg.llm.base_url = args.llm_url

    # 5. Start Backend Pipeline
    pipeline = DiarizeFlowPipeline(cfg)
    pipeline.start()
    app = create_app(cfg, pipeline)

    is_whisper = "whisper" in getattr(cfg.asr, "engine", "").lower()
    asr_name = "Faster-Whisper Large-v2" if is_whisper else "SenseVoiceSmall 多語言 ASR"
    if is_whisper:
        asr_prec = getattr(cfg.asr, "whisper_precision", "FP16").upper()
    elif getattr(pipeline.asr, "is_fp16", False):
        asr_prec = "FP16 (半精度加速)"
    elif "int8" in str(getattr(cfg.asr, "model_path", "")).lower():
        asr_prec = "INT8 (整數輕量化)"
    else:
        asr_prec = "FP32 (單精度基準)"

    diar_prec = "FP16 (CUDA Tensor Core 加速)" if getattr(pipeline.diarizer, "is_fp16", False) else ("INT8 (CPU 輕量化)" if "int8" in str(getattr(cfg.diarization, "model_path", "")).lower() else "FP32 (單精度基準)")
    diar_ep = getattr(pipeline.diarizer, "active_provider", "Unknown")

    print("=" * 75)
    print("🚀 啟動 DiarizeFlow 一體化即時語者分離翻譯應用")
    print(f"   後端服務: http://127.0.0.1:{actual_port}")
    print(f"   語者分離: NVIDIA Nemotron-3 Diarization 【精度: {diar_prec} | {diar_ep}】")
    print(f"   語音辨識: {asr_name} 【精度: {asr_prec}】")
    print(f"   音訊處理: 智慧音量動態調節 (AGC) [{'已啟用' if getattr(cfg.audio, 'agc_enabled', True) else '已停用'}]")
    print(f"   翻譯引擎: {cfg.llm.provider} -> 【{cfg.llm.target_language}】")
    print("=" * 75)

    server, thread = start_server_in_thread(app, "127.0.0.1", actual_port)
    time.sleep(0.5)

    # 6. Start Frontend HUD
    try:
        from diarizeflow.app.frontend.desktop_overlay import run_overlay_app
        print("[*] 正在啟動 PySide6 原生透明飄浮字幕視窗 (Windows & Linux 相容)...")
        code = run_overlay_app(cfg, pipeline=pipeline)
        return code if isinstance(code, int) else 0
    except Exception as e:
        print(f"[!] 桌面視窗啟動失敗: {e}")
        print("[!] 請確認系統具備桌面圖形環境 (如 X11/Wayland/Windows 桌面)。")
        return 1


main = run_cli


if __name__ == "__main__":
    sys.exit(run_cli())

