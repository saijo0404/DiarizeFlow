"""WebSocket background workers and REST API helpers for DiarizeFlow frontend."""

import json
import queue
import threading
import time
from typing import Callable, Optional
import urllib.parse
import urllib.request
import numpy as np
from websockets.sync.client import connect as ws_connect

from diarizeflow.audio.protocol import pack_audio_frame
from diarizeflow.config import AppConfig


def sync_config_to_remote_backend(
    host: str,
    port: int,
    new_cfg: AppConfig,
) -> threading.Thread:
    """Asynchronously push updated configuration to remote backend via POST /api/config."""
    if host == "0.0.0.0":
        host = "127.0.0.1"

    def _send():
        try:
            url = f"http://{host}:{port}/api/config"
            payload = json.dumps(new_cfg.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    print("[✓] 前端設定已成功同步至後端伺服器 (POST /api/config)")
        except Exception as e:
            print(f"[!] 同步設定至後端伺服器失敗: {e}")

    t = threading.Thread(target=_send, daemon=True)
    t.start()
    return t


def fetch_remote_backend_config(
    host: str,
    port: int,
    on_success: Callable[[AppConfig], None],
    on_error: Optional[Callable[[Exception], None]] = None,
) -> threading.Thread:
    """Fetch remote backend configuration via GET /api/config and notify callback."""
    if host == "0.0.0.0":
        host = "127.0.0.1"

    def _fetch():
        try:
            url = f"http://{host}:{port}/api/config"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    raw = resp.read().decode("utf-8")
                    remote_dict = json.loads(raw)
                    remote_cfg = AppConfig.from_dict(remote_dict)
                    print("[✓] 成功從後端伺服器拉取最新配置並對齊前端 (GET /api/config)")
                    on_success(remote_cfg)
        except Exception as e:
            print(f"[!] 從後端拉取最新配置失敗: {e}")
            if on_error:
                on_error(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    return t


def send_remote_rename_api(
    port: int,
    old_name: str,
    new_name: str,
    color: Optional[str] = None,
) -> None:
    """Send speaker rename request to backend via POST /api/speakers/{name}/rename."""
    try:
        url = f"http://127.0.0.1:{port}/api/speakers/{urllib.parse.quote(old_name)}/rename"
        payload = json.dumps({"name": new_name, "color": color}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0):
            pass
    except Exception as e:
        print(f"[!] 無法透過 REST API 同步講者重命名: {e}")


def send_remote_delete_api(port: int, speaker_id: str) -> None:
    """Send speaker deletion request to backend via DELETE /api/speakers/{speaker_id}."""
    try:
        url = f"http://127.0.0.1:{port}/api/speakers/{urllib.parse.quote(speaker_id)}"
        req = urllib.request.Request(
            url,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=2.0):
            pass
    except Exception as e:
        print(f"[!] 無法透過 REST API 同步講者刪除: {e}")


def run_subtitles_client(
    host: str,
    port: int,
    is_running: Callable[[], bool],
    on_status: Callable[[str, str], None],
    on_message: Callable[[dict], None],
    on_connected: Optional[Callable[[], None]] = None,
) -> None:
    """Background loop listening for translated subtitle events via WebSocket."""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"ws://{host}:{port}/ws/subtitles"

    while is_running():
        try:
            with ws_connect(url, open_timeout=2.0) as ws:
                on_status("● 已連線", "#38bdf8")
                if on_connected:
                    on_connected()
                while is_running():
                    try:
                        msg = ws.recv(timeout=1.0)
                        data = json.loads(msg)
                        if data.get("type") != "connection":
                            on_message(data)
                    except TimeoutError:
                        continue
        except Exception:
            if is_running():
                on_status("● 等待後端", "#64748b")
                time.sleep(1.5)


def run_audio_client(
    host: str,
    port: int,
    audio_queue: queue.Queue,
    is_running: Callable[[], bool],
) -> None:
    """Background loop streaming captured audio chunks to backend via WebSocket."""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"ws://{host}:{port}/ws/audio"

    while is_running():
        try:
            with ws_connect(url, open_timeout=2.0) as ws:
                while is_running():
                    try:
                        data = audio_queue.get(timeout=0.5)
                        if isinstance(data, (tuple, list)):
                            chunk = data[0]
                            track = data[1] if len(data) > 1 else "mixed"
                            sr = data[2] if len(data) > 2 else 16000
                            payload = pack_audio_frame(chunk, track=track, sample_rate=sr)
                        elif isinstance(data, np.ndarray):
                            payload = pack_audio_frame(data, track="mixed", sample_rate=16000)
                        elif isinstance(data, (bytes, bytearray)):
                            payload = data
                        else:
                            continue
                        ws.send(payload)
                    except queue.Empty:
                        continue
        except Exception:
            if is_running():
                time.sleep(1.5)


__all__ = [
    "sync_config_to_remote_backend",
    "fetch_remote_backend_config",
    "send_remote_rename_api",
    "send_remote_delete_api",
    "run_subtitles_client",
    "run_audio_client",
]
