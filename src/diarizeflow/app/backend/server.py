"""FastAPI Backend Server with WebSocket Audio Streaming and Subtitle Broadcasting.

Provides:
- /ws/audio: Receives raw binary 16kHz float32 PCM chunks from frontend
- /ws/subtitles: Broadcasts real-time translated subtitles to floating HUDs
- /api/devices: Discovers available microphones and system loopback devices
- /api/config: Reads and updates runtime configuration
- /api/test/audio: Test pipeline with uploaded or example audio files
- /: API service status and discovery
"""

import asyncio
from pathlib import Path
from typing import List, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import librosa
import io

from diarizeflow.app.config import AppConfig
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline, SubtitleEvent
from diarizeflow.app.audio.devices import list_audio_devices


def create_app(config: AppConfig, pipeline: DiarizeFlowPipeline) -> FastAPI:
    app = FastAPI(title="DiarizeFlow Real-time Translation API", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Active subtitle WebSocket connections
    active_subtitle_clients: Set[WebSocket] = set()

    async def broadcast_subtitle(event: SubtitleEvent):
        """Broadcast new subtitle event to all connected UI clients."""
        payload = event.to_dict()
        closed_clients = set()
        for client in active_subtitle_clients:
            try:
                await client.send_json(payload)
            except Exception:
                closed_clients.add(client)
        active_subtitle_clients.difference_update(closed_clients)

    pipeline.on_subtitle_broadcast = broadcast_subtitle

    @app.on_event("startup")
    async def on_startup():
        if not pipeline.is_running:
            pipeline.start(asyncio.get_event_loop())

    @app.on_event("shutdown")
    async def on_shutdown():
        # Do not forcefully kill pipeline if managed by desktop application
        pass

    # --- WebSocket Endpoints ---

    @app.websocket("/ws/audio")
    async def audio_websocket(websocket: WebSocket):
        """WebSocket endpoint receiving binary 16kHz float32 PCM audio chunks."""
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    continue
                # Parse raw bytes as float32 array
                chunk = np.frombuffer(data, dtype=np.float32)
                pipeline.process_audio_chunk(chunk)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[!] Audio WebSocket error: {e}")

    @app.websocket("/ws/subtitles")
    async def subtitles_websocket(websocket: WebSocket):
        """WebSocket endpoint streaming translated subtitle events to UIs."""
        await websocket.accept()
        active_subtitle_clients.add(websocket)
        try:
            # Send initial ping / connection ack
            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "target_language": config.llm.target_language,
                "provider": config.llm.provider,
            })
            while True:
                # Keep-alive receive
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            active_subtitle_clients.discard(websocket)
        except Exception:
            active_subtitle_clients.discard(websocket)

    # --- REST Endpoints ---

    @app.get("/api/status")
    async def get_status():
        return {
            "status": "running",
            "active_ui_clients": len(active_subtitle_clients),
            "diarizer_provider": pipeline.diarizer.active_provider,
            "asr_provider": pipeline.asr.active_provider,
            "llm_provider": config.llm.provider,
            "target_language": config.llm.target_language,
        }

    @app.get("/api/config")
    async def get_config():
        active_cfg = pipeline.config if pipeline and hasattr(pipeline, "config") else config
        return active_cfg.to_dict()

    @app.post("/api/config")
    async def update_config(payload: dict):
        try:
            new_cfg = AppConfig.from_dict(payload)
            if pipeline:
                pipeline.update_config(new_cfg)
            return {"status": "updated", "config": new_cfg.to_dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid configuration payload: {e}")

    @app.get("/api/devices")
    async def get_devices():
        mics, loopbacks = list_audio_devices()
        return {
            "microphones": [m.__dict__ for m in mics],
            "loopbacks": [l.__dict__ for l in loopbacks],
        }

    @app.post("/api/test/audio")
    async def test_audio_file(request: Request):
        """Upload an audio file to test the translation pipeline without a microphone."""
        content_type = request.headers.get("content-type", "")
        filename = "test_audio.wav"
        if "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            contents = await file.read()
            filename = getattr(file, "filename", filename)
        else:
            contents = await request.body()

        audio, sr = sf.read(io.BytesIO(contents))
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)

        # Trigger VAD callback directly with whole audio
        duration = len(audio) / 16000.0
        pipeline._on_speech_utterance(audio, duration)

        return {"status": "enqueued", "filename": filename, "duration": round(duration, 2)}

    # --- Speaker Voiceprint Profiles Endpoints ---

    @app.get("/api/speakers")
    async def get_speakers():
        """Get all saved persistent speaker voiceprint profiles."""
        profiles = pipeline.get_speaker_profiles()
        return {"profiles": profiles, "count": len(profiles)}

    @app.post("/api/speakers/{speaker_id}/rename")
    async def rename_speaker(speaker_id: str, payload: dict):
        """Rename a speaker and pin their profile to disk."""
        new_name = payload.get("name") or payload.get("new_name")
        if not new_name or not str(new_name).strip():
            return JSONResponse(status_code=400, content={"detail": "Missing 'name' in request body"})
        new_name = str(new_name).strip()
        color = payload.get("color")

        success = pipeline.rename_speaker(speaker_id, new_name, color)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"Speaker '{speaker_id}' not found"})

        profiles = pipeline.get_speaker_profiles()
        updated = next((p for p in profiles if p["id"] == speaker_id or p["name"] == new_name), None)

        # Broadcast update to connected subtitle clients
        closed_clients = set()
        for client in active_subtitle_clients:
            try:
                await client.send_json({
                    "type": "speaker_renamed",
                    "old_speaker": speaker_id,
                    "new_speaker": new_name,
                    "color": color,
                })
            except Exception:
                closed_clients.add(client)
        active_subtitle_clients.difference_update(closed_clients)

        return {"status": "success", "profile": updated}

    @app.delete("/api/speakers/{speaker_id}")
    async def delete_speaker(speaker_id: str):
        """Delete a speaker profile permanently from disk and cache."""
        success = pipeline.delete_speaker(speaker_id)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"Speaker '{speaker_id}' not found"})

        # Broadcast deletion to connected subtitle clients
        closed_clients = set()
        for client in active_subtitle_clients:
            try:
                await client.send_json({
                    "type": "speaker_deleted",
                    "speaker_id": speaker_id,
                })
            except Exception:
                closed_clients.add(client)
        active_subtitle_clients.difference_update(closed_clients)

        return {"status": "deleted", "speaker_id": speaker_id}

    # --- Root API Status ---
    @app.get("/")
    async def root():
        return {
            "service": "DiarizeFlow API",
            "version": "0.1.0",
            "docs": "/docs",
            "status": "running",
        }

    return app


def run_cli():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="DiarizeFlow 後端伺服器")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.server.host = args.host
    cfg.server.port = args.port

    pipeline = DiarizeFlowPipeline(cfg)
    app = create_app(cfg, pipeline)
    uvicorn.run(app, host=args.host, port=args.port)
