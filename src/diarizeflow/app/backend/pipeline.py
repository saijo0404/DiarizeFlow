"""Real-time Translation & Diarization Pipeline Orchestrator.

Processes streaming audio chunks through:
Audio Stream -> VAD -> Nemotron Diarization -> SenseVoice ASR -> LLM Translation -> Subtitle Broadcast
"""

import asyncio
from dataclasses import dataclass, asdict, field
import queue
import threading
import time
from typing import Callable, List, Optional
import uuid
import numpy as np

from diarizeflow.app.config import AppConfig
from diarizeflow.app.audio.vad import EnergyVADSegmenter
from diarizeflow.app.audio.agc import apply_speech_agc, StreamingInputAGC
from diarizeflow.app.backend.diarizer import NemotronDiarizer
from diarizeflow.app.backend.asr import SenseVoiceASR, create_asr_engine
from diarizeflow.app.backend.translator import LLMTranslator


@dataclass
class SubtitleEvent:
    id: str
    speaker: str
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    duration: float
    timestamp: float
    latency: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class DiarizeFlowPipeline:
    """End-to-End Orchestrator for Real-time Diarization, ASR and Translation."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        on_subtitle_broadcast: Optional[Callable[[SubtitleEvent], None]] = None,
    ):
        self.config = config or AppConfig()
        self.on_subtitle_broadcast = on_subtitle_broadcast

        # Initialize sub-modules
        print("[*] Initializing DiarizeFlow Backend Pipeline...")
        self.diarizer = NemotronDiarizer(self.config.diarization)
        self.asr = create_asr_engine(self.config.asr)
        self.translator = LLMTranslator(self.config.llm)

        # Setup Real-time Streaming Input AGC (Dynamic Volume Leveling for VAD)
        self.stream_agc = StreamingInputAGC(
            target_rms=getattr(self.config.audio, "agc_target_rms", 0.06),
            max_gain=getattr(self.config.audio, "agc_max_gain", 25.0),
            min_gain=getattr(self.config.audio, "agc_min_gain", 0.15),
        )

        # Setup VAD segmenter
        self.vad = EnergyVADSegmenter(
            sample_rate=self.config.audio.sample_rate,
            energy_threshold=self.config.vad.energy_threshold,
            min_speech_ms=self.config.vad.min_speech_ms,
            silence_timeout_ms=self.config.vad.silence_timeout_ms,
            max_speech_s=self.config.vad.max_speech_s,
            on_speech_utterance=self._on_speech_utterance,
        )

        self._speech_queue: queue.Queue = queue.Queue(maxsize=100)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task] = None
        self.is_running = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start the async pipeline worker."""
        if self.is_running:
            return
        self.is_running = True
        if loop is not None:
            self._loop = loop
            self._worker_task = self._loop.create_task(self._pipeline_worker())
        else:
            try:
                self._loop = asyncio.get_running_loop()
                self._worker_task = self._loop.create_task(self._pipeline_worker())
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                self._loop = new_loop
                def _run_worker_loop():
                    asyncio.set_event_loop(new_loop)
                    self._worker_task = new_loop.create_task(self._pipeline_worker())
                    new_loop.run_forever()
                self._thread = threading.Thread(target=_run_worker_loop, daemon=True)
                self._thread.start()
        print("[✓] DiarizeFlow Pipeline background worker started.")

    def stop(self):
        """Stop the pipeline worker."""
        self.is_running = False
        if self._worker_task:
            self._worker_task.cancel()
        try:
            self._speech_queue.put_nowait((None, 0.0, 0.0))
        except Exception:
            pass
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        print("[*] DiarizeFlow Pipeline stopped.")

    def on_subtitle(self, callback: Callable[[SubtitleEvent], None]):
        """Register or chain a subtitle callback."""
        prev_cb = self.on_subtitle_broadcast
        if prev_cb is None:
            self.on_subtitle_broadcast = callback
        else:
            async def chained(evt):
                if prev_cb is not None:
                    try:
                        if asyncio.iscoroutinefunction(prev_cb):
                            await prev_cb(evt)
                        else:
                            prev_cb(evt)
                    except Exception as e:
                        print(f"[!] Error in prev subtitle callback: {e}")
                if callback is not None:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(evt)
                        else:
                            callback(evt)
                    except Exception as e:
                        print(f"[!] Error in subtitle callback: {e}")
            self.on_subtitle_broadcast = chained

    def push_audio(self, chunk: np.ndarray, rms: Optional[float] = None):
        """Pass audio chunk into pipeline (alias for process_audio_chunk)."""
        self.process_audio_chunk(chunk, rms)

    def process_audio_chunk(self, chunk: np.ndarray, rms: Optional[float] = None):
        """Ingest real-time 16kHz mono audio chunk."""
        if not self.is_running:
            print("[*] Audio received while pipeline inactive. Auto-starting pipeline...")
            self.start()

        if rms is None:
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        # Real-time Streaming AGC: dynamically boost quiet audio & tame loud audio before VAD
        proc_chunk = chunk
        proc_rms = rms
        if getattr(self.config.audio, "agc_enabled", True):
            proc_chunk, current_gain, proc_rms = self.stream_agc.process(chunk)

        self.vad.process_chunk(proc_chunk, proc_rms)

    def _on_speech_utterance(self, audio_segment: np.ndarray, duration: float):
        """Callback from VAD when a speech utterance completes."""
        if not self.is_running:
            return

        print(f"[*] 檢測到語音活動: 長度 {duration:.2f} 秒, 正在送入辨識...")
        try:
            self._speech_queue.put_nowait((audio_segment, duration, time.time()))
        except queue.Full:
            print("[!] 語音隊列已滿，丟棄片段")

    async def _pipeline_worker(self):
        """Worker loop that runs Diarization, ASR, and LLM translation."""
        while self.is_running:
            try:
                item = await asyncio.to_thread(self._speech_queue.get)
                if not self.is_running or item is None or item[0] is None:
                    break
                audio_segment, duration, timestamp = item
            except (asyncio.CancelledError, Exception):
                if not self.is_running:
                    break
                continue

            t_process_start = time.perf_counter()
            try:
                # 0. Automatic Gain Control (AGC) & Volume Normalization
                proc_audio = audio_segment
                if getattr(self.config.audio, "agc_enabled", True):
                    target_rms = getattr(self.config.audio, "agc_target_rms", 0.08)
                    max_gain = getattr(self.config.audio, "agc_max_gain", 4.0)
                    min_gain = getattr(self.config.audio, "agc_min_gain", 0.25)
                    proc_audio, applied_gain = apply_speech_agc(
                        audio_segment,
                        target_rms=target_rms,
                        max_gain=max_gain,
                        min_gain=min_gain,
                    )
                    if applied_gain > 1.25 or applied_gain < 0.8:
                        action = "放大" if applied_gain > 1.0 else "縮小"
                        print(f"  [AGC 自動調音] 自動{action}輸入音訊 {applied_gain:.2f}x (目標能量: {target_rms})")

                # 1. Speaker Diarization & Turn Segmentation (Detects single or multiple speakers)
                t_diar_start = time.perf_counter()
                speaker_segments = await asyncio.to_thread(
                    self.diarizer.diarize_and_split,
                    proc_audio,
                    self.config.audio.sample_rate,
                )
                t_diar_ms = (time.perf_counter() - t_diar_start) * 1000.0

                for seg_audio, speaker_label, confidence in speaker_segments:
                    seg_dur = len(seg_audio) / self.config.audio.sample_rate
                    if seg_dur < 0.25:
                        continue

                    # 2. ASR Transcription for this speaker segment
                    t_asr_start = time.perf_counter()
                    orig_text, detected_lang = await asyncio.to_thread(
                        self.asr.transcribe,
                        seg_audio,
                        self.config.asr.language,
                    )
                    t_asr_ms = (time.perf_counter() - t_asr_start) * 1000.0

                    if not orig_text or not orig_text.strip():
                        print(f"[*] 語音辨識為空 (ASR耗時: {t_asr_ms:.0f}ms，可能為背景雜音或非人聲)")
                        continue

                    print(f"[{speaker_label}] [{detected_lang.upper()}]: {orig_text}")

                    # 3. LLM Translation
                    target_lang = self.config.llm.target_language
                    t_trans_start = time.perf_counter()
                    translated_text = await self.translator.translate(
                        text=orig_text,
                        target_language=target_lang,
                        source_language=detected_lang,
                    )
                    t_trans_ms = (time.perf_counter() - t_trans_start) * 1000.0

                    t_total_ms = (time.perf_counter() - t_process_start) * 1000.0
                    print(f"  └─> [翻譯 ({target_lang})]: {translated_text}")

                    is_fw = hasattr(self.asr, "active_precision")
                    asr_tag = f"Whisper {self.asr.active_precision.upper()}" if is_fw else ("SenseVoice FP16" if getattr(self.asr, "is_fp16", False) else ("SenseVoice INT8" if "int8" in str(getattr(self.config.asr, "model_path", "")).lower() else "SenseVoice FP32"))
                    diar_tag = "Nemotron FP16" if getattr(self.diarizer, "is_fp16", False) else ("Nemotron INT8" if "int8" in str(getattr(self.config.diarization, "model_path", "")).lower() else "Nemotron FP32")

                    print(f"  ⚡ [延遲診斷] 音訊: {seg_dur:.2f}s | ASR辨識 ({asr_tag}): {t_asr_ms:.0f}ms | 語者分離 ({diar_tag}): {t_diar_ms:.0f}ms | LLM翻譯: {t_trans_ms:.0f}ms | 總處理耗時: {t_total_ms:.0f}ms")

                    latency_dict = {
                        "duration_s": round(seg_dur, 2),
                        "asr_ms": round(t_asr_ms, 1),
                        "diar_ms": round(t_diar_ms, 1),
                        "trans_ms": round(t_trans_ms, 1),
                        "total_ms": round(t_total_ms, 1),
                    }

                    # 4. Construct Subtitle Event
                    event = SubtitleEvent(
                        id=str(uuid.uuid4())[:8],
                        speaker=speaker_label,
                        original_text=orig_text,
                        translated_text=translated_text,
                        source_lang=detected_lang,
                        target_lang=target_lang,
                        confidence=confidence,
                        duration=round(seg_dur, 2),
                        timestamp=timestamp,
                        latency=latency_dict,
                    )

                    # 5. Broadcast to connected UI clients
                    if self.on_subtitle_broadcast:
                        try:
                            if asyncio.iscoroutinefunction(self.on_subtitle_broadcast):
                                await self.on_subtitle_broadcast(event)
                            else:
                                self.on_subtitle_broadcast(event)
                        except Exception as e:
                            print(f"[!] Error broadcasting subtitle event: {e}")

            except Exception as e:
                print(f"[!] Error in pipeline processing: {e}")

    def update_config(self, new_config: AppConfig):
        """Dynamically update runtime configuration."""
        old_engine = getattr(self.config.asr, "engine", "sensevoice")
        old_prec = getattr(self.config.asr, "whisper_precision", "float16")
        self.config = new_config
        self.diarizer.update_config(new_config.diarization)
        if getattr(new_config.asr, "engine", "sensevoice") != old_engine or getattr(new_config.asr, "whisper_precision", "float16") != old_prec:
            self.asr = create_asr_engine(new_config.asr)
        else:
            self.asr.config = new_config.asr
        self.translator.config = new_config.llm
        self.vad.energy_threshold = new_config.vad.energy_threshold
        self.vad.min_speech_ms = new_config.vad.min_speech_ms
        self.vad.silence_timeout_ms = new_config.vad.silence_timeout_ms
        self.vad.max_speech_s = new_config.vad.max_speech_s
        if hasattr(self, "stream_agc"):
            self.stream_agc.target_rms = getattr(new_config.audio, "agc_target_rms", 0.06)
            self.stream_agc.max_gain = getattr(new_config.audio, "agc_max_gain", 25.0)
            self.stream_agc.min_gain = getattr(new_config.audio, "agc_min_gain", 0.15)
        print("[✓] Pipeline configuration updated.")
