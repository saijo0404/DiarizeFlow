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

from diarizeflow.config import AppConfig
from diarizeflow.audio.segmenter import StreamingDiarizationSegmenter
from diarizeflow.audio.agc import StreamingInputAGC
from diarizeflow.engine.diarizer import NemotronDiarizer
from diarizeflow.engine.asr import SenseVoiceASR, create_asr_engine
from diarizeflow.engine.translator import LLMTranslator


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

        # Setup Real-time Streaming Input AGC (Dynamic Volume Leveling)
        self.stream_agc = StreamingInputAGC(
            target_rms=getattr(self.config.audio, "agc_target_rms", 0.06),
            max_gain=getattr(self.config.audio, "agc_max_gain", 25.0),
            min_gain=getattr(self.config.audio, "agc_min_gain", 0.15),
        )

        # Setup Target-Speaker Extraction (TSE) for overlap speech waveform separation
        tse_cfg = getattr(self.config, "tse", None)
        self.tse = None
        if tse_cfg is not None and getattr(tse_cfg, "enabled", True):
            try:
                from diarizeflow.audio.tse import TargetSpeakerExtractor
                self.tse = TargetSpeakerExtractor(
                    config=tse_cfg,
                    sample_rate=self.config.audio.sample_rate,
                )
            except Exception as e:
                print(f"[!] Warning: TargetSpeakerExtractor initialization skipped: {e}")

        # Setup Streaming Diarization-Driven Segmenter (replaces traditional VAD)
        self.segmenter = StreamingDiarizationSegmenter(
            sample_rate=self.config.audio.sample_rate,
            sad_threshold=getattr(self.config.diarization, "sad_threshold", 0.50),
            silence_timeout_ms=self.config.vad.silence_timeout_ms,
            min_speech_ms=self.config.vad.min_speech_ms,
            max_speech_s=self.config.vad.max_speech_s,
            pre_pad_ms=getattr(self.config.vad, "pre_pad_ms", 150),
            post_pad_ms=getattr(self.config.vad, "post_pad_ms", 150),
            diarizer=self.diarizer,
            on_utterance=self._on_diarized_utterance,
            tse_extractor=self.tse,
            tse_enabled=getattr(tse_cfg, "enabled", True) if tse_cfg else False,
        )
        # Keep self.vad alias for backward compatibility
        self.vad = self.segmenter

        self._speech_queue: queue.Queue = queue.Queue(maxsize=100)
        self._raw_chunk_queue: queue.Queue = queue.Queue(maxsize=200)
        self.last_audio_source: str = "mixed"
        self.audio_stats: Dict[str, int] = {
            "mic_chunks": 0,
            "loopback_chunks": 0,
            "mixed_chunks": 0,
            "total_chunks": 0,
        }
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._segmenter_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start the async pipeline worker and dedicated segmenter worker thread."""
        if self.is_running:
            return
        self.is_running = True

        if self._segmenter_thread is None or not self._segmenter_thread.is_alive():
            self._segmenter_thread = threading.Thread(target=self._segmenter_worker, daemon=True)
            self._segmenter_thread.start()

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
        try:
            self._raw_chunk_queue.put_nowait(None)
        except Exception:
            pass
        if self._segmenter_thread and self._segmenter_thread.is_alive():
            self._segmenter_thread.join(timeout=0.5)
        self._segmenter_thread = None

        if self._worker_task:
            self._worker_task.cancel()
        try:
            self._speech_queue.put_nowait(None)
        except Exception:
            pass
        if hasattr(self.translator, "close"):
            try:
                if self._loop and self._loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(self.translator.close(), self._loop)
                    try:
                        fut.result(timeout=0.5)
                    except Exception:
                        pass
                elif hasattr(self.translator, "close_sync"):
                    self.translator.close_sync()
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

    def push_audio(
        self,
        chunk: np.ndarray,
        rms: Optional[float] = None,
        source: Optional[str] = None,
    ):
        """Pass audio chunk into pipeline (alias for process_audio_chunk)."""
        self.process_audio_chunk(chunk, rms, source=source)

    def process_audio_chunk(
        self,
        chunk: np.ndarray,
        rms: Optional[float] = None,
        source: Optional[str] = None,
    ):
        """Ingest real-time 16kHz mono audio chunk asynchronously into non-blocking queue."""
        if not self.is_running:
            print("[*] Audio received while pipeline inactive. Auto-starting pipeline...")
            self.start()

        if rms is None:
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        src = source or "mixed"
        self.last_audio_source = src
        self.audio_stats["total_chunks"] += 1
        if src == "mic":
            self.audio_stats["mic_chunks"] += 1
        elif src in ("loopback", "system"):
            self.audio_stats["loopback_chunks"] += 1
        else:
            self.audio_stats["mixed_chunks"] += 1

        item = (chunk, rms, src)
        try:
            self._raw_chunk_queue.put_nowait(item)
        except queue.Full:
            try:
                self._raw_chunk_queue.get_nowait()
            except queue.Empty:
                pass
            self._raw_chunk_queue.put_nowait(item)

    def _segmenter_worker(self):
        """Dedicated background thread pulling raw audio chunks and executing streaming SAD."""
        while self.is_running:
            try:
                item = self._raw_chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self.is_running or item is None:
                break

            if len(item) == 3:
                chunk, rms, source = item
            else:
                chunk, rms = item
                source = "mixed"

            try:
                # Real-time Streaming AGC: dynamically boost quiet audio & tame loud audio before SAD
                proc_chunk = chunk
                proc_rms = rms
                if getattr(self.config.audio, "agc_enabled", True):
                    proc_chunk, current_gain, proc_rms = self.stream_agc.process(chunk)

                # Ingest into streaming diarization-driven segmenter (multi-track SAD)
                self.segmenter.process_chunk(proc_chunk, proc_rms, source=source)
            except Exception as e:
                print(f"[!] Error in segmenter worker: {e}")

    def _on_diarized_utterance(
        self,
        audio_segment: np.ndarray,
        speaker_label: str,
        confidence: float,
        duration: float,
        is_neural_sad: Optional[bool] = None,
    ):
        """Callback from StreamingDiarizationSegmenter when a speaker's utterance completes."""
        if not self.is_running:
            return

        if is_neural_sad is None:
            is_neural_sad = (
                getattr(self.segmenter, "last_emission_source", "neural_sad") == "neural_sad"
            )

        print(f"[*] [Sortformer SAD] 檢測到講者發話 ({speaker_label}): 長度 {duration:.2f} 秒, 正在送入辨識...")
        try:
            self._speech_queue.put_nowait(
                (audio_segment, speaker_label, confidence, duration, time.time(), is_neural_sad)
            )
        except queue.Full:
            print("[!] 語音隊列已滿，丟棄片段")

    def _on_speech_utterance(self, audio_segment: np.ndarray, duration: float):
        """Direct audio utterance ingestion (e.g. for whole-audio testing / API)."""
        if not self.is_running:
            return

        print(f"[*] 檢測到音訊輸入: 長度 {duration:.2f} 秒, 正在送入分離與辨識...")
        try:
            self._speech_queue.put_nowait((audio_segment, None, None, duration, time.time(), False))
        except queue.Full:
            print("[!] 語音隊列已滿，丟棄片段")

    async def _pipeline_worker(self):
        """Worker loop that runs Diarization, ASR, and LLM translation."""
        while self.is_running:
            try:
                item = await asyncio.to_thread(self._speech_queue.get)
                if not self.is_running or item is None or item[0] is None:
                    break
                audio_segment = item[0]
                spk_label = item[1]
                conf = item[2]
                duration = item[3]
                timestamp = item[4]
                is_neural_sad = item[5] if len(item) > 5 else (spk_label is not None)
            except (asyncio.CancelledError, Exception):
                if not self.is_running:
                    break
                continue

            t_process_start = time.perf_counter()
            try:
                proc_audio = audio_segment
                seg_rms = float(np.sqrt(np.mean(proc_audio ** 2) + 1e-9))
                # Skip segments whose energy is at the ambient noise floor
                if seg_rms < 0.004:
                    continue

                # 1. Speaker Diarization (Single-Pass Feature Reuse)
                t_diar_start = time.perf_counter()
                if is_neural_sad and spk_label is not None:
                    # Single-pass SAD Track Reuse: Segment was already cleanly isolated and tagged by Sortformer SAD
                    # Bypasses duplicate Sortformer ONNX forward pass (diarize_and_split / identify_speaker)
                    speaker_segments = [(proc_audio, spk_label, conf if conf is not None else 1.0)]
                elif self.diarizer is not None:
                    # Fallback path (Energy VAD or direct audio input)
                    if duration >= 2.2 and hasattr(self.diarizer, "diarize_and_split"):
                        speaker_segments = await asyncio.to_thread(
                            self.diarizer.diarize_and_split,
                            proc_audio,
                            self.config.audio.sample_rate,
                        )
                    elif hasattr(self.diarizer, "identify_speaker"):
                        spk, c_val, _ = await asyncio.to_thread(
                            self.diarizer.identify_speaker,
                            proc_audio,
                            self.config.audio.sample_rate,
                        )
                        speaker_segments = [(proc_audio, spk, c_val)]
                    else:
                        speaker_segments = [(proc_audio, spk_label or "講者 1", conf if conf is not None else 1.0)]
                else:
                    speaker_segments = [(proc_audio, spk_label or "講者 1", conf if conf is not None else 1.0)]
                t_diar_ms = (time.perf_counter() - t_diar_start) * 1000.0

                # 2. Sequential ASR & Concurrent LLM Translation + Subtitle Broadcast
                target_lang = self.config.llm.target_language
                translation_tasks = []

                async def _translate_and_broadcast(item: dict):
                    t_trans_start = time.perf_counter()
                    try:
                        translated_text = await self.translator.translate(
                            text=item["orig_text"],
                            target_language=target_lang,
                            source_language=item["detected_lang"],
                        )
                    except Exception as trans_err:
                        print(f"[!] Translation error: {trans_err}")
                        translated_text = item["orig_text"]

                    t_trans_ms = (time.perf_counter() - t_trans_start) * 1000.0
                    t_total_ms = (time.perf_counter() - t_process_start) * 1000.0
                    print(f"  └─> [翻譯 ({target_lang})]: {translated_text}")

                    is_fw = hasattr(self.asr, "active_precision")
                    asr_tag = (
                        f"Whisper {self.asr.active_precision.upper()}"
                        if is_fw
                        else (
                            "SenseVoice FP16"
                            if getattr(self.asr, "is_fp16", False)
                            else (
                                "SenseVoice INT8"
                                if "int8" in str(getattr(self.config.asr, "model_path", "")).lower()
                                else "SenseVoice FP32"
                            )
                        )
                    )
                    diar_tag = (
                        "Nemotron FP16"
                        if getattr(self.diarizer, "is_fp16", False)
                        else (
                            "Nemotron INT8"
                            if "int8" in str(getattr(self.config.diarization, "model_path", "")).lower()
                            else "Nemotron FP32"
                        )
                    )

                    print(
                        f"  ⚡ [延遲診斷] 音訊: {item['seg_dur']:.2f}s | "
                        f"ASR辨識 ({asr_tag}): {item['t_asr_ms']:.0f}ms | "
                        f"語者分離 ({diar_tag}): {t_diar_ms:.0f}ms | "
                        f"LLM翻譯: {t_trans_ms:.0f}ms | 總處理耗時: {t_total_ms:.0f}ms"
                    )

                    latency_dict = {
                        "duration_s": round(item["seg_dur"], 2),
                        "asr_ms": round(item["t_asr_ms"], 1),
                        "diar_ms": round(t_diar_ms, 1),
                        "trans_ms": round(t_trans_ms, 1),
                        "total_ms": round(t_total_ms, 1),
                    }

                    # Construct Subtitle Event
                    event = SubtitleEvent(
                        id=str(uuid.uuid4())[:8],
                        speaker=item["speaker_label"],
                        original_text=item["orig_text"],
                        translated_text=translated_text,
                        source_lang=item["detected_lang"],
                        target_lang=target_lang,
                        confidence=item["confidence"],
                        duration=round(item["seg_dur"], 2),
                        timestamp=timestamp,
                        latency=latency_dict,
                    )

                    # Broadcast to connected UI clients
                    if self.on_subtitle_broadcast:
                        try:
                            if asyncio.iscoroutinefunction(self.on_subtitle_broadcast):
                                await self.on_subtitle_broadcast(event)
                            else:
                                self.on_subtitle_broadcast(event)
                        except Exception as e:
                            print(f"[!] Error broadcasting subtitle event: {e}")

                for seg_audio, speaker_label, confidence in speaker_segments:
                    seg_dur = len(seg_audio) / self.config.audio.sample_rate
                    if seg_dur < 0.20:
                        continue

                    # ASR Transcription for this speaker segment
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

                    clean_orig = orig_text.strip()
                    # Reject isolated single-token noise hallucinations on longer segments with low acoustic energy
                    if len(clean_orig) <= 1 and seg_dur >= 1.5 and seg_rms < 0.015:
                        print(f"[*] 略過單字雜音幻覺 (長度 {seg_dur:.2f}s, 辨識='{clean_orig}', ASR耗時: {t_asr_ms:.0f}ms)")
                        continue

                    print(f"[{speaker_label}] [{detected_lang.upper()}]: {orig_text}")

                    # Spawn concurrent translation task immediately
                    item_data = {
                        "seg_audio": seg_audio,
                        "seg_dur": seg_dur,
                        "speaker_label": speaker_label,
                        "confidence": confidence,
                        "orig_text": orig_text,
                        "detected_lang": detected_lang,
                        "t_asr_ms": t_asr_ms,
                    }
                    task = asyncio.create_task(_translate_and_broadcast(item_data))
                    translation_tasks.append(task)

                # Concurrently await all translations for the current utterance
                if translation_tasks:
                    try:
                        await asyncio.gather(*translation_tasks, return_exceptions=True)
                    except asyncio.CancelledError:
                        for t in translation_tasks:
                            if not t.done():
                                t.cancel()
                        raise

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
        if hasattr(self.translator, "update_config"):
            self.translator.update_config(new_config.llm)
        else:
            self.translator.config = new_config.llm
        if hasattr(self, "segmenter"):
            self.segmenter.sad_threshold = getattr(new_config.diarization, "sad_threshold", 0.50)
            self.segmenter.silence_timeout_ms = new_config.vad.silence_timeout_ms
            self.segmenter.min_speech_ms = new_config.vad.min_speech_ms
            self.segmenter.max_speech_s = new_config.vad.max_speech_s
            self.segmenter.pre_pad_ms = getattr(new_config.vad, "pre_pad_ms", 150)
            self.segmenter.post_pad_ms = getattr(new_config.vad, "post_pad_ms", 150)
            self.segmenter.energy_threshold = new_config.vad.energy_threshold
            if hasattr(new_config, "tse"):
                self.segmenter.tse_enabled = getattr(new_config.tse, "enabled", True)
        if hasattr(self, "tse") and self.tse is not None and hasattr(new_config, "tse"):
            self.tse.update_config(new_config.tse)
        if hasattr(self, "stream_agc"):
            self.stream_agc.target_rms = getattr(new_config.audio, "agc_target_rms", 0.06)
            self.stream_agc.max_gain = getattr(new_config.audio, "agc_max_gain", 25.0)
            self.stream_agc.min_gain = getattr(new_config.audio, "agc_min_gain", 0.15)
        print("[✓] Pipeline configuration updated.")

    def rename_speaker(self, old_name_or_id: str, new_name: str, color: Optional[str] = None) -> bool:
        """Rename a speaker, pin their voiceprint profile, and persist to disk."""
        if self.diarizer is not None and hasattr(self.diarizer, "rename_speaker"):
            return self.diarizer.rename_speaker(old_name_or_id, new_name, color)
        return False

    def delete_speaker(self, speaker_id_or_name: str) -> bool:
        """Delete a speaker profile from disk and in-memory cache."""
        if self.diarizer is not None and hasattr(self.diarizer, "delete_speaker"):
            return self.diarizer.delete_speaker(speaker_id_or_name)
        return False

    def get_speaker_profiles(self) -> List[dict]:
        """Retrieve all persistent speaker voiceprint profiles."""
        if self.diarizer is not None and hasattr(self.diarizer, "get_speaker_profiles"):
            return self.diarizer.get_speaker_profiles()
        return []
