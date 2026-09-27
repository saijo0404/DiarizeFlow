"""Cross-platform Audio Capture Streamer (Mic & System Audio Loopback).

Captures audio in real-time, resamples to 16kHz mono float32,
and pushes PCM frames to a callback/queue for WebSocket transmission or pipeline ingestion.
"""

import platform
import queue
import threading
import time
from typing import Callable, Optional, Tuple, Union
import numpy as np
import sounddevice as sd
import scipy.signal


class AudioCaptureStream:
    """Manages audio streaming from microphone and/or system loopback."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        chunk_ms: int = 250,
        gain: float = 1.0,
        on_audio_chunk: Optional[Callable[[np.ndarray, float], None]] = None,
        stall_timeout: float = 1.0,
        max_drift_chunks: int = 4,
        max_buffer_chunks: int = 8,
    ):
        self.target_sr = target_sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_samples = int(self.target_sr * (self.chunk_ms / 1000.0))
        self.gain = gain
        self.on_audio_chunk = on_audio_chunk
        self.stall_timeout = stall_timeout
        self.max_drift_samples = self.chunk_samples * max_drift_chunks
        self.max_buffer_samples = self.chunk_samples * max_buffer_chunks

        self.running = False
        self.mic_stream: Optional[sd.InputStream] = None
        self.loopback_stream: Optional[sd.InputStream] = None
        self._loopback_thread: Optional[threading.Thread] = None
        self._loopback_active: bool = False

        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._worker_thread: Optional[threading.Thread] = None

        # Level meters for UI feedback (0.0 to 1.0)
        self.current_mic_rms: float = 0.0
        self.current_loopback_rms: float = 0.0

    def _create_stream(self, device_id: int, is_loopback: bool, callback: Callable) -> sd.InputStream:
        """Create a sounddevice input stream with appropriate OS-specific settings and fallbacks."""
        dev_info = sd.query_devices(device_id)
        native_sr = int(dev_info.get("default_samplerate", 48000))
        out_ch = int(dev_info.get("max_output_channels", 2))
        in_ch = max(1, min(2, int(dev_info.get("max_input_channels", 1))))

        extra_settings = None
        if is_loopback and platform.system().lower() == "windows":
            # WASAPI Loopback mode on Windows
            try:
                hostapis = sd.query_hostapis()
                hostapi_name = hostapis[dev_info.get("hostapi", 0)]["name"]
                if "wasapi" not in hostapi_name.lower():
                    # Find WASAPI counterpart for this device name
                    dev_name = dev_info.get("name", "")
                    all_devs = sd.query_devices()
                    for idx, d in enumerate(all_devs):
                        h_name = hostapis[d.get("hostapi", 0)]["name"]
                        if "wasapi" in h_name.lower() and (d.get("name") == dev_name or dev_name in d.get("name", "")):
                            print(f"[*] 自動切換系統聲音至 WASAPI 端點: #{idx} {d.get('name')}")
                            device_id = idx
                            dev_info = d
                            native_sr = int(dev_info.get("default_samplerate", 48000))
                            out_ch = int(dev_info.get("max_output_channels", 2))
                            break

                extra_settings = sd.WasapiSettings(loopback=True)
            except Exception as e:
                print(f"[!] Warning: Failed to configure WasapiSettings(loopback=True): {e}")

            # Try candidate parameters for WASAPI loopback (channels, sample rates, blocksize)
            channel_candidates = [2]
            if out_ch > 0 and out_ch != 2:
                channel_candidates.append(out_ch)
            if 1 not in channel_candidates:
                channel_candidates.append(1)

            sr_candidates = [native_sr]
            for sr in [48000, 44100, 96000]:
                if sr not in sr_candidates:
                    sr_candidates.append(sr)

            last_err = None
            for sr in sr_candidates:
                for ch in channel_candidates:
                    for bs in [0, int(sr * (self.chunk_ms / 1000.0))]:
                        try:
                            stream = sd.InputStream(
                                device=device_id,
                                samplerate=sr,
                                channels=ch,
                                dtype="float32",
                                blocksize=bs,
                                extra_settings=extra_settings,
                                callback=callback,
                            )
                            print(f"[✓] 成功以採樣率 {sr}Hz, {ch} 聲道, blocksize={bs} 建立 WASAPI 系統聲音串流: #{device_id} ({dev_info.get('name')})")
                            return stream
                        except Exception as e:
                            last_err = e
                            continue
            raise RuntimeError(f"無法建立 WASAPI 系統聲音串流 (#{device_id} {dev_info.get('name')}): {last_err}")

        # Normal microphone input stream
        blocksize = int(native_sr * (self.chunk_ms / 1000.0))
        try:
            stream = sd.InputStream(
                device=device_id,
                samplerate=native_sr,
                channels=in_ch,
                dtype="float32",
                blocksize=blocksize,
                extra_settings=extra_settings,
                callback=callback,
            )
            return stream
        except Exception:
            # Fallback to blocksize=0 (driver default)
            return sd.InputStream(
                device=device_id,
                samplerate=native_sr,
                channels=in_ch,
                dtype="float32",
                blocksize=0,
                extra_settings=extra_settings,
                callback=callback,
            )

    def _resample_to_16k(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Convert multi-channel audio to mono and resample to 16kHz cleanly and quickly."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)  # downmix to mono

        if orig_sr != self.target_sr and len(audio) > 0:
            num_target_samples = int(len(audio) * (self.target_sr / orig_sr))
            if num_target_samples > 0:
                x_orig = np.linspace(0.0, 1.0, len(audio), endpoint=False)
                x_target = np.linspace(0.0, 1.0, num_target_samples, endpoint=False)
                audio = np.interp(x_target, x_orig, audio)

        return audio.astype(np.float32)

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass
        arr = indata.copy()
        rms = float(np.sqrt(np.mean(arr ** 2) + 1e-9))
        self.current_mic_rms = min(1.0, rms * 5.0)

        sr = int(self.mic_stream.samplerate) if self.mic_stream else 16000
        self._audio_queue.put(("mic", arr, sr))

    def _loopback_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            pass
        arr = indata.copy()
        rms = float(np.sqrt(np.mean(arr ** 2) + 1e-9))
        self.current_loopback_rms = min(1.0, rms * 5.0)

        sr = int(self.loopback_stream.samplerate) if self.loopback_stream else 16000
        self._audio_queue.put(("loopback", arr, sr))

    def _processing_loop(self):
        """Worker thread to pull chunks, resample, mix and dispatch."""
        buffer_mic = np.zeros(0, dtype=np.float32)
        buffer_loop = np.zeros(0, dtype=np.float32)
        last_log_time = 0.0
        last_mic_time = time.time()
        last_loop_time = time.time()

        while self.running:
            # 1. Pull incoming audio chunks from queue
            try:
                src, raw_data, sr = self._audio_queue.get(timeout=0.05)
                processed = self._resample_to_16k(raw_data, sr) * self.gain
                now = time.time()
                if src == "mic":
                    buffer_mic = np.concatenate([buffer_mic, processed])
                    last_mic_time = now
                else:
                    buffer_loop = np.concatenate([buffer_loop, processed])
                    last_loop_time = now

                # Drain any additional queued chunks immediately to minimize latency
                while True:
                    try:
                        src, raw_data, sr = self._audio_queue.get_nowait()
                        processed = self._resample_to_16k(raw_data, sr) * self.gain
                        now = time.time()
                        if src == "mic":
                            buffer_mic = np.concatenate([buffer_mic, processed])
                            last_mic_time = now
                        else:
                            buffer_loop = np.concatenate([buffer_loop, processed])
                            last_loop_time = now
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            has_mic = self.mic_stream is not None
            has_loop = self._loopback_active or (self.loopback_stream is not None)

            now = time.time()

            # Periodic diagnostic heartbeat if sound is playing
            if now - last_log_time >= 5.0:
                cur_rms = max(self.current_mic_rms, self.current_loopback_rms)
                if cur_rms > 0.004:
                    print(f"[*] 音訊串流活躍監聽中 - 喇叭強度: {self.current_loopback_rms:.3f}, 麥克風強度: {self.current_mic_rms:.3f} (智慧 AGC 自動增益調節中)")
                    last_log_time = now

            if has_mic and not has_loop:
                while len(buffer_mic) >= self.chunk_samples:
                    chunk = buffer_mic[: self.chunk_samples]
                    buffer_mic = buffer_mic[self.chunk_samples :]
                    self._dispatch_chunk(chunk)
                if len(buffer_mic) > self.max_buffer_samples:
                    buffer_mic = buffer_mic[-self.max_buffer_samples :]

            elif has_loop and not has_mic:
                while len(buffer_loop) >= self.chunk_samples:
                    chunk = buffer_loop[: self.chunk_samples]
                    buffer_loop = buffer_loop[self.chunk_samples :]
                    self._dispatch_chunk(chunk)
                if len(buffer_loop) > self.max_buffer_samples:
                    buffer_loop = buffer_loop[-self.max_buffer_samples :]

            elif has_mic and has_loop:
                # 1. Primary synchronous draining: mix when both streams have accumulated a full chunk
                while len(buffer_mic) >= self.chunk_samples and len(buffer_loop) >= self.chunk_samples:
                    m_chunk = buffer_mic[: self.chunk_samples]
                    buffer_mic = buffer_mic[self.chunk_samples :]
                    l_chunk = buffer_loop[: self.chunk_samples]
                    buffer_loop = buffer_loop[self.chunk_samples :]

                    mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
                    self._dispatch_chunk(mixed)

                # 2. Stall fallback / Drift compensation:
                # If one stream has stopped producing data for > stall_timeout or buffer drift exceeds max_drift_samples,
                # drain the active stream with zero-padding for the stalled stream to avoid freezing/latency buildup.
                if len(buffer_mic) >= self.chunk_samples:
                    loop_stalled = (now - last_loop_time >= self.stall_timeout) or (
                        len(buffer_mic) - len(buffer_loop) >= self.max_drift_samples and len(buffer_loop) < self.chunk_samples
                    )
                    if loop_stalled:
                        while len(buffer_mic) >= self.chunk_samples:
                            m_chunk = buffer_mic[: self.chunk_samples]
                            buffer_mic = buffer_mic[self.chunk_samples :]
                            if len(buffer_loop) >= self.chunk_samples:
                                l_chunk = buffer_loop[: self.chunk_samples]
                                buffer_loop = buffer_loop[self.chunk_samples :]
                            else:
                                l_chunk = np.zeros(self.chunk_samples, dtype=np.float32)
                            mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
                            self._dispatch_chunk(mixed)

                if len(buffer_loop) >= self.chunk_samples:
                    mic_stalled = (now - last_mic_time >= self.stall_timeout) or (
                        len(buffer_loop) - len(buffer_mic) >= self.max_drift_samples and len(buffer_mic) < self.chunk_samples
                    )
                    if mic_stalled:
                        while len(buffer_loop) >= self.chunk_samples:
                            l_chunk = buffer_loop[: self.chunk_samples]
                            buffer_loop = buffer_loop[self.chunk_samples :]
                            if len(buffer_mic) >= self.chunk_samples:
                                m_chunk = buffer_mic[: self.chunk_samples]
                                buffer_mic = buffer_mic[self.chunk_samples :]
                            else:
                                m_chunk = np.zeros(self.chunk_samples, dtype=np.float32)
                            mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
                            self._dispatch_chunk(mixed)

                # Bound max buffer size to prevent memory leaks and unrecoverable latency lag
                if len(buffer_mic) > self.max_buffer_samples:
                    buffer_mic = buffer_mic[-self.max_buffer_samples :]
                if len(buffer_loop) > self.max_buffer_samples:
                    buffer_loop = buffer_loop[-self.max_buffer_samples :]

        # Flush remaining complete chunks upon shutdown
        if has_mic and has_loop:
            while len(buffer_mic) >= self.chunk_samples and len(buffer_loop) >= self.chunk_samples:
                m_chunk = buffer_mic[: self.chunk_samples]
                buffer_mic = buffer_mic[self.chunk_samples :]
                l_chunk = buffer_loop[: self.chunk_samples]
                buffer_loop = buffer_loop[self.chunk_samples :]
                mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
                self._dispatch_chunk(mixed)
        elif has_mic:
            while len(buffer_mic) >= self.chunk_samples:
                chunk = buffer_mic[: self.chunk_samples]
                buffer_mic = buffer_mic[self.chunk_samples :]
                self._dispatch_chunk(chunk)
        elif has_loop:
            while len(buffer_loop) >= self.chunk_samples:
                chunk = buffer_loop[: self.chunk_samples]
                buffer_loop = buffer_loop[self.chunk_samples :]
                self._dispatch_chunk(chunk)

    def _dispatch_chunk(self, chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))
        if self.on_audio_chunk:
            try:
                self.on_audio_chunk(chunk, rms)
            except Exception as e:
                print(f"[!] Error in on_audio_chunk callback: {e}")

    def _run_soundcard_loopback(self, dev_id_or_name: Union[int, str]):
        """Background thread running native Windows WASAPI loopback capture via soundcard."""
        try:
            import soundcard as sc
            target_mic = None
            all_mics = sc.all_microphones(include_loopback=True)
            for m in all_mics:
                if m.isloopback and (str(dev_id_or_name) == m.name or str(dev_id_or_name) in m.name or str(dev_id_or_name) in str(m.id)):
                    target_mic = m
                    break
            if not target_mic:
                for m in all_mics:
                    if m.isloopback:
                        target_mic = m
                        break
            if not target_mic:
                target_mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

            print(f"[✓] 成功啟用 Windows 原生 WASAPI Loopback 系統聲音錄音: {target_mic.name}")
            self._loopback_active = True
            sr = 48000
            chunk_frames = int(sr * (self.chunk_ms / 1000.0))

            with target_mic.recorder(samplerate=sr, channels=2) as rec:
                while self.running:
                    data = rec.record(numframes=chunk_frames)
                    if not self.running:
                        break
                    arr = np.ascontiguousarray(data, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(arr ** 2) + 1e-9))
                    self.current_loopback_rms = min(1.0, rms * 5.0)
                    self._audio_queue.put(("loopback", arr, sr))
        except Exception as e:
            print(f"[!] 原生 WASAPI Loopback 錄音異常: {e}")
            self._loopback_active = False

    def start(self, mic_device: Optional[Union[int, str]] = None, loopback_device: Optional[Union[int, str]] = None):
        """Start capturing from mic, loopback, or both."""
        if self.running:
            self.stop()

        self.running = True
        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

        # 1. Microphone capture
        if mic_device is not None and mic_device != -1 and mic_device != "-1":
            try:
                self.mic_stream = self._create_stream(mic_device, is_loopback=False, callback=self._mic_callback)
                self.mic_stream.start()
                info = sd.query_devices(mic_device)
                print(f"[✓] 麥克風音訊已成功啟動: #{mic_device} ({info.get('name')})")
            except Exception as e:
                print(f"[!] 麥克風啟動失敗 (#{mic_device}): {e}")
                self.mic_stream = None

        # 2. System loopback capture
        if loopback_device is not None and loopback_device != -1 and loopback_device != "-1":
            started = False
            # Method A: Try native WASAPI Loopback via soundcard (100% reliable on Windows)
            try:
                import soundcard as sc
                self._loopback_thread = threading.Thread(
                    target=self._run_soundcard_loopback,
                    args=(loopback_device,),
                    daemon=True,
                )
                self._loopback_thread.start()
                started = True
            except Exception as e:
                print(f"[-] soundcard loopback not available: {e}")

            # Method B: Fallback to sounddevice
            if not started and isinstance(loopback_device, int):
                try:
                    self.loopback_stream = self._create_stream(loopback_device, is_loopback=True, callback=self._loopback_callback)
                    self.loopback_stream.start()
                    info = sd.query_devices(loopback_device)
                    print(f"[✓] 系統聲音 (Loopback) 已成功啟動: #{loopback_device} ({info.get('name')})")
                    self._loopback_active = True
                except Exception as e:
                    print(f"[!] 系統聲音啟動失敗 (#{loopback_device}): {e}")
                    self.loopback_stream = None

    def stop(self):
        """Stop all capture streams."""
        self.running = False
        self._loopback_active = False

        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None

        if self.loopback_stream:
            try:
                self.loopback_stream.stop()
                self.loopback_stream.close()
            except Exception:
                pass
            self.loopback_stream = None

        if self._loopback_thread and self._loopback_thread.is_alive():
            self._loopback_thread.join(timeout=0.5)
        self._loopback_thread = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        print("[*] Audio capture stopped.")
