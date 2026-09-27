"""Streaming Diarization-Driven ASR Segmenter.

Replaces traditional energy-based VAD with neural Speaker Activity Detection (SAD)
powered by NVIDIA Nemotron-3 (Sortformer) multi-track posterior probabilities.

Maintains independent per-speaker audio buffers:
- Continuous audio streaming chunks are ingested.
- Neural SAD evaluates multi-speaker activity posterior probabilities P(t, k).
- Audio is routed to independent buffers for each active speaker channel.
- Overlap speech is cloned to all active speakers' buffers without collision.
- Silence hangover / falling edge detection terminates a speaker's turn.
- Context padding (pre-pad & post-pad) is attached to preserve phonetic boundaries.
- Emits completed speaker segments with accurate speaker identities directly to ASR.
"""

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class SpeakerChannelBuffer:
    """Manages audio accumulation and silence hang-over for a single speaker channel."""

    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        self.chunks: List[np.ndarray] = []
        self.in_speech: bool = False
        self.speech_samples: int = 0
        self.silence_samples: int = 0
        self.start_timestamp: float = 0.0

    def start_speech(self, pre_pad_chunks: List[np.ndarray], initial_chunk: Optional[np.ndarray] = None):
        """Begin an utterance, initializing buffer with pre-pad context chunks."""
        self.chunks = list(pre_pad_chunks)
        self.speech_samples = sum(len(c) for c in self.chunks)
        self.silence_samples = 0
        self.in_speech = True
        if initial_chunk is not None:
            self.chunks.append(initial_chunk)
            self.speech_samples += len(initial_chunk)

    def add_speech_chunk(self, chunk: np.ndarray):
        """Append an active speech chunk and reset silence hangover counter."""
        self.chunks.append(chunk)
        self.speech_samples += len(chunk)
        self.silence_samples = 0

    def add_silence_chunk(self, chunk: np.ndarray):
        """Append trailing audio during silence hangover (serves as post-padding)."""
        self.chunks.append(chunk)
        self.silence_samples += len(chunk)
        self.speech_samples += len(chunk)

    def get_utterance_audio(self) -> np.ndarray:
        """Concatenate all buffered chunks into a contiguous 1D float32 audio segment."""
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).astype(np.float32)

    def reset(self):
        """Reset buffer state for the next utterance."""
        self.chunks.clear()
        self.in_speech = False
        self.speech_samples = 0
        self.silence_samples = 0


class StreamingDiarizationSegmenter:
    """Streaming multi-track audio segmenter driven by Nemotron Sortformer SAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        sad_threshold: float = 0.40,
        silence_timeout_ms: int = 350,
        min_speech_ms: int = 200,
        max_speech_s: float = 6.0,
        pre_pad_ms: int = 150,
        post_pad_ms: int = 150,
        diarizer: Optional[Any] = None,
        on_utterance: Optional[Callable[[np.ndarray, str, float, float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.sad_threshold = sad_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_s = max_speech_s
        self.pre_pad_ms = pre_pad_ms
        self.post_pad_ms = post_pad_ms
        self.diarizer = diarizer
        self.on_utterance = on_utterance

        # Sample limits
        self.silence_timeout_samples = int((silence_timeout_ms / 1000.0) * self.sample_rate)
        self.min_speech_samples = int((min_speech_ms / 1000.0) * self.sample_rate)
        self.max_speech_samples = int(max_speech_s * self.sample_rate)
        self.pre_pad_samples = int((pre_pad_ms / 1000.0) * self.sample_rate)

        # 8-channel speaker buffer pool (one for each Sortformer output channel)
        self.num_channels = 8
        self.channel_buffers: Dict[int, SpeakerChannelBuffer] = {
            i: SpeakerChannelBuffer(channel_id=i) for i in range(self.num_channels)
        }

        # Pre-pad ring buffer
        self._pre_buffer_max_chunks = max(2, int(self.pre_pad_samples / 4000) + 1)
        self._pre_buffer: deque = deque(maxlen=self._pre_buffer_max_chunks)

        # Rolling audio context window for Sortformer inference
        self._window_max_samples = int(1.04 * self.sample_rate)  # 1.04s buffer
        self._audio_window: List[np.ndarray] = []
        self._window_total_samples = 0

        # Acoustic Energy Fallback state (used when ONNX model is unavailable)
        self.noise_floor: float = 0.002
        self.noise_alpha: float = 0.05
        self.energy_threshold: float = 0.008

    def process_chunk(
        self, chunk: np.ndarray, rms: Optional[float] = None
    ) -> List[Tuple[np.ndarray, str, float, float]]:
        """Ingest streaming audio chunk and return any completed speaker utterances.

        Args:
            chunk: 1D float32 audio array (typically 100ms - 250ms).
            rms: Optional pre-computed RMS energy.

        Returns:
            List of (audio_segment, speaker_label, confidence, duration).
        """
        if len(chunk) == 0:
            return []

        if rms is None:
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        completed_utterances: List[Tuple[np.ndarray, str, float, float]] = []

        # Check if neural Sortformer SAD is available
        has_onnx = (
            self.diarizer is not None
            and getattr(self.diarizer, "session", None) is not None
            and hasattr(self.diarizer, "forward_streaming_step")
        )

        if has_onnx:
            completed_utterances = self._process_neural_sad(chunk)
        else:
            completed_utterances = self._process_fallback_energy(chunk, rms)

        # Append to pre-pad ring buffer for future speech onset
        self._pre_buffer.append(chunk)

        # Trigger callback if registered
        if self.on_utterance:
            for seg_audio, spk, conf, dur in completed_utterances:
                try:
                    self.on_utterance(seg_audio, spk, conf, dur)
                except Exception as e:
                    print(f"[!] Error in on_utterance callback: {e}")

        return completed_utterances

    def _process_neural_sad(self, chunk: np.ndarray) -> List[Tuple[np.ndarray, str, float, float]]:
        """Process chunk using Sortformer Frame-level SAD probabilities."""
        self._audio_window.append(chunk)
        self._window_total_samples += len(chunk)

        # Trim window to maximum 1.04s context
        while self._window_total_samples > self._window_max_samples and len(self._audio_window) > 1:
            removed = self._audio_window.pop(0)
            self._window_total_samples -= len(removed)

        window_audio = np.concatenate(self._audio_window).astype(np.float32)

        # Execute streaming forward step
        probs = self.diarizer.forward_streaming_step(window_audio, self.sample_rate)

        active_channels = set()
        if probs is not None and len(probs) > 0:
            # 1 Sortformer output frame = 40ms = 640 audio samples
            # Determine how many output frames correspond to the newly arrived chunk
            chunk_frames = max(1, int(np.ceil(len(chunk) / 640)))
            recent_probs = probs[-chunk_frames:, :]  # shape (F, 8)

            for ch in range(min(self.num_channels, recent_probs.shape[1])):
                max_prob = float(np.max(recent_probs[:, ch]))
                if max_prob >= self.sad_threshold:
                    active_channels.add(ch)

        completed: List[Tuple[np.ndarray, str, float, float]] = []

        # Route audio and update per-channel buffers
        for ch, buf in self.channel_buffers.items():
            if ch in active_channels:
                if not buf.in_speech:
                    buf.start_speech(list(self._pre_buffer), initial_chunk=chunk)
                else:
                    buf.add_speech_chunk(chunk)
            else:
                if buf.in_speech:
                    buf.add_silence_chunk(chunk)
                    # Check if silence timeout or max duration reached
                    is_silence_timeout = buf.silence_samples >= self.silence_timeout_samples
                    is_max_duration = buf.speech_samples >= self.max_speech_samples

                    if is_silence_timeout or is_max_duration:
                        if buf.speech_samples >= self.min_speech_samples:
                            seg_audio = buf.get_utterance_audio()
                            spk_label, conf = self._identify_speaker_segment(seg_audio, ch)
                            dur = round(len(seg_audio) / self.sample_rate, 2)
                            completed.append((seg_audio, spk_label, conf, dur))
                        buf.reset()

        return completed

    def _process_fallback_energy(
        self, chunk: np.ndarray, rms: float
    ) -> List[Tuple[np.ndarray, str, float, float]]:
        """Acoustic energy SAD fallback when ONNX model is missing or disabled."""
        # Dynamically track background noise floor
        buf = self.channel_buffers[0]
        if not buf.in_speech:
            if rms < self.noise_floor:
                self.noise_floor = 0.8 * self.noise_floor + 0.2 * rms
            else:
                self.noise_floor = (1.0 - self.noise_alpha) * self.noise_floor + self.noise_alpha * min(
                    rms, self.noise_floor * 1.5
                )
        else:
            if rms < self.noise_floor:
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms

        noise_adaptive_thresh = self.noise_floor * 1.65 + 0.002
        if self.noise_floor < 0.003:
            effective_thresh = max(0.0035, min(self.energy_threshold, noise_adaptive_thresh))
        else:
            effective_thresh = max(self.energy_threshold, noise_adaptive_thresh)

        is_speech = rms >= effective_thresh
        completed: List[Tuple[np.ndarray, str, float, float]] = []

        if is_speech:
            if not buf.in_speech:
                buf.start_speech(list(self._pre_buffer), initial_chunk=chunk)
            else:
                buf.add_speech_chunk(chunk)
        else:
            if buf.in_speech:
                buf.add_silence_chunk(chunk)
                is_silence_timeout = buf.silence_samples >= self.silence_timeout_samples
                is_max_duration = buf.speech_samples >= self.max_speech_samples

                if is_silence_timeout or is_max_duration:
                    if buf.speech_samples >= self.min_speech_samples:
                        seg_audio = buf.get_utterance_audio()
                        spk_label, conf = self._identify_speaker_segment(seg_audio, 0)
                        dur = round(len(seg_audio) / self.sample_rate, 2)
                        completed.append((seg_audio, spk_label, conf, dur))
                    buf.reset()

        return completed

    def _identify_speaker_segment(self, audio: np.ndarray, channel_hint: int) -> Tuple[str, float]:
        """Identify speaker identity using diarizer's voiceprint database or fallback."""
        if self.diarizer is not None and hasattr(self.diarizer, "identify_speaker"):
            try:
                spk, conf, _ = self.diarizer.identify_speaker(audio, self.sample_rate)
                return spk, conf
            except Exception as e:
                print(f"[!] Warning in segmenter speaker identification: {e}")

        return f"講者 {channel_hint + 1}", 1.0

    def flush(self) -> List[Tuple[np.ndarray, str, float, float]]:
        """Flush and return any pending speaker utterances in active buffers."""
        completed: List[Tuple[np.ndarray, str, float, float]] = []
        for ch, buf in self.channel_buffers.items():
            if buf.in_speech and buf.speech_samples >= self.min_speech_samples:
                seg_audio = buf.get_utterance_audio()
                spk, conf = self._identify_speaker_segment(seg_audio, ch)
                dur = round(len(seg_audio) / self.sample_rate, 2)
                completed.append((seg_audio, spk, conf, dur))
                buf.reset()
        return completed
