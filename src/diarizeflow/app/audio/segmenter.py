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
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class SpeakerChannelBuffer:
    """Manages audio accumulation, silence hang-over, and pre-encode embeddings for a speaker channel."""

    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        self.chunks: List[np.ndarray] = []
        self.in_speech: bool = False
        self.speech_samples: int = 0
        self.silence_samples: int = 0
        self.start_timestamp: float = 0.0
        self.embeddings: List[np.ndarray] = []

    def start_speech(
        self,
        pre_pad_chunks: List[np.ndarray],
        initial_chunk: Optional[np.ndarray] = None,
        initial_embedding: Optional[np.ndarray] = None,
    ):
        """Begin an utterance, initializing buffer with pre-pad context chunks."""
        self.chunks = list(pre_pad_chunks)
        self.speech_samples = sum(len(c) for c in self.chunks)
        self.silence_samples = 0
        self.in_speech = True
        self.embeddings.clear()
        if initial_chunk is not None:
            self.chunks.append(initial_chunk)
            self.speech_samples += len(initial_chunk)
        if initial_embedding is not None and isinstance(initial_embedding, np.ndarray):
            self.embeddings.append(initial_embedding)

    def add_speech_chunk(self, chunk: np.ndarray, embedding: Optional[np.ndarray] = None):
        """Append an active speech chunk and reset silence hangover counter."""
        self.chunks.append(chunk)
        self.speech_samples += len(chunk)
        self.silence_samples = 0
        if embedding is not None and isinstance(embedding, np.ndarray):
            self.embeddings.append(embedding)

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

    def get_utterance_embedding(self) -> Optional[np.ndarray]:
        """Compute the mean normalized speaker embedding accumulated across this utterance."""
        if not self.embeddings:
            return None
        mean_emb = np.mean(self.embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 1e-6:
            mean_emb /= norm
        return mean_emb

    def reset(self):
        """Reset buffer state for the next utterance."""
        self.chunks.clear()
        self.in_speech = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.embeddings.clear()


class StreamingDiarizationSegmenter:
    """Streaming multi-track audio segmenter driven by Nemotron Sortformer SAD."""

    def __init__(
        self,
        sample_rate: int = 16000,
        sad_threshold: float = 0.50,
        silence_timeout_ms: int = 350,
        min_speech_ms: int = 200,
        max_speech_s: float = 6.0,
        pre_pad_ms: int = 150,
        post_pad_ms: int = 150,
        diarizer: Optional[Any] = None,
        on_utterance: Optional[Callable[[np.ndarray, str, float, float], None]] = None,
        enable_deep_identification: bool = False,
        tse_extractor: Optional[Any] = None,
        tse_enabled: bool = True,
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
        self.enable_deep_identification = enable_deep_identification
        self.tse_extractor = tse_extractor
        self.tse_enabled = tse_enabled

        # Overlap and TSE separation statistics
        self.total_overlap_chunks: int = 0
        self.separated_overlap_chunks: int = 0
        self.channel_last_embeddings: Dict[int, np.ndarray] = {}
        self.last_chunk_source: str = "mixed"

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

        # Acoustic Energy Fallback state (used when ONNX model is unavailable or for energy gating)
        self.noise_floor: float = 0.002
        self.noise_alpha: float = 0.05
        self.energy_threshold: float = 0.008
        self.last_emission_source: str = "neural_sad"

    @property
    def is_neural_sad_active(self) -> bool:
        """Whether neural Sortformer SAD is currently loaded and available."""
        return (
            self.diarizer is not None
            and getattr(self.diarizer, "session", None) is not None
            and hasattr(self.diarizer, "forward_streaming_step")
        )

    def process_chunk(
        self,
        chunk: np.ndarray,
        rms: Optional[float] = None,
        source: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, str, float, float]]:
        """Ingest streaming audio chunk and return any completed speaker utterances.

        Args:
            chunk: 1D float32 audio array (typically 100ms - 250ms).
            rms: Optional pre-computed RMS energy.
            source: Optional audio track source tag ('mic', 'loopback', 'mixed').

        Returns:
            List of (audio_segment, speaker_label, confidence, duration).
        """
        if len(chunk) == 0:
            return []

        if source is not None:
            self.last_chunk_source = source

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
            self.last_emission_source = "neural_sad"
            completed_utterances = self._process_neural_sad(chunk, rms)
        else:
            self.last_emission_source = "fallback_energy"
            completed_utterances = self._process_fallback_energy(chunk, rms)

        # Append to pre-pad ring buffer for future speech onset
        self._pre_buffer.append(chunk)

        # Trigger callback if registered
        if self.on_utterance:
            is_neural = (self.last_emission_source == "neural_sad")
            sig = inspect.signature(self.on_utterance)
            accepts_5 = len(sig.parameters) >= 5 or any(
                p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for p in sig.parameters.values()
            ) or ("is_neural_sad" in sig.parameters)
            for seg_audio, spk, conf, dur in completed_utterances:
                try:
                    if accepts_5:
                        self.on_utterance(seg_audio, spk, conf, dur, is_neural)
                    else:
                        self.on_utterance(seg_audio, spk, conf, dur)
                except Exception as e:
                    print(f"[!] Error in on_utterance callback: {e}")

        return completed_utterances

    def _process_neural_sad(
        self, chunk: np.ndarray, rms: float
    ) -> List[Tuple[np.ndarray, str, float, float]]:
        """Process chunk using Sortformer Frame-level SAD probabilities and energy gating."""
        self._audio_window.append(chunk)
        self._window_total_samples += len(chunk)

        # Trim window to maximum 1.04s context
        while self._window_total_samples > self._window_max_samples and len(self._audio_window) > 1:
            removed = self._audio_window.pop(0)
            self._window_total_samples -= len(removed)

        window_audio = np.concatenate(self._audio_window).astype(np.float32)

        # Execute streaming forward step
        step_emb = None
        probs = None
        if hasattr(self.diarizer, "forward_streaming_step"):
            res = self.diarizer.forward_streaming_step(window_audio, self.sample_rate)
            if isinstance(res, tuple) and len(res) == 2:
                probs, cand_emb = res
                if isinstance(cand_emb, np.ndarray):
                    step_emb = cand_emb
            else:
                probs = res
                cand_emb = getattr(self.diarizer, "last_pre_encode_embs", None)
                if isinstance(cand_emb, np.ndarray):
                    step_emb = cand_emb

        # Dynamically track background noise floor to prevent false triggers on quiet hiss / ambient room noise
        any_in_speech = any(buf.in_speech for buf in self.channel_buffers.values())
        if not any_in_speech:
            if rms < self.noise_floor:
                self.noise_floor = 0.85 * self.noise_floor + 0.15 * rms
            else:
                self.noise_floor = (1.0 - self.noise_alpha) * self.noise_floor + self.noise_alpha * min(
                    rms, self.noise_floor * 1.5
                )
        else:
            if rms < self.noise_floor:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms

        min_speech_energy = max(0.004, self.noise_floor * 1.5)
        has_audible_energy = rms >= min_speech_energy

        active_channels = set()
        if probs is not None and len(probs) > 0:
            # 1 Sortformer output frame = 40ms = 640 audio samples
            chunk_frames = max(1, int(np.ceil(len(chunk) / 640)))
            recent_probs = probs[-chunk_frames:, :]  # shape (F, 8)

            for ch in range(min(self.num_channels, recent_probs.shape[1])):
                ch_p = recent_probs[:, ch]
                is_currently_in_speech = self.channel_buffers[ch].in_speech

                if is_currently_in_speech:
                    # Hysteresis for ongoing speech so natural pauses/dips in words aren't cut prematurely
                    thresh = max(0.30, self.sad_threshold - 0.10)
                    if float(np.max(ch_p)) >= thresh:
                        active_channels.add(ch)
                else:
                    # Onset of new speech requires audible acoustic energy AND sustained probability
                    if has_audible_energy and (
                        float(np.mean(ch_p)) >= self.sad_threshold
                        or np.sum(ch_p >= self.sad_threshold) >= 2
                        or float(np.max(ch_p)) >= min(0.90, self.sad_threshold + 0.20)
                    ):
                        active_channels.add(ch)

        completed: List[Tuple[np.ndarray, str, float, float]] = []
        is_overlap = len(active_channels) > 1

        if is_overlap:
            self.total_overlap_chunks += 1

        # Route audio and update per-channel buffers
        for ch, buf in self.channel_buffers.items():
            if ch in active_channels:
                routed_chunk = chunk
                if is_overlap and self.tse_extractor is not None and self.tse_enabled:
                    # Target-Speaker Extraction (TSE) conditioned on target speaker's embedding
                    target_emb = buf.get_utterance_embedding()
                    if target_emb is None:
                        target_emb = self.channel_last_embeddings.get(ch)
                    if target_emb is None:
                        target_emb = step_emb

                    if target_emb is not None:
                        try:
                            routed_chunk = self.tse_extractor.extract(
                                chunk, target_emb, sample_rate=self.sample_rate
                            )
                            self.separated_overlap_chunks += 1
                        except Exception as e:
                            print(f"[!] Warning: TSE extraction failed on channel {ch}: {e}")
                            routed_chunk = chunk
                elif not is_overlap and step_emb is not None:
                    self.channel_last_embeddings[ch] = step_emb

                if not buf.in_speech:
                    buf.start_speech(list(self._pre_buffer), initial_chunk=routed_chunk, initial_embedding=step_emb)
                else:
                    buf.add_speech_chunk(routed_chunk, embedding=step_emb)

                # Enforce max duration cutoff even during continuous speech
                if buf.speech_samples >= self.max_speech_samples:
                    if buf.speech_samples >= self.min_speech_samples:
                        seg_audio = buf.get_utterance_audio()
                        mean_emb = buf.get_utterance_embedding()
                        if mean_emb is not None:
                            self.channel_last_embeddings[ch] = mean_emb
                        spk_label, conf = self._identify_speaker_segment(seg_audio, ch, embedding=mean_emb)
                        dur = round(len(seg_audio) / self.sample_rate, 2)
                        completed.append((seg_audio, spk_label, conf, dur))
                    buf.reset()
                    # Seamlessly continue speech accumulation for next chunk
                    buf.start_speech([], initial_chunk=None, initial_embedding=step_emb)
            else:
                if buf.in_speech:
                    buf.add_silence_chunk(chunk)
                    # Check if silence timeout or max duration reached
                    is_silence_timeout = buf.silence_samples >= self.silence_timeout_samples
                    is_max_duration = buf.speech_samples >= self.max_speech_samples

                    if is_silence_timeout or is_max_duration:
                        if buf.speech_samples >= self.min_speech_samples:
                            seg_audio = buf.get_utterance_audio()
                            mean_emb = buf.get_utterance_embedding()
                            if mean_emb is not None:
                                self.channel_last_embeddings[ch] = mean_emb
                            spk_label, conf = self._identify_speaker_segment(seg_audio, ch, embedding=mean_emb)
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

            # Enforce max duration cutoff in fallback mode
            if buf.speech_samples >= self.max_speech_samples:
                if buf.speech_samples >= self.min_speech_samples:
                    seg_audio = buf.get_utterance_audio()
                    spk_label, conf = self._identify_speaker_segment(seg_audio, 0, embedding=None)
                    dur = round(len(seg_audio) / self.sample_rate, 2)
                    completed.append((seg_audio, spk_label, conf, dur))
                buf.reset()
                buf.start_speech([], initial_chunk=None)
        else:
            if buf.in_speech:
                buf.add_silence_chunk(chunk)
                is_silence_timeout = buf.silence_samples >= self.silence_timeout_samples
                is_max_duration = buf.speech_samples >= self.max_speech_samples

                if is_silence_timeout or is_max_duration:
                    if buf.speech_samples >= self.min_speech_samples:
                        seg_audio = buf.get_utterance_audio()
                        spk_label, conf = self._identify_speaker_segment(seg_audio, 0, embedding=None)
                        dur = round(len(seg_audio) / self.sample_rate, 2)
                        completed.append((seg_audio, spk_label, conf, dur))
                    buf.reset()

        return completed

    def _identify_speaker_segment(
        self,
        audio: np.ndarray,
        channel_hint: int,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """Identify speaker identity using cached pre-encode embedding, channel hint, or fallback.

        When Sortformer pre-encode embedding is available, performs instantaneous vector similarity
        matching against persistent and temporary profiles, eliminating redundant ONNX forward passes.
        """
        duration_s = float(len(audio) / self.sample_rate)

        # 1. Fast embedding reuse: Match against persistent/temporary voiceprints without ONNX forward
        if (
            embedding is not None
            and isinstance(embedding, np.ndarray)
            and self.diarizer is not None
            and hasattr(self.diarizer, "identify_speaker_from_embedding")
        ):
            try:
                res = self.diarizer.identify_speaker_from_embedding(
                    embedding, duration_s=duration_s
                )
                if isinstance(res, tuple) and len(res) >= 2 and isinstance(res[0], str):
                    return res[0], float(res[1])
            except Exception as e:
                print(f"[!] Warning in fast speaker identification from embedding: {e}")

        # 2. Deep identification fallback (legacy or when embedding is not available)
        if (
            self.enable_deep_identification
            and self.diarizer is not None
            and hasattr(self.diarizer, "identify_speaker")
        ):
            try:
                spk, conf, _ = self.diarizer.identify_speaker(audio, self.sample_rate)
                return spk, conf
            except Exception as e:
                print(f"[!] Warning in segmenter deep speaker identification: {e}")

        # 3. Default channel-based label
        return f"講者 {channel_hint + 1}", 1.0

    def separate_utterance(
        self, audio: np.ndarray, speaker_embedding: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply Target-Speaker Extraction (TSE) to separate clean target waveform from mixture."""
        if self.tse_extractor is not None and self.tse_enabled and speaker_embedding is not None:
            return self.tse_extractor.extract(audio, speaker_embedding, sample_rate=self.sample_rate)
        return audio

    @property
    def overlap_stats(self) -> Dict[str, int]:
        """Return overlap speech detection and separation statistics."""
        return {
            "total_overlap_chunks": self.total_overlap_chunks,
            "separated_overlap_chunks": self.separated_overlap_chunks,
        }

    def flush(self) -> List[Tuple[np.ndarray, str, float, float]]:
        """Flush and return any pending speaker utterances in active buffers."""
        completed: List[Tuple[np.ndarray, str, float, float]] = []
        for ch, buf in self.channel_buffers.items():
            if buf.in_speech and buf.speech_samples >= self.min_speech_samples:
                seg_audio = buf.get_utterance_audio()
                mean_emb = buf.get_utterance_embedding()
                if mean_emb is not None:
                    self.channel_last_embeddings[ch] = mean_emb
                spk, conf = self._identify_speaker_segment(seg_audio, ch, embedding=mean_emb)
                dur = round(len(seg_audio) / self.sample_rate, 2)
                completed.append((seg_audio, spk, conf, dur))
                buf.reset()
        return completed
