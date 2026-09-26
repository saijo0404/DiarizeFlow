"""Voice Activity Detection (VAD) and Speech Segmenter.

Segments real-time streaming audio chunks into continuous speech utterances
with pre-padding and silence hang-over protection.
"""

from collections import deque
from typing import Callable, List, Optional
import numpy as np


class EnergyVADSegmenter:
    """Detects voice activity using adaptive energy threshold and speech state machine."""

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.008,
        min_speech_ms: int = 250,
        silence_timeout_ms: int = 300,
        max_speech_s: float = 3.5,
        pre_pad_ms: int = 350,
        on_speech_utterance: Optional[Callable[[np.ndarray, float], None]] = None,
    ):
        self.sr = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_ms = min_speech_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.max_speech_s = max_speech_s
        self.on_speech_utterance = on_speech_utterance

        # Adaptive background noise tracking (low initial floor to immediately catch soft speech)
        self.noise_floor: float = 0.002
        self.noise_alpha: float = 0.05

        # Pre-pad buffer to preserve the onset of speech
        self.pre_pad_frames = max(3, int(pre_pad_ms / 100))
        self._pre_buffer = deque(maxlen=self.pre_pad_frames)

        self._in_speech = False
        self._current_utterance: List[np.ndarray] = []
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0

    def process_chunk(self, chunk: np.ndarray, rms: float) -> Optional[np.ndarray]:
        """Process incoming 16kHz audio chunk with adaptive ambient noise tracking."""
        chunk_ms = (len(chunk) / self.sr) * 1000.0

        # Dynamically track background noise floor
        if not self._in_speech:
            if rms < self.noise_floor:
                self.noise_floor = 0.8 * self.noise_floor + 0.2 * rms
            else:
                self.noise_floor = (1.0 - self.noise_alpha) * self.noise_floor + self.noise_alpha * min(rms, self.noise_floor * 1.5)
        else:
            if rms < self.noise_floor:
                self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms

        # Speech threshold must rise dynamically above ambient noise floor:
        # In quiet environments (noise_floor ~0.001), threshold is ~0.0035 to catch soft speech (0.006).
        # In noisy/game environments (noise_floor ~0.02), threshold rises to ~0.035 to avoid false triggers.
        noise_adaptive_thresh = self.noise_floor * 1.65 + 0.002
        if self.noise_floor < 0.003:
            effective_thresh = max(0.0035, min(self.energy_threshold, noise_adaptive_thresh))
        else:
            effective_thresh = max(self.energy_threshold, noise_adaptive_thresh)

        is_speech = rms >= effective_thresh

        completed_segment: Optional[np.ndarray] = None

        if not self._in_speech:
            self._pre_buffer.append(chunk)
            if is_speech:
                self._speech_duration_ms += chunk_ms
                if self._speech_duration_ms >= self.min_speech_ms:
                    # Speech onset triggered!
                    self._in_speech = True
                    self._current_utterance = list(self._pre_buffer)
                    self._pre_buffer.clear()
                    self._silence_duration_ms = 0.0
            else:
                self._speech_duration_ms = 0.0
        else:
            # Currently inside a speech utterance
            self._current_utterance.append(chunk)
            self._speech_duration_ms += chunk_ms

            if is_speech:
                self._silence_duration_ms = 0.0
            else:
                self._silence_duration_ms += chunk_ms

            # Check if utterance is complete
            silence_timeout = self._silence_duration_ms >= self.silence_timeout_ms
            max_duration_reached = (self._speech_duration_ms / 1000.0) >= self.max_speech_s

            if silence_timeout or max_duration_reached:
                if len(self._current_utterance) > 0:
                    completed_segment = np.concatenate(self._current_utterance)
                    duration_s = len(completed_segment) / self.sr
                    if self.on_speech_utterance and duration_s >= (self.min_speech_ms / 1000.0):
                        try:
                            self.on_speech_utterance(completed_segment, duration_s)
                        except Exception as e:
                            print(f"[!] Error in on_speech_utterance callback: {e}")

                # If maximum duration reached during continuous speech, remain in speech state
                if max_duration_reached and is_speech:
                    self._in_speech = True
                    self._current_utterance = []
                    self._speech_duration_ms = 0.0
                    self._silence_duration_ms = 0.0
                else:
                    # Natural silence end
                    self._in_speech = False
                    self._current_utterance = []
                    self._speech_duration_ms = 0.0
                    self._silence_duration_ms = 0.0

        return completed_segment

    def flush(self) -> Optional[np.ndarray]:
        """Flush any remaining speech in buffer."""
        if self._in_speech and self._current_utterance:
            completed_segment = np.concatenate(self._current_utterance)
            duration_s = len(completed_segment) / self.sr
            self._in_speech = False
            self._current_utterance = []
            if self.on_speech_utterance and duration_s >= (self.min_speech_ms / 1000.0):
                self.on_speech_utterance(completed_segment, duration_s)
            return completed_segment
        return None
