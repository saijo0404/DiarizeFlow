"""Automatic Gain Control (AGC) and Dynamic Range Normalization for DiarizeFlow.

Ensures real-time audio inputs remain in the optimal dynamic range (-22 dBFS)
for ASR (SenseVoice, Faster-Whisper) and Speaker Diarization (Nemotron-3):
- Continuously scales streaming audio chunks (StreamingInputAGC) to guarantee VAD trigger
- Boosts soft speech / distant whispers so phonemes aren't lost
- Attenuates loud speech / game sound effects to prevent digital clipping
- Provides soft-knee peak limiting to guarantee zero distortion
"""

from typing import Tuple
import numpy as np


class StreamingInputAGC:
    """Real-time Streaming Automatic Gain Control (AGC) for audio chunks.

    Continuously tracks streaming input signal envelope and smoothly scales audio chunks:
    - Automatically amplifies quiet sound sources (e.g. low Windows volume, soft whispers)
    - Dynamically compresses loud sound sources (e.g. game sound effects, loud music)
    - Distinguishes active speech/sound from ambient floor so background silence is not amplified into noise
    - Smoothly transitions gain using asymmetric attack/release to prevent clipping and pumping
    - Soft-knee limiter ensures peak amplitude stays below ceiling (zero digital distortion)
    """

    def __init__(
        self,
        target_rms: float = 0.06,
        max_gain: float = 25.0,
        min_gain: float = 0.15,
        attack_alpha: float = 0.60,   # Fast attack when signal gets louder to avoid clipping
        release_alpha: float = 0.40,  # Fast boost when quiet speech arrives so first syllables are captured
        peak_limit: float = 0.92,
        noise_floor_init: float = 0.001,
    ):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.attack_alpha = attack_alpha
        self.release_alpha = release_alpha
        self.peak_limit = peak_limit

        self.current_gain: float = 1.0
        self.noise_floor: float = noise_floor_init

    def process(self, chunk: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Process incoming 16kHz audio chunk with real-time dynamic gain.

        Args:
            chunk: 1D float32 audio numpy array.

        Returns:
            Tuple of (processed_chunk, current_gain, output_rms)
        """
        if len(chunk) == 0:
            return chunk, self.current_gain, 0.0

        raw_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-9))

        # Dynamically track background ambient noise floor
        if raw_rms < self.noise_floor:
            self.noise_floor = 0.85 * self.noise_floor + 0.15 * raw_rms
        else:
            self.noise_floor = 0.98 * self.noise_floor + 0.02 * min(raw_rms, self.noise_floor * 1.5)

        # Ambient silence gate: avoid inflating digital zero or pure background hiss
        is_silence = raw_rms < max(0.0003, self.noise_floor * 1.4)

        if is_silence:
            # During silence, gently decay high gain back towards neutral 1.0~2.5 to avoid noise breathing
            desired_gain = min(self.current_gain, 2.5)
            alpha = 0.02
        else:
            # Sound activity detected: calculate desired gain to achieve target RMS
            desired_gain = float(np.clip(self.target_rms / max(raw_rms, 1e-5), self.min_gain, self.max_gain))
            # Asymmetric smoothing: fast attack when sound becomes louder, smooth release when quieter
            alpha = self.attack_alpha if desired_gain < self.current_gain else self.release_alpha

        # Exponential moving average gain adjustment
        self.current_gain = (1.0 - alpha) * self.current_gain + alpha * desired_gain

        # Apply gain
        processed = chunk * self.current_gain

        # Soft-knee peak limiting to strictly prevent digital saturation / clipping
        peak = float(np.max(np.abs(processed)))
        if peak > self.peak_limit:
            mask = np.abs(processed) > self.peak_limit
            excess = np.abs(processed[mask]) - self.peak_limit
            headroom = max(1.0 - self.peak_limit, 1e-4)
            processed[mask] = np.sign(processed[mask]) * (self.peak_limit + headroom * np.tanh(excess / headroom))

        out_rms = float(np.sqrt(np.mean(processed ** 2) + 1e-9))
        return processed.astype(np.float32), self.current_gain, out_rms

    def reset(self):
        """Reset internal gain state."""
        self.current_gain = 1.0
        self.noise_floor = 0.001


def apply_speech_agc(
    audio: np.ndarray,
    target_rms: float = 0.08,
    max_gain: float = 6.0,
    min_gain: float = 0.20,
    peak_limit: float = 0.90,
) -> Tuple[np.ndarray, float]:
    """Dynamically adjust audio volume for optimal ASR feature extraction.

    Args:
        audio: 1D float32 audio waveform (16kHz).
        target_rms: Target RMS energy level (default: 0.08, ~-22 dBFS).
        max_gain: Maximum amplification factor (default: 6.0, +15.5 dB).
        min_gain: Minimum attenuation factor (default: 0.20, -14 dB).
        peak_limit: Maximum allowed absolute amplitude (default: 0.90).

    Returns:
        Tuple of (normalized_audio, applied_gain_factor)
    """
    if len(audio) == 0:
        return audio, 1.0

    orig_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    orig_peak = float(np.max(np.abs(audio)))

    # Ignore pure silence
    if orig_rms < 1e-4:
        return audio, 1.0

    # Desired gain to hit target RMS
    desired_gain = target_rms / orig_rms
    gain = float(np.clip(desired_gain, min_gain, max_gain))

    # Peak ceiling protection
    if orig_peak * gain > peak_limit:
        gain = peak_limit / max(orig_peak, 1e-6)

    processed = audio * gain

    # Soft-knee limiter on extreme peaks to prevent any digital hard clipping
    mask = np.abs(processed) > peak_limit
    if np.any(mask):
        excess = np.abs(processed[mask]) - peak_limit
        headroom = max(1.0 - peak_limit, 1e-4)
        processed[mask] = np.sign(processed[mask]) * (peak_limit + headroom * np.tanh(excess / headroom))

    return processed.astype(np.float32), gain
