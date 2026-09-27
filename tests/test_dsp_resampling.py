"""Unit tests for DSP audio resampling and AGC processing (Issue #4)."""

import unittest
import numpy as np

from diarizeflow.app.audio.capture import AudioCaptureStream
from diarizeflow.app.config import AppConfig
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline


class TestDSPAudioResampling(unittest.TestCase):
    """Test polyphase resampling anti-aliasing and pipeline AGC coherence."""

    def test_resampling_anti_aliasing_attenuates_above_nyquist(self):
        """Verify that audio frequencies above 16kHz Nyquist (8kHz) are attenuated, not aliased."""
        stream = AudioCaptureStream(target_sample_rate=16000, chunk_ms=250)

        # Generate a 10kHz sine wave at 48kHz (Nyquist at 16kHz is 8kHz, so 10kHz must be filtered)
        orig_sr = 48000
        duration = 0.25
        t = np.arange(int(orig_sr * duration)) / float(orig_sr)
        freq = 10000.0
        input_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        resampled = stream._resample_to_16k(input_audio, orig_sr)
        expected_len = int(len(input_audio) * (16000 / orig_sr))
        self.assertEqual(len(resampled), expected_len)

        # In a proper anti-aliasing filter, frequencies > 8kHz should be heavily attenuated
        # With naive linear interpolation (np.interp), 10kHz aliases to 6kHz with RMS ~ 0.5-0.7.
        # With polyphase Kaiser-windowed sinc filter, RMS should be < 0.05.
        rms = float(np.sqrt(np.mean(resampled ** 2)))
        self.assertLess(
            rms,
            0.08,
            f"Expected heavy attenuation above Nyquist (RMS < 0.08), but got RMS={rms:.4f}. Aliasing detected!",
        )

    def test_resampling_preserves_passband_signal(self):
        """Verify that passband audio (e.g. 1kHz) is accurately preserved across 48k->16k and 44.1k->16k."""
        stream = AudioCaptureStream(target_sample_rate=16000, chunk_ms=250)

        for orig_sr in (48000, 44100):
            t = np.arange(int(orig_sr * 0.25)) / float(orig_sr)
            input_audio = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

            resampled = stream._resample_to_16k(input_audio, orig_sr)
            expected_len = int(len(input_audio) * (16000 / orig_sr))

            # Length should match expected target sample count within rounding
            self.assertAlmostEqual(len(resampled), expected_len, delta=2)

            # RMS of pure 1kHz sine wave should remain close to 1 / sqrt(2) ≈ 0.707
            input_rms = float(np.sqrt(np.mean(input_audio ** 2)))
            resampled_rms = float(np.sqrt(np.mean(resampled ** 2)))
            self.assertAlmostEqual(resampled_rms, input_rms, delta=0.08)

    def test_stereo_to_mono_downmixing(self):
        """Verify multi-channel stereo downmix to mono."""
        stream = AudioCaptureStream(target_sample_rate=16000, chunk_ms=250)
        orig_sr = 48000
        n_samples = int(orig_sr * 0.1)

        # Stereo: Left channel 0.4, Right channel 0.6 -> Mean mono 0.5
        stereo = np.zeros((n_samples, 2), dtype=np.float32)
        stereo[:, 0] = 0.4
        stereo[:, 1] = 0.6

        resampled = stream._resample_to_16k(stereo, orig_sr)
        self.assertEqual(resampled.ndim, 1)
        # Exclude FIR transient boundary effects (first/last few samples)
        np.testing.assert_allclose(resampled[20:-20], 0.5, atol=1e-3)


class TestPipelineAGCCoherence(unittest.TestCase):
    """Test that pipeline AGC is not cascaded twice."""

    def test_pipeline_worker_avoids_double_agc(self):
        """Verify that speech segment is not passed through a second AGC stage in pipeline worker."""
        import inspect
        from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline

        source = inspect.getsource(DiarizeFlowPipeline._pipeline_worker)
        # Verify apply_speech_agc is not called inside _pipeline_worker
        self.assertNotIn(
            "apply_speech_agc",
            source,
            "Redundant double AGC call (apply_speech_agc) detected in _pipeline_worker!",
        )


if __name__ == "__main__":
    unittest.main()
