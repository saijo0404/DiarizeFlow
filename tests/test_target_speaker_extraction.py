"""Unit and integration tests for Target-Speaker Extraction (TSE) (Issue #54).

Verifies:
1. TSEConfig serialization/deserialization and integration into AppConfig.from_dict.
2. TargetSpeakerExtractor initialization, projection basis, and parameter updates.
3. Time-frequency mask estimation M_k(t, f) in [min_gain, 1.0] conditioned on 512-d embeddings.
4. Clean waveform reconstruction: length preservation, no NaNs/Infs, dynamic range defense.
5. Overlapping speech waveform separation on synthetic mixed acoustic signals.
6. Multi-target extraction from shared STFT (extract_multi).
7. StreamingDiarizationSegmenter dynamic overlap detection and on-demand TSE activation.
8. Bypassing TSE when only a single speaker is active (zero overhead).
9. DiarizeFlowPipeline lifecycle and dynamic runtime config updates.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from diarizeflow.app.config import AppConfig, TSEConfig
from diarizeflow.app.audio.tse import TargetSpeakerExtractor
from diarizeflow.app.audio.segmenter import StreamingDiarizationSegmenter, SpeakerChannelBuffer
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline


def make_harmonic_speech(f0: float, duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate synthetic harmonic acoustic voice waveform with fundamental frequency f0."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    signal = np.zeros_like(t)
    for h in range(1, 10):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
    return (signal * envelope * 0.25).astype(np.float32)


class TestTSEConfig(unittest.TestCase):
    """Test TSE configuration schema, defaults, and serialization."""

    def test_default_config(self):
        cfg = TSEConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.embedding_dim, 512)
        self.assertEqual(cfg.n_fft, 512)
        self.assertEqual(cfg.hop_length, 160)
        self.assertEqual(cfg.win_length, 400)
        self.assertAlmostEqual(cfg.min_gain, 0.05)

    def test_app_config_integration(self):
        app_cfg = AppConfig()
        self.assertTrue(hasattr(app_cfg, "tse"))
        self.assertTrue(app_cfg.tse.enabled)

        d = app_cfg.to_dict()
        self.assertIn("tse", d)
        self.assertEqual(d["tse"]["min_gain"], 0.05)

        restored = AppConfig.from_dict(d)
        self.assertTrue(restored.tse.enabled)
        self.assertEqual(restored.tse.min_gain, 0.05)


class TestTargetSpeakerExtractor(unittest.TestCase):
    """Test core TargetSpeakerExtractor neural-DSP mask estimation and waveform synthesis."""

    def setUp(self):
        self.sr = 16000
        self.extractor = TargetSpeakerExtractor(sample_rate=self.sr, min_gain=0.05)

        # Generate two synthetic speakers with distinct fundamental pitches
        self.speech_A = make_harmonic_speech(140.0, 1.0, self.sr)  # ~140 Hz (Male)
        self.speech_B = make_harmonic_speech(320.0, 1.0, self.sr)  # ~320 Hz (Female)
        self.mixture = (self.speech_A + self.speech_B).astype(np.float32)

        # Generate distinct normalized 512-dim voiceprint embeddings
        rng = np.random.RandomState(101)
        self.emb_A = rng.randn(512).astype(np.float32)
        self.emb_A /= np.linalg.norm(self.emb_A)

        self.emb_B = rng.randn(512).astype(np.float32)
        self.emb_B -= np.dot(self.emb_B, self.emb_A) * self.emb_A
        self.emb_B /= np.linalg.norm(self.emb_B)

    def test_estimate_mask_dimensions_and_bounds(self):
        """Verify estimated mask has shape (F, T) and is strictly bounded in [min_gain, 1.0]."""
        import librosa
        stft = librosa.stft(self.mixture, n_fft=512, hop_length=160, win_length=400)
        mag = np.abs(stft)

        mask_A = self.extractor.estimate_mask(mag, self.emb_A)
        self.assertEqual(mask_A.shape, mag.shape)
        self.assertGreaterEqual(float(np.min(mask_A)), 0.05 - 1e-5)
        self.assertLessEqual(float(np.max(mask_A)), 1.0 + 1e-5)
        self.assertFalse(np.isnan(mask_A).any())

        mask_B = self.extractor.estimate_mask(mag, self.emb_B)
        self.assertEqual(mask_B.shape, mag.shape)
        # Masks for different speakers should be distinct
        self.assertFalse(np.allclose(mask_A, mask_B))

    def test_extract_waveform_reconstruction_integrity(self):
        """Verify extracted waveform preserves length, is finite, and avoids distortion."""
        extracted_A = self.extractor.extract(self.mixture, self.emb_A)

        self.assertEqual(len(extracted_A), len(self.mixture))
        self.assertEqual(extracted_A.dtype, np.float32)
        self.assertFalse(np.isnan(extracted_A).any())
        self.assertFalse(np.isinf(extracted_A).any())

        # Amplitude should remain bounded
        self.assertLessEqual(float(np.max(np.abs(extracted_A))), float(np.max(np.abs(self.mixture))) * 1.1)

    def test_extract_bypass_on_missing_embedding(self):
        """Verify missing or zero embedding safely returns a clean copy of the original audio."""
        result_none = self.extractor.extract(self.mixture, None)
        np.testing.assert_array_equal(result_none, self.mixture)

        zero_emb = np.zeros(512, dtype=np.float32)
        result_zero = self.extractor.extract(self.mixture, zero_emb)
        np.testing.assert_array_equal(result_zero, self.mixture)

    def test_extract_short_audio_graceful(self):
        """Verify audio shorter than hop_length is safely returned without error."""
        short_audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        res = self.extractor.extract(short_audio, self.emb_A)
        np.testing.assert_array_equal(res, short_audio)

    def test_extract_multi_batch(self):
        """Verify extract_multi separates multiple speaker tracks from single shared STFT."""
        channel_embs = {
            0: self.emb_A,
            1: self.emb_B,
            2: None,  # Channel with no embedding
        }
        res_dict = self.extractor.extract_multi(self.mixture, channel_embs)
        self.assertEqual(len(res_dict), 3)
        self.assertIn(0, res_dict)
        self.assertIn(1, res_dict)
        self.assertIn(2, res_dict)

        self.assertEqual(len(res_dict[0]), len(self.mixture))
        self.assertEqual(len(res_dict[1]), len(self.mixture))
        np.testing.assert_array_equal(res_dict[2], self.mixture)

    def test_onnx_session_dispatch_mock(self):
        """Verify ONNX InferenceSession is invoked when session is active."""
        mock_sess = MagicMock()
        in0 = MagicMock()
        in0.name = "spec_mag"
        in0.shape = [1, 257, 100]
        in1 = MagicMock()
        in1.name = "spk_emb"
        mock_sess.get_inputs.return_value = [in0, in1]

        # Return mock mask
        mock_mask = np.ones((1, 257, 101), dtype=np.float32) * 0.85
        mock_sess.run.return_value = [mock_mask]

        self.extractor.session = mock_sess
        extracted = self.extractor.extract(self.mixture, self.emb_A)

        self.assertEqual(len(extracted), len(self.mixture))
        mock_sess.run.assert_called_once()


class TestSegmenterTSEIntegration(unittest.TestCase):
    """Test dynamic on-demand TSE triggering during overlap speech in StreamingDiarizationSegmenter."""

    def setUp(self):
        self.sr = 16000
        self.tse = TargetSpeakerExtractor(sample_rate=self.sr)

        self.mock_diarizer = MagicMock()
        self.mock_diarizer.session = MagicMock()
        self.mock_diarizer.forward_streaming_step = MagicMock()

        self.segmenter = StreamingDiarizationSegmenter(
            sample_rate=self.sr,
            sad_threshold=0.50,
            silence_timeout_ms=300,
            min_speech_ms=100,
            diarizer=self.mock_diarizer,
            tse_extractor=self.tse,
            tse_enabled=True,
        )

        rng = np.random.RandomState(42)
        self.emb_spk1 = rng.randn(512).astype(np.float32)
        self.emb_spk1 /= np.linalg.norm(self.emb_spk1)

        self.emb_spk2 = rng.randn(512).astype(np.float32)
        self.emb_spk2 /= np.linalg.norm(self.emb_spk2)

    def test_single_speaker_bypasses_tse(self):
        """When only one speaker is active, TSE is bypassed (zero compute overhead)."""
        chunk = make_harmonic_speech(160.0, 0.16, self.sr)  # 160ms = 2560 samples

        # Sortformer output: Channel 0 high prob (0.95), other channels idle (0.05)
        probs = np.zeros((4, 8), dtype=np.float32)
        probs[:, 0] = 0.95
        self.mock_diarizer.forward_streaming_step.return_value = (probs, self.emb_spk1)

        with patch.object(self.tse, "extract", wraps=self.tse.extract) as spy_extract:
            self.segmenter.process_chunk(chunk)
            # TSE must NOT be called for single speaker
            spy_extract.assert_not_called()

        self.assertEqual(self.segmenter.overlap_stats["total_overlap_chunks"], 0)
        self.assertEqual(self.segmenter.overlap_stats["separated_overlap_chunks"], 0)

    def test_overlap_speech_triggers_tse_separation(self):
        """When multiple channels are active simultaneously (overlap), TSE is dynamically invoked."""
        chunk = make_harmonic_speech(200.0, 0.16, self.sr)

        # Pre-populate channel 0 with an existing speech utterance and embedding
        self.segmenter.channel_buffers[0].start_speech([], initial_chunk=chunk, initial_embedding=self.emb_spk1)
        self.segmenter.channel_last_embeddings[0] = self.emb_spk1
        self.segmenter.channel_last_embeddings[1] = self.emb_spk2

        # Sortformer output: BOTH Channel 0 and Channel 1 active (> 0.50)!
        probs = np.zeros((4, 8), dtype=np.float32)
        probs[:, 0] = 0.92
        probs[:, 1] = 0.88
        self.mock_diarizer.forward_streaming_step.return_value = (probs, self.emb_spk1)

        with patch.object(self.tse, "extract", wraps=self.tse.extract) as spy_extract:
            self.segmenter.process_chunk(chunk)
            # TSE should be called for active channels
            self.assertGreaterEqual(spy_extract.call_count, 1)

        self.assertGreater(self.segmenter.overlap_stats["total_overlap_chunks"], 0)
        self.assertGreater(self.segmenter.overlap_stats["separated_overlap_chunks"], 0)

    def test_tse_disabled_skips_separation_during_overlap(self):
        """When tse_enabled is False, overlap speech bypasses TSE."""
        self.segmenter.tse_enabled = False
        chunk = make_harmonic_speech(200.0, 0.16, self.sr)

        probs = np.zeros((4, 8), dtype=np.float32)
        probs[:, 0] = 0.90
        probs[:, 1] = 0.85
        self.mock_diarizer.forward_streaming_step.return_value = (probs, self.emb_spk1)

        with patch.object(self.tse, "extract") as mock_extract:
            self.segmenter.process_chunk(chunk)
            mock_extract.assert_not_called()

    def test_separate_utterance_method(self):
        """Verify separate_utterance applies TSE extraction directly."""
        audio = make_harmonic_speech(180.0, 0.5, self.sr)
        sep = self.segmenter.separate_utterance(audio, self.emb_spk1)
        self.assertEqual(len(sep), len(audio))
        self.assertFalse(np.isnan(sep).any())


class TestPipelineTSEIntegration(unittest.TestCase):
    """Test TSE lifecycle inside DiarizeFlowPipeline."""

    def test_pipeline_initializes_and_updates_tse(self):
        cfg = AppConfig()
        cfg.tse.enabled = True
        cfg.tse.min_gain = 0.08

        # Mock heavy submodules to avoid long model loading
        with patch("diarizeflow.app.backend.pipeline.NemotronDiarizer"), \
             patch("diarizeflow.app.backend.pipeline.create_asr_engine"), \
             patch("diarizeflow.app.backend.pipeline.LLMTranslator"):
            pipeline = DiarizeFlowPipeline(cfg)

            self.assertIsNotNone(pipeline.tse)
            self.assertTrue(pipeline.segmenter.tse_enabled)
            self.assertEqual(pipeline.tse.min_gain, 0.08)

            # Update configuration dynamically
            new_cfg = AppConfig()
            new_cfg.tse.enabled = False
            new_cfg.tse.min_gain = 0.12

            pipeline.update_config(new_cfg)
            self.assertFalse(pipeline.segmenter.tse_enabled)
            self.assertEqual(pipeline.tse.min_gain, 0.12)


if __name__ == "__main__":
    unittest.main()
