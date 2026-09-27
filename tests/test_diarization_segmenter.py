"""Unit tests for End-to-End Streaming Diarization-Driven ASR Segmentation (Issue #13)."""

from collections import deque
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from diarizeflow.app.config import AppConfig, DiarizationConfig
from diarizeflow.app.audio.segmenter import StreamingDiarizationSegmenter, SpeakerChannelBuffer
from diarizeflow.app.backend.diarizer import NemotronDiarizer


class TestDiarizationDrivenSegmenter(unittest.TestCase):
    """Test suite verifying neural SAD multi-track segmentation and per-speaker decoupling."""

    def setUp(self):
        self.sample_rate = 16000
        self.cfg = AppConfig()

    def test_segmenter_initialization(self):
        """Verify default parameters, per-channel buffer pool, and threshold settings."""
        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.45,
            silence_timeout_ms=350,
            min_speech_ms=200,
            max_speech_s=6.0,
            pre_pad_ms=150,
            post_pad_ms=150,
        )
        self.assertEqual(segmenter.sample_rate, 16000)
        self.assertEqual(segmenter.sad_threshold, 0.45)
        self.assertEqual(segmenter.silence_timeout_ms, 350)
        self.assertEqual(len(segmenter.channel_buffers), 8)

    def test_single_speaker_emission_with_silence(self):
        """Verify a single speaker utterance is emitted when silence hangover is reached."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()
        mock_diarizer.identify_speaker.return_value = ("講者 1", 0.95, [0.95] + [0.0] * 7)

        # Mock forward_streaming_step to indicate Channel 0 is active for 4 chunks, then inactive
        active_probs = np.zeros((6, 8), dtype=np.float32)
        active_probs[:, 0] = 0.90  # Channel 0 active

        inactive_probs = np.zeros((6, 8), dtype=np.float32)
        inactive_probs[:, 0] = 0.05  # Silence

        mock_diarizer.forward_streaming_step.side_effect = [
            active_probs,
            active_probs,
            active_probs,
            active_probs,
            inactive_probs,
            inactive_probs,
        ]

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            min_speech_ms=200,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk_len = 4000  # 250ms at 16kHz
        chunk = np.ones(chunk_len, dtype=np.float32) * 0.1

        # Feed 4 active chunks (1.0s speech)
        for _ in range(4):
            segmenter.process_chunk(chunk, rms=0.1)

        # Not emitted yet because speech is still active
        self.assertEqual(len(emitted), 0)

        # Feed 2 inactive chunks (500ms silence > 300ms timeout)
        silence_chunk = np.zeros(chunk_len, dtype=np.float32)
        segmenter.process_chunk(silence_chunk, rms=0.0)
        segmenter.process_chunk(silence_chunk, rms=0.0)

        # Should now be emitted
        self.assertEqual(len(emitted), 1)
        audio_seg, spk_label, conf, dur = emitted[0]
        self.assertEqual(spk_label, "講者 1")
        self.assertGreaterEqual(dur, 1.0)
        self.assertGreater(len(audio_seg), 4 * chunk_len)  # includes padding

    def test_multi_speaker_turn_taking_no_leakage(self):
        """Verify Speaker 1 and Speaker 2 speaking sequentially are segregated into distinct buffers."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()

        def mock_identify(audio, sr=16000):
            # If audio has value 0.1 -> Speaker 1; if 0.2 -> Speaker 2
            mean_val = np.mean(np.abs(audio[audio != 0])) if np.any(audio != 0) else 0.1
            if mean_val < 0.15:
                return ("講者 1", 0.92, [0.92] + [0.0] * 7)
            else:
                return ("講者 2", 0.89, [0.0, 0.89] + [0.0] * 6)

        mock_diarizer.identify_speaker.side_effect = mock_identify

        ch0_probs = np.zeros((6, 8), dtype=np.float32)
        ch0_probs[:, 0] = 0.90  # Channel 0 active

        ch1_probs = np.zeros((6, 8), dtype=np.float32)
        ch1_probs[:, 1] = 0.90  # Channel 1 active

        silence_probs = np.zeros((6, 8), dtype=np.float32)

        # Speaker 1 speaks (3 chunks), then silence (2 chunks), then Speaker 2 speaks (2 chunks), then silence (2 chunks)
        mock_diarizer.forward_streaming_step.side_effect = [
            ch0_probs, ch0_probs, ch0_probs,  # Spk 1 (750ms)
            silence_probs, silence_probs,     # Spk 1 finishes
            ch1_probs, ch1_probs,             # Spk 2 (500ms)
            silence_probs, silence_probs,     # Spk 2 finishes
        ]

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            min_speech_ms=200,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk_len = 4000  # 250ms
        spk1_chunk = np.ones(chunk_len, dtype=np.float32) * 0.10
        spk2_chunk = np.ones(chunk_len, dtype=np.float32) * 0.20
        silence_chunk = np.zeros(chunk_len, dtype=np.float32)

        # 1. Spk 1 speaks
        for _ in range(3):
            segmenter.process_chunk(spk1_chunk, rms=0.10)

        # 2. Spk 1 silence hangover -> triggers Spk 1 emission
        segmenter.process_chunk(silence_chunk, rms=0.0)
        segmenter.process_chunk(silence_chunk, rms=0.0)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1], "講者 1")

        # 3. Spk 2 speaks immediately
        for _ in range(2):
            segmenter.process_chunk(spk2_chunk, rms=0.20)

        # 4. Spk 2 silence hangover -> triggers Spk 2 emission
        segmenter.process_chunk(silence_chunk, rms=0.0)
        segmenter.process_chunk(silence_chunk, rms=0.0)

        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[1][1], "講者 2")

        # Crucial check: Speaker 1's segment should not contain Speaker 2's 0.20 audio!
        self.assertTrue(np.all(emitted[0][0] <= 0.15), "Speaker 1's audio was contaminated with Speaker 2's voice!")

    def test_overlap_speech_decoupling(self):
        """Verify that when two speakers talk simultaneously, both buffers capture the overlap audio."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()

        def mock_identify(audio, sr=16000):
            # Identify based on channel buffer caller
            return ("講者", 0.90, [0.90] + [0.0] * 7)

        mock_diarizer.identify_speaker.side_effect = mock_identify

        # Both Ch0 and Ch1 active (overlap)
        overlap_probs = np.zeros((6, 8), dtype=np.float32)
        overlap_probs[:, 0] = 0.85
        overlap_probs[:, 1] = 0.85

        silence_probs = np.zeros((6, 8), dtype=np.float32)

        mock_diarizer.forward_streaming_step.side_effect = [
            overlap_probs, overlap_probs,  # 500ms overlap speech
            silence_probs, silence_probs,  # silence timeout
        ]

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            min_speech_ms=200,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk = np.ones(4000, dtype=np.float32) * 0.15
        silence = np.zeros(4000, dtype=np.float32)

        segmenter.process_chunk(chunk, rms=0.15)
        segmenter.process_chunk(chunk, rms=0.15)
        segmenter.process_chunk(silence, rms=0.0)
        segmenter.process_chunk(silence, rms=0.0)

        # Both speakers emitted
        self.assertEqual(len(emitted), 2, "Both overlapping speakers should have their utterances emitted independently!")

    def test_context_padding_preserves_phonemes(self):
        """Verify context padding is prepended and appended around speech bounds."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()
        mock_diarizer.identify_speaker.return_value = ("講者 1", 1.0, [1.0] + [0.0] * 7)

        silence_probs = np.zeros((6, 8), dtype=np.float32)
        active_probs = np.zeros((6, 8), dtype=np.float32)
        active_probs[:, 0] = 0.95

        mock_diarizer.forward_streaming_step.side_effect = [
            silence_probs,  # pre-speech silence (stored in pre-buffer)
            active_probs,   # speech chunk 1
            active_probs,   # speech chunk 2
            silence_probs,  # post-speech silence chunk 1
            silence_probs,  # post-speech silence chunk 2 -> triggers timeout
        ]

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            pre_pad_ms=150,
            post_pad_ms=150,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        noise_pre = np.ones(4000, dtype=np.float32) * 0.01
        speech = np.ones(4000, dtype=np.float32) * 0.20
        silence = np.zeros(4000, dtype=np.float32)

        segmenter.process_chunk(noise_pre, rms=0.01)
        segmenter.process_chunk(speech, rms=0.20)
        segmenter.process_chunk(speech, rms=0.20)
        segmenter.process_chunk(silence, rms=0.0)
        segmenter.process_chunk(silence, rms=0.0)

        self.assertEqual(len(emitted), 1)
        audio_seg = emitted[0][0]
        # Must be longer than 2 chunks of pure speech (8000 samples) due to pre/post context padding
        self.assertGreater(len(audio_seg), 8000)

    def test_acoustic_fallback_when_onnx_unavailable(self):
        """Verify acoustic voiceprint fallback functions properly when ONNX session is None."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = None  # No ONNX model
        mock_diarizer.identify_speaker.return_value = ("講者 1", 0.85, [0.85] + [0.0] * 7)

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            min_speech_ms=200,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk_speech = np.ones(4000, dtype=np.float32) * 0.15
        chunk_silence = np.zeros(4000, dtype=np.float32)

        # 3 speech chunks (750ms)
        for _ in range(3):
            segmenter.process_chunk(chunk_speech, rms=0.15)

        # 2 silence chunks (500ms > 300ms)
        segmenter.process_chunk(chunk_silence, rms=0.0)
        segmenter.process_chunk(chunk_silence, rms=0.0)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1], "講者 1")

    def test_continuous_speech_cut_by_max_speech_s(self):
        """Verify continuous speech without pauses is forcefully cut and emitted at max_speech_s."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()
        mock_diarizer.identify_speaker.return_value = ("講者 1", 0.95, [0.95] + [0.0] * 7)

        # Continually active Channel 0 (no silence at all!)
        active_probs = np.zeros((6, 8), dtype=np.float32)
        active_probs[:, 0] = 0.90
        mock_diarizer.forward_streaming_step.return_value = active_probs

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.50,
            silence_timeout_ms=300,
            min_speech_ms=200,
            max_speech_s=1.0,  # 1.0 second max speech duration
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk = np.ones(4000, dtype=np.float32) * 0.15  # 250ms chunks

        # Feed 10 consecutive chunks of speech (2.5s total with NO silence)
        for _ in range(10):
            segmenter.process_chunk(chunk, rms=0.15)

        # Must have been cut into multiple segments at 1.0s boundaries!
        self.assertGreaterEqual(len(emitted), 2, "Continuous speech should have been cut by max_speech_s!")
        for _, _, _, dur in emitted:
            self.assertLessEqual(dur, 1.5, "No emitted utterance should exceed max_speech_s + padding!")

    def test_ambient_noise_rejection(self):
        """Verify quiet ambient noise below acoustic energy floor does not trigger speech onset."""
        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration):
            emitted.append((audio_seg, spk_label, conf, duration))

        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()

        # Neural network has occasional false spike of 0.55 on ambient noise
        spike_probs = np.zeros((6, 8), dtype=np.float32)
        spike_probs[0, 0] = 0.55

        mock_diarizer.forward_streaming_step.return_value = spike_probs

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.50,
            silence_timeout_ms=300,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        # Feed 4 chunks of near-silent background noise (RMS 0.001)
        quiet_chunk = np.ones(4000, dtype=np.float32) * 0.001
        for _ in range(4):
            segmenter.process_chunk(quiet_chunk, rms=0.001)

        # Nothing should be emitted or triggered
        self.assertEqual(len(emitted), 0)
        self.assertFalse(segmenter.channel_buffers[0].in_speech)


if __name__ == "__main__":
    unittest.main()
