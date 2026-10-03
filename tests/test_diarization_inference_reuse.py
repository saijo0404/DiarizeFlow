"""Unit and integration tests for Sortformer single-pass SAD track reuse and embedding caching (Issue #31).

Verifies:
1. Pre-encode embedding extraction and caching in NemotronDiarizer.forward_streaming_step.
2. Fast profile matching via identify_speaker_from_embedding without ONNX forward passes.
3. SpeakerChannelBuffer accumulation of step embeddings into normalized utterance embeddings.
4. StreamingDiarizationSegmenter fast identification using cached pre-encode embeddings.
5. DiarizeFlowPipeline._pipeline_worker bypassing diarize_and_split and identify_speaker for SAD tracks.
6. Fallback energy VAD and direct audio inputs appropriately invoking diarize_and_split.
7. diarize_and_split reusing streaming speaker vectors for single-speaker segments without duplicate calls.
"""

import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from diarizeflow.app.audio.segmenter import SpeakerChannelBuffer, StreamingDiarizationSegmenter
from diarizeflow.app.backend.diarizer import NemotronDiarizer
from diarizeflow.app.backend.pipeline import DiarizeFlowPipeline, SubtitleEvent
from diarizeflow.app.backend.voiceprint import SpeakerProfile
from diarizeflow.app.config import AppConfig, DiarizationConfig


class TestNemotronEmbeddingCachingAndFastIdentification(unittest.TestCase):
    """Test suite verifying NemotronDiarizer pre-encode embedding caching and fast identification."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.profiles_path = Path(self.tmp_dir.name) / "profiles.json"
        self.config = DiarizationConfig(profiles_path=str(self.profiles_path))
        self.diarizer = NemotronDiarizer(self.config)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_forward_streaming_step_caches_and_returns_embeddings(self):
        """Verify forward_streaming_step normalizes pre-encode embeddings and caches them."""
        # Mock session to return Sortformer outputs
        mock_session = MagicMock()
        # outputs[0]: (1, 528, 8) frame probs
        raw_preds = np.zeros((1, 528, 8), dtype=np.float32)
        raw_preds[0, :24, 0] = 5.0  # sigmoid(5.0) ~ 0.99 for channel 0
        # outputs[1]: (1, 264, 512) pre-encode embeddings
        pre_embs = np.ones((1, 264, 512), dtype=np.float32) * 0.5
        # outputs[2]: (1,) valid length
        valid_len = np.array([12], dtype=np.int64)

        mock_session.run.return_value = [raw_preds, pre_embs, valid_len]
        self.diarizer.session = mock_session

        audio = np.ones(3200, dtype=np.float32) * 0.1

        # 1. Standard call returns probs and updates last_pre_encode_embs
        probs = self.diarizer.forward_streaming_step(audio, sample_rate=16000)
        self.assertIsNotNone(probs)
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape[1], 8)

        cached_emb = self.diarizer.get_last_pre_encode_embs()
        self.assertIsNotNone(cached_emb)
        self.assertEqual(cached_emb.shape, (512,))
        # Assert L2 normalized
        self.assertAlmostEqual(float(np.linalg.norm(cached_emb)), 1.0, places=4)

        # 2. Call with return_embeddings=True returns tuple (probs, step_emb)
        probs_tuple, step_emb = self.diarizer.forward_streaming_step(
            audio, sample_rate=16000, return_embeddings=True
        )
        self.assertIsNotNone(probs_tuple)
        self.assertIsNotNone(step_emb)
        self.assertEqual(step_emb.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(step_emb)), 1.0, places=4)

        # 3. Reset clears cached pre-encode embedding
        self.diarizer.reset()
        self.assertIsNone(self.diarizer.get_last_pre_encode_embs())

    def test_identify_speaker_from_embedding_matches_pinned_profile(self):
        """Verify identify_speaker_from_embedding identifies pinned profile with zero ONNX forward."""
        anchor_vec = np.zeros(512, dtype=np.float32)
        anchor_vec[0] = 1.0  # Unit vector along axis 0

        # Pin "Alice" to voiceprint database
        alice_profile = SpeakerProfile(
            id="spk_alice",
            name="Alice",
            is_pinned=True,
            anchor_emb=anchor_vec.copy(),
            sample_count=10,
        )
        self.diarizer.voiceprint_db.add_or_update(alice_profile)
        self.diarizer._sync_known_speakers()

        # Query with embedding close to Alice
        query_vec = anchor_vec.copy()
        query_vec += np.random.normal(0, 0.01, size=512).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        # Assert no session or ONNX forward run is needed
        self.diarizer.session = None

        spk, conf = self.diarizer.identify_speaker_from_embedding(query_vec, duration_s=2.5)
        self.assertEqual(spk, "Alice")
        self.assertGreater(conf, 0.95)

    def test_identify_speaker_from_embedding_registers_temporary_speaker(self):
        """Verify identify_speaker_from_embedding registers new temporary speaker when unpinned."""
        emb = np.zeros(512, dtype=np.float32)
        emb[42] = 1.0

        self.diarizer.session = None
        spk, conf = self.diarizer.identify_speaker_from_embedding(emb, duration_s=1.0)
        self.assertEqual(spk, "講者 1")
        self.assertEqual(conf, 1.0)
        self.assertEqual(len(self.diarizer.temporary_speakers), 1)
        self.assertEqual(self.diarizer.temporary_speakers[0].name, "講者 1")


class TestSpeakerChannelBufferEmbeddingAccumulation(unittest.TestCase):
    """Test suite verifying SpeakerChannelBuffer accumulates step embeddings."""

    def test_buffer_embedding_accumulation_and_normalization(self):
        """Verify buffer computes normalized mean embedding across speech steps."""
        buf = SpeakerChannelBuffer(channel_id=0)

        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0

        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0

        chunk = np.ones(1600, dtype=np.float32) * 0.1

        buf.start_speech([], initial_chunk=chunk, initial_embedding=emb1)
        self.assertEqual(len(buf.embeddings), 1)

        buf.add_speech_chunk(chunk, embedding=emb2)
        self.assertEqual(len(buf.embeddings), 2)

        mean_emb = buf.get_utterance_embedding()
        self.assertIsNotNone(mean_emb)
        self.assertEqual(mean_emb.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(mean_emb)), 1.0, places=4)
        # Components along axis 0 and 1 should be equal (1 / sqrt(2) ~ 0.7071)
        self.assertAlmostEqual(mean_emb[0], float(1.0 / np.sqrt(2)), places=3)
        self.assertAlmostEqual(mean_emb[1], float(1.0 / np.sqrt(2)), places=3)

        buf.reset()
        self.assertEqual(len(buf.embeddings), 0)
        self.assertIsNone(buf.get_utterance_embedding())


class TestSegmenterFastFeatureReuse(unittest.TestCase):
    """Test suite verifying StreamingDiarizationSegmenter uses cached embeddings for instant identification."""

    def test_segmenter_identifies_speaker_via_embedding_reuse(self):
        """Verify segmenter calls identify_speaker_from_embedding instead of identify_speaker."""
        mock_diarizer = MagicMock()
        mock_diarizer.session = MagicMock()

        # Step embedding
        test_step_emb = np.zeros(512, dtype=np.float32)
        test_step_emb[5] = 1.0

        mock_diarizer.last_pre_encode_embs = test_step_emb
        mock_diarizer.identify_speaker_from_embedding.return_value = ("Bob", 0.94)

        active_probs = np.zeros((6, 8), dtype=np.float32)
        active_probs[:, 2] = 0.90  # Channel 2 active

        inactive_probs = np.zeros((6, 8), dtype=np.float32)

        mock_diarizer.forward_streaming_step.side_effect = [
            active_probs,
            active_probs,
            inactive_probs,
            inactive_probs,
        ]

        emitted = []

        def on_utterance(audio_seg, spk_label, conf, duration, is_neural_sad=True):
            emitted.append((spk_label, conf, is_neural_sad))

        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.40,
            silence_timeout_ms=300,
            min_speech_ms=200,
            diarizer=mock_diarizer,
            on_utterance=on_utterance,
        )

        chunk_len = 4000
        chunk = np.ones(chunk_len, dtype=np.float32) * 0.1

        # 2 active speech chunks
        segmenter.process_chunk(chunk, rms=0.1)
        segmenter.process_chunk(chunk, rms=0.1)

        # 2 silence chunks triggering hangover timeout
        silence_chunk = np.zeros(chunk_len, dtype=np.float32)
        segmenter.process_chunk(silence_chunk, rms=0.0)
        segmenter.process_chunk(silence_chunk, rms=0.0)

        self.assertEqual(len(emitted), 1)
        spk_label, conf, is_neural = emitted[0]
        self.assertEqual(spk_label, "Bob")
        self.assertAlmostEqual(conf, 0.94, places=2)
        self.assertTrue(is_neural)

        # Assert identify_speaker_from_embedding was called
        mock_diarizer.identify_speaker_from_embedding.assert_called_once()
        # Assert identify_speaker (heavy ONNX pass) was NOT called
        mock_diarizer.identify_speaker.assert_not_called()


class TestPipelineSinglePassDiarizationWorker(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying DiarizeFlowPipeline worker bypasses duplicate inference on SAD tracks."""

    async def test_neural_sad_utterance_bypasses_diarize_and_split(self):
        """Verify pipeline worker trusts SAD speaker label and bypasses diarize_and_split."""
        cfg = AppConfig()
        cfg.llm.provider = "bypass"

        broadcasted = []

        def on_broadcast(evt: SubtitleEvent):
            broadcasted.append(evt)

        pipeline = DiarizeFlowPipeline(cfg, on_subtitle_broadcast=on_broadcast)
        pipeline.diarizer = MagicMock()
        pipeline.diarizer.diarize_and_split = MagicMock()
        pipeline.diarizer.identify_speaker = MagicMock()
        pipeline.translator.translate = AsyncMock(side_effect=lambda text, **kwargs: f"譯: {text}")

        # Mock ASR
        pipeline.asr.transcribe = MagicMock(return_value=("這是串流 SAD 語音", "zh"))

        pipeline.start(asyncio.get_running_loop())

        # Feed 3.0s audio segment from StreamingDiarizationSegmenter with is_neural_sad=True
        audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        pipeline._on_diarized_utterance(
            audio_segment=audio,
            speaker_label="講者 2",
            confidence=0.96,
            duration=3.0,
            is_neural_sad=True,
        )

        # Wait for worker to process
        for _ in range(30):
            if len(broadcasted) >= 1:
                break
            await asyncio.sleep(0.05)

        pipeline.stop()

        self.assertEqual(len(broadcasted), 1)
        self.assertEqual(broadcasted[0].speaker, "講者 2")
        self.assertEqual(broadcasted[0].original_text, "這是串流 SAD 語音")

        # CRITICAL ASSERTION: diarize_and_split and identify_speaker MUST NOT be called!
        pipeline.diarizer.diarize_and_split.assert_not_called()
        pipeline.diarizer.identify_speaker.assert_not_called()

    async def test_fallback_energy_vad_triggers_diarize_and_split(self):
        """Verify fallback energy VAD (or direct audio feed) still executes diarize_and_split."""
        cfg = AppConfig()
        cfg.llm.provider = "bypass"

        broadcasted = []

        def on_broadcast(evt: SubtitleEvent):
            broadcasted.append(evt)

        pipeline = DiarizeFlowPipeline(cfg, on_subtitle_broadcast=on_broadcast)
        pipeline.diarizer = MagicMock()
        sub_audio1 = np.ones(16000 * 1, dtype=np.float32) * 0.1
        sub_audio2 = np.ones(16000 * 1, dtype=np.float32) * 0.2
        pipeline.diarizer.diarize_and_split = MagicMock(
            return_value=[(sub_audio1, "講者 1", 0.9), (sub_audio2, "講者 2", 0.9)]
        )
        pipeline.translator.translate = AsyncMock(side_effect=lambda text, **kwargs: f"譯: {text}")

        pipeline.asr.transcribe = MagicMock(return_value=("測試發話", "zh"))

        pipeline.start(asyncio.get_running_loop())

        # Feed 3.0s direct audio segment without SAD track information
        audio = np.ones(16000 * 3, dtype=np.float32) * 0.1
        pipeline._on_speech_utterance(audio, duration=3.0)

        for _ in range(30):
            if len(broadcasted) >= 2:
                break
            await asyncio.sleep(0.05)

        pipeline.stop()

        self.assertEqual(len(broadcasted), 2)
        # diarize_and_split MUST be called for fallback / direct audio
        pipeline.diarizer.diarize_and_split.assert_called_once()


class TestDiarizeAndSplitVectorReuse(unittest.TestCase):
    """Test suite verifying diarize_and_split reuses spk_vec for single speaker paths."""

    def test_single_speaker_path_reuses_spk_vec_without_second_call(self):
        """Verify diarize_and_split does not invoke identify_speaker when spk_vec is available."""
        tmp_dir = tempfile.TemporaryDirectory()
        cfg = DiarizationConfig(profiles_path=str(Path(tmp_dir.name) / "profiles.json"))
        diarizer = NemotronDiarizer(cfg)

        diarizer.session = MagicMock()

        # Mock _stream_process_audio returning single active channel probs and spk_vec
        single_spk_probs = np.zeros((20, 8), dtype=np.float32)
        single_spk_probs[:, 0] = 0.85  # Only Channel 0 active

        mock_spk_vec = np.zeros(512, dtype=np.float32)
        mock_spk_vec[0] = 1.0

        diarizer._stream_process_audio = MagicMock(return_value=(single_spk_probs, mock_spk_vec))
        diarizer.identify_speaker = MagicMock()

        audio = np.ones(16000 * 2, dtype=np.float32) * 0.1
        result = diarizer.diarize_and_split(audio, sample_rate=16000)

        self.assertEqual(len(result), 1)
        sub_audio, spk, conf = result[0]
        self.assertEqual(spk, "講者 1")

        # identify_speaker must NOT be called because spk_vec was reused directly
        diarizer.identify_speaker.assert_not_called()

        tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
