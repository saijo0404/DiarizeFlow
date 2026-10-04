"""Unit and integration tests for WebSocket audio streaming protocol and metadata header (Issue #52).

Verifies:
1. Binary framing packing and unpacking:
   - 8-byte framing header format (>BBBBHH) with magic byte 0xDF and version 0x01.
   - Track tag mapping (0x01: mic, 0x02: loopback, 0x03: mixed).
   - Sample rate, channels, and flags preservation.
   - Raw float32 PCM audio payload preservation.
2. Backward compatibility & fallback mechanisms:
   - Seamless fallback to raw float32 PCM when header is absent or length < 8.
   - Fallback when magic byte or version does not match.
   - Robust error handling for partial or malformed packets.
3. Backend WebSocket /ws/audio endpoint:
   - Correctly unpacks framed audio chunks and routes track source to pipeline.
   - Correctly processes raw float32 PCM chunks (fallback to "mixed").
   - Status endpoint /api/status exposes last_audio_source and audio_stats.
4. Pipeline and segmenter multi-track awareness:
   - process_audio_chunk and push_audio track sources and update counters.
   - Segmenter receives source tag during streaming processing.
5. Frontend integration:
   - AudioCaptureStream dispatches track source in callbacks.
   - TransparentSubtitleOverlay packs framed audio chunks when streaming over network.
   - run_audio_client formats framed payloads from tuples or arrays.
"""

import queue
import struct
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from fastapi.testclient import TestClient

from diarizeflow.audio.protocol import (
    AUDIO_FRAME_HEADER_MAGIC,
    AUDIO_FRAME_HEADER_SIZE,
    AUDIO_FRAME_VERSION,
    AudioTrackTag,
    is_framed_audio_packet,
    pack_audio_frame,
    unpack_audio_frame,
)
from diarizeflow.audio.capture import AudioCaptureStream
from diarizeflow.audio.segmenter import StreamingDiarizationSegmenter
from diarizeflow.engine.pipeline import DiarizeFlowPipeline
from diarizeflow.engine.server import create_app
from diarizeflow.config import AppConfig
from diarizeflow.ui.network import run_audio_client


class TestAudioProtocolBinaryFraming(unittest.TestCase):
    """Test suite for binary audio frame packing, unpacking, and validation."""

    def test_audio_track_tag_enum_values(self):
        """Verify AudioTrackTag enum maps to expected 1-byte integer values."""
        self.assertEqual(AudioTrackTag.MIC, 0x01)
        self.assertEqual(AudioTrackTag.LOOPBACK, 0x02)
        self.assertEqual(AudioTrackTag.MIXED, 0x03)

    def test_pack_and_unpack_audio_frame_roundtrip(self):
        """Verify packing and unpacking preserves track tag, sample rate, channels, and PCM data."""
        original_chunk = np.array([0.1, -0.2, 0.5, -0.8, 0.0], dtype=np.float32)

        for track_name, expected_tag in [
            ("mic", AudioTrackTag.MIC),
            ("loopback", AudioTrackTag.LOOPBACK),
            ("system", AudioTrackTag.LOOPBACK),
            ("mixed", AudioTrackTag.MIXED),
        ]:
            packet = pack_audio_frame(
                original_chunk,
                track=track_name,
                sample_rate=16000,
                channels=1,
            )
            # Check packet structure
            self.assertEqual(len(packet), AUDIO_FRAME_HEADER_SIZE + original_chunk.nbytes)
            self.assertTrue(is_framed_audio_packet(packet))

            # Unpack and verify fields
            chunk, track, sr, ch = unpack_audio_frame(packet)
            self.assertEqual(track, AudioTrackTag.to_string(expected_tag))
            self.assertEqual(sr, 16000)
            self.assertEqual(ch, 1)
            np.testing.assert_allclose(chunk, original_chunk, rtol=1e-5)

    def test_pack_audio_frame_header_bytes(self):
        """Verify binary layout of the 8-byte header conforms exactly to protocol specification."""
        chunk = np.array([0.0], dtype=np.float32)
        packet = pack_audio_frame(chunk, track=AudioTrackTag.LOOPBACK, sample_rate=48000, channels=2, flags=0x05)

        magic, version, tag, flags, sr, ch = struct.unpack(">BBBBHH", packet[:8])
        self.assertEqual(magic, AUDIO_FRAME_HEADER_MAGIC)
        self.assertEqual(version, AUDIO_FRAME_VERSION)
        self.assertEqual(tag, 0x02)
        self.assertEqual(flags, 0x05)
        self.assertEqual(sr, 48000)
        self.assertEqual(ch, 2)

    def test_unpack_audio_frame_fallback_for_raw_pcm(self):
        """Verify backward compatibility: raw float32 PCM without header is parsed cleanly as mixed."""
        raw_pcm = np.array([0.25, -0.5, 0.75, -0.1], dtype=np.float32).tobytes()

        self.assertFalse(is_framed_audio_packet(raw_pcm))
        chunk, track, sr, ch = unpack_audio_frame(raw_pcm)

        self.assertEqual(track, "mixed")
        self.assertEqual(sr, 16000)
        self.assertEqual(ch, 1)
        np.testing.assert_allclose(chunk, np.frombuffer(raw_pcm, dtype=np.float32))

    def test_unpack_audio_frame_invalid_magic_fallback(self):
        """Verify packets with non-matching magic byte fallback to raw PCM."""
        fake_header = struct.pack(">BBBBHH", 0xAA, 0x01, 0x01, 0x00, 16000, 1)
        audio_data = np.array([0.1, 0.2], dtype=np.float32).tobytes()
        packet = fake_header + audio_data

        self.assertFalse(is_framed_audio_packet(packet))
        chunk, track, sr, ch = unpack_audio_frame(packet)
        self.assertEqual(track, "mixed")
        self.assertEqual(sr, 16000)
        self.assertEqual(ch, 1)

    def test_unpack_audio_frame_short_packet_fallback(self):
        """Verify packets shorter than header size (< 8 bytes) are handled gracefully."""
        short_packet = b"\x00\x01\x02"
        self.assertFalse(is_framed_audio_packet(short_packet))
        chunk, track, sr, ch = unpack_audio_frame(short_packet)
        self.assertEqual(track, "mixed")
        self.assertEqual(len(chunk), 0)


class TestBackendWebSocketAudioEndpoint(unittest.TestCase):
    """Test suite for FastAPI /ws/audio streaming endpoint with framing protocol."""

    def setUp(self):
        self.config = AppConfig()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.config = self.config
        self.mock_pipeline.is_running = True
        self.mock_pipeline.last_audio_source = "mixed"
        self.mock_pipeline.audio_stats = {
            "total_chunks": 0,
            "mic_chunks": 0,
            "loopback_chunks": 0,
            "mixed_chunks": 0,
        }
        self.app = create_app(self.config, self.mock_pipeline)
        self.client = TestClient(self.app)

    def test_ws_audio_with_framed_mic_packet(self):
        """Verify /ws/audio receives framed 'mic' packet and passes track='mic' to pipeline."""
        chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        packet = pack_audio_frame(chunk, track="mic", sample_rate=16000)

        with self.client.websocket_connect("/ws/audio") as ws:
            ws.send_bytes(packet)

        self.mock_pipeline.process_audio_chunk.assert_called_once()
        called_chunk, kwargs = self.mock_pipeline.process_audio_chunk.call_args[0], self.mock_pipeline.process_audio_chunk.call_args[1]
        np.testing.assert_allclose(called_chunk[0], chunk)
        self.assertEqual(kwargs.get("source"), "mic")

    def test_ws_audio_with_framed_loopback_packet(self):
        """Verify /ws/audio receives framed 'loopback' packet and passes track='loopback' to pipeline."""
        chunk = np.array([0.5, -0.5], dtype=np.float32)
        packet = pack_audio_frame(chunk, track="loopback", sample_rate=16000)

        with self.client.websocket_connect("/ws/audio") as ws:
            ws.send_bytes(packet)

        self.mock_pipeline.process_audio_chunk.assert_called_once()
        kwargs = self.mock_pipeline.process_audio_chunk.call_args[1]
        self.assertEqual(kwargs.get("source"), "loopback")

    def test_ws_audio_with_legacy_raw_pcm(self):
        """Verify /ws/audio seamlessly handles legacy raw float32 PCM with track='mixed'."""
        chunk = np.array([0.05, -0.05, 0.12], dtype=np.float32)
        raw_bytes = chunk.tobytes()

        with self.client.websocket_connect("/ws/audio") as ws:
            ws.send_bytes(raw_bytes)

        self.mock_pipeline.process_audio_chunk.assert_called_once()
        kwargs = self.mock_pipeline.process_audio_chunk.call_args[1]
        self.assertEqual(kwargs.get("source"), "mixed")

    def test_api_status_exposes_audio_source_and_stats(self):
        """Verify /api/status endpoint exposes last_audio_source and audio_stats."""
        self.mock_pipeline.last_audio_source = "mic"
        self.mock_pipeline.audio_stats = {
            "total_chunks": 10,
            "mic_chunks": 7,
            "loopback_chunks": 3,
            "mixed_chunks": 0,
        }

        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["last_audio_source"], "mic")
        self.assertIn("audio_stats", data)
        self.assertEqual(data["audio_stats"]["mic_chunks"], 7)


class TestPipelineAndSegmenterTrackAwareness(unittest.TestCase):
    """Test suite for pipeline and segmenter multi-track statistics and source passing."""

    def test_pipeline_process_audio_chunk_updates_stats(self):
        """Verify DiarizeFlowPipeline.process_audio_chunk tracks sources and updates counters."""
        cfg = AppConfig()
        # Mock components to avoid loading heavy models
        with patch("diarizeflow.engine.pipeline.NemotronDiarizer"), \
             patch("diarizeflow.engine.pipeline.create_asr_engine"), \
             patch("diarizeflow.engine.pipeline.LLMTranslator"), \
             patch("diarizeflow.engine.pipeline.StreamingDiarizationSegmenter"):
            pipeline = DiarizeFlowPipeline(cfg)
            pipeline.is_running = True

            chunk = np.zeros(160, dtype=np.float32)

            pipeline.process_audio_chunk(chunk, source="mic")
            self.assertEqual(pipeline.last_audio_source, "mic")
            self.assertEqual(pipeline.audio_stats["mic_chunks"], 1)

            pipeline.process_audio_chunk(chunk, source="loopback")
            self.assertEqual(pipeline.last_audio_source, "loopback")
            self.assertEqual(pipeline.audio_stats["loopback_chunks"], 1)

            pipeline.process_audio_chunk(chunk, source="mixed")
            self.assertEqual(pipeline.last_audio_source, "mixed")
            self.assertEqual(pipeline.audio_stats["mixed_chunks"], 1)
            self.assertEqual(pipeline.audio_stats["total_chunks"], 3)

    def test_streaming_diarization_segmenter_accepts_source(self):
        """Verify StreamingDiarizationSegmenter.process_chunk accepts source parameter."""
        segmenter = StreamingDiarizationSegmenter(
            sample_rate=16000,
            sad_threshold=0.50,
            silence_timeout_ms=300,
            min_speech_ms=200,
        )

        chunk = np.zeros(320, dtype=np.float32)
        segmenter.process_chunk(chunk, rms=0.01, source="mic")
        self.assertEqual(segmenter.last_chunk_source, "mic")

        segmenter.process_chunk(chunk, rms=0.01, source="loopback")
        self.assertEqual(segmenter.last_chunk_source, "loopback")


class TestFrontendCaptureAndNetworkClient(unittest.TestCase):
    """Test suite for audio capture source tagging and network client packing."""

    def test_audio_capture_stream_dispatch_chunk_includes_source(self):
        """Verify AudioCaptureStream._dispatch_chunk passes source parameter to callback."""
        received_sources = []

        def callback(chunk, rms, source=None):
            received_sources.append(source)

        stream = AudioCaptureStream(
            target_sample_rate=16000,
            on_audio_chunk=callback,
        )

        chunk = np.zeros(160, dtype=np.float32)
        stream._dispatch_chunk(chunk, 0.05, source="mic")
        stream._dispatch_chunk(chunk, 0.02, source="loopback")
        stream._dispatch_chunk(chunk, 0.07, source="mixed")

        self.assertEqual(received_sources, ["mic", "loopback", "mixed"])

    def test_audio_capture_stream_backward_compatible_with_2arg_callback(self):
        """Verify AudioCaptureStream._dispatch_chunk falls back cleanly for legacy 2-arg callbacks."""
        received_chunks = []

        def legacy_callback(chunk, rms):
            received_chunks.append(len(chunk))

        stream = AudioCaptureStream(
            target_sample_rate=16000,
            on_audio_chunk=legacy_callback,
        )

        chunk = np.zeros(160, dtype=np.float32)
        # Should not raise TypeError:
        stream._dispatch_chunk(chunk, 0.05, source="mic")
        self.assertEqual(received_chunks, [160])

    def test_run_audio_client_tuple_and_array_packing(self):
        """Verify run_audio_client formats tuple and numpy array inputs into framed bytes."""
        audio_q = queue.Queue()
        sent_messages = []

        class MockWebSocket:
            def send(self, data):
                sent_messages.append(data)

        chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Enqueue tuple (chunk, "mic", 16000)
        audio_q.put((chunk, "mic", 16000))
        # Enqueue ndarray
        audio_q.put(chunk)
        # Enqueue raw framed bytes
        audio_q.put(pack_audio_frame(chunk, track="loopback", sample_rate=16000))

        is_running_flag = [True]

        def stop_after_drained():
            if audio_q.empty() and len(sent_messages) == 3:
                is_running_flag[0] = False
            return is_running_flag[0]

        with patch("diarizeflow.ui.network.ws_connect") as mock_connect:
            mock_ws = MockWebSocket()
            mock_connect.return_value.__enter__.return_value = mock_ws

            run_audio_client(
                host="127.0.0.1",
                port=8765,
                audio_queue=audio_q,
                is_running=stop_after_drained,
            )

        self.assertEqual(len(sent_messages), 3)

        # Check message 1: from tuple
        chunk1, track1, sr1, ch1 = unpack_audio_frame(sent_messages[0])
        self.assertEqual(track1, "mic")
        np.testing.assert_allclose(chunk1, chunk)

        # Check message 2: from ndarray
        chunk2, track2, sr2, ch2 = unpack_audio_frame(sent_messages[1])
        self.assertEqual(track2, "mixed")
        np.testing.assert_allclose(chunk2, chunk)

        # Check message 3: from framed bytes
        chunk3, track3, sr3, ch3 = unpack_audio_frame(sent_messages[2])
        self.assertEqual(track3, "loopback")
        np.testing.assert_allclose(chunk3, chunk)


class TestHUDOverlayFramingStreaming(unittest.TestCase):
    """Test suite for HUD overlay audio chunk handling and signal emission."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def test_overlay_on_audio_chunk_in_process_pipeline(self):
        """Verify _on_audio_chunk passes chunk, rms, and source to pipeline when pipeline is active."""
        from diarizeflow.ui.desktop_overlay import TransparentSubtitleOverlay

        cfg = AppConfig()
        mock_pipeline = MagicMock()
        mock_pipeline.is_running = True

        overlay = TransparentSubtitleOverlay(
            config=cfg,
            pipeline=mock_pipeline,
            enable_network=False,
            auto_start_capture=False,
        )

        chunk = np.array([0.1, -0.1], dtype=np.float32)
        overlay._on_audio_chunk(chunk, 0.05, source="mic")

        mock_pipeline.push_audio.assert_called_once_with(chunk, 0.05, source="mic")
        overlay.close()

    def test_overlay_on_audio_chunk_remote_streaming_packs_frame(self):
        """Verify _on_audio_chunk emits framed packet when pipeline is None (network mode)."""
        from diarizeflow.ui.desktop_overlay import TransparentSubtitleOverlay

        cfg = AppConfig()
        overlay = TransparentSubtitleOverlay(
            config=cfg,
            pipeline=None,
            enable_network=False,
            auto_start_capture=False,
        )

        emitted_packets = []
        overlay.audio_chunk_signal.connect(lambda data: emitted_packets.append(data))

        chunk = np.array([0.2, -0.3], dtype=np.float32)
        overlay._on_audio_chunk(chunk, 0.08, source="loopback")

        self.assertEqual(len(emitted_packets), 1)
        unpacked_chunk, track, sr, ch = unpack_audio_frame(emitted_packets[0])
        self.assertEqual(track, "loopback")
        self.assertEqual(sr, cfg.audio.sample_rate)
        np.testing.assert_allclose(unpacked_chunk, chunk)
        overlay.close()

    def test_audio_track_tag_from_string(self):
        """Verify AudioTrackTag.from_string resolves canonical and alias names correctly."""
        self.assertEqual(AudioTrackTag.from_string("mic"), AudioTrackTag.MIC)
        self.assertEqual(AudioTrackTag.from_string("microphone"), AudioTrackTag.MIC)
        self.assertEqual(AudioTrackTag.from_string("loopback"), AudioTrackTag.LOOPBACK)
        self.assertEqual(AudioTrackTag.from_string("system"), AudioTrackTag.LOOPBACK)
        self.assertEqual(AudioTrackTag.from_string("mixed"), AudioTrackTag.MIXED)
        self.assertEqual(AudioTrackTag.from_string("unknown_track"), AudioTrackTag.MIXED)

    def test_unpack_audio_frame_multichannel_downmix(self):
        """Verify unpack_audio_frame downmixes multichannel audio into mono."""
        # 2 channels: L=1.0, R=0.5 -> average 0.75
        stereo_pcm = np.array([1.0, 0.5, 0.8, 0.2], dtype=np.float32)
        packet = pack_audio_frame(stereo_pcm, track="mixed", sample_rate=16000, channels=2)

        chunk, track, sr, ch = unpack_audio_frame(packet)
        self.assertEqual(len(chunk), 2)
        np.testing.assert_allclose(chunk, np.array([0.75, 0.5], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
