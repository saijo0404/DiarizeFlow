"""Unit tests for AudioCaptureStream synchronization and mixing logic (Issue #1)."""

import queue
import time
import unittest
import numpy as np

from diarizeflow.audio.capture import AudioCaptureStream


class DummyStream:
    """Mock sounddevice stream."""
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate


class TestAudioCaptureStream(unittest.TestCase):
    def test_simultaneous_mic_and_loopback_no_expansion(self):
        """Test that simultaneous mic and loopback streams do not expand duration or interleave silence."""
        dispatched_chunks = []

        def on_chunk(chunk: np.ndarray, rms: float):
            dispatched_chunks.append(chunk.copy())

        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,  # 4000 samples per chunk
            gain=1.0,
            on_audio_chunk=on_chunk,
        )

        # Mock active mic and loopback
        stream.running = True
        stream.mic_stream = DummyStream(16000)
        stream._loopback_active = True

        # Start worker thread
        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        num_chunks = 4
        chunk_size = stream.chunk_samples  # 4000

        # Feed 4 mic chunks (value 0.4) and 4 loopback chunks (value 0.3) interleaved
        for i in range(num_chunks):
            mic_data = np.full(chunk_size, 0.4, dtype=np.float32)
            loop_data = np.full(chunk_size, 0.3, dtype=np.float32)
            stream._audio_queue.put(("mic", mic_data, 16000))
            time.sleep(0.01)
            stream._audio_queue.put(("loopback", loop_data, 16000))
            time.sleep(0.01)

        # Wait for queue to drain
        time.sleep(0.3)
        stream.running = False
        worker.join(timeout=1.0)

        # Expectation: exactly 4 chunks dispatched, NOT 8 chunks!
        # Duration should be 4 * 250ms = 1000ms, not 2000ms.
        self.assertEqual(
            len(dispatched_chunks),
            num_chunks,
            f"Expected {num_chunks} chunks, but got {len(dispatched_chunks)}. Temporal expansion detected!",
        )

        # Each chunk should be mixed: 0.4 + 0.3 = 0.7
        for idx, chunk in enumerate(dispatched_chunks):
            self.assertEqual(len(chunk), chunk_size)
            np.testing.assert_allclose(
                chunk,
                0.7,
                atol=1e-4,
                err_msg=f"Chunk {idx} was not synchronously mixed (found alternating silence or unmixed data)",
            )

    def test_single_stream_mic_only(self):
        """Test that mic-only mode dispatches expected chunks."""
        dispatched_chunks = []
        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,
            on_audio_chunk=lambda c, r: dispatched_chunks.append(c.copy()),
        )
        stream.running = True
        stream.mic_stream = DummyStream(16000)
        stream._loopback_active = False

        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        chunk_size = stream.chunk_samples
        for _ in range(3):
            stream._audio_queue.put(("mic", np.full(chunk_size, 0.5, dtype=np.float32), 16000))
            time.sleep(0.01)

        time.sleep(0.2)
        stream.running = False
        worker.join(timeout=1.0)

        self.assertEqual(len(dispatched_chunks), 3)
        for chunk in dispatched_chunks:
            np.testing.assert_allclose(chunk, 0.5, atol=1e-4)

    def test_single_stream_loopback_only(self):
        """Test that loopback-only mode dispatches expected chunks."""
        dispatched_chunks = []
        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,
            on_audio_chunk=lambda c, r: dispatched_chunks.append(c.copy()),
        )
        stream.running = True
        stream.mic_stream = None
        stream._loopback_active = True

        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        chunk_size = stream.chunk_samples
        for _ in range(3):
            stream._audio_queue.put(("loopback", np.full(chunk_size, 0.6, dtype=np.float32), 16000))
            time.sleep(0.01)

        time.sleep(0.2)
        stream.running = False
        worker.join(timeout=1.0)

        self.assertEqual(len(dispatched_chunks), 3)
        for chunk in dispatched_chunks:
            np.testing.assert_allclose(chunk, 0.6, atol=1e-4)

    def test_stall_fallback_when_loopback_halts(self):
        """Test that if loopback halts, mic chunks are not permanently blocked and are dispatched with silence."""
        dispatched_chunks = []
        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,
            stall_timeout=0.1,  # Short timeout for test
            on_audio_chunk=lambda c, r: dispatched_chunks.append(c.copy()),
        )
        stream.running = True
        stream.mic_stream = DummyStream(16000)
        stream._loopback_active = True

        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        chunk_size = stream.chunk_samples
        # Send 3 mic chunks, 0 loopback chunks
        for _ in range(3):
            stream._audio_queue.put(("mic", np.full(chunk_size, 0.8, dtype=np.float32), 16000))
            time.sleep(0.01)

        # Wait past stall_timeout
        time.sleep(0.25)
        stream.running = False
        worker.join(timeout=1.0)

        self.assertEqual(
            len(dispatched_chunks),
            3,
            f"Expected 3 fallback chunks dispatched, got {len(dispatched_chunks)}",
        )
        for chunk in dispatched_chunks:
            # 0.8 + 0.0 (silence) = 0.8
            np.testing.assert_allclose(chunk, 0.8, atol=1e-4)

    def test_stall_fallback_when_mic_halts(self):
        """Test that if mic halts, loopback chunks are not blocked and are dispatched with silence."""
        dispatched_chunks = []
        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,
            stall_timeout=0.1,  # Short timeout for test
            on_audio_chunk=lambda c, r: dispatched_chunks.append(c.copy()),
        )
        stream.running = True
        stream.mic_stream = DummyStream(16000)
        stream._loopback_active = True

        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        chunk_size = stream.chunk_samples
        # Send 3 loopback chunks, 0 mic chunks
        for _ in range(3):
            stream._audio_queue.put(("loopback", np.full(chunk_size, 0.4, dtype=np.float32), 16000))
            time.sleep(0.01)

        time.sleep(0.25)
        stream.running = False
        worker.join(timeout=1.0)

        self.assertEqual(
            len(dispatched_chunks),
            3,
            f"Expected 3 fallback chunks dispatched, got {len(dispatched_chunks)}",
        )
        for chunk in dispatched_chunks:
            np.testing.assert_allclose(chunk, 0.4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

