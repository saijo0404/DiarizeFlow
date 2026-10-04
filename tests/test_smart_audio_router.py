"""Unit tests for SmartAudioRouter and Dual-Track Decoupled Routing (Issue #30).

Verifies:
1. Pure Loopback turn-taking: passes system audio, completely mutes mic noise/bleed.
2. Pure Mic turn-taking: passes mic audio, completely mutes system background hum.
3. Cross-Talk / Overlap: passes both streams simultaneously when both speak.
4. Acoustic Bleed Suppression (AES): suppresses mic echo when speakers play loud audio into room.
5. Strong speech override: user speaking over loud audio is preserved as cross-talk.
6. Anti-Click cross-fading: linear ramp between routing state transitions ensures C0 continuity.
7. Mode switching: 'mix', 'mic_only', 'loopback_only' modes work as configured.
8. Dynamic parameter reconfiguration via update_config.
9. Buffer clock drift compensation and alignment in AudioCaptureStream.
"""

import time
import unittest
import numpy as np

from diarizeflow.audio.capture import AudioCaptureStream, SmartAudioRouter


class DummyStream:
    """Mock sounddevice stream."""
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate


class TestSmartAudioRouter(unittest.TestCase):
    """Test suite for SmartAudioRouter activity gating, bleed suppression, and cross-fading."""

    def setUp(self):
        self.router = SmartAudioRouter(
            routing_mode="smart",
            mic_threshold=0.008,
            loopback_threshold=0.008,
            bleed_suppression=True,
            bleed_ratio=0.40,
        )
        self.n_samples = 4000  # 250ms at 16kHz

    def test_pure_loopback_activity_mutes_mic_noise(self):
        """When only system audio is active, mic room noise is completely gated out."""
        # Mic has room ambient noise (RMS ~ 0.003, below threshold)
        mic_noise = np.random.uniform(-0.003, 0.003, self.n_samples).astype(np.float32)
        # Loopback has clean speech (RMS ~ 0.15)
        t = np.linspace(0, 0.25, self.n_samples)
        loop_speech = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # First chunk transitions from initial weights (1.0) to loopback target
        routed_1, src_1 = self.router.route(mic_noise, loop_speech)
        # Second chunk is fully settled in pure loopback mode
        routed_2, src_2 = self.router.route(mic_noise, loop_speech)

        self.assertEqual(src_2, "loopback")
        # In settled chunk, mic weight is 0.0, loop weight is 1.0
        np.testing.assert_allclose(routed_2, loop_speech, atol=1e-5)

    def test_pure_mic_activity_mutes_loopback(self):
        """When only microphone is active, system loopback is gated out."""
        t = np.linspace(0, 0.25, self.n_samples)
        mic_speech = (0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        loop_silence = np.full(self.n_samples, 0.001, dtype=np.float32)

        # Settle router
        self.router.route(mic_speech, loop_silence)
        routed, src = self.router.route(mic_speech, loop_silence)

        self.assertEqual(src, "mic")
        np.testing.assert_allclose(routed, mic_speech, atol=1e-5)

    def test_cross_talk_overlap_preserves_both(self):
        """When both streams are genuinely active, both are preserved for Sortformer separation."""
        t = np.linspace(0, 0.25, self.n_samples)
        mic_speech = (0.10 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        loop_speech = (0.12 * np.cos(2 * np.pi * 500 * t)).astype(np.float32)

        routed, src = self.router.route(mic_speech, loop_speech)
        self.assertEqual(src, "both")
        expected = np.clip(mic_speech + loop_speech, -1.0, 1.0)
        np.testing.assert_allclose(routed, expected, atol=1e-4)

    def test_acoustic_bleed_suppression_cancels_speaker_echo(self):
        """Loud loopback audio picked up by room mic at lower volume is recognized as echo and suppressed."""
        # Computer speaker playing loud audio (RMS ~ 0.10)
        t = np.linspace(0, 0.25, self.n_samples)
        loop_speech = (0.14 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # Room mic picking up speaker echo bleed at attenuated volume (e.g. 20% of loopback, RMS ~ 0.02)
        mic_bleed = 0.20 * loop_speech

        # Settle
        self.router.route(mic_bleed, loop_speech)
        routed, src = self.router.route(mic_bleed, loop_speech)

        # Bleed suppression should detect echo leakage and route loopback exclusively
        self.assertEqual(src, "loopback")
        np.testing.assert_allclose(routed, loop_speech, atol=1e-5)

    def test_strong_user_speech_overrides_bleed_suppression(self):
        """When user speaks loudly close to mic during playback, it is treated as genuine cross-talk."""
        t = np.linspace(0, 0.25, self.n_samples)
        loop_speech = (0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # User speaking directly into mic with substantial energy (RMS ~ 0.06 >= 0.030 threshold)
        mic_voice = (0.08 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        routed, src = self.router.route(mic_voice, loop_speech)
        self.assertEqual(src, "both")

    def test_anti_click_cross_fade_continuity(self):
        """Verify smooth linear ramp during track transitions without hard waveform step jumps."""
        # Start settled in loopback mode
        chunk_silent = np.zeros(self.n_samples, dtype=np.float32)
        chunk_loop = np.full(self.n_samples, 0.1, dtype=np.float32)
        self.router.route(chunk_silent, chunk_loop)
        self.router.route(chunk_silent, chunk_loop)
        self.assertEqual(self.router.prev_w_mic, 0.0)
        self.assertEqual(self.router.prev_w_loop, 1.0)

        # Transition: Mic suddenly speaks, loopback stops
        chunk_mic = np.full(self.n_samples, 0.1, dtype=np.float32)
        routed, src = self.router.route(chunk_mic, chunk_silent)
        self.assertEqual(src, "mic")

        # The output chunk begins with mic ramped up from 0.0 (prev weight) to 1.0 (settled weight)
        self.assertAlmostEqual(routed[0], 0.0, places=3)
        self.assertAlmostEqual(routed[-1], 0.1, places=3)
        # Check that ramp is smooth monotonically across transition without jumps
        diffs = np.diff(routed)
        self.assertTrue(np.all(diffs >= 0), "Ramp should be monotonically non-decreasing")
        self.assertTrue(np.all(np.abs(diffs) < 1e-4), "Discontinuous jump detected in cross-fade!")

    def test_routing_modes_mix_mic_only_loopback_only(self):
        """Verify explicit routing_mode overrides work as expected."""
        chunk_mic = np.full(self.n_samples, 0.2, dtype=np.float32)
        chunk_loop = np.full(self.n_samples, 0.3, dtype=np.float32)

        # Test 'mix' mode
        self.router.update_config(routing_mode="mix")
        routed, src = self.router.route(chunk_mic, chunk_loop)
        self.assertEqual(src, "both")
        np.testing.assert_allclose(routed, 0.5, atol=1e-5)

        # Test 'mic_only' mode
        self.router.update_config(routing_mode="mic_only")
        routed, src = self.router.route(chunk_mic, chunk_loop)
        self.assertEqual(src, "mic")
        np.testing.assert_allclose(routed, 0.2, atol=1e-5)

        # Test 'loopback_only' mode
        self.router.update_config(routing_mode="loopback_only")
        routed, src = self.router.route(chunk_mic, chunk_loop)
        self.assertEqual(src, "loopback")
        np.testing.assert_allclose(routed, 0.3, atol=1e-5)


class TestAudioCaptureStreamSmartRoutingIntegration(unittest.TestCase):
    """Integration tests for AudioCaptureStream with smart routing and drift alignment."""

    def test_stream_drift_compensation_realignment(self):
        """Verify that when one buffer drifts beyond max_drift_samples, it is realigned without freezing."""
        dispatched = []
        stream = AudioCaptureStream(
            target_sample_rate=16000,
            chunk_ms=250,  # 4000 samples
            max_drift_chunks=2,  # 8000 samples max drift
            on_audio_chunk=lambda c, r: dispatched.append(c.copy()),
        )
        stream.running = True
        stream.mic_stream = DummyStream(16000)
        stream._loopback_active = True

        import threading
        worker = threading.Thread(target=stream._processing_loop, daemon=True)
        worker.start()

        chunk_size = stream.chunk_samples  # 4000

        # Feed 6 mic chunks and only 1 loop chunk initially (causing 5 chunks = 20000 samples drift)
        for _ in range(6):
            stream._audio_queue.put(("mic", np.full(chunk_size, 0.15, dtype=np.float32), 16000))
        stream._audio_queue.put(("loopback", np.full(chunk_size, 0.15, dtype=np.float32), 16000))

        # Give worker a moment to process and align
        time.sleep(0.3)
        stream.running = False
        worker.join(timeout=1.0)

        # Verify stream processed chunks without hanging or crashing
        self.assertGreater(len(dispatched), 0)


if __name__ == "__main__":
    unittest.main()
