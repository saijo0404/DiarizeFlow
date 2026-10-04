"""Unit tests and benchmarks for Mel Spectrogram filter matrix and window caching (Issue #49).

Verifies:
1. Static Mel filterbank matrix (_mel_basis) and Hamming window (_mel_window) are pre-computed in __init__.
2. _extract_mel reuses cached structures and does not re-invoke librosa.filters.mel or window generators.
3. Numerical consistency between cached _extract_mel and librosa.feature.melspectrogram (< 1e-6 max absolute difference).
4. Robustness across diverse audio lengths, silence, impulses, and multi-sample-rate dynamic caching.
5. Execution speedup in streaming inference loops.
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np
import scipy.signal

from diarizeflow.engine.diarizer import NemotronDiarizer
from diarizeflow.config import DiarizationConfig


class TestMelSpectrogramCache(unittest.TestCase):
    """Test suite verifying Mel Spectrogram filter matrix and window function caching in NemotronDiarizer."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.profiles_path = Path(self.tmp_dir.name) / "profiles.json"
        self.config = DiarizationConfig(profiles_path=str(self.profiles_path))
        self.diarizer = NemotronDiarizer(self.config)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_mel_structures_cached_on_initialization(self):
        """Verify _mel_basis and _mel_window are precomputed and cached on diarizer initialization."""
        self.assertIsNotNone(self.diarizer._mel_basis)
        self.assertIsNotNone(self.diarizer._mel_window)
        self.assertIn(16000, self.diarizer._mel_cache)

        # Expected dimensions for 16kHz, n_fft=512, n_mels=128, win_length=int(16000 * 0.025)=400
        self.assertEqual(self.diarizer._mel_basis.shape, (128, 257))
        self.assertEqual(self.diarizer._mel_window.shape, (400,))
        self.assertEqual(self.diarizer._mel_cache[16000][0].shape, (128, 257))
        self.assertEqual(self.diarizer._mel_cache[16000][1].shape, (400,))

    def test_extract_mel_numerical_equivalence_to_librosa_baseline(self):
        """Verify _extract_mel produces exact numerical output compared to standard librosa (< 1e-6 diff)."""
        rng = np.random.RandomState(42)
        sample_rate = 16000
        audio = rng.randn(sample_rate).astype(np.float32)

        # 1. Baseline librosa.feature.melspectrogram calculation
        n_fft = 512
        win_length = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.010)

        baseline_mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=128,
            window="hamming",
            power=2.0,
        )
        baseline_log_mel = np.log(np.maximum(baseline_mel_spec, 1e-5)).T.astype(np.float32)

        # 2. Optimized cached _extract_mel
        cached_log_mel = self.diarizer._extract_mel(audio, sample_rate=sample_rate)

        # 3. Assert shapes match
        self.assertEqual(cached_log_mel.shape, baseline_log_mel.shape)

        # 4. Assert max absolute difference is strictly < 1e-6
        max_abs_diff = float(np.max(np.abs(cached_log_mel - baseline_log_mel)))
        self.assertLess(max_abs_diff, 1e-6, f"Max difference {max_abs_diff} exceeds 1e-6 threshold")

    def test_extract_mel_on_different_audio_lengths(self):
        """Verify cached _extract_mel functions reliably and matches baseline across multiple chunk durations."""
        sample_rate = 16000
        chunk_lengths = [
            1600,   # 100ms
            3200,   # 200ms
            5120,   # 320ms (ultra-low latency profile buffer)
            10240,  # 640ms (very-low latency profile buffer)
            16640,  # 1.04s (low latency official default buffer)
            32000,  # 2.0s
        ]

        rng = np.random.RandomState(123)
        for num_samples in chunk_lengths:
            audio = rng.randn(num_samples).astype(np.float32)

            baseline_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=512,
                hop_length=160,
                win_length=400,
                n_mels=128,
                window="hamming",
                power=2.0,
            )
            baseline_log_mel = np.log(np.maximum(baseline_spec, 1e-5)).T.astype(np.float32)

            cached_log_mel = self.diarizer._extract_mel(audio, sample_rate=sample_rate)

            self.assertEqual(cached_log_mel.shape, baseline_log_mel.shape)
            self.assertEqual(cached_log_mel.dtype, np.float32)
            max_diff = float(np.max(np.abs(cached_log_mel - baseline_log_mel)))
            self.assertLess(max_diff, 1e-6, f"Mismatch on chunk size {num_samples}: {max_diff}")

    def test_extract_mel_edge_cases_silence_and_impulse(self):
        """Verify cached _extract_mel handles edge cases such as complete silence and impulse signals."""
        sample_rate = 16000

        # Silence (all zeros)
        silence = np.zeros(8000, dtype=np.float32)
        base_silence = librosa.feature.melspectrogram(
            y=silence,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=128,
            window="hamming",
            power=2.0,
        )
        base_log_silence = np.log(np.maximum(base_silence, 1e-5)).T.astype(np.float32)
        cached_log_silence = self.diarizer._extract_mel(silence, sample_rate=sample_rate)
        self.assertLess(float(np.max(np.abs(cached_log_silence - base_log_silence))), 1e-6)

        # Impulse signal
        impulse = np.zeros(8000, dtype=np.float32)
        impulse[4000] = 1.0
        base_impulse = librosa.feature.melspectrogram(
            y=impulse,
            sr=sample_rate,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=128,
            window="hamming",
            power=2.0,
        )
        base_log_impulse = np.log(np.maximum(base_impulse, 1e-5)).T.astype(np.float32)
        cached_log_impulse = self.diarizer._extract_mel(impulse, sample_rate=sample_rate)
        self.assertLess(float(np.max(np.abs(cached_log_impulse - base_log_impulse))), 1e-6)

    def test_mel_cache_reuses_precomputed_arrays_without_rebuilding(self):
        """Verify _extract_mel reuses cached arrays and does not recompute librosa.filters.mel or scipy windows."""
        audio = np.random.randn(5120).astype(np.float32)

        # Patch librosa.filters.mel and scipy.signal.windows.hamming to detect calls
        with patch("librosa.filters.mel") as mock_mel, patch(
            "scipy.signal.windows.hamming"
        ) as mock_hamming:
            # Execute multiple extraction passes
            for _ in range(10):
                mel = self.diarizer._extract_mel(audio, sample_rate=16000)
                self.assertEqual(mel.shape[-1], 128)

            # Neither should have been called because 16000 is cached
            mock_mel.assert_not_called()
            mock_hamming.assert_not_called()

    def test_multi_sample_rate_dynamic_caching(self):
        """Verify dynamic caching of mel filters for alternative sample rates (e.g. 8000 Hz)."""
        sr_8k = 8000
        audio_8k = np.random.randn(sr_8k).astype(np.float32)

        # 8000 Hz should not be initially cached
        self.assertNotIn(sr_8k, self.diarizer._mel_cache)

        # First call triggers lazy initialization and caching
        mel_8k_first = self.diarizer._extract_mel(audio_8k, sample_rate=sr_8k)
        self.assertIn(sr_8k, self.diarizer._mel_cache)
        basis_8k, win_8k = self.diarizer._mel_cache[sr_8k]
        self.assertEqual(basis_8k.shape, (128, 257))
        self.assertEqual(win_8k.shape, (int(sr_8k * 0.025),))  # 200 samples

        # Check numerical accuracy at 8000 Hz
        base_8k = librosa.feature.melspectrogram(
            y=audio_8k,
            sr=sr_8k,
            n_fft=512,
            hop_length=int(sr_8k * 0.010),
            win_length=int(sr_8k * 0.025),
            n_mels=128,
            window="hamming",
            power=2.0,
        )
        base_log_8k = np.log(np.maximum(base_8k, 1e-5)).T.astype(np.float32)
        self.assertLess(float(np.max(np.abs(mel_8k_first - base_log_8k))), 1e-6)

        # Second call reuses cached 8000 Hz filters without recomputation
        with patch("librosa.filters.mel") as mock_mel:
            mel_8k_second = self.diarizer._extract_mel(audio_8k, sample_rate=sr_8k)
            mock_mel.assert_not_called()
            np.testing.assert_array_equal(mel_8k_first, mel_8k_second)

        # 16000 Hz cached filters remain intact and unchanged
        self.assertIn(16000, self.diarizer._mel_cache)

    def test_manual_override_of_mel_structures(self):
        """Verify that explicitly setting _mel_basis or _mel_window on diarizer is respected."""
        audio = np.random.randn(3200).astype(np.float32)
        custom_basis = np.ones((128, 257), dtype=np.float32) * 0.1
        custom_window = np.ones(400, dtype=np.float32)

        self.diarizer._mel_basis = custom_basis
        self.diarizer._mel_window = custom_window

        ret_basis, ret_win = self.diarizer._get_mel_filters(16000)
        self.assertIs(ret_basis, custom_basis)
        self.assertIs(ret_win, custom_window)

    def test_caching_speedup_benchmark(self):
        """Benchmark cached extraction vs repeated generation to verify meaningful performance speedup."""
        sample_rate = 16000
        audio = np.random.randn(5120).astype(np.float32)  # 320ms chunk
        iterations = 50

        # Benchmark un-cached (recreating filters each time)
        n_fft = 512
        win_length = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.010)

        t0 = time.perf_counter()
        for _ in range(iterations):
            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=128,
                window="hamming",
                power=2.0,
            )
            _ = np.log(np.maximum(spec, 1e-5)).T
        t_uncached = time.perf_counter() - t0

        # Benchmark cached _extract_mel
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.diarizer._extract_mel(audio, sample_rate=sample_rate)
        t_cached = time.perf_counter() - t0

        speedup = t_uncached / max(t_cached, 1e-6)
        print(f"\n[Benchmark] Uncached: {t_uncached*1000/iterations:.3f}ms/call | Cached: {t_cached*1000/iterations:.3f}ms/call | Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
