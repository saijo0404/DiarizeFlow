"""Nemotron-3-Diarization (Sortformer) Speaker Separation and Diarization Engine.

Implements NVIDIA's official Low-Latency Streaming Diarization specification:
- Low-latency mode: 1.04s input buffer latency (CHUNK_LEN=9, RIGHT_CONTEXT=4, FIFO_LEN=264, SPKCACHE_LEN=264, UPDATE_PERIOD=222)
- Very low-latency mode: 0.64s input buffer latency (CHUNK_LEN=6, RIGHT_CONTEXT=2)
- Ultra-low latency mode: 0.32s input buffer latency (CHUNK_LEN=3, RIGHT_CONTEXT=1)
- Supports stateful streaming speaker cache and temporal speaker turn splitting.
"""

from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import librosa
import numpy as np
import onnxruntime as ort

from diarizeflow.app.config import DiarizationConfig, resolve_app_path



# NVIDIA Nemotron-3-Diarization official streaming configurations
# Reference: https://huggingface.co/nvidia/Nemotron-3-Diarization
# Latency formula: (CHUNK_LEN + RIGHT_CONTEXT) * 80 ms
NEMOTRON_STREAMING_PROFILES: Dict[str, Dict[str, Any]] = {
    "low_latency": {
        "name": "Low latency (1.04s 官方預設)",
        "latency_s": 1.04,
        "chunk_len": 9,             # 9 * 80ms = 720ms active chunk
        "right_context": 4,         # 4 * 80ms = 320ms lookahead context
        "spkcache_len": 264,
        "fifo_len": 264,
        "update_period": 222,
        "mel_chunk_frames": 72,     # 9 * 8 = 72 mel frames (10ms hop)
        "mel_right_frames": 32,     # 4 * 8 = 32 mel frames
        "mel_buffer_frames": 104,   # (9 + 4) * 8 = 104 mel frames (1.04s)
        "sample_buffer_len": 16640, # 1.04s at 16kHz
    },
    "very_low_latency": {
        "name": "Very low latency (0.64s 超低延遲)",
        "latency_s": 0.64,
        "chunk_len": 6,             # 6 * 80ms = 480ms active chunk
        "right_context": 2,         # 2 * 80ms = 160ms lookahead context
        "spkcache_len": 264,
        "fifo_len": 264,
        "update_period": 222,
        "mel_chunk_frames": 48,     # 6 * 8 = 48 mel frames
        "mel_right_frames": 16,     # 2 * 8 = 16 mel frames
        "mel_buffer_frames": 64,    # (6 + 2) * 8 = 64 mel frames (0.64s)
        "sample_buffer_len": 10240, # 0.64s at 16kHz
    },
    "ultra_low_latency": {
        "name": "Ultra-low latency (0.32s 極限延遲)",
        "latency_s": 0.32,
        "chunk_len": 3,             # 3 * 80ms = 240ms active chunk
        "right_context": 1,         # 1 * 80ms = 80ms lookahead context
        "spkcache_len": 264,
        "fifo_len": 264,
        "update_period": 222,
        "mel_chunk_frames": 24,     # 3 * 8 = 24 mel frames
        "mel_right_frames": 8,      # 1 * 8 = 8 mel frames
        "mel_buffer_frames": 32,    # (3 + 1) * 8 = 32 mel frames (0.32s)
        "sample_buffer_len": 5120,  # 0.32s at 16kHz
    },
    "offline": {
        "name": "Offline (30.4s 長批次)",
        "latency_s": 30.4,
        "chunk_len": 340,
        "right_context": 40,
        "spkcache_len": 264,
        "fifo_len": 40,
        "update_period": 300,
        "mel_chunk_frames": 2720,
        "mel_right_frames": 320,
        "mel_buffer_frames": 3040,
        "sample_buffer_len": 486400,
    },
}


class NemotronDiarizer:
    """Sortformer-based streaming speaker diarization with ONNX Runtime."""

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self.session: Optional[ort.InferenceSession] = None
        self.active_provider: str = "CPU"
        self.is_fp16: bool = False
        self._input_frame_dim = 2112  # Standard chunk frame buffer for Sortformer
        self._target_mel_bins = 128
        self._mel_basis: Optional[np.ndarray] = None
        self.known_speakers: List[Dict[str, Any]] = []
        self.similarity_threshold: float = getattr(self.config, "speaker_threshold", 0.82)
        if self.similarity_threshold < 0.5:
            self.similarity_threshold = 0.82

        self.streaming_mode = getattr(self.config, "streaming_mode", "low_latency")
        self.active_profile = NEMOTRON_STREAMING_PROFILES.get(
            self.streaming_mode, NEMOTRON_STREAMING_PROFILES["low_latency"]
        )
        self.fifo_dim: int = self.active_profile["fifo_len"]

        self._lock = threading.RLock()
        self._load_model()
        self._init_streaming_state()

    def _create_temp_cache(self, clone: bool = True) -> Dict[str, np.ndarray]:
        """Create an isolated temporary cache dictionary for non-streaming inferences."""
        with self._lock:
            if clone:
                return {
                    "spkcache": self.spkcache.copy(),
                    "spkcache_lengths": self.spkcache_lengths.copy(),
                    "fifo": self.fifo.copy(),
                    "fifo_lengths": self.fifo_lengths.copy(),
                }
            else:
                return {
                    "spkcache": np.zeros_like(self.spkcache),
                    "spkcache_lengths": np.array([0], dtype=np.int64),
                    "fifo": np.zeros_like(self.fifo),
                    "fifo_lengths": np.array([0], dtype=np.int64),
                }

    def _init_streaming_state(self):
        """Initialize stateful streaming memory buffers for Sortformer according to active latency profile."""
        dtype = np.float16 if self.is_fp16 else np.float32
        spkcache_cap = self.active_profile.get("spkcache_len", 264)
        fifo_cap = self.active_profile.get("fifo_len", 264)

        # Defensive check against fixed ONNX graph shapes
        if self.session is not None:
            for inp in self.session.get_inputs():
                if inp.name == "fifo" and len(inp.shape) > 1:
                    d = inp.shape[1]
                    if isinstance(d, int) and d >= 0:
                        fifo_cap = d
                        self.fifo_dim = d

        self.spkcache = np.zeros((1, spkcache_cap, 512), dtype=dtype)
        self.spkcache_lengths = np.array([0], dtype=np.int64)
        self.fifo = np.zeros((1, fifo_cap, 512), dtype=dtype)
        self.fifo_lengths = np.array([0], dtype=np.int64)
        self.update_period = self.active_profile.get("update_period", 222)
        self.fifo_accumulated = 0

    def update_config(self, config: DiarizationConfig):
        """Update runtime configuration for diarization."""
        with self._lock:
            self.config = config
            new_thresh = getattr(config, "speaker_threshold", 0.82)
            self.similarity_threshold = new_thresh if new_thresh >= 0.5 else 0.82

            new_mode = getattr(config, "streaming_mode", "low_latency")
            if new_mode != self.streaming_mode and new_mode in NEMOTRON_STREAMING_PROFILES:
                self.streaming_mode = new_mode
                self.active_profile = NEMOTRON_STREAMING_PROFILES[new_mode]
                self._init_streaming_state()
                print(f"[*] Nemotron 串流分離模式切換為: {self.active_profile['name']}")
            else:
                print(f"[*] 語者分離門檻更新為: {self.similarity_threshold:.2f}")

    def reset(self):
        """Reset speaker memory profiles and Sortformer streaming cache."""
        with self._lock:
            self.known_speakers.clear()
            self._init_streaming_state()
            print("[*] Speaker diarization memory and Sortformer streaming cache reset.")


    def _load_model(self):
        """Find and load the appropriate ONNX model (FP16 on GPU, FP32 fallback)."""
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        target_path = resolve_app_path(self.config.model_path)
        if self.config.use_fp16 and "CUDAExecutionProvider" in providers:
            fp16_path = resolve_app_path(self.config.fp16_model_path)
            if fp16_path.exists():
                target_path = fp16_path

        if not target_path.exists():
            alt = resolve_app_path("models/nemotron_diarization/Nemotron-3-Diarization.onnx")
            if alt.exists():
                target_path = alt
            else:
                print(f"[!] Diarization ONNX model not found at {target_path}. Diarization will use acoustic voiceprints.")
                return

        print(f"[*] Loading Nemotron Diarization ONNX: {target_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(target_path), sess_options=sess_opts, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.is_fp16 = "float16" in self.session.get_inputs()[0].type
        print(
            f"[✓] Nemotron Diarizer initialized on {self.active_provider} "
            f"(FP16={self.is_fp16}, Mode={self.active_profile['name']})"
        )

    def _extract_mel(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Compute 128-channel log-mel spectrogram (25ms window, 10ms hop)."""
        n_fft = 512
        win_length = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.010)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=self._target_mel_bins,
            window="hamming",
            power=2.0,
        )
        log_mel = np.log(np.maximum(mel_spec, 1e-5)).T
        return log_mel.astype(np.float32)

    def _forward_chunk(
        self,
        chunk_mel: np.ndarray,
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Execute a single Sortformer forward step and update streaming FIFO / SPKCACHE.

        Args:
            chunk_mel: mel spectrogram slice, shape (N, 128).
            cache: Optional isolated temporary cache dict with keys 'spkcache',
                   'spkcache_lengths', 'fifo', 'fifo_lengths'. If None, the instance's
                   live streaming cache attributes are read and updated.

        Returns:
            Tuple of (channel_probs, pre_encode_embs, valid_embs_len).
        """
        if self.session is None:
            return None, None, 0

        num_frames = chunk_mel.shape[0]
        chunk_len = min(num_frames, self._input_frame_dim)

        dtype = np.float16 if self.is_fp16 else np.float32
        chunk_input = np.zeros((1, self._input_frame_dim, self._target_mel_bins), dtype=dtype)
        chunk_input[0, :chunk_len, :] = chunk_mel[:chunk_len]
        chunk_lengths = np.array([chunk_len], dtype=np.int64)

        with self._lock:
            target_spkcache = self.spkcache if cache is None else cache["spkcache"]
            target_spkcache_lengths = (
                self.spkcache_lengths if cache is None else cache["spkcache_lengths"]
            )
            target_fifo = self.fifo if cache is None else cache["fifo"]
            target_fifo_lengths = self.fifo_lengths if cache is None else cache["fifo_lengths"]

            feed = {
                "chunk": chunk_input,
                "chunk_lengths": chunk_lengths,
                "spkcache": target_spkcache,
                "spkcache_lengths": target_spkcache_lengths,
                "fifo": target_fifo,
                "fifo_lengths": target_fifo_lengths,
            }

            try:
                outputs = self.session.run(None, feed)
            except Exception:
                # Defensive fallback if FIFO shape differs from graph expectation
                if target_fifo.shape[1] != 0:
                    feed["fifo"] = np.zeros((1, 0, 512), dtype=dtype)
                    feed["fifo_lengths"] = np.array([0], dtype=np.int64)
                    outputs = self.session.run(None, feed)
                else:
                    raise

            if not outputs or outputs[0] is None:
                return None, None, 0

            # Sortformer outputs[0]: shape (1, 528, 8) frame predictions (1 frame per 40ms)
            raw_preds = outputs[0][0]
            valid_frames = max(1, chunk_len // 4)
            sub_preds = raw_preds[:valid_frames, :].astype(np.float32)

            if np.max(sub_preds) > 1.0 or np.min(sub_preds) < 0.0:
                probs = 1.0 / (1.0 + np.exp(-np.clip(sub_preds, -20.0, 20.0)))
            else:
                probs = sub_preds

            # Pre-encode speaker embeddings: outputs[1] shape (1, 264, 512)
            pre_embs = None
            valid_embs_len = 0
            if len(outputs) > 1 and outputs[1] is not None:
                pre_embs = outputs[1][0]
                valid_embs_len = (
                    max(1, int(outputs[2][0]))
                    if len(outputs) > 2 and outputs[2] is not None
                    else min(264, max(1, valid_frames // 2))
                )

                # Update Sortformer streaming speaker cache buffer (spkcache ring buffer)
                cur_cache_len = int(target_spkcache_lengths[0])
                spk_cap = target_spkcache.shape[1]
                if cur_cache_len + valid_embs_len <= spk_cap:
                    target_spkcache[0, cur_cache_len : cur_cache_len + valid_embs_len, :] = outputs[1][
                        0, :valid_embs_len, :
                    ].astype(dtype)
                    new_spk_lengths = np.array([cur_cache_len + valid_embs_len], dtype=np.int64)
                else:
                    keep = max(0, spk_cap - valid_embs_len)
                    if keep > 0:
                        target_spkcache[0, :keep, :] = target_spkcache[
                            0, cur_cache_len - keep : cur_cache_len, :
                        ]
                    target_spkcache[0, keep:spk_cap, :] = outputs[1][0, :valid_embs_len, :].astype(dtype)
                    new_spk_lengths = np.array([spk_cap], dtype=np.int64)

                if cache is None:
                    self.spkcache_lengths = new_spk_lengths
                else:
                    cache["spkcache_lengths"] = new_spk_lengths

                # Update FIFO queue buffer if enabled
                fifo_cap = target_fifo.shape[1]
                if fifo_cap > 0:
                    cur_fifo_len = int(target_fifo_lengths[0])
                    if cur_fifo_len + valid_embs_len <= fifo_cap:
                        target_fifo[0, cur_fifo_len : cur_fifo_len + valid_embs_len, :] = outputs[1][
                            0, :valid_embs_len, :
                        ].astype(dtype)
                        new_fifo_lengths = np.array([cur_fifo_len + valid_embs_len], dtype=np.int64)
                    else:
                        keep_f = max(0, fifo_cap - valid_embs_len)
                        if keep_f > 0:
                            target_fifo[0, :keep_f, :] = target_fifo[
                                0, cur_fifo_len - keep_f : cur_fifo_len, :
                            ]
                        target_fifo[0, keep_f:fifo_cap, :] = outputs[1][0, :valid_embs_len, :].astype(dtype)
                        new_fifo_lengths = np.array([fifo_cap], dtype=np.int64)

                    if cache is None:
                        self.fifo_lengths = new_fifo_lengths
                    else:
                        cache["fifo_lengths"] = new_fifo_lengths

            return probs, pre_embs, valid_embs_len

    def forward_streaming_step(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """Execute a streaming forward step on audio window and return frame probabilities.

        Args:
            audio: 1D numpy array of 16kHz float32 audio.
            sample_rate: sampling rate (default: 16000).

        Returns:
            probs array of shape (valid_frames, 8) with posterior probabilities in [0.0, 1.0],
            or None if inference is unavailable.
        """
        if self.session is None or len(audio) < 1600:
            return None

        mel = self._extract_mel(audio, sample_rate)
        probs, _, _ = self._forward_chunk(mel, cache=None)
        return probs

    def _stream_process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Stream process audio according to the official NVIDIA Low-Latency stepping specification.

        Uses an isolated temporary cache to avoid polluting the persistent streaming state.

        Formula:
          - Step stride = CHUNK_LEN * 8 mel frames (e.g. 72 frames = 720ms for low_latency)
          - Lookahead context = RIGHT_CONTEXT * 8 mel frames (e.g. 32 frames = 320ms for low_latency)
          - Total input buffer = 104 mel frames = 1.04s

        Returns:
            Tuple of (concatenated_frame_probs, mean_speaker_embedding).
            frame_probs shape: (total_frames, 8) where each frame is 40ms.
            speaker_embedding shape: (512,).
        """
        mel = self._extract_mel(audio, sample_rate)
        total_mel_frames = mel.shape[0]

        if cache is None:
            cache = self._create_temp_cache(clone=True)

        step_frames = self.active_profile.get("mel_chunk_frames", 72)
        lookahead_frames = self.active_profile.get("mel_right_frames", 32)
        buffer_frames = self.active_profile.get("mel_buffer_frames", 104)

        # If audio is within single buffer limit, execute one forward pass
        if total_mel_frames <= buffer_frames:
            probs, pre_embs, vlen = self._forward_chunk(mel, cache=cache)
            if probs is None:
                return None, None
            spk_vec = None
            if pre_embs is not None and vlen > 0:
                spk_vec = np.mean(pre_embs[:vlen], axis=0).astype(np.float32)
                norm = float(np.linalg.norm(spk_vec))
                if norm > 1e-6:
                    spk_vec /= norm
            return probs, spk_vec

        # Audio exceeds buffer: execute multi-chunk streaming generator
        scored_probs_list: List[np.ndarray] = []
        embs_list: List[np.ndarray] = []

        start_frame = 0
        while start_frame < total_mel_frames:
            end_frame = min(total_mel_frames, start_frame + buffer_frames)
            chunk_slice = mel[start_frame:end_frame]
            is_last_chunk = (end_frame >= total_mel_frames) or ((start_frame + step_frames) >= total_mel_frames)

            chunk_probs, chunk_embs, vlen = self._forward_chunk(chunk_slice, cache=cache)
            if chunk_probs is not None:
                # Scored frames exclude look-ahead context unless it's the last chunk of the session
                # 8 mel frames downsample to 2 Sortformer output frames (40ms per Sortformer output frame)
                if not is_last_chunk:
                    scored_out_frames = max(1, step_frames // 4)
                    scored_probs_list.append(chunk_probs[:scored_out_frames])
                else:
                    scored_probs_list.append(chunk_probs)

            if chunk_embs is not None and vlen > 0:
                embs_list.append(np.mean(chunk_embs[:vlen], axis=0))

            if is_last_chunk:
                break
            start_frame += step_frames

        if not scored_probs_list:
            return None, None

        all_probs = np.concatenate(scored_probs_list, axis=0)

        # Compute normalized speaker embedding from streaming chunks
        spk_vec = None
        if embs_list:
            spk_vec = np.mean(embs_list, axis=0).astype(np.float32)
            norm = float(np.linalg.norm(spk_vec))
            if norm > 1e-6:
                spk_vec /= norm

        return all_probs, spk_vec

    def _extract_voiceprint(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract multi-attribute acoustic voiceprint vector fallback.
        Uses CMVN-normalized MFCCs (excluding power band 0), dynamic delta transitions,
        and spectral contrast to ensure robust speaker discrimination.
        """
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
            mfcc_no_energy = mfcc[1:, :]
            cmvn = (mfcc_no_energy - np.mean(mfcc_no_energy, axis=1, keepdims=True)) / (
                np.std(mfcc_no_energy, axis=1, keepdims=True) + 1e-6
            )
            mfcc_mean = np.mean(cmvn, axis=1)
            mfcc_std = np.std(cmvn, axis=1)

            delta_mfcc = librosa.feature.delta(cmvn)
            delta_mean = np.mean(delta_mfcc, axis=1)

            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=512, hop_length=160)
            contrast_mean = np.mean(contrast, axis=1)

            feat = np.concatenate([mfcc_mean, mfcc_std, delta_mean, contrast_mean]).astype(np.float32)
            norm = float(np.linalg.norm(feat))
            if norm > 1e-6:
                feat /= norm
            return feat
        except Exception as e:
            print(f"[!] Warning in _extract_voiceprint: {e}")
            return np.zeros(65, dtype=np.float32)

    def identify_speaker(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Tuple[str, float, List[float]]:
        """Identify the primary speaker in the audio segment using native Nemotron Sortformer
        and 512-dimensional deep neural speaker embeddings with Low-latency streaming.

        Args:
            audio: 1D numpy array of 16kHz float32 audio.
            sample_rate: sampling rate (default: 16000).

        Returns:
            Tuple of (speaker_label, confidence, all_speaker_probs).
            e.g. ("講者 1", 0.92, [0.88, 0.05, 0.02, ...])
        """
        if len(audio) < 1600:
            return "講者 1", 1.0, [1.0] + [0.0] * 7

        # 1. Native Nemotron Sortformer Diarization with 512-d Deep Speaker Embeddings
        if self.session is not None:
            try:
                probs, spk_vec = self._stream_process_audio(audio, sample_rate)
                if probs is not None and spk_vec is not None:
                    channel_scores = np.mean(probs, axis=0)

                    # Match or register speaker with Nemotron deep embedding
                    speaker_label, best_sim = self._match_or_register_speaker(
                        spk_vec, f"Nemotron {self.active_profile['name']}"
                    )

                    spk_probs = [float(s) for s in channel_scores[:8]]
                    while len(spk_probs) < 8:
                        spk_probs.append(0.0)

                    channel_fmt = ", ".join(
                        [f"S{i+1}: {channel_scores[i]:.2f}" for i in range(min(4, len(channel_scores)))]
                    )
                    print(f"    8 聲軌活躍分佈: [{channel_fmt}]")

                    return speaker_label, best_sim, spk_probs

            except Exception as ex:
                print(f"[!] Nemotron ONNX 推論異常，轉為聲學特徵分離: {ex}")

        # 2. Acoustic Voiceprint Fallback (Used only if ONNX model is missing or fails)
        try:
            emb = self._extract_voiceprint(audio, sample_rate)
            speaker_label, best_sim = self._match_or_register_speaker(emb, "聲學聲紋分離")
            return speaker_label, best_sim, [best_sim] + [0.0] * 7
        except Exception as e:
            print(f"[!] Diarization inference failed: {e}")
            return "講者 1", 0.5, [0.5] + [0.0] * 7

    def _match_or_register_speaker(
        self, spk_vec: np.ndarray, engine_tag: str = "Nemotron 語者分離"
    ) -> Tuple[str, float]:
        """Match speaker against known speaker database using cosine similarity, or register a new speaker."""
        with self._lock:
            if not self.known_speakers:
                self.known_speakers.append({"id": 1, "emb": spk_vec, "count": 1})
                speaker_label = "講者 1"
                print(f"[+] [{engine_tag}] 註冊首位發話者: {speaker_label}")
                return speaker_label, 1.0

            sims = [float(np.dot(spk_vec, spk["emb"])) for spk in self.known_speakers]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            if best_sim >= self.similarity_threshold:
                spk = self.known_speakers[best_idx]
                spk["emb"] = 0.85 * spk["emb"] + 0.15 * spk_vec
                spk_n = float(np.linalg.norm(spk["emb"]))
                if spk_n > 1e-6:
                    spk["emb"] /= spk_n
                spk["count"] += 1
                speaker_label = f"講者 {spk['id']}"
                print(f"[*] [{engine_tag}] 判定為 {speaker_label} (聲線相似度: {best_sim:.3f})")
                return speaker_label, best_sim
            else:
                if len(self.known_speakers) < self.config.max_speakers:
                    new_id = len(self.known_speakers) + 1
                    self.known_speakers.append({"id": new_id, "emb": spk_vec, "count": 1})
                    speaker_label = f"講者 {new_id}"
                    print(
                        f"[+] [{engine_tag}] 檢測到新聲線！註冊為 {speaker_label} "
                        f"(最高相似度: {best_sim:.3f} < 門檻 {self.similarity_threshold:.2f})"
                    )
                    return speaker_label, best_sim
                else:
                    speaker_label = f"講者 {self.known_speakers[best_idx]['id']}"
                    print(f"[*] [{engine_tag}] 達到講者上限，指派最接近的 {speaker_label} (相似度: {best_sim:.3f})")
                    return speaker_label, best_sim

    def diarize_and_split(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[Tuple[np.ndarray, str, float]]:
        """Identify speakers and split audio into multi-speaker intervals using Sortformer SAD.

        Replaces rigid split heuristics with neural multi-track Speaker Activity Detection,
        enabling short-turn detection (<0.65s), multi-speaker turns (>2 speakers),
        and natural overlap speech handling.

        Args:
            audio: 1D numpy array of 16kHz float32 audio.
            sample_rate: sampling rate (default: 16000).

        Returns:
            List of (sub_audio, speaker_label, confidence).
        """
        if len(audio) < int(sample_rate * 0.5) or self.session is None:
            spk, conf, _ = self.identify_speaker(audio, sample_rate)
            return [(audio, spk, conf)]

        try:
            probs, _ = self._stream_process_audio(audio, sample_rate)
            if probs is None or len(probs) < 5:
                spk, conf, _ = self.identify_speaker(audio, sample_rate)
                return [(audio, spk, conf)]

            valid_frames = len(probs)
            active_channels = []
            for ch in range(min(8, probs.shape[1])):
                if np.sum(probs[:, ch] >= 0.40) >= 5:  # at least 200ms speech
                    active_channels.append(ch)

            # If single active speaker detected, identify as single segment
            if len(active_channels) <= 1:
                spk, conf, _ = self.identify_speaker(audio, sample_rate)
                return [(audio, spk, conf)]

            # Multi-channel SAD: extract intervals for each active speaker
            intervals = []
            min_speech_frames = 5   # 200ms
            max_gap_frames = 6      # 240ms bridge

            for ch in active_channels:
                ch_active = (probs[:, ch] >= 0.40).astype(np.int32)
                # Bridge short gaps
                gap_start = -1
                for t in range(valid_frames):
                    if ch_active[t] == 0:
                        if gap_start < 0:
                            gap_start = t
                    else:
                        if gap_start >= 0 and (t - gap_start) <= max_gap_frames:
                            ch_active[gap_start:t] = 1
                        gap_start = -1

                # Find contiguous active regions
                diffs = np.diff(np.pad(ch_active, (1, 1), mode="constant"))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]

                for s, e in zip(starts, ends):
                    if (e - s) >= min_speech_frames:
                        intervals.append((s, e, ch))

            if len(intervals) >= 2:
                # Context padding: 150ms before and after
                pad_samples = int(0.150 * sample_rate)
                extracted_segments = []

                for s_frame, e_frame, ch in intervals:
                    s_samp = max(0, int(s_frame * 0.040 * sample_rate) - pad_samples)
                    e_samp = min(len(audio), int(e_frame * 0.040 * sample_rate) + pad_samples)
                    sub_audio = audio[s_samp:e_samp]
                    if len(sub_audio) >= int(0.20 * sample_rate):
                        spk, conf, _ = self.identify_speaker(sub_audio, sample_rate)
                        extracted_segments.append((s_samp, sub_audio, spk, conf))

                if extracted_segments:
                    # Sort by start sample
                    extracted_segments.sort(key=lambda x: x[0])
                    print(
                        f"[★] [Nemotron Sortformer SAD 多軌切分] 於 {len(audio)/sample_rate:.2f}s 音訊中"
                        f"分離出 {len(extracted_segments)} 段講者發言: "
                        f"{' -> '.join([seg[2] for seg in extracted_segments])}"
                    )
                    return [(seg[1], seg[2], seg[3]) for seg in extracted_segments]

            # Fallback to single primary speaker
            spk, conf, _ = self.identify_speaker(audio, sample_rate)
            return [(audio, spk, conf)]

        except Exception as e:
            print(f"[!] Warning in diarize_and_split: {e}")
            spk, conf, _ = self.identify_speaker(audio, sample_rate)
            return [(audio, spk, conf)]
