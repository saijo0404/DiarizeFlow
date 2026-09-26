"""Multilingual Automatic Speech Recognition (ASR) Engine.

Specialized for SenseVoiceSmall ONNX, providing low-latency multilingual speech recognition
(Mandarin Chinese, English, Japanese, Cantonese, Korean, etc.).
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from diarizeflow.app.config import ASRConfig, resolve_app_path


class SenseVoiceASR:
    """Ultra-fast multilingual ASR using SenseVoiceSmall ONNX model."""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[spm.SentencePieceProcessor] = None
        self.cmvn_means: Optional[np.ndarray] = None
        self.cmvn_vars: Optional[np.ndarray] = None
        self.is_fp16: bool = False
        self.active_provider: str = "CPU"
        self._fbank_opts = None

        self._init_engine()

    def _init_engine(self):
        """Initialize Kaldi fbank, CMVN table, SentencePiece tokenizer, and ONNX Runtime session."""
        model_dir = resolve_app_path(self.config.model_dir)

        # 1. Load CMVN table
        cmvn_path = resolve_app_path(self.config.cmvn_file)
        if not cmvn_path.exists():
            cmvn_path = model_dir / "am.mvn"

        if cmvn_path.exists():
            self._load_cmvn(cmvn_path)
        else:
            print(f"[!] Warning: CMVN file not found at {cmvn_path}")

        # 2. Load Tokenizer
        bpe_path = resolve_app_path(self.config.bpe_model)
        if not bpe_path.exists():
            bpe_path = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"

        if bpe_path.exists():
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(bpe_path))
            print(f"[✓] SenseVoice Tokenizer loaded ({self.tokenizer.get_piece_size()} vocab)")
        else:
            print(f"[!] Warning: SentencePiece model not found at {bpe_path}")

        # 3. Setup Kaldi Native Fbank
        try:
            import kaldi_native_fbank as knf
            opts = knf.FbankOptions()
            opts.frame_opts.samp_freq = 16000
            opts.frame_opts.dither = 0.0
            opts.frame_opts.window_type = "hamming"
            opts.frame_opts.frame_shift_ms = 10.0
            opts.frame_opts.frame_length_ms = 25.0
            opts.mel_opts.num_bins = 80
            opts.energy_floor = 0.0
            opts.frame_opts.snip_edges = True
            opts.mel_opts.debug_mel = False
            self._fbank_opts = opts
        except Exception as e:
            print(f"[!] Warning: kaldi-native-fbank unavailable: {e}")

        # 4. Load ONNX model
        available = ort.get_available_providers()
        providers = []
        if self.config.use_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        target_model = resolve_app_path(self.config.model_path)
        if not target_model.exists():
            for candidate in [
                model_dir / "SenseVoiceSmall_int8.onnx",
                model_dir / "SenseVoiceSmall_fp16.onnx",
                model_dir / "SenseVoiceSmall.onnx",
                model_dir / "model_int8.onnx",
                model_dir / "model.onnx",
            ]:
                if candidate.exists():
                    target_model = candidate
                    break

        if target_model.exists():
            print(f"[*] Loading SenseVoice ONNX: {target_model}")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(target_model), sess_options=sess_opts, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            self.is_fp16 = "float16" in self.session.get_inputs()[0].type
            print(f"[✓] SenseVoice ASR loaded on {self.active_provider} (FP16={self.is_fp16})")
        else:
            print(f"[!] SenseVoice ONNX model not found in {model_dir}")

    def _load_cmvn(self, cmvn_file: Path):
        """Parse Kaldi CMVN file without any external dependencies."""
        with open(cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if not line_item:
                continue
            if line_item[0] == "<AddShift>":
                if i + 1 < len(lines):
                    next_item = lines[i + 1].split()
                    if next_item and next_item[0] == "<LearnRateCoef>":
                        means_list = next_item[3 : (len(next_item) - 1)]
            elif line_item[0] == "<Rescale>":
                if i + 1 < len(lines):
                    next_item = lines[i + 1].split()
                    if next_item and next_item[0] == "<LearnRateCoef>":
                        vars_list = next_item[3 : (len(next_item) - 1)]

        self.cmvn_means = np.array(means_list, dtype=np.float32)
        self.cmvn_vars = np.array(vars_list, dtype=np.float32)

    def _extract_fbank(self, audio: np.ndarray) -> np.ndarray:
        """Extract 80-bin filterbank features."""
        import kaldi_native_fbank as knf
        waveform = audio * (1 << 15)  # Scale to 16-bit integer scale
        fbank_fn = knf.OnlineFbank(self._fbank_opts)
        fbank_fn.accept_waveform(16000, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, 80], dtype=np.float32)
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        return mat

    def _apply_lfr_cmvn(self, fbank: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
        """Apply Linear Filter Reconstruction (LFR) and Cepstral Mean & Variance Normalization (CMVN)."""
        T = fbank.shape[0]
        T_lfr = int(np.ceil(T / lfr_n))
        left_padding = np.tile(fbank[0], ((lfr_m - 1) // 2, 1))
        inputs = np.vstack((left_padding, fbank))
        T = T + (lfr_m - 1) // 2

        LFR_inputs = []
        for i in range(T_lfr):
            if lfr_m <= T - i * lfr_n:
                LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
            else:
                num_padding = lfr_m - (T - i * lfr_n)
                frame = inputs[i * lfr_n :].reshape(-1)
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)

        feat = np.vstack(LFR_inputs).astype(np.float32)  # shape: (T_lfr, 560)

        # CMVN
        if self.cmvn_means is not None and self.cmvn_vars is not None:
            feat = (feat + self.cmvn_means) * self.cmvn_vars

        return feat

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Tuple[str, str]:
        """Transcribe speech waveform (16kHz float32) into text.

        Returns:
            Tuple of (clean_text, detected_language).
        """
        if self.session is None or self.tokenizer is None or len(audio) < 1600:
            return "", "auto"

        try:
            # 1. Feature extraction
            fb = self._extract_fbank(audio)
            if fb.shape[0] < 5:
                return "", "auto"

            feats = self._apply_lfr_cmvn(fb)

            # 2. Forward pass
            dtype = np.float16 if self.is_fp16 else np.float32
            feed = {
                "speech": feats[np.newaxis, ...].astype(dtype),
                "speech_lengths": np.array([feats.shape[0]], dtype=np.int32),
                "language": np.array([0], dtype=np.int32),  # 0: auto
                "textnorm": np.array([15], dtype=np.int32),  # 15: woitn/with punct
            }

            logits = self.session.run(None, feed)[0]  # (1, T, vocab_size)

            # 3. CTC Greedy Decoding
            preds = np.argmax(logits[0], axis=-1)
            tokens = []
            prev = -1
            for p in preds:
                if p != prev:
                    if p != 0:  # skip blank token (id 0)
                        tokens.append(int(p))
                    prev = p

            raw_text = self.tokenizer.decode(tokens)

            # 4. Clean tags and detect language
            detected_lang = "auto"
            lang_match = re.search(r"<\|(zh|en|ja|ko|yue)\|>", raw_text)
            if lang_match:
                detected_lang = lang_match.group(1)

            # Strip special tags like <|zh|>, <|NEUTRAL|>, <|HAPPY|>, <|Speech|>, etc.
            clean_text = re.sub(r"<\|[^|>]+?\|>", "", raw_text).strip()
            # Normalize whitespace
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            return clean_text, detected_lang

        except Exception as e:
            print(f"[!] ASR transcription failed: {e}")
            return "", "auto"


class FasterWhisperASR:
    """Multilingual Automatic Speech Recognition Engine using Faster-Whisper (CTranslate2).

    Supports high-accuracy multilingual recognition with precision scaling:
    float32, float16, int8, fp8, w4a16, nvfp4, mxfp4.
    """

    # Mapping of user-requested precision names to CTranslate2 compute_types
    PRECISION_MAP = {
        "float32": ("float32", "FP32 全精度無損"),
        "fp32": ("float32", "FP32 全精度無損"),
        "float16": ("float16", "FP16 半精度加速"),
        "fp16": ("float16", "FP16 半精度加速"),
        "int8": ("int8_float16", "INT8 混合精度高壓縮"),
        "fp8": ("float8_e4m3fn", "FP8 E4M3FN 新代顯卡加速"),
        "w4a16": ("int8_float16", "W4A16 4-bit 權重量化"),
        "nvfp4": ("int8_float16", "NVFP4 Blackwell FP4 區塊量化"),
        "mxfp4": ("int8_float16", "MXFP4 OCP 顯微浮點量化"),
    }

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model = None
        self.active_precision = getattr(self.config, "whisper_precision", "float16").lower()
        self.active_compute_type = "float16"
        self.device = "cuda" if getattr(self.config, "use_gpu", True) else "cpu"
        self._fallback_sensevoice: Optional[SenseVoiceASR] = None
        self._init_engine()

    def _init_engine(self):
        """Initialize Faster-Whisper model with selected precision."""
        model_target = getattr(self.config, "whisper_model", "models/faster-whisper-large-v2")
        resolved_path = resolve_app_path(model_target)

        # Determine target model identifier or directory
        if resolved_path.exists():
            model_id_or_path = str(resolved_path)
        else:
            # Fallback to local models dir or HuggingFace ID
            cand = Path("models") / Path(model_target).name
            if cand.exists():
                model_id_or_path = str(cand)
            else:
                model_id_or_path = str(model_target)

        # Check GPU availability
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
            except Exception:
                pass

        # Resolve compute_type based on precision and device
        target_prec = self.active_precision
        ct2_compute, desc = self.PRECISION_MAP.get(target_prec, ("float16", "FP16"))

        if self.device == "cpu":
            if ct2_compute in ["float16", "float8_e4m3fn", "int8_float16"]:
                ct2_compute = "int8" if "int8" in ct2_compute else "float32"

        print(f"[*] 正在載入 Faster-Whisper 模型: {model_id_or_path}")
        print(f"    運算裝置: {self.device.upper()} | 設定精度: {target_prec.upper()} ({desc}) | 核心計算: {ct2_compute}")

        try:
            from faster_whisper import WhisperModel

            # Attempt loading with preferred compute_type, falling back gracefully
            for compute_attempt in [ct2_compute, "int8_float16", "float16", "int8", "float32"]:
                try:
                    self.model = WhisperModel(
                        model_id_or_path,
                        device=self.device,
                        compute_type=compute_attempt,
                    )
                    self.active_compute_type = compute_attempt
                    print(f"[✓] Faster-Whisper 成功載入！(運算模式: {self.active_compute_type})")
                    return
                except Exception as load_err:
                    print(f"    [!] 嘗試 compute_type={compute_attempt} 失敗: {load_err}，嘗試備用精度...")

            print(f"[!] 無法使用任何精度載入 Faster-Whisper 模型: {model_id_or_path}")

        except ImportError:
            print("[!] 未安裝 faster-whisper 套件。請使用: pip install faster-whisper")
        except Exception as e:
            print(f"[!] Faster-Whisper 初始化異常: {e}")

        print("[*] 啟用 SenseVoiceSmall 作為即時備用 ASR 引擎...")
        self._fallback_sensevoice = SenseVoiceASR(self.config)

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Tuple[str, str]:
        """Transcribe audio waveform (16kHz float32) using Faster-Whisper.

        Returns:
            Tuple of (clean_text, detected_language).
        """
        if self.model is None:
            if self._fallback_sensevoice:
                return self._fallback_sensevoice.transcribe(audio, language)
            return "", "auto"

        if len(audio) < 1600:
            return "", "auto"

        try:
            # Normalize to 1D float32
            audio_f32 = audio.astype(np.float32)
            if audio_f32.ndim > 1:
                audio_f32 = audio_f32.squeeze()

            lang_param = None if (not language or language == "auto") else language.lower()
            beam_size = getattr(self.config, "beam_size", 1)

            segments, info = self.model.transcribe(
                audio_f32,
                beam_size=beam_size,
                language=lang_param,
                condition_on_previous_text=False,
                vad_filter=False,  # VAD already handled upstream by DiarizeFlow VAD
            )

            texts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
            full_text = " ".join(texts).strip()
            detected_lang = info.language if info and info.language else "auto"

            return full_text, detected_lang

        except Exception as e:
            print(f"[!] Faster-Whisper 語音推論失敗: {e}")
            if self._fallback_sensevoice:
                return self._fallback_sensevoice.transcribe(audio, language)
            return "", "auto"


def create_asr_engine(config: Optional[ASRConfig] = None):
    """Factory creating appropriate ASR engine based on configuration."""
    cfg = config or ASRConfig()
    engine_type = cfg.engine.lower()
    if engine_type in ["faster-whisper", "whisper", "faster_whisper"]:
        return FasterWhisperASR(cfg)
    return SenseVoiceASR(cfg)

