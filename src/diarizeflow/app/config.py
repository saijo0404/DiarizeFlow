"""Configuration models and loader for DiarizeFlow Real-time Translation App."""

from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import sys
from typing import Optional, Dict, Any


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 250  # milliseconds per streaming chunk
    mic_device: Optional[int] = None
    loopback_device: Optional[int] = None
    mix_sources: bool = True
    gain: float = 1.0
    agc_enabled: bool = True  # 自動音量調節 (AGC): 太大聲自動縮小、太小聲自動放大
    agc_target_rms: float = 0.06  # ASR 最佳黃金輸入能量 (-24 dBFS)
    agc_max_gain: float = 25.0  # 最大放大倍數 (+28 dB, 即使音量 0.006 也能自動拉高至人聲檢測區間)
    agc_min_gain: float = 0.15  # 最小縮小倍數 (-16.5 dB, 抑制爆音防破音)


@dataclass
class VADConfig:
    enabled: bool = True
    energy_threshold: float = 0.008  # 靈敏度門檻 (由 0.02 下調至 0.008，結合動態環境底噪自適應)
    min_speech_ms: int = 250
    silence_timeout_ms: int = 300   # 靜音停頓判定間隔 (由 400ms 下調至 300ms，換句更明快)
    max_speech_s: float = 3.5       # 音訊累積上限 (由 8.5s 縮短至 3.5s，字句快速彈出且大幅減少多講者混雜)


@dataclass
class DiarizationConfig:
    enabled: bool = True
    model_path: str = "models/nemotron_diarization/Nemotron-3-Diarization.onnx"
    fp16_model_path: str = "models/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx"
    use_fp16: bool = True
    streaming_mode: str = "low_latency"  # "low_latency" (1.04s, 官方預設), "very_low_latency" (0.64s), "ultra_low_latency" (0.32s), "offline" (30.4s)
    speaker_threshold: float = 0.82
    max_speakers: int = 8


@dataclass
class ASRConfig:
    engine: str = "sensevoice"  # "sensevoice", "faster-whisper"
    model_dir: str = "models/sensevoice_small"
    model_path: str = "models/sensevoice_small/SenseVoiceSmall.onnx"
    cmvn_file: str = "models/sensevoice_small/am.mvn"
    bpe_model: str = "models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model"
    language: str = "auto"  # "auto", "zh", "en", "ja", "ko", "yue"
    use_gpu: bool = True
    whisper_model: str = "models/faster-whisper-large-v2"
    whisper_precision: str = "float16"  # "float32", "float16", "int8", "fp8", "w4a16", "nvfp4", "mxfp4"
    beam_size: int = 1


@dataclass
class LLMConfig:
    provider: str = "vllm"  # "vllm", "llama.cpp", "openai", "claude", "ollama", "bypass"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "auto"  # "auto" will dynamically discover model from /v1/models
    target_language: str = "繁體中文"  # "繁體中文", "English", "日本語", "한국어", "簡體中文"
    temperature: float = 0.1
    max_tokens: int = 512
    system_prompt: str = (
        "你是一位高水準的即時語音字幕翻譯專家。請將發話內容直接翻譯成【{target_language}】。\n"
        "重要規則：\n"
        "1. 輸入文本來自即時語音辨識（ASR），可能包含語音同音錯字、非標準口語、無標點、截斷尾音或外來語音譯。\n"
        "2. 請根據整句話的語意脈絡與生活常理，自動理解真實含意並轉化為通順道地的表達，嚴禁生硬音譯無關事物。\n"
        "3. 嚴禁無中生有之過度腦補，遇到不完整或模糊的語句，以符合上下文的最簡明自然方式收尾即可。\n"
        "4. 僅輸出翻譯結果，嚴禁包含思考過程、引號、拼音或任何額外說明。"
    )


@dataclass
class UIConfig:
    fade_out_seconds: float = 5.0
    font_size: int = 22
    opacity: float = 0.50
    always_on_top: bool = True
    click_through: bool = False
    show_original: bool = True
    accent_color: str = "#38bdf8"
    window_width: int = 760
    window_height: int = 180


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    ws_endpoint: str = "/ws/audio"
    subtitles_endpoint: str = "/ws/subtitles"



def get_config_path(filename: str = "config.json") -> Path:
    """Return persistent, writable path for configuration file."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        return exe_dir / filename
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        return project_root / filename
    except Exception:
        return Path(filename).resolve()


def resolve_app_path(path_str: str | Path) -> Path:
    """Resolve a relative resource path across dev, installed package, and PyInstaller environments."""
    if not path_str:
        return Path(".")
    path_str = os.path.expanduser(str(path_str))
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    # 1. Check PyInstaller bundle directory
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        cand = exe_dir / p
        if cand.exists():
            return cand
        cand = exe_dir / "_internal" / p
        if cand.exists():
            return cand
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cand = Path(meipass) / p
            if cand.exists():
                return cand

    # 2. Check current working directory
    if p.exists():
        return p.resolve()

    # 3. Check project root
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        cand = project_root / p
        if cand.exists():
            return cand
    except Exception:
        pass

    return p


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    hardware_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        diar_dict = dict(data.get("diarization", {}))
        if diar_dict.get("speaker_threshold", 0) < 0.5:
            diar_dict["speaker_threshold"] = 0.82
        ui_dict = dict(data.get("ui", {}))
        if ui_dict.get("opacity", 0.5) > 0.85:
            ui_dict["opacity"] = 0.50
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            vad=VADConfig(**data.get("vad", {})),
            diarization=DiarizationConfig(**diar_dict),
            asr=ASRConfig(**data.get("asr", {})),
            llm=LLMConfig(**data.get("llm", {})),
            ui=UIConfig(**ui_dict),
            server=ServerConfig(**data.get("server", {})),
            hardware_calibrated=data.get("hardware_calibrated", False),
        )

    def save(self, filepath: Optional[str] = None) -> None:
        """Save configuration persistently to disk."""
        if filepath is None or filepath == "config.json":
            path = get_config_path("config.json")
        else:
            path = Path(filepath)
            if not path.is_absolute():
                path = get_config_path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[✓] 設定已永久儲存至: {path}")

    @classmethod
    def load(cls, filepath: str = "config.json") -> "AppConfig":
        """Load configuration, prioritizing persistent user config next to executable."""
        persistent_path = get_config_path(filepath)
        if persistent_path.exists():
            try:
                with open(persistent_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                print(f"[!] Warning reading persistent config {persistent_path}: {e}")

        path = resolve_app_path(filepath)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                print(f"[!] Warning reading bundled config {path}: {e}")

        default_config = cls()
        default_config.save(filepath)
        return default_config

