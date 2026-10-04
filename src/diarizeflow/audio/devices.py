"""Cross-platform Audio Device Enumeration and Loopback Detection.

Supports:
- Windows: MME, DirectSound, WASAPI Loopback (capturing system audio, games, browser).
- Linux: ALSA, PulseAudio / PipeWire Monitor sources.
"""

from dataclasses import dataclass
import platform
import sys
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class AudioDeviceInfo:
    id: Union[int, str]
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_loopback: bool = False
    device_type: str = "input"  # "microphone", "loopback", "output"

    def display_name(self) -> str:
        tag = "[系統音訊/Loopback]" if self.is_loopback else "[麥克風]"
        return f"{tag} {self.name} ({self.hostapi})"


def list_audio_devices() -> Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
    """Enumerate audio devices separated into (microphones, loopbacks).

    Returns:
        Tuple of (microphones_list, system_audio_loopbacks_list).
    """
    microphones: List[AudioDeviceInfo] = []
    loopbacks: List[AudioDeviceInfo] = []

    # 1. First attempt native WASAPI loopback detection via soundcard
    try:
        import soundcard as sc
        for m in sc.all_microphones(include_loopback=True):
            if m.isloopback:
                loopbacks.append(
                    AudioDeviceInfo(
                        id=m.name,
                        name=m.name,
                        hostapi="WASAPI Loopback",
                        max_input_channels=2,
                        max_output_channels=2,
                        default_sample_rate=48000.0,
                        is_loopback=True,
                        device_type="loopback",
                    )
                )
    except Exception:
        pass

    try:
        import sounddevice as sd
    except Exception as e:
        print(f"[!] Warning: Cannot import sounddevice: {e}", file=sys.stderr)
        return microphones, loopbacks

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"[!] Warning: Failed to query sound devices: {e}", file=sys.stderr)
        return microphones, loopbacks

    os_type = platform.system().lower()

    for idx, dev in enumerate(devices):
        hostapi_idx = dev.get("hostapi", 0)
        hostapi_name = hostapis[hostapi_idx]["name"] if hostapi_idx < len(hostapis) else "Unknown"
        name = dev.get("name", f"Device #{idx}")
        in_ch = dev.get("max_input_channels", 0)
        out_ch = dev.get("max_output_channels", 0)
        sample_rate = dev.get("default_samplerate", 16000.0)

        # Check if device is a loopback device
        is_loopback = False
        name_lower = name.lower()

        # Windows WASAPI Loopback detection
        if "windows" in os_type:
            if "wasapi" in hostapi_name.lower():
                # On Windows WASAPI, output devices can be opened in loopback mode
                if out_ch > 0 and in_ch == 0:
                    is_loopback = True
                elif "loopback" in name_lower or "stereo mix" in name_lower or "立體聲混音" in name_lower:
                    is_loopback = True

        # Linux PulseAudio / PipeWire monitor detection
        elif "linux" in os_type:
            if "monitor" in name_lower or ".monitor" in name_lower:
                is_loopback = True
            elif "pulse" in hostapi_name.lower() and "monitor" in name_lower:
                is_loopback = True

        # Generic name check
        if any(keyword in name_lower for keyword in ["stereo mix", "what u hear", "loopback", "monitor of", "立體聲混音"]):
            is_loopback = True

        if is_loopback:
            if not any(l.name == name for l in loopbacks):
                info = AudioDeviceInfo(
                    id=idx,
                    name=name,
                    hostapi=hostapi_name,
                    max_input_channels=in_ch if in_ch > 0 else 2,
                    max_output_channels=out_ch,
                    default_sample_rate=sample_rate,
                    is_loopback=True,
                    device_type="loopback",
                )
                loopbacks.append(info)
        elif in_ch > 0:
            info = AudioDeviceInfo(
                id=idx,
                name=name,
                hostapi=hostapi_name,
                max_input_channels=in_ch,
                max_output_channels=out_ch,
                default_sample_rate=sample_rate,
                is_loopback=False,
                device_type="microphone",
            )
            microphones.append(info)

    return microphones, loopbacks


def get_default_device_indices() -> Tuple[Optional[int], Optional[int]]:
    """Return default (mic_index, loopback_index)."""
    mics, loopbacks = list_audio_devices()
    default_mic = mics[0].id if mics else None
    default_loopback = loopbacks[0].id if loopbacks else None
    return default_mic, default_loopback
