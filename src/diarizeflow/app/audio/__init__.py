"""Audio capture and VAD module."""

from diarizeflow.app.audio.devices import AudioDeviceInfo, list_audio_devices, get_default_device_indices
from diarizeflow.app.audio.capture import AudioCaptureStream, SmartAudioRouter
from diarizeflow.app.audio.vad import EnergyVADSegmenter
from diarizeflow.app.audio.segmenter import StreamingDiarizationSegmenter, SpeakerChannelBuffer
from diarizeflow.app.audio.tse import TargetSpeakerExtractor
from diarizeflow.app.audio.protocol import (
    AudioTrackTag,
    pack_audio_frame,
    unpack_audio_frame,
    is_framed_audio_packet,
)

__all__ = [
    "AudioDeviceInfo",
    "list_audio_devices",
    "get_default_device_indices",
    "AudioCaptureStream",
    "SmartAudioRouter",
    "EnergyVADSegmenter",
    "StreamingDiarizationSegmenter",
    "SpeakerChannelBuffer",
    "TargetSpeakerExtractor",
    "AudioTrackTag",
    "pack_audio_frame",
    "unpack_audio_frame",
    "is_framed_audio_packet",
]
