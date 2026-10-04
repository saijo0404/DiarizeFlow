"""Audio engineering domain: capture, DSP routing, VAD/SAD, TSE, AGC and wire protocol."""

from diarizeflow.audio.devices import AudioDeviceInfo, list_audio_devices, get_default_device_indices
from diarizeflow.audio.capture import AudioCaptureStream, SmartAudioRouter
from diarizeflow.audio.vad import EnergyVADSegmenter
from diarizeflow.audio.segmenter import StreamingDiarizationSegmenter, SpeakerChannelBuffer
from diarizeflow.audio.tse import TargetSpeakerExtractor
from diarizeflow.audio.protocol import (
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
