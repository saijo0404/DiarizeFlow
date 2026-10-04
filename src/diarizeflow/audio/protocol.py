"""Binary Frame Protocol for DiarizeFlow Audio Streaming (/ws/audio).

Defines a lightweight 8-byte metadata header preceding raw float32 PCM audio:
- Byte 0: Magic byte (0xDF)
- Byte 1: Protocol Version (0x01)
- Byte 2: Source Track Tag (0x01 = Mic, 0x02 = Loopback, 0x03 = Mixed)
- Byte 3: Flags / Reserved (0x00 = standard float32 mono)
- Byte 4-5: Sample Rate (uint16 big-endian, e.g. 16000)
- Byte 6-7: Channels (uint16 big-endian, e.g. 1)
- Bytes 8+: Raw float32 PCM payload

Provides:
- pack_audio_frame: Pack numpy float32 chunk with 8-byte metadata header.
- unpack_audio_frame: Unpack binary frame, with automatic fallback to legacy raw PCM.
- is_framed_audio_packet: Quick magic bytes check.
- AudioTrackTag: Enumeration of track sources.
"""

from enum import IntEnum
import struct
from typing import Optional, Tuple, Union
import numpy as np


class AudioTrackTag(IntEnum):
    """Audio source track tags for binary framing."""
    MIC = 0x01        # Microphone (Track A)
    LOOPBACK = 0x02   # System / Loopback (Track B)
    MIXED = 0x03      # Mixed / Multi-track blended

    @classmethod
    def to_string(cls, tag: Union[int, "AudioTrackTag"]) -> str:
        """Convert tag enum/int to canonical track name string."""
        return TAG_TO_TRACK_NAME.get(int(tag), "mixed")

    @classmethod
    def from_string(cls, name: str) -> "AudioTrackTag":
        """Convert string name to AudioTrackTag enum."""
        cleaned = str(name).strip().lower()
        return TRACK_NAME_TO_TAG.get(cleaned, cls.MIXED)


MAGIC_BYTE = 0xDF
PROTOCOL_VERSION = 0x01
HEADER_FORMAT = ">BBBBHH"
HEADER_SIZE = 8

# Aliases for explicit naming
AUDIO_FRAME_HEADER_MAGIC = MAGIC_BYTE
AUDIO_FRAME_VERSION = PROTOCOL_VERSION
AUDIO_FRAME_HEADER_SIZE = HEADER_SIZE

TAG_TO_TRACK_NAME = {
    AudioTrackTag.MIC: "mic",
    AudioTrackTag.LOOPBACK: "loopback",
    AudioTrackTag.MIXED: "mixed",
}

TRACK_NAME_TO_TAG = {
    "mic": AudioTrackTag.MIC,
    "microphone": AudioTrackTag.MIC,
    "loopback": AudioTrackTag.LOOPBACK,
    "system": AudioTrackTag.LOOPBACK,
    "mixed": AudioTrackTag.MIXED,
    "both": AudioTrackTag.MIXED,
    "silence": AudioTrackTag.MIXED,
}


def pack_audio_frame(
    chunk: np.ndarray,
    track: Union[str, int, AudioTrackTag] = AudioTrackTag.MIXED,
    sample_rate: int = 16000,
    channels: int = 1,
    flags: int = 0,
) -> bytes:
    """Pack float32 audio chunk with 8-byte metadata header.

    Args:
        chunk: 1D or 2D numpy audio array (will be coerced to float32).
        track: Source track tag ('mic', 'loopback', 'mixed', or AudioTrackTag).
        sample_rate: Audio sample rate in Hz (default: 16000).
        channels: Channel count (default: 1).
        flags: Reserved bitflags (default: 0).

    Returns:
        bytes: 8-byte header followed by raw float32 PCM data.
    """
    if isinstance(track, str):
        tag_val = TRACK_NAME_TO_TAG.get(track.lower(), AudioTrackTag.MIXED)
    elif isinstance(track, AudioTrackTag):
        tag_val = track.value
    else:
        tag_val = int(track) if int(track) in (1, 2, 3) else AudioTrackTag.MIXED

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_BYTE,
        PROTOCOL_VERSION,
        tag_val,
        flags & 0xFF,
        int(sample_rate) & 0xFFFF,
        int(channels) & 0xFFFF,
    )
    payload = chunk.astype(np.float32, copy=False).tobytes()
    return header + payload


def is_framed_audio_packet(data: bytes) -> bool:
    """Check whether binary data starts with valid DiarizeFlow frame header."""
    if len(data) < HEADER_SIZE:
        return False
    return data[0] == MAGIC_BYTE and data[1] == PROTOCOL_VERSION


def unpack_audio_frame(
    data: bytes,
    default_sample_rate: int = 16000,
    default_channels: int = 1,
) -> Tuple[np.ndarray, str, int, int]:
    """Unpack audio frame bytes into audio array and metadata.

    Gracefully falls back to legacy raw float32 PCM decoding if the 8-byte header is absent.

    Args:
        data: Received WebSocket binary payload.
        default_sample_rate: Default sample rate if fallback decoding is used.
        default_channels: Default channels if fallback decoding is used.

    Returns:
        Tuple of (chunk_float32, track_name, sample_rate, channels).
    """
    if not data:
        return np.array([], dtype=np.float32), "mixed", default_sample_rate, default_channels

    # Check for valid DiarizeFlow framing
    if is_framed_audio_packet(data):
        try:
            magic, ver, tag_val, flags, sample_rate, channels = struct.unpack(
                HEADER_FORMAT, data[:HEADER_SIZE]
            )
            # Validate tag_val and basic sanity
            if tag_val in (1, 2, 3) and sample_rate > 0:
                track_name = TAG_TO_TRACK_NAME.get(tag_val, "mixed")
                payload = data[HEADER_SIZE:]
                valid_len = (len(payload) // 4) * 4
                chunk = np.frombuffer(payload[:valid_len], dtype=np.float32) if valid_len > 0 else np.array([], dtype=np.float32)

                # Defensive multichannel flattening if needed
                if channels > 1 and len(chunk) >= channels and len(chunk) % channels == 0:
                    chunk = chunk.reshape(-1, channels).mean(axis=1).astype(np.float32)

                sr = sample_rate if sample_rate > 0 else default_sample_rate
                ch = channels if channels > 0 else default_channels
                return chunk, track_name, sr, ch
        except Exception:
            # Fall back to raw decoding on any corrupt header
            pass

    # Fallback to legacy raw float32 PCM decoding
    valid_len = (len(data) // 4) * 4
    chunk = np.frombuffer(data[:valid_len], dtype=np.float32) if valid_len > 0 else np.array([], dtype=np.float32)
    return chunk, "mixed", default_sample_rate, default_channels
