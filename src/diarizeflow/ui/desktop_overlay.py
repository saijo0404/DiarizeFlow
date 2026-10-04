"""Backward-compatibility proxy for DiarizeFlow desktop overlay frontend.

This module re-exports components decomposed into:
- widgets: SPEAKER_COLORS, get_speaker_color, create_app_icon, format_vu_level
- cards: SpeakerBadge, SubtitleCardWidget
- settings_dialog: SettingsDialog
- overlay_window: TransparentSubtitleOverlay, run_overlay_app, run_cli
"""

from diarizeflow.ui.cards import (
    SpeakerBadge,
    SubtitleCardWidget,
)
from diarizeflow.ui.network import (
    fetch_remote_backend_config,
    run_audio_client,
    run_subtitles_client,
    send_remote_delete_api,
    send_remote_rename_api,
    sync_config_to_remote_backend,
)
from diarizeflow.ui.overlay_window import (
    TransparentSubtitleOverlay,
    run_cli,
    run_overlay_app,
)
from diarizeflow.ui.settings_dialog import SettingsDialog
from diarizeflow.ui.widgets import (
    SPEAKER_COLORS,
    create_app_icon,
    format_vu_level,
    get_speaker_color,
)

__all__ = [
    "SPEAKER_COLORS",
    "get_speaker_color",
    "create_app_icon",
    "format_vu_level",
    "SpeakerBadge",
    "SubtitleCardWidget",
    "SettingsDialog",
    "TransparentSubtitleOverlay",
    "run_overlay_app",
    "run_cli",
    "sync_config_to_remote_backend",
    "fetch_remote_backend_config",
    "send_remote_rename_api",
    "send_remote_delete_api",
    "run_subtitles_client",
    "run_audio_client",
]


if __name__ == "__main__":
    run_cli()
