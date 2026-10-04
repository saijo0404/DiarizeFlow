"""DiarizeFlow frontend package."""

from diarizeflow.app.frontend.cards import (
    SpeakerBadge,
    SubtitleCardWidget,
)
from diarizeflow.app.frontend.display_server import (
    DisplayServerInfo,
    detect_display_server,
    get_clickthrough_balloon_message,
    get_clickthrough_tooltip,
    get_clickthrough_tray_title,
    is_wayland_session,
    print_display_server_guidance,
)
from diarizeflow.app.frontend.overlay_window import (
    TransparentSubtitleOverlay,
    run_cli,
    run_overlay_app,
)
from diarizeflow.app.frontend.settings_dialog import SettingsDialog
from diarizeflow.app.frontend.widgets import (
    SPEAKER_COLORS,
    create_app_icon,
    format_vu_level,
    get_speaker_color,
)

__all__ = [
    "TransparentSubtitleOverlay",
    "run_overlay_app",
    "run_cli",
    "SubtitleCardWidget",
    "SpeakerBadge",
    "SettingsDialog",
    "get_speaker_color",
    "create_app_icon",
    "format_vu_level",
    "SPEAKER_COLORS",
    "DisplayServerInfo",
    "detect_display_server",
    "is_wayland_session",
    "print_display_server_guidance",
    "get_clickthrough_tooltip",
    "get_clickthrough_tray_title",
    "get_clickthrough_balloon_message",
]
