"""Display server environment detection and Wayland guidance utilities.

Provides:
- detect_display_server: Detects Linux Wayland, X11, Windows, macOS display server environments.
- DisplayServerInfo: Dataclass containing display protocol, desktop environment, and hotkey support status.
- is_wayland_session: Boolean helper to check for Wayland compositors.
- print_display_server_guidance: Emits friendly terminal banners guiding user on Wayland restrictions.
- get_clickthrough_tooltip: Dynamic UI tooltip based on display server and click-through state.
- get_clickthrough_tray_title: Tray menu action title reflecting platform capabilities.
- get_clickthrough_balloon_message: System tray balloon notification text.
"""

from dataclasses import dataclass
import os
import sys
from typing import Mapping, Optional


@dataclass(frozen=True)
class DisplayServerInfo:
    """Detailed metadata about the host operating system's display server."""
    platform: str
    session_type: str
    is_wayland: bool
    is_x11: bool
    wayland_display: str
    desktop_environment: str
    hotkeys_supported: bool
    guidance_message: Optional[str] = None


_GUIDANCE_PRINTED: bool = False


def detect_display_server(
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> DisplayServerInfo:
    """Detect current display server protocol and compositor capabilities.

    Args:
        environ: Optional environment dictionary override (defaults to os.environ).
        platform: Optional platform string override (defaults to sys.platform).

    Returns:
        DisplayServerInfo instance with detected attributes.
    """
    env = environ if environ is not None else os.environ
    plat = platform if platform is not None else sys.platform

    if plat == "win32":
        return DisplayServerInfo(
            platform="win32",
            session_type="win32",
            is_wayland=False,
            is_x11=False,
            wayland_display="",
            desktop_environment="Windows",
            hotkeys_supported=True,
            guidance_message=None,
        )

    if plat == "darwin":
        return DisplayServerInfo(
            platform="darwin",
            session_type="cocoa",
            is_wayland=False,
            is_x11=False,
            wayland_display="",
            desktop_environment="macOS",
            hotkeys_supported=True,
            guidance_message=None,
        )

    # Linux and Unix-like platforms
    session_type_raw = env.get("XDG_SESSION_TYPE", "").strip().lower()
    wayland_display = env.get("WAYLAND_DISPLAY", "").strip()
    x11_display = env.get("DISPLAY", "").strip()

    is_wayland = (session_type_raw == "wayland") or bool(wayland_display)
    is_x11 = not is_wayland and ((session_type_raw == "x11") or bool(x11_display))

    desktop = (
        env.get("XDG_CURRENT_DESKTOP")
        or env.get("DESKTOP_SESSION")
        or env.get("XDG_SESSION_DESKTOP")
        or "Unknown"
    ).strip()

    if is_wayland:
        resolved_session = "wayland"
        hotkeys_supported = False
        guidance = (
            "[*] 檢測到系統運行於 Linux Wayland 顯示環境。\n"
            "    提示：Wayland 安全機制限制全域鍵盤熱鍵攔截。\n"
            "    若需切換滑鼠穿透或控制字幕，請優先使用桌面系統托盤選單 (System Tray) 或頂部控制列操作。"
        )
    elif is_x11:
        resolved_session = "x11"
        hotkeys_supported = True
        guidance = None
    else:
        resolved_session = session_type_raw or "unknown"
        hotkeys_supported = True
        guidance = None

    return DisplayServerInfo(
        platform=plat,
        session_type=resolved_session,
        is_wayland=is_wayland,
        is_x11=is_x11,
        wayland_display=wayland_display,
        desktop_environment=desktop,
        hotkeys_supported=hotkeys_supported,
        guidance_message=guidance,
    )


def is_wayland_session(
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
) -> bool:
    """Return True if running in a Linux Wayland session."""
    return detect_display_server(environ=environ, platform=platform).is_wayland


def print_display_server_guidance(
    environ: Optional[Mapping[str, str]] = None,
    platform: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Print friendly guidance banner if running under Linux Wayland.

    Args:
        environ: Optional environment dictionary override.
        platform: Optional platform string override.
        force: If True, prints even if already printed previously.

    Returns:
        True if guidance message was printed, False otherwise.
    """
    global _GUIDANCE_PRINTED
    info = detect_display_server(environ=environ, platform=platform)
    if info.is_wayland and info.guidance_message and (force or not _GUIDANCE_PRINTED):
        print(info.guidance_message)
        _GUIDANCE_PRINTED = True
        return True
    return False


def get_clickthrough_tooltip(
    is_clickthrough: bool,
    is_wayland: Optional[bool] = None,
) -> str:
    """Return appropriate tooltip text for click-through button based on Wayland state."""
    wayland = is_wayland if is_wayland is not None else is_wayland_session()
    if wayland:
        if is_clickthrough:
            return "滑鼠穿透已開啟（在 Wayland 環境下請使用系統托盤解除穿透）"
        return "開啟/關閉滑鼠穿透（在 Wayland 環境下全域熱鍵受限，請使用此按鈕或系統托盤操作）"

    if is_clickthrough:
        return "滑鼠穿透已開啟（按 Alt+Shift+H 或托盤關閉）"
    return "開啟/關閉滑鼠穿透（快捷鍵: Alt+Shift+H）"


def get_clickthrough_tray_title(is_wayland: Optional[bool] = None) -> str:
    """Return system tray context menu action title for click-through."""
    wayland = is_wayland if is_wayland is not None else is_wayland_session()
    if wayland:
        return "🛡️ 切換滑鼠穿透 (Alt+Shift+H / Wayland 推薦)"
    return "🛡️ 切換滑鼠穿透 (Alt+Shift+H)"


def get_clickthrough_balloon_message(is_wayland: Optional[bool] = None) -> str:
    """Return system tray balloon notification message when click-through is enabled."""
    wayland = is_wayland if is_wayland is not None else is_wayland_session()
    if wayland:
        return (
            "🛡️ 已開啟滑鼠穿透模式！\n"
            "在 Wayland 環境下全域熱鍵受限，隨時右鍵點擊右下角系統托盤即可解除穿透。"
        )
    return (
        "🛡️ 已開啟滑鼠穿透模式！\n"
        "隨時按 Alt+Shift+H 或右鍵點擊右下角系統托盤即可解除穿透。"
    )
