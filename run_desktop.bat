@echo off
rem ==============================================================================
rem DiarizeFlow - Windows Native Qt Transparent Overlay Launcher
rem 跨平台 (Windows) 原生 Qt 桌面透明飄浮字幕一鍵啟動
rem ==============================================================================

chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================================
echo   🚀 啟動 DiarizeFlow 原生桌面飄浮視窗 (Windows Qt Overlay)
echo ==================================================================

rem 優先檢查 Windows 專屬虛擬環境 .venv-win
if exist ".venv-win\Scripts\python.exe" (
    echo [*] 使用 .venv-win 虛擬環境啟動中...
    ".venv-win\Scripts\python.exe" scripts/run_app.py --mode desktop %*
    goto :end
)

rem 檢查是否安裝 uv
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] 使用 uv 環境啟動中...
    uv run python scripts/run_app.py --mode desktop %*
    goto :end
)

rem 檢查本機虛擬環境 .venv
if exist ".venv\Scripts\python.exe" (
    echo [*] 使用 .venv 虛擬環境啟動中...
    ".venv\Scripts\python.exe" scripts/run_app.py --mode desktop %*
    goto :end
)

rem 嘗試系統 Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] 使用系統 Python 啟動中...
    python scripts/run_app.py --mode desktop %*
    goto :end
)

echo [!] 錯誤: 未找到 Python 或 uv，請確認已安裝 Python 3.10+ 或 uv。
pause

:end
