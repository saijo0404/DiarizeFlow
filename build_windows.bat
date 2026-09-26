@echo off
rem ==============================================================================
rem DiarizeFlow - Windows Executable Builder (PyInstaller)
rem 在 Windows 環境中打包為免 Python 的獨立執行檔 (.exe)
rem ==============================================================================

chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================================
echo   📦 正在為 Windows 打包 DiarizeFlow 獨立執行檔 (.exe)
echo ==================================================================

rem 檢查 uv 是否可用
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] 使用 uv 執行打包...
    uv run python scripts/build_executable.py
    goto :done
)

rem 檢查 Windows 專屬虛擬環境 .venv-win
if exist ".venv-win\Scripts\python.exe" (
    echo [*] 使用 .venv-win 虛擬環境執行打包...
    ".venv-win\Scripts\python.exe" scripts/build_executable.py
    goto :done
)

rem 檢查本機虛擬環境
if exist ".venv\Scripts\python.exe" (
    echo [*] 使用本機 .venv Python 執行打包...
    ".venv\Scripts\python.exe" scripts/build_executable.py
    goto :done
)

rem 檢查系統 Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    echo [*] 使用系統 Python 執行打包...
    python scripts/build_executable.py
    goto :done
)

echo [!] 未檢測到 Python 或 uv。請在 Windows PowerShell 執行以下命令安裝 uv:
echo     irm https://astral.sh/uv/install.ps1 ^| iex
pause
exit /b 1

:done
echo [✓] 打包完成！請至 dist\DiarizeFlow\ 資料夾查看 DiarizeFlow.exe。
pause
