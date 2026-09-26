@echo off
chcp 65001 >nul
title DiarizeFlow - 即時終端偵錯監控視窗
cd /d "%~dp0"

echo ==============================================================================
echo   🚀 DiarizeFlow 即時終端偵錯模式 (Live Debug Console)
echo ==============================================================================
echo.
echo [*] 程式目錄: %~dp0
echo [*] 日誌路徑: %~dp0diarizeflow.log
echo [*] 正在啟動 DiarizeFlow.exe 並即時顯示所有音訊與翻譯 Log...
echo.
echo ------------------------------------------------------------------------------

if exist "%~dp0.venv-win\Scripts\python.exe" if exist "%~dp0scripts\run_app.py" (
    echo [*] 優先使用 .venv-win 原生 Python 啟動最新原始碼...
    "%~dp0.venv-win\Scripts\python.exe" "%~dp0scripts\run_app.py" --mode desktop
    goto :end
)

if exist "%~dp0DiarizeFlow.exe" (
    echo [*] 正在啟動 DiarizeFlow.exe...
    "%~dp0DiarizeFlow.exe"
    goto :end
) else if exist "%~dp0dist\DiarizeFlow\DiarizeFlow.exe" (
    echo [*] 正在啟動 dist\DiarizeFlow\DiarizeFlow.exe...
    "%~dp0dist\DiarizeFlow\DiarizeFlow.exe"
    goto :end
) else (
    echo [!] 找不到 DiarizeFlow.exe 或 Python 環境！請確認檔案位置。
)

:end

echo.
echo ------------------------------------------------------------------------------
echo [*] DiarizeFlow 程式已結束。
echo [*] 若發生錯誤，完整 Log 已保存在: %~dp0diarizeflow.log
echo ==============================================================================
echo.
pause
