# [Architecture/Refactor]: 統一應用程式啟動入口邏輯：將 DualLogger、AttachConsole 與硬體自適應校準下沉至 launcher.py

**Labels**: `architecture`, `refactor`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
目前專案存在兩套平行的應用程式啟動入口：
1. 套件原生入口：`src/diarizeflow/app/launcher.py`（在 `pyproject.toml` 中註冊為 CLI 指令 `diarizeflow-app`）。
2. 根目錄腳本：`scripts/run_app.py`（在 `run_desktop.bat` 與 PyInstaller 打包時作為入口）。

### 根因分析 (Root Cause Analysis)：
兩者雖然都負責初始化後端伺服器與啟動 PySide6 HUD，但代碼邏輯嚴重分歧：
- `scripts/run_app.py` 實作了關鍵的防護與輔助功能：
  - `DualLogger`：雙向輸出至終端與 `diarizeflow.log`，並具備 5MB 檔案輪轉機制，防止日誌無限膨脹。
  - Windows `AttachConsole`：在 Windows GUI 模式下能自動掛載父終端以輸出 log。
  - 全域未捕獲異常處理 (`sys.excepthook = handle_exception`)。
  - **首次啟動硬體自適應校準**：呼叫 `ensure_calibrated_models(cfg)`，自動辨識 GPU/CPU 並生成最佳化 FP16/INT8 模型。
- 但 `launcher.py` 完全遺漏了上述所有功能！
- 如果使用者透過 `pip install` 安裝後執行 `diarizeflow-app`，將**無法觸發硬體自動量化校準**，亦沒有 `diarizeflow.log` 輪轉紀錄，導致 CLI 與腳本啟動行為不一致。

---

## 2. 建議解決方案 (Proposed Solution)

將核心啟動保護邏輯下沉至 `src/diarizeflow/app/launcher.py`，使之成為單一可信來源（Single Source of Truth）：

1. **下沉基礎設施到 `launcher.py`**：
   - 將 `DualLogger`、`init_logging()`、`AttachConsole` 與全域例外捕捉整合至 `diarizeflow.app.launcher`（或抽取為 `diarizeflow.app.logging`）。
   - 在 `launcher.run_cli()` 中於加載配置後立即執行 `ensure_calibrated_models(cfg)`。
2. **簡化 `scripts/run_app.py`**：
   - 將 `scripts/run_app.py` 簡化為薄包裝（Thin Wrapper）：
     ```python
     #!/usr/bin/env python3
     import sys
     from diarizeflow.app.launcher import run_cli

     if __name__ == "__main__":
         sys.exit(run_cli())
     ```
3. **保持 PyInstaller 打包與 CLI 指令行為 100% 對齊**：
   - 無論是執行 `uv run python scripts/run_app.py`、雙擊 `.exe` 或執行 `diarizeflow-app`，均走過同一套校準與日誌機制。

---

## 3. 預期效益 (Expected Benefits)
- 消除代碼重複（DRY 原則），避免日後修改啟動參數時遺漏其中一個入口。
- 保證所有啟動途徑皆具備硬體自適應校準與穩定可靠的日誌輸出。
