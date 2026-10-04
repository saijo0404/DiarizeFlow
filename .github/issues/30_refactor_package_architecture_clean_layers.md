# [Refactor/Architecture]: 專案套件架構分層重整：消除 app/ 冗餘層級與消除跨層雙向依賴

**Labels**: `refactor`, `architecture`

## 1. 需求背景與問題描述 (Problem Statement)
目前專案的原始碼目錄結構在 `src/diarizeflow/` 下存在顯著的組織混亂與層級倒置：
- 頂層散落著：`calibration.py`, `export_onnx.py`, `hardware.py`, `models.py`, `patches.py`, `quantize.py`。
- 同時又有一個 `app/` 子套件，裡面包裝著 `audio/`, `backend/`, `frontend/`, `config.py`, `launcher.py`。

### 核心架構缺陷：
1. **跨層級雙向依賴（Cyclic / Layer Inversion Dependency）**：
   - 頂層工具 `src/diarizeflow/calibration.py` 匯入了內部子模組 `diarizeflow.app.config.AppConfig`。
   - 子模組 `src/diarizeflow/app/launcher.py` 又匯入了頂層的 `diarizeflow.calibration.ensure_calibrated_models`。
   - 違反了清晰架構（Clean Architecture）單向依賴的原則。
2. **語意層級冗餘**：
   - 全專案本質上就是一個即時語者分離翻譯應用程式，多包一層 `app/` 導致匯入路徑冗長（如 `diarizeflow.app.backend.server`）。
3. **模型與工具職責割裂**：
   - `models.py`（模型下載）在頂層，但 `devices.py`（設備探測）卻在 `app/audio/`。

---

## 2. 建議解決方案 (Proposed Solution)
重整專案目錄為乾淨的四大分層核心子包，徹底消除 `app/` 冗餘層級：

```text
src/diarizeflow/
├── core/         # 底層運算核心：硬體感知、模型量化、ONNX 匯出與修補 (hardware, quantize, export, patches)
├── audio/        # 音訊工程領域：採集、DSP 路由、SAD、TSE、重採樣 (capture, router, segmenter, tse, protocol)
├── engine/       # AI 業務服務：ASR、Diarizer、Translator、聲紋資料庫、FastAPI (asr, diarizer, translator, voiceprint, server)
├── ui/           # 桌面圖形介面：PySide6 HUD 懸浮窗、卡片、對話框、托盤 (overlay, cards, settings, tray, widgets)
├── cli/          # 命令列與啟動器：桌面啟動器、模型下載 CLI、匯出 CLI (launcher, downloader)
└── config.py     # 全域統一配置中心 (作為領域基底，依賴朝向此模組)
```

### 具體步驟：
1. 移動 `app/backend/` 下的服務至 `engine/`。
2. 移動 `app/frontend/` 下的元件至 `ui/`。
3. 移動 `export_onnx.py`, `quantize.py`, `patches.py`, `hardware.py` 至 `core/`。
4. 移動 `launcher.py` 與 `models.py` 的 CLI 入口至 `cli/`。
5. 更新全專案的匯入語句（`import`）與 `pyproject.toml` 的 entrypoints。

---

## 3. 預期效益 (Expected Benefits)
- 徹底消除模組間跨層雙向依賴風險。
- 套件結構符合現代開源專案慣例，大幅降低新進開發者的理解負擔。
- 匯入路徑簡潔優雅（例如 `from diarizeflow.engine import DiarizeFlowPipeline`）。
