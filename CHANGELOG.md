# DiarizeFlow 更新日誌 (Changelog)

所有關於 DiarizeFlow 專案的顯著變更皆記錄於本文件中。
本專案遵循 [Semantic Versioning (語意化版本 2.0.0)](https://semver.org/lang/zh-TW/) 規範。

---

## [Unreleased]

### ♻️ 重構 (Refactor)

- **專案套件架構分層重整** ([#64](https://github.com/saijo0404/DiarizeFlow/issues/64))
  - 移除冗餘的 `app/` 層級，將 `src/diarizeflow/` 重整為 `config` / `core` / `audio` / `engine` / `ui` / `cli` 分層：
    - `app/backend/*` → `engine/`、`app/frontend/*` → `ui/`、`app/audio/*` → `audio/`、`app/config.py` → `config.py`
    - `export_onnx` / `quantize` / `patches` / `hardware` / `calibration` → `core/`
    - `app/launcher.py` → `cli/launcher.py`、`models.py` → `cli/downloader.py`
  - 消除 `calibration` ↔ `launcher` 跨層雙向依賴，依賴方向嚴格為 `cli → ui/engine → audio → config`、`cli → core → config`。
  - 同步更新 `pyproject.toml` entrypoints、scripts、測試與 README；新增 `tests/test_package_layering.py` 以 AST 自動驗證分層與無循環依賴。

---

## [2.0.0] - 2026-10-04

### 🚀 里程碑概述 (Milestone Overview)
DiarizeFlow 迎來重大架構躍進！自 v1.0.0 以來歷經 62 個 Commits 與 29 個 Issues 的深度打磨，專案已由初期的單體原型全面升級為**工業級即時語音分離、辨識與翻譯微核心架構**。單元與整合測試規模自 77 題擴增至 **254 題全數通過 (100% Passed)**，技術債與 DeprecationWarning 歸零。

---

### 🌟 重大特性 (Major Features)

1. **目標講者語音提取 (Target-Speaker Extraction / TSE)** ([#54](https://github.com/saijo0404/DiarizeFlow/issues/54))
   - 解決單收音音軌中「多人同步重疊說話 (Overlap / Cross-talk)」波形無法分離的業界難題。
   - 復用 Sortformer 512 維聲紋嵌入作為條件提示向量，結合時頻遮罩 (Time-Frequency Masking) 重建目標講者的乾淨單一人聲波形，使下游 ASR 辨識錯誤率 (WER) 大幅下降。
   - 採動態按需啟動機制：單人發話時零額外算力開銷，多講者重疊時即時介入分軌。

2. **SmartAudioRouter 雙軌智慧動態路由與串音抑制** ([#30](https://github.com/saijo0404/DiarizeFlow/issues/30))
   - 獨立排程本地麥克風 (Track A) 與系統音訊 Loopback (Track B)。
   - 結合動態能量門控、聲學回音串音抑制 (Bleed Suppression) 與跨 Chunk 平滑 Cross-fading，徹底消除爆音與時間膨脹。

3. **永久聲紋資料庫與自訂講者身份追蹤 (Voiceprint Database)** ([#11](https://github.com/saijo0404/DiarizeFlow/issues/11))
   - 支援將暫時講者命名並釘選至磁碟，提供自適應聲線更新與黃金錨點保護 (Anti-Drift)。
   - 即時將講者標籤與專屬色系同步至 HUD 卡片與遠端客戶端。

4. **串流神經 SAD 多軌分流 (Streaming Sortformer SAD)** ([#13](https://github.com/saijo0404/DiarizeFlow/issues/13))
   - 全面替換傳統能量 VAD，由 NVIDIA Nemotron-3 (Sortformer) 8 通道發話後驗機率驅動時間軸切分。
   - 支援語者邊界預編碼嵌入單次推論復用 ([#31](https://github.com/saijo0404/DiarizeFlow/issues/31))，消除重複 ONNX 前向傳播。

5. **輕量化一鍵模型下載與初次引導工具** ([#51](https://github.com/saijo0404/DiarizeFlow/issues/51))
   - 新增 `scripts/download_models.py`，支援自動偵測硬體並自 Hugging Face Hub / ModelScope 一鍵下載預轉換的 ONNX 模型、CMVN 均值方差表與 Tokenizer 模型。

6. **全新模組化桌面 HUD 懸浮視窗** ([#36](https://github.com/saijo0404/DiarizeFlow/issues/36))
   - 拆解龐大模組為 `cards.py`、`settings_dialog.py`、`widgets.py`、`network.py` 與 `overlay_window.py`。
   - 滑鼠穿透模式三重防死鎖安全保護：系統托盤選單 (`QSystemTrayIcon`)、全域快捷鍵 (`Alt+Shift+H` / `Ctrl+Shift+T`) 與頂部控制列獨立事件保護 ([#26](https://github.com/saijo0404/DiarizeFlow/issues/26))。
   - 視窗幾何記憶與多螢幕防跑版邊界保護 ([#34](https://github.com/saijo0404/DiarizeFlow/issues/34))。

7. **Linux Wayland 顯示環境深度適配** ([#53](https://github.com/saijo0404/DiarizeFlow/issues/53))
   - 啟動時自動感知 `XDG_SESSION_TYPE`，在 Wayland 環境下主動指引托盤與視窗控制操作。

8. **音訊串流協定擴充二進位標頭 (Binary Frame Protocol)** ([#52](https://github.com/saijo0404/DiarizeFlow/issues/52))
   - `/ws/audio` 支援 8-byte 二進位 Header (包含 Magic byte、版本握手、音軌來源標記與採樣率資訊)，並對舊版客戶端 100% 向下相容。

---

### ⚡ 效能與後端架構改進 (Performance & Architecture)

- **FastAPI Lifespan 現代化升級**：將 `server.py` 舊式 `startup`/`shutdown` 遷移至現代標準 `@asynccontextmanager async def lifespan`，全套測試告警歸零 ([#48](https://github.com/saijo0404/DiarizeFlow/issues/48))。
- **Mel Spectrogram 濾波矩陣與窗函數全快取**：預先快取 `mel_basis` 與 Hamming 窗，消除串流循環中重複構建開銷 ([#49](https://github.com/saijo0404/DiarizeFlow/issues/49))。
- **LLM 翻譯連線池化與並行化**：改用持久化 `aiohttp.ClientSession` 連線池與 `/models` 快取，請求延遲降低 50~150ms ([#15](https://github.com/saijo0404/DiarizeFlow/issues/15), [#33](https://github.com/saijo0404/DiarizeFlow/issues/33))。
- **前後端分離狀態同步**：前端設定變更透過非同步 REST API `POST /api/config` 即時推送到後端伺服器 ([#32](https://github.com/saijo0404/DiarizeFlow/issues/32))。
- **依賴輕量化解耦**：將 `nemo-toolkit` 與 `onnxsim` 移至 `[project.optional-dependencies] export`，普通使用者安裝體積縮減 80% 以上 ([#35](https://github.com/saijo0404/DiarizeFlow/issues/35))。
- **移除舊式 Web 前端**：精簡程式庫，全力深耕原生高效能 PySide6 懸浮 HUD ([#16](https://github.com/saijo0404/DiarizeFlow/issues/16))。

---

### 🛠️ 漏洞修復與強固性提升 (Bug Fixes & Robustness)

- **修復純 uv 環境打包失敗問題**：處理 `.venv` 缺失 `pip` 模組導致的自動安裝 PyInstaller 崩潰，並在 `pyproject.toml` 宣告 `build` 可選依賴群組 ([#57](https://github.com/saijo0404/DiarizeFlow/issues/57))。
- **修復音訊重採樣與混音漂移**：改採多相多項式重採樣 (`resample_poly`) 與滑動視窗時鐘漂移補償 ([#1](https://github.com/saijo0404/DiarizeFlow/issues/1), [#4](https://github.com/saijo0404/DiarizeFlow/issues/4))。
- **修復 WebSocket 事件處理遺漏**：過濾 `speaker_deleted` 等控制訊息，避免 HUD 渲染空白卡片 ([#28](https://github.com/saijo0404/DiarizeFlow/issues/28))。
- **配置反序列化安全防呆**：增強 `AppConfig.from_dict` 欄位過濾與檔案損毀安全復原 ([#2](https://github.com/saijo0404/DiarizeFlow/issues/2), [#29](https://github.com/saijo0404/DiarizeFlow/issues/29))。
- **補充執行期缺失依賴**：補齊 `soundcard`、`faster-whisper`、`ml-dtypes`、`sentencepiece`、`kaldi-native-fbank` 等執行期套件 ([#3](https://github.com/saijo0404/DiarizeFlow/issues/3))。

---

## [1.0.0] - 2026-09-25
- 初版釋出：NVIDIA Nemotron-3 Diarization ONNX 轉換管線與基礎 PySide6 HUD。
