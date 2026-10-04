# DiarizeFlow

[![CI](https://github.com/saijo0404/DiarizeFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/saijo0404/DiarizeFlow/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/saijo0404/DiarizeFlow/releases/tag/v2.0.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

`DiarizeFlow` 是一個專為即時語音辨識、多講者分離與即時翻譯設計的現代化桌面懸浮字幕（HUD）系統。核心技術整合 **NVIDIA Nemotron-3-Diarization (Sortformer)**、**Faster-Whisper Large-v2**、**SenseVoiceSmall**、**目標講者語音提取 (Target-Speaker Extraction / TSE)** 以及雙軌智慧音訊路由，並採用 `uv` 進行確定性套件與環境管理。

---

## 🌟 特點與核心技術亮點

### 1. 🎙️ 即時語者分離翻譯應用 (Real-time Diarization & Translation HUD)
- **NVIDIA Nemotron-3 Diarization 官方串流架構**：以 1.04 秒小區間滑動視窗分析，支援最多 8 位講者交替說話之即時毫秒級邊界切分 (`diarize_and_split`)，徹底解決多講者語音累積混疊難題。
- **目標講者語音提取 (TSE)**：引入講者特徵導向的重疊語音提取模型，在多人同頻重疊說話 (Overlap Speech) 時精準分離目標聲軌。
- **雙軌音訊智慧路由 (SmartAudioRouter)**：支援同時擷取本機麥克風與系統音訊 (WASAPI Loopback / PulseAudio / PipeWire)，具備動態活動門檻判定、耳機/喇叭串音抑制 (Bleed Suppression) 與防爆音平滑交叉淡化 (Cross-Fading)。
- **輕量化 8-Byte 二進位 WebSocket 訊框協定 (`/ws/audio`)**：制定 `>BBBBHH` 二進位標頭，攜帶來源音軌標記 (Mic / Loopback / Mixed)、採樣率與聲道元資料，並支援純 Float32 PCM 自動相容回退。
- **雙 ASR 語音辨識引擎自由切換**：
  - **SenseVoiceSmall**：超低延遲（60~80ms）多語言辨識，極低資源消耗。
  - **Faster-Whisper Large-v2**：旗艦級語音模型，支援 CUDA Tensor Core（FP16 / INT8）硬體加速。
- **LLM 即時翻譯與連線池優化**：相容 `vLLM`、`llama.cpp`、`OpenAI`、`Claude`、`Ollama`，採用持久化 `aiohttp.ClientSession` 連線池以降低 HTTP 連線與 TLS 握手延遲，並自動清理思考標籤（如 `<think>`）。
- **前後端動態設定即時同步**：前端 HUD 透過 REST API `POST /api/config` 即時更新後端狀態，無需重啟伺服器。
- **持久化聲紋資料庫 (Speaker Profiles)**：支援聲紋特徵永久釘選儲存、講者名稱自訂及顏色關聯，並具備即時 WebSocket 事件廣播。

### 2. 🪟 模組化原生 PySide6 懸浮字幕視窗 (Modular Desktop HUD)
- **模組化元件架構**：將介面拆解為獨立的 `cards.py`、`settings_dialog.py`、`widgets.py`、`network.py`、`display_server.py` 與 `overlay_window.py`，結構清晰且利於擴充。
- **多卡片滑動對話隊列 (Multi-Card Sliding Queue)**：支援同時呈現多則發話者對話卡片（`max_cards`），具備獨立平滑淡出與霓虹講者識別徽章。
- **滑鼠穿透模式 (Click-Through) 與三重防死鎖安全保護**：
  1. 頂部控制列常駐互動，隨時可於視窗本體直接關閉穿透。
  2. 應用程式全域快捷鍵切換：`Alt+Shift+H` 或 `Ctrl+Shift+T`。
  3. 桌面系統托盤選單 (System Tray) 防死鎖安全退出機制。
- **Linux Wayland / X11 顯示伺服器環境感知**：自動偵測 Wayland 合成器，於終端機、HUD 按鈕 Tooltip、氣球通知及設定對話框給予清晰操作指引。
- **跨平台多螢幕視窗幾何記憶 (Window Geometry Persistence)**：自動記錄視窗座標與尺寸，跨重啟自動還原，並具備多螢幕座標邊界校驗與螢幕解析度安全箝位。

### 3. ⚡ Nemotron-3 Diarization ONNX 轉換與量化管線
- **RoPE 與注意力機制 ONNX 相容轉換**：將 FlexAttention 降階對應至標準 `scaled_dot_product_attention`，保留 RoPE 旋轉位置編碼運算。
- **`aten.sort.stable` 自動分解**：自動將 Dynamo 缺乏之排序運算子分解為原生 ONNX `TopK(sorted=True)`。
- **多精度量化評測引擎**：支援 FP16、INT8、FP8、NVFP4、MXFP4、W4A16 六大精度，嚴格區分原生硬體算子與數值誤差模擬。
- **Mel Spectrogram 矩陣與窗函數快取**：預先計算並快取濾波矩陣與 Hamming 窗函數，消除推論迴圈中重複計算的開銷。

---

## 📁 專案目錄架構

```
DiarizeFlow/
├── pyproject.toml                     # uv 專案定義、核心依賴與可選群組 [export, dev, build]
├── README.md                          # 專案總體說明與架構文檔
├── config.json                        # 預設應用程式設定檔
├── scripts/                           # 常用執行與自動化工具腳本
│   ├── download_models.py             # 一鍵自動下載預訓練模型 (Nemotron, SenseVoice, Whisper)
│   ├── build_executable.py            # PyInstaller 原生桌面執行檔自動打包 (支援 uv/pip 自動回退)
│   ├── run_app.py                     # 一鍵啟動後端管線與 PySide6 原生懸浮視窗
│   ├── run_backend.py                 # 獨立啟動 FastAPI 後端伺服器 (包含 WebSocket 與 REST API)
│   ├── run_frontend.py                # 獨立啟動 PySide6 前端懸浮視窗 (連接遠端或既有後端)
│   ├── convert_nemo_to_onnx.py        # Nemotron-3 Diarization ONNX 模型轉換腳本
│   ├── quantize_onnx.py               # ONNX 模型多精度量化轉換腳本
│   ├── verify_onnx.py                 # ONNX 模型結構與數值推論檢驗腳本
│   ├── benchmark_all.py               # 語者分離量化模型全面基準測試排行榜
│   ├── benchmark_sensevoice.py        # SenseVoice 語音辨識基準測試排行榜
│   └── benchmark_whisper.py           # Faster-Whisper 推論基準評測腳本
├── src/
│   └── diarizeflow/
│       ├── __init__.py                # 套件導出 (AppConfig 與核心運算 API)
│       ├── config.py                  # 全域統一配置中心 (AppConfig, AudioConfig, UIConfig 等；依賴朝向此模組)
│       ├── core/                      # 底層運算核心：硬體感知、ONNX 匯出、量化與校準
│       │   ├── export_onnx.py         # 核心 ONNX 匯出、簡化與驗證邏輯
│       │   ├── quantize.py            # 多精度量化引擎 (FP16/INT8/FP8/NVFP4/MXFP4/W4A16)
│       │   ├── hardware.py            # 跨平台硬體偵測引擎 (CUDA Driver API / SM 架構感知)
│       │   ├── calibration.py         # 首次啟動硬體探測與自適應量化校準
│       │   └── patches.py             # PyTorch Dynamo / ONNX 運算子相容性修補
│       ├── audio/                     # 音訊工程領域：音訊擷取、DSP 處理與多軌路由
│       │   ├── capture.py             # 跨平台串流音訊擷取 (WASAPI / Pulse / PipeWire) 與 SmartAudioRouter
│       │   ├── protocol.py            # 8-Byte 二進位 WebSocket 音訊標頭訊框協定與解析
│       │   ├── segmenter.py           # StreamingDiarizationSegmenter 串流語者分割與能量回退
│       │   ├── tse.py                 # TargetSpeakerExtractor 目標講者重疊語音提取
│       │   ├── vad.py                 # 能量式語音活動檢測 (EnergyVADSegmenter)
│       │   ├── agc.py                 # 串流輸入動態音量自動調節 (StreamingInputAGC)
│       │   └── devices.py             # 音訊輸入裝置與系統回放裝置探測
│       ├── engine/                    # AI 業務服務：推論管線與 FastAPI 服務
│       │   ├── server.py              # FastAPI 應用、WebSocket (/ws/audio, /ws/subtitles) 與 REST API
│       │   ├── pipeline.py            # DiarizeFlowPipeline 核心流程調度器
│       │   ├── diarizer.py            # Nemotron-3 Diarization 串流推論與 Mel 濾波矩陣/窗函數快取
│       │   ├── voiceprint.py          # 持久化聲紋資料庫 (SpeakerProfile, VoiceprintDatabase)
│       │   ├── asr.py                 # SenseVoiceSmall 與 Faster-Whisper ASR 引擎工廠
│       │   └── translator.py          # LLM 翻譯器 (持久化 aiohttp.ClientSession 連線池)
│       ├── ui/                        # 桌面圖形介面：PySide6 原生懸浮字幕元件 (模組化架構)
│       │   ├── overlay_window.py      # TransparentSubtitleOverlay 懸浮視窗本體與滑鼠穿透
│       │   ├── cards.py               # SubtitleCardWidget 與 SpeakerBadge (多卡片對話隊列)
│       │   ├── settings_dialog.py     # SettingsDialog 視覺化設定對話框與 Wayland 提示
│       │   ├── display_server.py      # Linux Wayland / X11 / Windows / macOS 顯示伺服器協議偵測
│       │   ├── network.py             # 後台 WebSocket 接收/音訊發送執行緒與 REST API 同步
│       │   ├── widgets.py             # UI 輔助元件、圖示繪製、動態 VU 聲波計與霓虹色盤
│       │   └── desktop_overlay.py     # UI 公開介面 Facade (無縫向後相容)
│       └── cli/                       # 命令列與啟動器入口
│           ├── launcher.py            # 整合式啟動器 (雙重日誌、例外捕獲、動態連接埠衝突檢測)
│           └── downloader.py          # 語音模型註冊表、狀態檢查與下載管理 CLI
└── tests/                             # 完整單元與整合測試套件 (250+ 測試全數通過)
```

**套件分層與依賴方向（嚴格單向，無循環依賴；由 `tests/test_package_layering.py` 自動驗證）：**

```
cli ──► ui / engine ──► audio ──► config
 └────► core ──────────────────► config
```

匯入範例：`from diarizeflow.engine import DiarizeFlowPipeline`、`from diarizeflow.config import AppConfig`。

---

## 🏗️ 系統架構設計

```
                                      【音訊捕捉與輸入端】
                                ┌───────────────────────────────┐
                                │ 本機麥克風 (Microphone Input) │
                                └──────────────┬────────────────┘
                                               │
                                ┌──────────────▼────────────────┐
                                │ 系統音訊 (WASAPI/Pulse Monitor)│
                                └──────────────┬────────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────────┐
                              │        SmartAudioRouter           │
                              │  - 雙軌動態能量分析與分流         │
                              │  - 串音抑制 (Bleed Suppression)   │
                              │  - 平滑交叉淡化 (Anti-pop Fading) │
                              └────────────────┬──────────────────┘
                                               │
                                               │ (單軌/多軌標籤)
                                               ▼
                         ┌─────────────────────────────────────────────┐
                         │   二進位音訊訊框協定 (Binary Framing)       │
                         │   - 8-Byte Header (>BBBBHH)                 │
                         │   - Magic: 0xDF | Ver: 0x01 | Track: Mic/Sys│
                         │   - SampleRate: 16000 | Channels: 1         │
                         │   - 自動回退純 Float32 PCM 相容模式         │
                         └─────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
              (In-Process 直連模式)                            (前後端分離網路模式)
                       │                                               │
                       │                                               ▼
                       │                                WebSocket: /ws/audio
                       │                                               │
                       ▼                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DiarizeFlow 後端處理核心 (Backend Pipeline)                     │
│                                                                                             │
│  1. 音量動態平準化 ────────> StreamingInputAGC (動態增益與防破音壓限)                         │
│                                                                                             │
│  2. 串流語者分割 ──────────> StreamingDiarizationSegmenter                                   │
│                              ├─ Sortformer 8 講者神經 SAD                                   │
│                              └─ 能量降級回退 (Energy VAD Fallback)                           │
│                                                                                             │
│  3. 重疊語音分離 ──────────> TargetSpeakerExtractor (TSE 神經聲學波形分離)                    │
│                                                                                             │
│  4. 語者分離與聲紋比對 ────> NemotronDiarizer                                                │
│                              ├─ 官方 Low-Latency 1.04s 串流架構                              │
│                              ├─ diarize_and_split 講者毫秒級交替切分                         │
│                              └─ 持久化聲紋特徵庫 (Persistent Voiceprint Profiles)            │
│                                                                                             │
│  5. 語音辨識 (ASR) ────────> SenseVoiceSmall (60~80ms) / Faster-Whisper Large-v2            │
│                                                                                             │
│  6. 即時語言翻譯 (LLM) ────> LLMTranslator (持久化 aiohttp.ClientSession 連線池)             │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ├──────────────────────────────┐
                                               ▼                              ▼
                                     WebSocket: /ws/subtitles         REST API: /api/*
                                               │                              │
                                               ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          PySide6 模組化懸浮字幕視窗 (Modular Frontend HUD)                  │
│                                                                                             │
│  - TransparentSubtitleOverlay: 無邊框毛玻璃半透明浮動視窗 (支援自由拖曳與置頂)              │
│  - SubtitleCardWidget: 多卡片滑動隊列 (Multi-Card Queue, 獨立淡出, 霓虹講者徽章)            │
│  - DisplayServer: Linux Wayland / X11 環境感知偵測與快捷鍵受限操作導引                      │
│  - SettingsDialog: 視覺化即時設定對話框 (音訊裝置、ASR 引擎、LLM 翻譯、UI 外觀)             │
│  - Mouse Click-Through: 滑鼠穿透三重防死鎖保護 (頂部操作列 / Alt+Shift+H / 系統托盤選單)   │
│  - GeometryPersistence: 跨重啟視窗幾何座標自動記憶與多螢幕安全箝位                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 API 與前後端通訊規格

DiarizeFlow 採用前後端完全分離設計，前端 HUD 與外部第三方客戶端可透過標準 WebSocket 與 RESTful API 完全掌控後端服務。

### 1. WebSocket 串流端點

| 端點 (Endpoint) | 方向 | 協定格式 | 說明 |
| :--- | :--- | :--- | :--- |
| `/ws/audio` | Client ➔ Server | 二進位訊框 (Binary Frame) | 傳輸 16kHz Float32 音訊訊框。支援 8-Byte 元資料標頭或直接傳送純 PCM。 |
| `/ws/subtitles` | Server ➔ Client | JSON 事件廣播 | 即時推送語者分離識別、翻譯結果、講者更名與刪除事件。 |

#### `/ws/audio` 8-Byte 二進位標頭格式：
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Magic (0xDF) | Version(0x01) |   Track Tag   | Flags (0x00)  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Sample Rate (uint16)    |      Channels (uint16)        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|             Raw Float32 Little-Endian PCM Payload...          |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
- **Track Tag 定義**：`0x01` = 麥克風 (Mic), `0x02` = 系統聲音 (Loopback), `0x03` = 混合聲音 (Mixed)。
- **無縫向後相容**：若前 2 位元組非 `0xDF, 0x01`，系統自動退回純 Float32 PCM 模式，將整包數據視為 `mixed` 來源解碼。

#### `/ws/subtitles` JSON 事件範例：
```json
{
  "id": "e4f8d2a1-7c9b-4b2e-a3d8-1e9a2b3c4d5e",
  "speaker": "講者 1",
  "original_text": "こんにちは、今日の会議を始めましょう。",
  "translated_text": "你好，我們開始今天的會議吧。",
  "source_lang": "ja",
  "target_lang": "繁體中文",
  "confidence": 0.94,
  "duration": 2.45,
  "timestamp": 1727982000.123,
  "latency": {
    "audio_duration_ms": 2450,
    "diarization_ms": 32.4,
    "asr_ms": 78.1,
    "translation_ms": 115.6,
    "total_pipeline_ms": 226.1
  }
}
```

### 2. RESTful API 端點

| 路由 (Path) | 方法 (Method) | 描述 (Description) | 回應格式範例 (Response Example) |
| :--- | :--- | :--- | :--- |
| `/api/status` | `GET` | 查詢後端運作狀態、模型供應商、連線數與音訊統計 | `{"status": "running", "active_ui_clients": 1, "last_audio_source": "mic", ...}` |
| `/api/config` | `GET` | 取得當前後端作用中的完整設定檔 | `{"audio": {...}, "asr": {...}, "llm": {...}}` |
| `/api/config` | `POST` | 前端動態同步新設定至後端（即時生效免重啟） | `{"status": "updated", "config": {...}}` |
| `/api/devices` | `GET` | 列出系統可用的所有實體麥克風與 Loopback 回放裝置 | `{"microphones": [...], "loopbacks": [...]}` |
| `/api/speakers` | `GET` | 取得所有持久化儲存的發話者聲紋檔案庫清單 | `{"profiles": [...], "count": 2}` |
| `/api/speakers/{id}/rename` | `POST` | 為特定語者重新命名並自訂標籤顏色 | `{"status": "success", "profile": {"name": "Alice", ...}}` |
| `/api/speakers/{id}` | `DELETE` | 永久刪除特定語者聲紋檔案並通知前端清除卡片 | `{"status": "deleted", "speaker_id": "Speaker 1"}` |
| `/api/test/audio` | `POST` | 上傳 `.wav` 測試檔案進行非即時管線功能驗證 | `{"status": "enqueued", "duration": 4.5}` |

---

## 🎧 雙軌音訊智慧路由與 DSP 處理

DiarizeFlow 專為多聲道即時通訊環境研發專屬 DSP 音訊處理鏈：

### 1. 雙軌智慧路由 (`SmartAudioRouter`)
- **路由模式支援**：
  - `smart` (預設推薦)：動態監控麥克風與系統音訊能量，僅當有實際發話時才路由數據，徹底避免無聲底噪佔用算力。
  - `mix`：傳統加權混音模式。
  - `mic_only` / `loopback_only`：單一音訊軌道鎖定。
- **串音抑制 (Bleed Suppression)**：藉由即時比對系統回放訊號與麥克風訊號強度，若麥克風能量顯著低於系統音訊（預設比例 `0.40`），自動判定為耳機/喇叭漏音並予以降噪抑制，杜絕迴音干擾。
- **防爆音平滑交叉淡化 (Anti-pop Cross-Fading)**：在音軌切換時執行餘弦加權平滑淡入淡出，消除切換瞬態爆音。

### 2. Mel Spectrogram 濾波矩陣與窗函數快取 (`engine/diarizer.py`)
- 在高頻率的串流處理迴圈中，預先計算並快取 128-bin Mel 濾波矩陣與 Hamming 窗函數，消除重複構建開銷，使特徵提取耗時由 4.3ms 大幅降低至 0.12ms (36x 加速)。

### 3. 目標講者重疊提取 (`tse.py`)
- 當會議或多人對話出現「多位講者同時開口」的重疊語音 (Overlap Speech) 時，透過聲學特徵注意力比對，自混合波形中抽離特定目標講者的純淨語音，大幅提升後續 ASR 辨識準確率。

---

## 🖥️ 桌面懸浮字幕 (HUD) 操作指南

### 1. 快捷鍵清單

| 快捷鍵 | 作用 | 適用平台 |
| :--- | :--- | :--- |
| `Alt + Shift + H` | 開啟 / 關閉滑鼠穿透模式 (Click-Through) | 全平台 (Windows / X11 / 焦點應用) |
| `Ctrl + Shift + T` | 切換滑鼠穿透備用快捷鍵 | 全平台 |
| 滑鼠左鍵拖曳頂部標題列 | 自由移動懸浮窗至螢幕任意位置 | 全平台 |

### 2. Linux Wayland 環境感知提示
- 在採用 Wayland 顯示協議的現代 Linux 發行版（Ubuntu 22.04+、Fedora、Arch、Debian 12 等）中，系統 Compositor 基於安全性預設限制背景程式攔截全域鍵盤熱鍵。
- DiarizeFlow 內建 Wayland 感知模組：
  - 啟動時自動於終端機輸出友善說明。
  - HUD 穿透按鈕與系統托盤項目動態切換為「Wayland 推薦」提示。
  - 開啟滑鼠穿透時，系統托盤主動彈出通知，導引用戶隨時透過桌面右下角托盤選單解除穿透。

### 3. 多螢幕視窗位置記憶與安全箝位
- 視窗關閉或移動停頓 800ms 後，自動將目前視窗座標與幾何尺寸寫入 `config.json` 中的 `ui.window_geometry`。
- 每次啟動時自動校驗座標是否座落於目前已連線的任一顯示器可視範圍內；若外接螢幕已拔除，自動平滑還原至主螢幕中央底部，杜絕視窗飛出螢幕外的困境。

---

## 🚀 快速上手與部署

### 1. 安裝環境與同步 (uv)

本專案使用 `uv` 進行確定性套件管理：

```bash
# Clone 專案庫
git clone https://github.com/saijo0404/DiarizeFlow.git
cd DiarizeFlow

# 同步核心依賴 (自動建立並配置 .venv)
uv sync
```

### 2. 一鍵下載預訓練模型

執行內建引導腳本自動下載並就緒所有模型檔案（Nemotron Diarization、SenseVoiceSmall、Faster-Whisper）：

```bash
uv run python scripts/download_models.py
```

### 3. 一鍵啟動桌面懸浮字幕 HUD

```bash
# 啟動桌面即時語者分離翻譯應用 (預設目標語言: 繁體中文)
uv run python scripts/run_app.py

# 或指定翻譯目標語言與連接埠
uv run python scripts/run_app.py --target-lang "English" --port 8765
```

- **Linux 快捷啟動**：`./run_desktop.sh`
- **Windows 快捷啟動**：雙擊 `run_desktop.bat`

### 4. 打包免 Python 的獨立執行檔 (Executable)

專案提供自動環境檢測與打包工具，自動處理 `uv` 虛擬環境中缺少 `pip` 的相容性問題：

```bash
# 安裝打包可選依賴
uv sync --extra build

# 執行自動打包
uv run python scripts/build_executable.py
```
- **Linux 產物**：`dist/DiarizeFlow/DiarizeFlow` (ELF binary)
- **Windows 產物**：`dist/DiarizeFlow/DiarizeFlow.exe`

---

## ⚡ 模型轉換、量化與基準測試 (ONNX Conversion & Quantization)

### 1. 執行模型轉換與 onnxsim 簡化

```bash
uv run python scripts/convert_nemo_to_onnx.py \
    --nemo-path models/nemotron_diarization/Nemotron-3-Diarization.nemo \
    --output models/nemotron_diarization/Nemotron-3-Diarization.onnx
```

### 2. 跨平台硬體偵測與多精度量化

DiarizeFlow 內建 CUDA Driver API 硬體感知引擎，能自動偵測 GPU 架構（Compute Capability / SM 等級），並提供 **FP16、INT8、FP8、NVFP4、MXFP4、W4A16** 六大量化精度轉換：

```bash
# 1. 自動硬體檢測策略 (依據 GPU 架構自動選定最優精度)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision auto

# 2. 指定 FP16 量化 (推薦 GPU 部署)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision fp16

# 3. 指定 Dynamic INT8 量化 (推薦 CPU 部署)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision int8

# 4. 一鍵全量化精度排行榜評測 (Nemotron 語者分離)
uv run python scripts/benchmark_all.py --bench-runs 15 --warmup-runs 5

# 5. 一鍵全量化精度排行榜評測 (SenseVoiceSmall 語音識別)
uv run python scripts/benchmark_sensevoice.py
```

### 3. 量化架構與基準測試透明度說明

DiarizeFlow 明確區分**原生硬體圖算子 (Native Graph Operators)** 與 **數值誤差模擬量化 (Weight Emulation / Fake Quantization)**：
- **原生硬體加速量化 (Native)**：
  - **FP16**：利用 NVIDIA Tensor Cores (`CUDAExecutionProvider`) 實現真實硬體 2x 吞吐量加速。
  - **INT8**：生成原生 ONNX 8-bit 動態量化算子（`QuantizeLinear`、`QLinearMatMul`），體積縮減 50%～70%。
  - **W4A16**：利用 `MatMulNBitsQuantizer` 打包 4-bit 權重，大幅減省記憶體。
- **數值誤差模擬量化 (Emulation)**：
  - **FP8 / NVFP4 / MXFP4**：用於在前瞻微架構未完全成熟時，預先評估量化引入之數值誤差 (MSE) 與特徵餘弦保真度 (Cosine Similarity)，所有評測報表均嚴格註記標籤，不混淆實測與模擬數據。

### 4. ONNX 模型張量規格

| 張量名稱 | 維度形狀 (Shape) | 數值型態 | 說明 |
| :--- | :--- | :--- | :--- |
| `chunk` | `(batch_size, chunk_frames, 128)` | `float32` | 輸入音訊 Mel-spectrogram 特徵 |
| `chunk_lengths` | `(batch_size,)` | `int64` | `chunk` 的有效特徵幀長度 |
| `spkcache` | `(batch_size, spkcache_len, 512)` | `float32` | 語者快取特徵 (AOSC 記憶狀態) |
| `spkcache_lengths` | `(batch_size,)` | `int64` | `spkcache` 的有效長度 |
| `fifo` | `(batch_size, fifo_len, 512)` | `float32` | 最近幀 FIFO 隊列特徵 |
| `fifo_lengths` | `(batch_size,)` | `int64` | `fifo` 的有效長度 |
| `spkcache_fifo_chunk_preds` | `(batch_size, time, 8)` | `float32` | 各幀對應最多 8 位語者的發話機率預測值 |
| `chunk_pre_encode_embs` | `(batch_size, num_frames, 512)` | `float32` | 預編碼特徵向量（用於更新下一輪快取） |
| `chunk_pre_encode_lengths` | `(batch_size,)` | `int64` | 預編碼特徵長度 |

---

## 🛠️ 常見問題排查 (Troubleshooting)

1. **`ModuleNotFoundError: No module named 'lhotse.indexing'`**：
   - NeMo 最新架構依賴 `lhotse>=2.0.0a6`，專案 `pyproject.toml` 已預設固定此版本，執行 `uv sync` 即可正常解析。
2. **`ValueError: self_attention_model='rope' is not supported`**：
   - 官方 PyPI 舊版尚未包含 RoPE 實作，本專案直接連結 `NVIDIA-NeMo/Speech` 最新版源碼，完整原生支援 RoPE。
3. **ONNX Runtime 顯示 `CUDAExecutionProvider is not in available provider names`**：
   - 此為純 CPU 版 onnxruntime 提示，推論會自動 fallback 到 `CPUExecutionProvider`。若需 GPU 加速推論，可使用 `uv pip install onnxruntime-gpu`。
4. **Wayland 環境下按下 `Alt+Shift+H` 無法切換穿透**：
   - 此為 Wayland 安全沙盒限制全域按鍵攔截所致，請直接在桌面右下角系統托盤圖示點擊右鍵選單解除穿透。
5. **打包執行檔報 `No module named pip`**：
   - 請執行 `uv sync --extra build` 安裝打包群組，本專案已全面支援 `uv pip install` 自動打包相容回退。

---

## 📄 開源授權

本專案基於 [MIT License](LICENSE) 條款開源發布。
