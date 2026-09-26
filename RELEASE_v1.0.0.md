# DiarizeFlow v1.0.0 正式發布 (Release Notes)

🚀 **DiarizeFlow v1.0.0** 是一個專為即時語音辨識、多語者分離與即時翻譯設計的現代化桌面懸浮字幕（HUD）應用程式。結合 **NVIDIA Nemotron-3-Diarization (Sortformer)**、**Faster-Whisper Large-v2**、**SenseVoiceSmall** 以及智慧音量自動增益（AGC），帶來極致低延遲、高保真度的多講者即時翻譯體驗。

---

## 🌟 核心功能與亮點

### 1. 🎙️ NVIDIA Nemotron-3 Diarization 官方 Low-Latency 架構
- **官方 1.04s 串流延遲規格**：嚴格實作 NVIDIA 官方 Low-latency 設定（`CHUNK_LEN=9`、`RIGHT_CONTEXT=4`、`FIFO_LEN=264`、`SPKCACHE_LEN=264`、`UPDATE_PERIOD=222`）。
- **即時多人對話交替切分 (`diarize_and_split`)**：採用 Sortformer 8 聲軌時間維度預測與滑動視窗，當多人持續交替發話時，自動在毫秒級時間邊界切分不同講者，分別送入 ASR 與翻譯，徹底解決「長時間累積音訊被混成同一個講者」的問題。
- **支援多元延遲模式**：
  - `low_latency`：1.04s（官方推薦預設）
  - `very_low_latency`：0.64s（極速模式）
  - `ultra_low_latency`：0.32s（超即時極限模式）
  - `offline`：30.4s（批次長音訊基準）

### 2. ⚡ 雙 ASR 語音辨識引擎自由切換
- **SenseVoiceSmall**：超低延遲（60~80ms）多語言語音辨識，支援中、英、日、韓、粵等語言，資源消耗極低。
- **Faster-Whisper Large-v2**：旗艦級高辨識率模型，支援 CUDA Tensor Core（FP16 / INT8）硬體加速，適合對長語意與專業名詞要求嚴苛的場景。

### 3. 🔊 智慧動態音量調節 (AGC)
- 內建二階自適應自動增益控制，自動將微弱發音（如能量 0.006）智慧放大至最佳人聲區間（黃金目標 -24 dBFS），並對爆音瞬間壓限防破音，大幅提升 VAD 檢測率與 ASR 辨識準確率。

### 4. 🪟 原生 PySide6 透明懸浮字幕 HUD
- **滑鼠穿透模式 (Click-Through)**：在玩全螢幕遊戲、看實況或開會時，懸浮視窗不搶焦點、不干擾操作。
- **即時延遲診斷 Tooltip**：隨時查看當前音訊段長、ASR 辨識耗時、Nemotron 語者分離耗時、LLM 翻譯耗時與總處理毫秒數。
- **視覺化雙語對照**：即時顯示「講者編號 + 原始語音 + 翻譯字幕」，支援自訂字型大小、透明度與自動淡出秒數。

### 5. 🛠️ 首次啟動硬體自適應量化校準
- 支援首次啟動時自動探測本機 GPU 架構（Ada Lovelace / Blackwell / Ampere / Turing），自動生成最佳量化模型（FP16 Tensor Core / INT8），未來啟動直接秒開。

---

## 📦 安裝與啟動

### 方式 A：透過 uv 一鍵啟動 (推薦)
```bash
# Clone 專案
git clone https://github.com/saijo0404/DiarizeFlow.git
cd DiarizeFlow

# 同步依賴並啟動
uv sync
uv run python scripts/run_app.py
```

### 方式 B：Windows 免 Python 執行檔打包
```bash
# 執行打包腳本生成免安裝目錄 (dist\DiarizeFlow)
build_windows.bat
```

---

## 📋 提交變更摘要 (Git Changelog)
- `feat(diarizer)`: 依據 NVIDIA 官方 HuggingFace 模型卡全面升級 Nemotron-3 Low-Latency 串流規格與狀態緩存機制。
- `feat(turn-detection)`: 實作 `diarize_and_split` 時間維度講者輪替切分，防止多講者音訊堆疊混雜。
- `feat(vad)`: 優化 VAD 累積上限為 3.5s、停頓間隔 300ms，換句更靈敏、字幕即刻輸出。
- `feat(ui)`: 設定視窗新增 Nemotron 串流分離延遲切換選單與 HUD 診斷浮動資訊。
- `chore(security)`: 全面清理外部絕對路徑與本地日誌，加固 `.gitignore` 確保零隱私洩漏。
