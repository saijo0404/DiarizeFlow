# DiarizeFlow

`DiarizeFlow` 是一個專為 NVIDIA **Nemotron-3-Diarization**（Sortformer 語者分離架構）打造的 ONNX 轉換與結構簡化管線。使用 `uv` 進行現代化、快速且確定性的環境與依賴管理。

---

## 🌟 特點與核心技術亮點

NVIDIA Nemotron-3 Diarization 是基於 31 層 Transformer Encoder 的即時串流語者分離架構（支援最多 8 名發話者），直接匯出 ONNX 時常遇到框架限制。`DiarizeFlow` 實現了關鍵技術修補，保證轉換無縫完成且數值 100% 精確對齊：

1. **RoPE 與注意力機制 ONNX 相容轉換**：
   - 模型預設使用 FlexAttention，包含 PyTorch 未實作 ONNX 降階的高階運算子（Higher-order ops）。
   - `DiarizeFlow` 自動將雙向完整注意力機制（`attn_mode='full'`）對應至標準 `torch.nn.functional.scaled_dot_product_attention`，完全保留 RoPE 旋轉位置編碼運算，並轉譯為標準 ONNX 算子。
2. **`aten.sort.stable` 運算子自動分解（Decomposition）**：
   - 針對 PyTorch Dynamo 匯出器缺乏 `aten.sort.stable` 降階規則的問題，自動註冊將其分解為原生 ONNX 支援的 `TopK(sorted=True)` 運算子。
3. **模型簡化與常數摺疊（ONNX Simplifier）**：
   - 使用 `onnxsim` 自動摺疊靜態子圖與權重、清理多餘節點，產出結構精簡的高效能 ONNX 模型。
4. **自動數值驗證（Numerical Verification）**：
   - 轉換後自動以 ONNX Runtime 載入並與 PyTorch 原生推論輸出進行逐張量絕對誤差比對，確保模型精度毫無損失。
5. **完整 `uv` 環境管理**：
   - 透過 `pyproject.toml` 與 `[tool.uv.sources]` 自動配置支援 RoPE 的最新 `NVIDIA-NeMo/Speech`，一鍵安裝所有相依套件。

---

## 📁 專案架構

```
DiarizeFlow/
├── pyproject.toml              # uv 專案定義與依賴設定
├── README.md                   # 專案說明文件
├── src/
│   └── diarizeflow/
│       ├── __init__.py         # 套件導出
│       ├── patches.py          # ONNX 匯出相容性修補 (aten.sort 與 SDPA)
│       └── export_onnx.py      # 核心匯出、簡化與驗證邏輯
└── scripts/
    ├── convert_nemo_to_onnx.py # 轉換 CLI 執行腳本
    └── verify_onnx.py          # ONNX 模型結構與推論驗證腳本
```

---

## 🚀 快速上手

### 1. 環境安裝與同步 (uv)

本專案使用 `uv` 管理虛擬環境與套件：

```bash
# Clone 並進入專案目錄
git clone https://github.com/saijo0404/DiarizeFlow.git
cd DiarizeFlow

# 同步並安裝所有依賴（會自動建立 .venv）
uv sync
```

### 2. 執行模型轉換與 onnxsim 簡化

使用內建腳本將 `Nemotron-3-Diarization.nemo` 轉換為 ONNX 並自動執行 `onnxsim`：

```bash
uv run python scripts/convert_nemo_to_onnx.py \
    --nemo-path models/nemotron_diarization/Nemotron-3-Diarization.nemo \
    --output models/nemotron_diarization/Nemotron-3-Diarization.onnx
```

也可透過 `pyproject.toml` 註冊的 CLI 命令執行：

```bash
uv run diarizeflow-export \
    --nemo-path models/nemotron_diarization/Nemotron-3-Diarization.nemo \
    --output models/nemotron_diarization/Nemotron-3-Diarization.onnx
```

#### 轉換參數說明：

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `--nemo-path` | 輸入的 `.nemo` 檢查點路徑 | `models/nemotron_diarization/Nemotron-3-Diarization.nemo` |
| `--output`, `-o` | 簡化後輸出的 ONNX 模型路徑 | `models/nemotron_diarization/Nemotron-3-Diarization.onnx` |
| `--raw-output` | 可選：保留未簡化的原始 ONNX 模型路徑 | `None`（暫存後自動清理） |
| `--batch-size` | 追蹤使用的 Batch 大小 | `1` |
| `--opset` | ONNX Opset 版本 | `18` |
| `--device` | 指定執行運算裝置 (`cuda` 或 `cpu`) | 自動選取（優先使用 `cuda`） |
| `--skip-onnxsim` | 跳過 `onnxsim` 簡化步驟 | 關閉 |
| `--skip-verify` | 跳過 ONNX Runtime 數值驗證步驟 | 關閉 |
| `--tolerance` | 數值比對容許最大誤差 | `1e-3` |

---

### 3. 驗證 ONNX 模型

轉換完成後，可使用檢驗腳本確認模型張量規格並進行推論測試：

```bash
uv run python scripts/verify_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx
```

---

## ⚡ 跨平台硬體偵測、多精度量化與數值相似度檢驗

DiarizeFlow 內建跨平台（Linux、Windows、WSL2）CUDA Driver API 硬體感知引擎，能自動偵測 GPU 架構（Compute Capability / SM 等級）、裝置名稱與顯存容量，並提供 **FP16、INT8、FP8、NVFP4、MXFP4、W4A16** 六大量化精度轉換與量化前後數值相似度檢驗。

模型全部存放在專案目錄下的 `models/` 子目錄進行嚴格模型隔離：
* `models/nemotron_diarization/`：語者分離模型（Nemotron Sortformer 各精度）
* `models/sensevoice_small/`：多語言語音辨識模型（SenseVoiceSmall 各精度）

### 執行量化與評測指令：

```bash
# 1. 自動硬體檢測策略 (依據 GPU 架構自動選定最優精度)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision auto

# 2. 指定 FP16 量化 (推薦 GPU 部署)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision fp16

# 3. 指定 Dynamic INT8 量化 (推薦 CPU 部署)
uv run python scripts/quantize_onnx.py --model models/nemotron_diarization/Nemotron-3-Diarization.onnx --precision int8

# 4. 一鍵全量化精度（FP16/INT8/FP8/NVFP4/MXFP4/W4A16）綜合排行榜評測 (Nemotron 語者分離)
uv run python scripts/benchmark_all.py --bench-runs 15 --warmup-runs 5

# 5. 一鍵全量化精度綜合排行榜評測 (SenseVoiceSmall 語音識別)
uv run python scripts/benchmark_sensevoice.py
```

### Python API 調用：

```python
from diarizeflow.quantize import auto_quantize_and_verify

# 執行自動偵測量化、相似度比對與推論加速評測
result_path, dev_info, metrics, benchmark = auto_quantize_and_verify(
    input_model="models/nemotron_diarization/Nemotron-3-Diarization.onnx",
    precision="fp16",  # "auto", "fp16", "int8", "fp8", "nvfp4", "mxfp4", "w4a16"
    min_cosine_threshold=0.90,
    bench_runs=15,
)

print(f"推理加速比: {benchmark.speedup:.2f}x (延遲降低 {benchmark.latency_reduction_pct:.1f}%)")
print(f"基準 FPS: {benchmark.orig_fps:.2f} chunks/s | 量化 FPS: {benchmark.quant_fps:.2f} chunks/s")
```

---

### 💎 量化架構與基準測試透明度說明 (Quantization Architecture & Benchmark Transparency)

DiarizeFlow 堅持軟體架構與基準測試數據的完全透明度。在模型量化與評測體系中，我們明確區分**原生硬體圖算子 (Native Graph Operators)** 與 **數值誤差模擬量化 (Weight Emulation / Fake Quantization)**：

#### 1. 原生硬體加速量化 (Native Hardware Quantization)
* **FP16 (Half Precision)**：轉換 ONNX 圖權重至 IEEE 754 半精度，利用 NVIDIA Tensor Cores (`CUDAExecutionProvider`) 實現真實硬體 2x 吞吐量加速。
* **INT8 (Dynamic Quantization)**：生成原生 ONNX 8-bit 動態量化算子（`QuantizeLinear`、`QLinearMatMul`、`MatMulInteger`），檔案體積真實縮減 50%～70%，適用於 CPU (AVX-512 / VNNI) 與 GPU 邊緣設備。
* **W4A16 (Weight-Only MatMulNBits)**：使用 ONNX Runtime 官方 `MatMulNBitsQuantizer`，將矩陣權重以 4 位元打包儲存，模型實質縮減約 75% 磁碟與顯存佔用，推論時透過硬體原生解包至 16 位元進行激活乘加。

#### 2. 數值誤差模擬量化 (Weight Emulation / Fake Quantization)
* **涵蓋精度**：`FP8` (E4M3FN)、`NVFP4` (NVIDIA Blackwell E2M1)、`MXFP4` (OCP Microscaling E2M1 + E8M0)。
* **技術原理**：當前開源推論引擎（如 ONNX Runtime、CTranslate2）對最新 4-bit / 8-bit 微架構原生硬體算子的支援尚在演進中。DiarizeFlow 採用權重數值投影模擬（Fake Quantization）：依據 Blackwell 雙層縮放或 OCP 規範，將浮點權重投影至目標格點後以浮點 Initializer 存儲。
* **核心用途**：可在一般硬體環境下精確量測各量化格式所引入之數值誤差（MSE、Max Error）及語者分離/語音辨識之特徵向量餘弦保真度（Cosine Similarity），作為前期架構選型與精度評估的重要依據。
* **透明度規範**：因以浮點初值儲存，該模式下模型容量未作位元打包壓縮，推論仍依浮點節點執行。專案內所有評測腳本均嚴格標記 `[實測硬體 (Native)]` 與 `[數值模擬 (Weight Emulation)]*`，堅決不混淆實際硬體測速與理論模擬數據。

#### 3. 透明度感知 API (Python)
```python
from diarizeflow import (
    is_native_quantization,
    is_simulated_quantization,
    get_quantization_execution_mode,
)

# 查詢精度執行模式
print(get_quantization_execution_mode("fp16"))   # "實測硬體 (Native Graph Operators)"
print(get_quantization_execution_mode("nvfp4"))  # "數值模擬 (Weight Emulation / Fake Quant)"
print(is_simulated_quantization("mxfp4"))         # True
```

---

## 📊 ONNX 模型輸入與輸出規格

轉換後的 ONNX 模型對應串流語者分離核心子圖：

### 輸入張量 (Inputs)

| 名稱 | 維度形狀 (Shape) | 數值型態 (dtype) | 說明 |
| :--- | :--- | :--- | :--- |
| `chunk` | `(batch_size, chunk_frames, 128)` | `float32` | 當前輸入的音訊特徵緩衝區（Mel-spectrogram 特徵） |
| `chunk_lengths` | `(batch_size,)` | `int64` | `chunk` 的有效特徵幀長度 |
| `spkcache` | `(batch_size, spkcache_len, 512)` | `float32` | 發話者快取特徵（AOSC 記憶狀態） |
| `spkcache_lengths` | `(batch_size,)` | `int64` | `spkcache` 的有效長度 |
| `fifo` | `(batch_size, fifo_len, 512)` | `float32` | 最近幀 FIFO 隊列特徵 |
| `fifo_lengths` | `(batch_size,)` | `int64` | `fifo` 的有效長度 |

### 輸出張量 (Outputs)

| 名稱 | 維度形狀 (Shape) | 數值型態 (dtype) | 說明 |
| :--- | :--- | :--- | :--- |
| `spkcache_fifo_chunk_preds` | `(batch_size, time, 8)` | `float32` | 各幀對應最多 8 位語者的發話機率預測值 |
| `chunk_pre_encode_embs` | `(batch_size, num_frames, 512)` | `float32` | 當前 chunk 經過預編碼後的嵌入特徵（用於更新下一輪快取） |
| `chunk_pre_encode_lengths` | `(batch_size,)` | `int64` | 預編碼特徵的長度 |

---

## 🛠️ 常見問題排查 (Troubleshooting)

1. **`ModuleNotFoundError: No module named 'lhotse.indexing'`**：
   - NeMo 最新架構依賴 `lhotse>=2.0.0a6`，專案 `pyproject.toml` 已預設固定此版本，執行 `uv sync` 即可正常解析。
2. **`ValueError: self_attention_model='rope' is not supported`**：
   - 官方 PyPI 舊版尚未包含 RoPE 實作，本專案直接連結 `NVIDIA-NeMo/Speech` 最新版源碼，完整原生支援 RoPE。
3. **ONNX Runtime 顯示 `CUDAExecutionProvider is not in available provider names`**：
   - 此為純 CPU 版 onnxruntime 警告，推論會自動 fallback 到 `CPUExecutionProvider`。若需 GPU 加速推論，可使用 `uv pip install onnxruntime-gpu`。

---

## 🎙️ 即時語者分離翻譯應用 (Real-time Diarization & Translation App)

DiarizeFlow 內建支援 **Windows 與 Linux** 的前後端分離即時語音分離與翻譯系統：

```
                               ┌──────────────────────────────────────────────┐
                               │                 前端 (Frontend)               │
                               │  - 麥克風 / 系統音訊 (WASAPI/Pulse Loopback) │
                               │  - PySide6 原生透明飄浮 HUD (防畫面干擾)     │
                               └──────────────────────┬───────────────────────┘
                                                      │ WebSocket (/ws/audio, /ws/subtitles)
                               ┌──────────────────────▼───────────────────────┐
                               │                 後端 (Backend)               │
                               │  1. VAD 動態語音端點偵測                     │
                               │  2. Nemotron-3 Diarization 語者分離 (8人)    │
                               │  3. SenseVoiceSmall 多語言 ASR (低延遲)      │
                               │  4. 可選 LLM API 翻譯 (vLLM/llama.cpp/OpenAI)│
                               └──────────────────────────────────────────────┘
```

### 1. 核心功能特性

1. **多來源音訊即時捕捉 (Cross-Platform Audio Capture)**：
   - **Windows**：支援 WASAPI Loopback 捕捉遊戲、瀏覽器、Discord 聲音，並可與麥克風同時混合收音。
   - **Linux**：原生相容 PulseAudio 與 PipeWire Monitor 來源，輕鬆擷取系統內播音訊。
2. **極致美觀的透明飄浮 UI (Floating HUD)**：
   - 無邊框、背景毛玻璃半透明、釘選於最上層（Always on Top）、支援自由拖曳位置。
   - **滑鼠穿透模式（Click-through）與防死鎖安全機制**：開啟後滑鼠點擊直接穿透到底層遊戲或視窗，不影響操作。內建三重防死鎖保護：
     - **頂部控制列獨立保護**：字幕容器穿透時，頂部操作按鈕依然接受點擊與拖曳，可直接在視窗上解除。
     - **全域快捷鍵切換**：隨時按下 `Alt+Shift+H` 或 `Ctrl+Shift+T` 一鍵切換穿透狀態。
     - **系統托盤常駐 (QSystemTrayIcon)**：桌面右下角托盤提供快捷右鍵選單（切換穿透、開始/停止收音、設定、日誌、退出），即使完全穿透亦能輕鬆掌控。
   - **自動淡出消失（Auto-dismiss）**：可自訂保留時間（預設 5 秒），發話結束自動淡出，徹底解決字幕長時間殘留干擾畫面的問題。
   - **發話者顏色識別**：各語者對應專屬高對比霓虹色系（`講者 1`、`講者 2`...）。
3. **高效端到端後端管線 (Backend Pipeline)**：
   - **語者分離**：NVIDIA Nemotron-3 Diarization ONNX 模型（GPU FP16 / CPU INT8）。
   - **語音辨識**：SenseVoiceSmall 多語言 ASR（自動偵測中/英/日/粵/韓語）。
   - **多後端 LLM 翻譯**：相容 `vLLM`、`llama.cpp`、`OpenAI`、`Claude`、`Ollama`，自動清洗思考標籤（如 `<think>`）。

---

### 2. 桌面透明飄浮視窗啟動方式 (Windows & Linux)

#### 方式一：一鍵雙擊/指令啟動（最簡便）

- **Linux 使用者**：
  直接在終端機或檔案管理員中執行可執行腳本：
  ```bash
  ./run_desktop.sh
  ```
  *(亦可將 `DiarizeFlow.desktop` 放入 `~/.local/share/applications/` 透過應用程式選單啟動)*

- **Windows 使用者**：
  直接雙擊專案目錄下的 `run_desktop.bat` 即可啟動！

#### 方式二：Python / uv 命令啟動

```bash
# 啟動 PySide6 原生透明飄浮 HUD（預設繁體中文，直連後端）
uv run python scripts/run_app.py
```

#### 方式三：打包為免 Python 的獨立執行檔 (Executable)

專案提供自動打包支援（Windows 為 `.exe`，Linux 為 ELF 執行檔）：

1. **打包 Linux 執行檔（在 Linux 或 WSL 內運行）**：
   ```bash
   uv run python scripts/build_executable.py
   ```
   產物位於 `dist/DiarizeFlow/DiarizeFlow`。

2. **在 WSL 內直接打包 Windows 執行檔 (`.exe`)**：
   WSL 原生支援呼叫 Windows 宿主機環境，只需執行：
   ```bash
   ./build_windows_from_wsl.sh
   ```
   腳本會自動透過 Windows PowerShell 調用 Windows Python/uv 進行打包，產物直接生成為 `dist/DiarizeFlow/DiarizeFlow.exe`！

3. **在 Windows 宿主機中打包**：
   直接雙擊 `build_windows.bat` 即可一鍵完成打包！

---

### 3. 設定與配置 (`config.json`)

系統啟動會讀取 `config.json`，亦可隨時點擊懸浮視窗右上角 `⚙ 設定` 按鈕即時調整：

| 參數 | 說明 | 預設 / 建議值 |
| :--- | :--- | :--- |
| `llm.provider` | 翻譯供應商 | `"vllm"`, `"llama.cpp"`, `"openai"`, `"claude"`, `"bypass"` |
| `llm.base_url` | 翻譯 API 端點 | `"http://127.0.0.1:8000/v1"` |
| `llm.target_language`| 目標語言 | `"繁體中文"`, `"English"`, `"日本語"`, `"한국어"` |
| `llm.max_tokens` | 翻譯輸出上限 | `512` (已自動關閉推理思考 tokens，極速直出) |
| `vad.silence_timeout_ms`| 發話停頓偵測時間 | `350` (低於 400ms 可達成極速語音切分) |
| `ui.fade_out_seconds`| 字幕自動淡出消失秒數 | `5.0` (發話結束自動淡出，不遮蔽遊戲或螢幕) |
| `ui.font_size` | 字幕字型大小 | `22` |
| `audio.mic_device` | 麥克風裝置 ID | `null` (自動選取) 或整數 ID |
| `audio.loopback_device`| 系統音訊 (Loopback) | `null` (自動選取) 或整數 ID |
