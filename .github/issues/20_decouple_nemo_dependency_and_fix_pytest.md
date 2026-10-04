# [Packaging/DevOps]: 將 nemo-toolkit 與 onnxsim 移至可選依賴 [project.optional-dependencies] export 並修復 Pytest 測試路徑

**Labels**: `packaging`, `dependencies`, `devops`

## 1. 需求背景與問題描述 (Problem Statement)
目前在 `pyproject.toml` 中，`nemo-toolkit`（自 GitHub 倉庫 `https://github.com/NVIDIA-NeMo/Speech.git` 拉取）被直接列在專案核心 `dependencies` 清單中。

### 根因分析 (Root Cause Analysis)：
1. **龐大的 NeMo 依賴問題**：
   - 專案在執行即時字幕、語者分離（`NemotronDiarizer`）、語音辨識（`SenseVoiceASR` / `FasterWhisperASR`）以及桌面 HUD 時，**底層全數使用 `onnxruntime` 載入 `.onnx` 執行**，完全不需要 NeMo！
   - NeMo 僅在開發者需要將原始 `.nemo` 檢查點轉換為 ONNX 格式（`export_onnx.py`）時才會用到。
   - 將其放在基礎依賴中，導致一般使用者或只想使用桌面字幕應用的人，在 `uv sync` 時被迫花費數分鐘下載並編譯超過 500MB 的 NeMo 源碼及其複雜的 C++/CUDA 依賴項。
2. **Pytest 收集問題**：
   - `pyproject.toml` 的 `[tool.pytest.ini_options]` 缺少 `testpaths = ["tests"]`，且未在依賴中定義測試套件。
   - 在專案根目錄執行 `pytest` 時，會誤爬到 `scratch/test_*.py`（其中有腳本在頂層建立 `QApplication`），導致測試收集直接崩潰。

---

## 2. 建議解決方案 (Proposed Solution)

1. **依賴解耦重構**：
   - 在 `pyproject.toml` 中將 `nemo-toolkit`、`onnxsim` 與 `onnxscript` 移出基礎 `dependencies`，改置於可選依賴組：
     ```toml
     [project.optional-dependencies]
     export = [
         "nemo-toolkit",
         "onnxsim>=0.4.36",
         "onnxscript>=0.7.2",
     ]
     dev = [
         "pytest>=8.0.0",
         "pytest-asyncio>=0.23.0",
     ]

     [dependency-groups]
     dev = [
         "pytest>=8.0.0",
         "pytest-asyncio>=0.23.0",
     ]
     ```
   - 需要匯出 `.nemo` 的開發者只需執行 `uv sync --extra export` 即可。
2. **防呆匯出提示**：
   - 在 `scripts/convert_nemo_to_onnx.py` 中增加相容檢查：若未安裝 `nemo-toolkit`，印出友善提示：
     ```python
     try:
         import nemo
     except ImportError:
         print("[!] 尚未安裝模型轉換依賴。請執行: uv sync --extra export (或 pip install -e '.[export]')")
         sys.exit(1)
     ```
3. **規範 Pytest 目錄**：
   - 在 `pyproject.toml` 中配置：
     ```toml
     [tool.pytest.ini_options]
     asyncio_mode = "auto"
     testpaths = ["tests"]
     norecursedirs = ["scratch", "build", "dist", ".venv"]
     ```

---

## 3. 預期效益 (Expected Benefits)
- 一般使用者與開發者的 `uv sync` 時間從數分鐘縮短至 **5 秒鐘極速同步**！
- 根除環境中的 PyTorch / NeMo 編譯衝突風險。
- 根目錄下執行 `pytest` 100% 綠燈，不再受 `scratch/` 腳本干擾。
