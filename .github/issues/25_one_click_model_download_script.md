# [Tooling/UX]: 實作一鍵模型下載與初次引導腳本 (scripts/download_models.py)

**Labels**: `enhancement`, `tooling`, `ux`

## 1. 需求背景與問題描述 (Problem Statement)
DiarizeFlow 仰賴兩大深度學習模型以實現完整的即時語音分離與辨識能力：
1. **語者分離**：NVIDIA Nemotron-3 Diarization ONNX 模型（位於 `models/nemotron_diarization/`）。
2. **語音辨識**：SenseVoiceSmall ONNX 模型及其 Tokenizer/CMVN 設定（位於 `models/sensevoice_small/`）。

### 目前痛點：
- **門檻過高**：新使用者 clone 專案後，若沒有預先轉換好的 ONNX 權重，需要手動下載龐大的 `.nemo` 檔案並安裝複雜的相依套件執行轉換。
- **缺乏自動化引導**：直接執行 `uv run diarizeflow-app` 時，若模型檔案不存在，後端僅在終端機輸出警告並降級為純聲學能量 VAD，初次使用的使用者可能無法直觀得知需要哪些模型或如何取得。

---

## 2. 建議解決方案 (Proposed Solution)

1. **編寫 `scripts/download_models.py`**：
   - 提供標準 CLI 工具，支援從 Hugging Face Hub 或 ModelScope 等開源模型倉庫一鍵拉取預先轉換好的 ONNX 模型檔案與設定檔：
     - `models/nemotron_diarization/Nemotron-3-Diarization.onnx`
     - `models/sensevoice_small/SenseVoiceSmall.onnx`
     - `models/sensevoice_small/am.mvn`
     - `models/sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model`
   - 支援斷點續傳、SHA256 完整性校驗與終端進度條顯示。
2. **自動硬體精度下載支援**：
   - 支援 `--precision auto/fp16/int8` 參數，若偵測到 NVIDIA GPU 則推薦並下載 FP16 模型；純 CPU 環境則下載 INT8 量化模型。
3. **啟動器首次偵測友善提示**：
   - 在 `launcher.py` 或應用啟動時，若偵測到 `models/` 目錄缺失必要模型，主動在終端機印出清晰友好的引導指令：
     ```bash
     [!] 偵測到尚未下載預訓練語音模型。請執行以下指令一鍵下載：
         uv run python scripts/download_models.py
     ```

---

## 3. 預期效益 (Expected Benefits)
- 將專案的初次安裝部署時間從原本數十分鐘的繁瑣手動操作大幅縮短至「一鍵幾分鐘內開箱即用」。
- 顯著降低非深度學習專業使用者的上手門檻。
