# [Refactor/Backend]: 語者分離引擎職責解耦：拆分 SortformerEngine、SpeakerIdentityTracker 與 IntervalSplitter

**Labels**: `refactor`, `diarization`, `architecture`

## 1. 需求背景與問題描述 (Problem Statement)
目前語者分離模組的核心類別 `NemotronDiarizer`（`src/diarizeflow/app/backend/diarizer.py`，955 行）聚合了過多的異質職責：

### 核心問題：
1. **模型推論與聲紋追蹤高度耦合**：
   - 類別內部同時負責：ONNX Runtime Session 的載入與推論、Sortformer 串流狀態快取（`spkcache` 與 `fifo`）、Mel 頻譜特徵運算。
   - 同時又負責：聲紋特徵的餘弦距離比對、EMA 動態更新、黃金錨點防漂移（Anti-Drift）、講者上限指派以及與磁碟資料庫（`VoiceprintDatabase`）的持久化溝通。
2. **切分演算法與模型綁定**：
   - 時間軸音訊切分演算法（`diarize_and_split`，連通域合併與過度切分橋接）直接嵌入在推論類別內。
3. **模型可替換性極差**：
   - 若未來希望引入新的分離模型（例如 CAM++、3D-Speaker 或 PyAnnote），必須連同聲紋追蹤、EMA 演算法與切分邏輯一起重寫或大幅改動。

---

## 2. 建議解決方案 (Proposed Solution)
依據單一職責與組合優於繼承（Composition over Inheritance）原則，拆解為三個獨立元件：

1. **`SortformerInferenceEngine`（純推論層）**：
   - 專門負責模型載入、硬體 ExecutionProvider 選擇、維護 `spkcache` 與 `fifo` 環形快取。
   - 輸入音訊/Mel，輸出原始發話機率張量 `(frames, 8)` 與預編碼特徵。
2. **`SpeakerIdentityTracker`（聲紋追蹤與度量學習層）**：
   - 純粹接收 512 維嵌入向量，負責餘弦相似度比對、暫存講者池維護、EMA 動態同化與黃金錨點保護。
   - 與底層推論模型完全解耦，可適用於任何提供 embedding 的模型。
3. **`TimeIntervalSplitter`（時間區間切分層）**：
   - 純演算法模組：輸入多軌發話機率矩陣，輸出時間切片 `(start_sample, end_sample, channel)`。
4. **`NemotronDiarizer` 作為外觀門面（Facade）**：
   - 組合上述三個元件，保持與下游 `pipeline.py` 的對外 API 完全相容。

---

## 3. 預期效益 (Expected Benefits)
- 使聲紋追蹤與防漂移演算法獨立可測，不受大型 ONNX 模型推論的依賴干擾。
- 為未來引進其他開源語者辨識與分離模型鋪平架構道路。
