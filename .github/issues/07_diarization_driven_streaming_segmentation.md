# [Architecture/Diarization]: End-to-End Streaming Diarization-Driven ASR Segmentation (移除傳統 VAD，由 Nemotron Sortformer SAD 驅動多軌音訊分流)

**Labels**: `architecture`, `diarization`, `enhancement`

## 1. 需求背景與問題描述 (Problem Statement)
目前 DiarizeFlow 在語音切分與講者判定流程中，容易出現「**前面是講者 1 的話、後面是講者 2 的話，但最後整段結果全部被判定為講者 1**」的現象。

### 根因分析 (Root Cause Analysis)：
1. **前端 VAD 純能量盲切（RMS Energy-based VAD）**：
   - 位於 `src/diarizeflow/app/audio/vad.py` 的 `EnergyVADSegmenter` 僅監控音訊音量能量，缺乏語者特徵感知能力。
   - 當講者 1 說完一句話、講者 2 迅速回應或插話時，只要彼此停頓小於 `silence_timeout_ms` (預設 300ms)，VAD 就會將兩人聲音打包為同一個 `completed_segment`（最長累積達 3.5s）。
2. **後端 `diarize_and_split` 切分條件過於嚴苛**：
   - 位於 `src/diarizeflow/app/backend/diarizer.py` 的切分演算法要求片段時長 $\ge 1.4\text{s}$，且切分點兩側均須 $\ge 0.65\text{s}$。
   - 若講者 2 只是簡短應答（例如「對啊」、「好的」，小於 0.65 秒），或者受環境雜音影響導致分數無法跨過 1.38 門檻，便直接放棄切分。
3. **單一聲紋匹配特徵被先發者主導**：
   - 切分失敗後，系統將整段音訊的特徵向量取平均（`spk_vec = np.mean(embs_list, axis=0)`），講者 1 佔據優勢後整段被貼上「講者 1」標籤，甚至在更新聲紋庫時造成講者 1 特徵被污染。
   - ASR 對整段音訊進行轉錄，最終輸出包含兩人發言的完整文字，但整句都被歸屬給講者 1。

---

## 2. 建議解決方案：端到端流式分離驅動架構 (Diarization-Driven Segmentation)

NeMo / Nemotron 語者分離架構（特別是 Sortformer 類別的端到端語者分離）內部本質上已具備 **Frame-level 的多講者活動偵測能力（Speaker Activity Detection, SAD）**，等同於天然具備「按講者拆分的多軌 VAD」。

因此，**建議完全移除前端傳統的 Energy VAD，改由 Diarization 引擎直接驅動串流分流**，這是工業級端到端流式分離（End-to-End Streaming Diarization）的標準實踐。

```
[連續音訊串流 (Chunk 100~500ms)]
                 │
                 ▼
    [Nemotron Sortformer (ONNX)]
                 │
   Frame-level 後驗機率 P(t, k)
   & Channel-to-Global Identity Mapping
                 │
        ┌────────┴────────┐
        ▼                 ▼
 [講者 A 獨立 Buffer]   [講者 B 獨立 Buffer]
 (檢測到 400ms 下緣靜音)  (持續說話累積)
        │
        ▼
   [加入前後 150ms Padding]
        │
        ▼
   [ASR 轉錄佇列 (SenseVoice / Whisper)]
```

### A. 為什麼可行 (Feasibility)
- Nemotron Diarizer 輸出的是每一幀（Frame-level，每幀 40ms）各個講者存在的機率矩陣 $P(t, k)$（$t$ 為時間軸，$k$ 為聲軌/講者索引）。
- 它同時解決了「有沒有人在說話」與「是誰在說話」，因此前端傳統的 Energy VAD 確實可以完全拿掉。

### B. 做法邏輯 (Implementation Logic)
1. **串流音訊切塊送入**：音訊以固定 Chunk（如 100ms ~ 500ms）連續送入 Diarizer。
2. **多軌活動判斷**：根據輸出的後驗機率 $P_k(t) > \text{threshold}$，動態判斷當前幀屬於哪些講者。
3. **每位活躍講者維護獨立音訊緩衝區（Buffer）**：
   - 建立 Per-Speaker Audio Buffer 池。
   - 當講者 $k$ 出現連續停頓（例如靜音達 400ms）或達到緩衝區上限（如 8s）時，觸發該講者的切分事件，將該講者的 Buffer 剪裁打包丟入 ASR 佇列。

### C. 核心優勢 (Key Advantages)
1. **徹底解決兩人發言黏連問題**：
   - 講者 1 講完、講者 2 緊接著插話時，講者 1 的軌道出現下緣（Falling Edge），講者 2 的軌道出現上緣（Rising Edge）。
   - 講者 1 立即觸發中斷結算，講者 2 另起新 Buffer，兩者在資料結構層面完全物理隔離。
2. **重疊說話（Overlap Speech）自然解耦**：
   - 兩人同時說話時，該區段音訊會分別拷貝複製給講者 1 與講者 2 的獨立 Buffer，各自送 ASR 辨識，完全不會互相覆蓋或漏字。

### D. 實作關鍵細節 (Implementation Details)
1. **聲軌黏性（Permutation Tracking / Speaker Identity Tracking）**：
   - Sortformer 等模型的輸出通道順序可能隨 Chunk 窗格跳動（例如上一秒是 Slot 0，下一秒跳到 Slot 1）。
   - 需要搭配原有的 Online Matching（聲紋或 Speaker Embedding）機制，確保把幀正確累積到全域的「講者 A」，而不是「模型當前輸出的 Slot 0」。
2. **音訊切除與前置邊界（Context Padding）**：
   - 送進 ASR 時，若切得太貼近講者邊界，容易掉字（例如發音起頭或喉音結尾）。
   - 建議在打包給 ASR 時，前後各保留 100ms ~ 150ms 的 Context Padding，保障 ASR 轉錄之完整性。

---

## 3. 預期效益 (Expected Benefits)
- 徹底消除「講者交替對話時整段被誤判為單一講者」的架構性瑕疵。
- 支援多人重疊說話時的雙軌並行辨識。
- 簡化音訊處理管線，不再需要調校脆弱的 Energy VAD 能量門檻與停頓時長。
