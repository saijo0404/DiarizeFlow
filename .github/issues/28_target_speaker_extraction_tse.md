# [Feature/Audio-DSP]: 引入目標講者語音提取 (Target-Speaker Extraction / TSE) 模型：解決單音軌多講者同步重疊說話 (Overlap Speech) 波形分離難題

**Labels**: `enhancement`, `audio`, `diarization`, `architecture`

## 1. 需求背景與問題描述 (Problem Statement)
目前系統採用 NVIDIA Nemotron-3 Diarization (Sortformer) 進行串流語者活動偵測（SAD）與聲紋萃取。

### 技術本質與現狀盲區：
1. **Diarization 不等於 Waveform Separation**：
   Sortformer 的本質是「時間軸上的發話機率預測器（Who spoke when）」，輸出維度為 `(時間幀, 8人)` 的發話機率矩陣，**其本身完全不具備產生或分離音訊波形的能力**。
2. **重疊說話 (Overlap / Cross-talk) 時波形未分離**：
   當同一個麥克風前有兩人以上同時說話時（例如會議插話、對談激辯），`StreamingDiarizationSegmenter`（`segmenter.py` 第 282-287 行）雖然能正確判定 Channel 0 與 Channel 1 同時活躍，但其處理方式為：
   ```python
   # 原始混音 chunk 被無差別複製派發給多個 Channel 緩衝區
   buf.add_speech_chunk(chunk, embedding=step_emb)
   ```
3. **ASR 辨識率崩潰**：
   當發話結束時，Channel 0 與 Channel 1 送入下游語音辨識模型（SenseVoice / Faster-Whisper）的音訊，**本質上依然是包含了兩人聲音的同一段混合波形**。ASR 面對重疊人聲時，極易發生字詞漏失、發話混雜或嚴重的文字幻覺，直接影響翻譯品質。

---

## 2. 建議解決方案 (Proposed Solution)
引入輕量級 **目標講者語音提取 (Target-Speaker Extraction / TSE)** 神經網路架構（如 VoiceFilter-Lite、SpEx+ 或基於 512 維嵌入引導的輕量神經遮罩網路）：

### 核心架構流程：
1. **聲紋條件提示 (Voiceprint Conditioning Vector)**：
   - 充分復用 Sortformer 已萃取出的 512 維 Speaker Embedding（或永久聲紋庫中的黃金錨點），直接作為 TSE 神經網路的參考聲紋條件向量。
2. **條件式時頻遮罩 (Time-Frequency Masking) 或時域濾波**：
   - 輸入：混合音訊的 STFT 頻譜幅值 $X(t, f)$ + 目標講者 512 維嵌入特徵 $e_k$。
   - 網路：透過輕量 Conv1D / Bi-GRU / Conformer 模組，預測目標講者的專屬幅值遮罩 $M_k(t, f) \in [0, 1]$。
3. **波形乾淨重建**：
   - 將目標遮罩與原始混合頻譜相乘：$\hat{S}_k(t, f) = M_k(t, f) \odot X(t, f)$。
   - 結合原始相角（Phase）透過逆傅立葉變換（iSTFT）重建出僅含目標講者的乾淨單一人聲波形 $\hat{s}_k(n)$。
4. **智慧動態觸發 (Dynamic On-Demand Trigger)**：
   - 平時單人說話時旁路（Bypass）TSE，零額外運算開銷。
   - 僅在 Sortformer 判定多個通道同時活躍（Overlap 檢測成功）時，才動態啟動 TSE 為各活躍通道分離出獨立的純人聲波形，再分別派送給 ASR 辨識。

---

## 3. 預期效益 (Expected Benefits)
- **攻克重疊語音痛點**：徹底解決「單一收音設備下、多人同步說話」時 ASR 語音辨識率驟降的業界難題。
- **免除盲分離的講者排列問題 (No Permutation Problem)**：傳統盲音源分離（BSS）輸出順序不固定，而 TSE 以特定聲紋為條件，分離出的音軌直接綁定目標講者，與現有聲紋資料庫完美相容。
- **極致輕量與低延遲**：TSE 參數量遠小於全功能音源盲分離模型，以 ONNX Runtime INT8/FP16 推論可在 20~40ms 內完成，保持全鏈路即時串流體驗。
