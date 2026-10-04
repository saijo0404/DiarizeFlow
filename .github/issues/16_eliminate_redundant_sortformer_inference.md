# [Performance/Diarization]: 消除 Sortformer 重複推論開銷：復用串流 SAD 聲軌資訊並優化 identify_speaker 嵌入萃取

**Labels**: `performance`, `diarization`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
目前管線在實作了串流 SAD（`StreamingDiarizationSegmenter`）之後，同一段語音在端到端流程中存在多次重複的 Nemotron-3 (Sortformer) ONNX 模型推論。

### 根因分析 (Root Cause Analysis)：
1. **第一輪推論**：
   在 `StreamingDiarizationSegmenter.process_chunk()` 中，音訊以 250ms 為單位送入，呼叫 `self.diarizer.forward_streaming_step()` 取得 8 聲軌的後驗機率 $P(t, k)$，並由對應的聲軌緩衝區收集發話音訊。
2. **第二輪推論**：
   當發話結束送入 `DiarizeFlowPipeline._pipeline_worker()` 時：
   ```python
   if duration >= 2.2 and hasattr(self.diarizer, "diarize_and_split"):
       speaker_segments = await asyncio.to_thread(
           self.diarizer.diarize_and_split,
           proc_audio,
           self.config.audio.sample_rate,
       )
   ```
   在 `diarize_and_split()` 內部，又再次對整段 `proc_audio` 執行了一次完整的 `_stream_process_audio()`！
3. **第三輪推論**：
   在 `diarize_and_split()` 內部切割出各個子片段後，對每個子片段又呼叫了一次 `identify_speaker()`，而 `identify_speaker()` 內部又對子音訊再執行了一次 `_stream_process_audio()`！

> **衝擊**：一段 2~3 秒的語音在發話結束時，在後端被重複跑了 3 到 4 次 ONNX 模型推論！這不僅導致 GPU 佔用飆高、顯存活動頻繁，在純 CPU 模式下更會造成高達 600~900ms 的無謂延遲堆積。

---

## 2. 建議解決方案 (Proposed Solution)

設計「**單次推論 / 特徵嵌入復用（Single-Pass Feature Reuse）**」策略：

1. **直接信任串流 SAD 的聲軌結果**：
   - 當 `StreamingDiarizationSegmenter` 的 Channel $k$ 觸發發話結束時，該段語音本來就已經隸屬於聲軌 $k$。
   - 不需要再退回全局呼叫 `diarize_and_split` 重新切分。
2. **特徵向量快取復用**：
   - 在 `StreamingDiarizationSegmenter` 運行的過程中，若需辨識講者身份（比對永久聲紋資料庫），應直接保存最後一輪或前幾輪前編碼特徵（`chunk_pre_encode_embs`），做平均後直接進行餘弦相似度比對。
   - 避免在發話結束後又拿整段 wav 重新計算 Mel-spectrogram 並重新執行 ONNX Forward。
3. **單一路徑決策**：
   - 只有在傳統 VAD 回退模式（Fallback Energy VAD）下，才需要在發話後執行 `diarize_and_split`；在神經網路 SAD 模式下，直接使用既有聲軌與特徵，達到真正的 0 重複推論。

---

## 3. 預期效益 (Expected Benefits)
- 端到端語者分離推論次數自 3~4 次降為 1 次。
- GPU 推論時間減少 60% 以上，純 CPU 模式下的處理延遲減少 400~800ms。
- 字幕產出更快、更即時，整體流暢度大幅提升。
