# [Performance/Pipeline]: LLM 翻譯改用 asyncio.gather 並行非同步請求，降低多講者與多段落串聯延遲

**Labels**: `performance`, `pipeline`, `enhancement`

## 1. 需求背景與問題描述 (Problem Statement)
目前 DiarizeFlow 在背景處理管線（`src/diarizeflow/app/backend/pipeline.py`）中，翻譯請求採用了「**單一 Worker 序列化循序處理**」的機制。

### 根因分析 (Root Cause Analysis)：
在 `DiarizeFlowPipeline._pipeline_worker` 中：
```python
speaker_segments = await asyncio.to_thread(self.diarizer.diarize_and_split, proc_audio, ...)

for seg_audio, speaker_label, confidence in speaker_segments:
    # 1. 執行 ASR
    orig_text, detected_lang = await asyncio.to_thread(self.asr.transcribe, seg_audio, ...)
    ...
    # 2. 循序等待翻譯
    translated_text = await self.translator.translate(text=orig_text, ...)
    ...
    # 3. 廣播
    await self.on_subtitle_broadcast(event)
```
- 當對話中出現講者輪替或切出多個子片段（例如 `part1` 講者 1 與 `part2` 講者 2）時，管線採用標準 `for` 迴圈阻塞式 `await`。
- 系統必須等待講者 1 的 ASR + LLM 翻譯全部完成並發布後，才會開始對講者 2 進行 ASR 與翻譯。
- **總耗時為線性累加**：$T_{\text{total}} = T_{\text{diar}} + (T_{\text{asr1}} + T_{\text{trans1}}) + (T_{\text{asr2}} + T_{\text{trans2}})$。
- 尤其當 LLM 翻譯延遲較高（例如 500ms ~ 1200ms）時，講者 2 必須乾等講者 1 翻譯完畢，造成第二位講者的字幕延遲明顯拉長。

---

## 2. 建議解決方案 (Proposed Solution)

**透過 `asyncio.gather` 或並行 Task 將翻譯請求改為並發送出，以進一步壓低端到端延遲**。

### A. 並行翻譯調度架構 (Concurrent Translation Flow)
1. **ASR 批量轉錄完成後並發翻譯**：
   - 先行完成各活躍子片段的 ASR（或由各軌獨立 ASR 產出文本）。
   - 對產出的多筆文本，使用 `asyncio.gather` 同時向 LLM 伺服器發起翻譯請求：
     ```python
     async def _translate_and_broadcast(orig_text, detected_lang, speaker_label, ...):
         translated = await self.translator.translate(orig_text, target_lang, detected_lang)
         event = SubtitleEvent(speaker=speaker_label, original_text=orig_text, translated_text=translated, ...)
         await self.on_subtitle_broadcast(event)

     # 並發執行多個講者段落的翻譯與廣播
     tasks = [
         _translate_and_broadcast(text, lang, spk, ...)
         for (text, lang, spk, ...) in asr_results
     ]
     await asyncio.gather(*tasks)
     ```
2. **即時流式推播（First-Come, First-Served）**：
   - 亦可使用 `asyncio.as_completed` 或獨立 `asyncio.create_task`，哪位講者的翻譯先返回就立刻推送到前端 HUD，不必互相等待。

### B. 本地部署並發保護機制 (Concurrency Limiter / Semaphore)
- **顯存防護**：
  考量到使用者多半使用本機部署的 LLM（例如 vLLM、llama.cpp、Ollama），突發過多並發請求可能導致 KV-cache 佔滿或 GPU 計算塞車。
- **解法**：在 `LLMTranslator` 內部引入 `asyncio.Semaphore(concurrency_limit)`（預設 2 或 3），在大幅降低等待延遲的同時，確保本機 GPU 不會超載崩潰。

---

## 3. 預期效益 (Expected Benefits)
- 多講者切分時，翻譯階段耗時由線性累加降為最大值：$T_{\text{trans}} \approx \max(T_1, T_2)$。
- 講者 2 的字幕等待時間平均可縮短 30% ~ 50%（減少 400ms ~ 1000ms 延遲），顯著提升即時對話翻譯的流暢感。
