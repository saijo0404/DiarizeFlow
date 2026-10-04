# [Refactor/Performance]: 規範管線執行緒與非同步邊界：改採原生 Async Stream Pipeline 減少佇列與執行緒切換

**Labels**: `refactor`, `performance`, `pipeline`

## 1. 需求背景與問題描述 (Problem Statement)
在目前的後端管線排程中（`src/diarizeflow/app/backend/pipeline.py` 與 `src/diarizeflow/app/audio/capture.py`），音訊資料流經歷了過多次的執行緒切換與隊列中轉：

### 數據流經路徑：
```text
1. 音效卡驅動回呼 (Audio Driver Thread) -> 寫入 DualAudioCapture 內部隊列
2. DualAudioCapture 重採樣線程 (Processing Loop Thread) -> 寫入 Pipeline._raw_chunk_queue
3. Segmenter 背景工作線程 (_segmenter_worker Thread) -> 寫入 Pipeline._speech_queue
4. Pipeline 非同步主迴圈 (Async EventLoop) -> 透過 await asyncio.to_thread(_speech_queue.get) 取出
```

### 架構缺陷：
1. **執行緒排程開銷與延遲累積**：一個 100ms 的音訊區塊在到達語音辨識（ASR）之前，經歷了 4 個不同線程與 3 次 Queue 記憶體拷貝，產生非必要的上下文切換（Context Switching）延遲。
2. **在 AsyncIO 內部阻塞執行緒池**：在 `async def _pipeline_worker` 協程中，使用了 `await asyncio.to_thread(self._speech_queue.get)`，這會長期霸佔 `concurrent.futures.ThreadPoolExecutor` 的工作執行緒，違反現代非同步程式設計最佳實踐。

---

## 2. 建議解決方案 (Proposed Solution)
重構為清晰的**生產者-消費者非同步串流管線（Async Stream Pipeline）**：

1. **統一佇列類型**：
   - 在非同步邊界處全面改採原生的 `asyncio.Queue`。
   - 音訊採集線程在推入區塊時，使用 `loop.call_soon_threadsafe(queue.put_nowait, item)`，不再需要中轉線程與阻塞式等待。
2. **非同步工作管線簡化**：
   - 移除 `_segmenter_worker` 獨立執行緒，將 SAD 特徵評估與切分任務直接以輕量任務排入管線協程（或透過統一的 CPU Worker 池調度）。
3. **消除 `asyncio.to_thread(queue.get)`**：
   - 協程直接執行 `item = await self._speech_queue.get()`，完全釋放線程池資源。

---

## 3. 預期效益 (Expected Benefits)
- 消除 1~2 個過渡背景線程，降低記憶體佔用與 CPU 排班競爭。
- 減少佇列排隊帶來的額外延遲，使端到端音訊處理延遲進一步降低 10~20ms。
