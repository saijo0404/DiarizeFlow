# [Refactor/Backend]: FastAPI Lifespan 現代化升級：遷移 on_event 為 lifespan context manager 並消除 DeprecationWarning

**Labels**: `refactor`, `backend`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
在目前的後端服務實作中（`src/diarizeflow/app/backend/server.py` 第 56 與 61 行），應用程式生命週期管理依然採用舊版的 FastAPI 事件裝飾器：
```python
@app.on_event("startup")
async def on_startup():
    if not pipeline.is_running:
        pipeline.start(asyncio.get_event_loop())

@app.on_event("shutdown")
async def on_shutdown():
    if hasattr(pipeline.translator, "close"):
        try:
            await pipeline.translator.close()
        except Exception:
            pass
```

### 根本原因與技術痛點：
1. **官方全面廢棄**：FastAPI (>=0.141.1) 與底層 Starlette 已正式宣布廢棄 `@app.on_event("startup")` 與 `@app.on_event("shutdown")`，並將在未來的重大版本（如 FastAPI 1.0）中徹底移除。
2. **測試日誌干擾**：在執行單元與整合測試（`pytest`）時，因測試客戶端頻繁建立應用實例，每次測試都會噴出超過 40 條 `DeprecationWarning: on_event is deprecated, use lifespan event handlers instead`，嚴重干擾除錯與 CI/CD 輸出。
3. **缺乏原子性保證**：舊式事件處理器無法透過結構化的 context manager 確保啟動前準備與關閉後資源釋放（如連線池清理）具備異常回退與原子性。

---

## 2. 建議解決方案 (Proposed Solution)
使用現代標準的 ASGI Lifespan 機制重構 `create_app`：

1. **引入 `@asynccontextmanager`**：
   在 `server.py` 中定義 lifespan context manager：
   ```python
   from contextlib import asynccontextmanager

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # 啟動時：初始化 pipeline 背景工作線程/任務
       if not pipeline.is_running:
           pipeline.start(asyncio.get_event_loop())
       yield
       # 關閉時：優雅釋放翻譯連線池與網路資源
       if hasattr(pipeline.translator, "close"):
           try:
               await pipeline.translator.close()
           except Exception:
               pass
   ```
2. **綁定至 FastAPI 應用**：
   將 `lifespan` 傳入 `FastAPI(..., lifespan=lifespan)`，並徹底移除 `@app.on_event`。
3. **相容性驗證**：
   確保既有的 `TestClient`、桌面整合啟動流程以及非同步事件迴圈在無警告情況下 100% 通過測試。

---

## 3. 預期效益 (Expected Benefits)
- 消滅全套測試中 46 個 `DeprecationWarning`，保持終端與測試日誌乾淨清爽。
- 順暢銜接 FastAPI 與 Starlette 最新規範，避免未來框架升級產生破壞性損壞（Breaking Changes）。
- 確保伺服器關閉時的連線池與管線資源能以最高可靠度優雅釋放。
