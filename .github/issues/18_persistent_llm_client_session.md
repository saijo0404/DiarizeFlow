# [Performance/Translation]: LLMTranslator 採用持久化 aiohttp.ClientSession 連線池以降低 HTTP 連線與 TLS 握手延遲

**Labels**: `performance`, `translation`

## 1. 需求背景與問題描述 (Problem Statement)
目前在 `src/diarizeflow/app/backend/translator.py` 中，無論是呼叫 OpenAI 相容端點（vLLM、llama.cpp、Ollama）還是 Claude API，都是在每個請求中臨時建立並銷毀一個 `aiohttp.ClientSession`：
```python
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8.0)) as session:
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            # ...
```

### 根因分析 (Root Cause Analysis)：
- 頻繁建立臨時 session 會造成：
  1. **失去 HTTP 連線複用 (Keep-Alive)**：每次發話翻譯都需要重新建立 TCP 連線，若是連線到雲端 API（如 OpenAI / Anthropic）還必須重新進行 TLS/SSL 握手協商。
  2. **重複 DNS 查詢**：未復用連線池，每次呼叫都有 DNS 解析延遲。
- 在即時字幕這種高頻短文本翻譯的情境下（一句話通常每 1~3 秒一次），每次重新握手會額外增加 **50ms ~ 200ms** 的無謂網路等待時間。
- 此外，在 `_resolve_model_name()` 中，如果端點未設定模型名稱且遠端伺服器不支援 `/v1/models`，該函式每次發言皆會執行一次逾時為 2.0 秒的探測，嚴重拖慢翻譯反應時間。

---

## 2. 建議解決方案 (Proposed Solution)

在 `LLMTranslator` 中維護事件迴圈等級的持久化 ClientSession 連線池：

1. **連線池管理（Lazy Persistent Session）**：
   - 實作 `_get_session()`，將 `aiohttp.ClientSession(connector=TCPConnector(limit=10, keepalive_timeout=30))` 快取在實例中。
   - 支援安全重置：當使用者在執行期更改 `base_url` 或 `api_key` 時，關閉舊 session 並建立新 session。
   - 在應用程式或 Pipeline 停止時正確關閉 session (`close()`)。
2. **快取 `_resolve_model_name` 失敗結果**：
   - 若遠端端點對 `/models` 查詢失敗一次，標記為 `fallback_used = True`，避免每筆翻譯請求都經歷 2 秒的逾時嘗試。
3. **更精細的逾時設定**：
   - 將連線逾時（Connect Timeout）設為 1.5s，讀取逾時（Socket Read Timeout）設為 6.0s，避免因網路擁塞卡住管線。

---

## 3. 預期效益 (Expected Benefits)
- 連接雲端或內網 LLM 伺服器時，單句翻譯延遲可降低 50ms ~ 150ms。
- 顯著減少作業系統的 Socket 資源反覆開啟與銷毀負擔。
