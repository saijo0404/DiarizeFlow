# [Architecture/Feature]: 補齊前後端分離模式下的設定同步機制：前端透過 REST API POST /api/config 即時更新後端狀態

**Labels**: `architecture`, `frontend`, `api`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
DiarizeFlow 支援「一體化本地直連模式」與「前後端分離模式（FastAPI 後端 + PySide6 遠端前端）」。後端伺服器在 `src/diarizeflow/app/backend/server.py` 中已經實作了配置更新端點：
```python
@app.post("/api/config")
async def update_config(payload: dict):
    new_cfg = AppConfig.from_dict(payload)
    pipeline.update_config(new_cfg)
    return {"status": "updated", "config": new_cfg.to_dict()}
```

### 根因分析 (Root Cause Analysis)：
然而在前端 `src/diarizeflow/app/frontend/desktop_overlay.py` 的 `_apply_updated_config` 中：
```python
def _apply_updated_config(self, new_cfg: AppConfig):
    self.config = new_cfg
    if self.pipeline:
        self.pipeline.update_config(new_cfg)
    # 本地 UI 元件重繪更新...
```
- 當使用者在前後端分離模式下執行時（例如使用 `scripts/run_frontend.py`，此時 `self.pipeline is None`），使用者打開 `⚙ 設定` 對話框調整了以下設定：
  - ASR 引擎（SenseVoice / Faster-Whisper）
  - LLM 翻譯供應商與 API URL
  - 語者分離門檻（speaker_threshold）
  - VAD 斷句與能量門檻
- 前端**僅將變更儲存到前端電腦本機的 `config.json`**，卻**完全沒有呼叫後端的 `POST /api/config` REST 端點**！
- 這導致後端伺服器毫不知情，仍維持舊設定運作，前後端狀態徹底脫節。

---

## 2. 建議解決方案 (Proposed Solution)

在獨立前端模式下建立非同步 REST API 設定推送機制：

1. **實作非同步 API 同步方法**：
   在 `desktop_overlay.py` 中增加 `_sync_config_to_remote_backend(new_cfg)`：
   ```python
   def _sync_config_to_remote_backend(self, new_cfg: AppConfig):
       def _send():
           try:
               port = self.config.server.port
               url = f"http://127.0.0.1:{port}/api/config"
               payload = json.dumps(new_cfg.to_dict()).encode("utf-8")
               req = urllib.request.Request(
                   url,
                   data=payload,
                   headers={"Content-Type": "application/json"},
                   method="POST",
               )
               with urllib.request.urlopen(req, timeout=3.0) as resp:
                   if resp.status == 200:
                       print("[✓] 前端設定已成功同步至後端伺服器")
           except Exception as e:
               print(f"[!] 同步設定至後端伺服器失敗: {e}")

       threading.Thread(target=_send, daemon=True).start()
   ```
2. **在 `_apply_updated_config` 中觸發**：
   ```python
   if self.pipeline:
       self.pipeline.update_config(new_cfg)
   elif getattr(self, "enable_network", True):
       self._sync_config_to_remote_backend(new_cfg)
   ```
3. **啟動時拉取後端最新配置（雙向對齊）**：
   在前端連線至後端成功時，透過 `GET /api/config` 查詢後端當前生效的設定，自動同步至前端 UI，避免版本不一致。

---

## 3. 預期效益 (Expected Benefits)
- 使「前後端分離模式」真正具備全功能可用性，輕薄筆電端調整設定能立即在 GPU 後端生效。
- 完善 REST API 的雙向同步生命週期，為多客戶端協同作業奠定穩健架構。
