# [Bug/Frontend]: 修正 WebSocket 事件處理遺漏 speaker_deleted 導致 HUD 渲染空白字幕卡片問題

**Labels**: `bug`, `ui`, `frontend`

## 1. 需求背景與問題描述 (Problem Statement)
在 DiarizeFlow 的後端聲紋管理 API 中，當使用者刪除某個講者檔案時，後端會向所有連線的 WebSocket 客戶端廣播刪除事件：
```python
# src/diarizeflow/app/backend/server.py
await client.send_json({
    "type": "speaker_deleted",
    "speaker_id": speaker_id,
})
```

### 根因分析 (Root Cause Analysis)：
在前端 `src/diarizeflow/app/frontend/desktop_overlay.py` 的 `_handle_subtitle_event(event)` 中：
```python
if event.get("type") == "speaker_renamed":
    # 處理重命名邏輯...
    return

# 未過濾 speaker_deleted，直接往下執行字幕生成
speaker = event.get("speaker", "講者 1")
orig = event.get("original_text", "")
trans = event.get("translated_text", "")
conf = event.get("confidence", 1.0)
self.show_subtitle(speaker, orig, trans, conf)
```
- 因為該方法**只對 `speaker_renamed` 進行了過濾攔截**，當接收到 `speaker_deleted` 時，事件被視為一筆普通的字幕訊息。
- 由於刪除事件中沒有 `speaker`、`original_text` 或 `translated_text` 欄位，讀取預設值後產生了一筆講者為「講者 1」、內容為空白字串的無效字幕，並呼叫 `show_subtitle` 在 HUD 上彈出一個毫無內容的空白卡片！

---

## 2. 建議解決方案 (Proposed Solution)

在 `desktop_overlay.py` 的 `_handle_subtitle_event` 中建立結構化的事件分發機制：

1. **嚴格過濾與分流事件類型**：
   ```python
   evt_type = event.get("type")
   if evt_type == "speaker_renamed":
       self._handle_speaker_renamed_event(event)
       return
   elif evt_type == "speaker_deleted":
       self._handle_speaker_deleted_event(event)
       return
   elif evt_type in ["connection", "ping", "pong"]:
       return  # 忽略心跳與系統交握封包
   ```
2. **優化 `speaker_deleted` 處理**：
   - 當講者被刪除時，若畫面上正好有該講者的活躍卡片，可將該卡片的講者徽章改為「未知講者」或預設名稱，而非新建空白卡片。
3. **字幕空值防禦**：
   - 在 `show_subtitle` 中增加基本防呆：若 `not translated.strip() and not original.strip()`，則直接忽略不渲染卡片。

---

## 3. 預期效益 (Expected Benefits)
- 徹底杜絕後端操作講者資料庫時在前端 HUD 上產生幽靈空白卡片的 Bug。
- 增強 WebSocket 通訊協議的解析健壯性，防止未來新增控制封包時引發畫面異常。
