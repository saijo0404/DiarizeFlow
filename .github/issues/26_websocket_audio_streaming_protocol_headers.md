# [Architecture/Network]: 擴充 WebSocket 音訊串流協定 (/ws/audio)：支援軌道標籤與元資料標頭

**Labels**: `architecture`, `network`, `audio`

## 1. 需求背景與問題描述 (Problem Statement)
目前前後端分離模式下（前端獨立桌面 HUD 透過網路連接遠端後端伺服器），音訊串流通訊協議（`src/diarizeflow/app/frontend/network.py` 與 `src/diarizeflow/app/backend/server.py`）採用最原始的二進位位元組流：
- 前端發送：`ws.send(chunk.tobytes())`
- 後端接收：
  ```python
  data = await websocket.receive_bytes()
  chunk = np.frombuffer(data, dtype=np.float32)
  pipeline.process_audio_chunk(chunk)
  ```

### 目前限制與架構痛點：
1. **缺乏協議元資料（Metadata）與交握（Handshake）**：傳輸的二進位完全沒有任何 Header，假設前端必然發送 16kHz mono float32。一旦傳輸發生封包丟失、採樣率不合或格式異常，後端無法校驗。
2. **無法傳遞雙軌音訊資訊**：前端若同時採集了麥克風與系統音訊，在發送前被迫必須先在前端本地執行混音（混合成單軌），才能送出 raw bytes。這使得後端 GPU 無法取得獨立的雙軌音訊，無法在伺服器端執行多軌聲學回音消除（AEC）或結合音訊來源標記的精準語者分離。

---

## 2. 建議解決方案 (Proposed Solution)

1. **定義輕量化二進位協定標頭（Binary Frame Protocol）**：
   在音訊 PCM 數據前方預留 8 個位元組的 Header：
   - Byte 0: Magic byte (`0xDF`)
   - Byte 1: Protocol Version (`0x01`)
   - Byte 2: 來源軌道標記 (`0x01` = Mic, `0x02` = Loopback, `0x03` = Mixed)
   - Byte 3: 保留欄位 / Flag
   - Byte 4-5: 採樣率（如 16000）
   - Byte 6-7: 聲道數（如 1）
   - 後續位元組：原始 float32 PCM payload。
2. **向下相容回退機制**：
   若收到的二進位數據前兩個位元組非 `0xDF, 0x01`，則自動降級為傳統無標頭 Raw PCM 解碼，確保舊版客戶端不受影響。
3. **後端多軌感知**：
   後端接收到帶有軌道標記的音訊後，可將來源標籤傳遞給 `pipeline.process_audio_chunk(chunk, source=...)`，賦予後端動態路由能力。

---

## 3. 預期效益 (Expected Benefits)
- 使前後端分離模式與本機直連模式具備完全對等的雙軌音訊分離與路由潛力。
- 提高音訊網路傳輸的健壯性與協定擴展彈性。
