# [Feature]: Persistent Voiceprint Profiles and Custom Speaker Identification (永久聲紋資料庫與自訂講者身份辨識)

**Labels**: `enhancement`, `diarization`, `ui`

## 1. 需求背景與問題描述 (Problem Statement)
目前 DiarizeFlow 的語者分離模組 ([src/diarizeflow/app/backend/diarizer.py](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/backend/diarizer.py)) 已經能透過 NVIDIA Nemotron-3 Diarization (Sortformer) 即時提取高品質的 512 維語者特徵向量（Speaker Embedding），並使用餘弦相似度（Cosine Similarity）進行講者匹配。

然而，現有實作存在以下限制：
1. **重啟即重設（In-Memory Only）**：`self.known_speakers` 僅存於 Python 執行階段記憶體中。應用程式重啟或呼叫 `reset()` 後，所有辨識過的講者聲紋特徵立即遺失。
2. **預設固定編號（Hardcoded Labels）**：每次啟動都強制從「講者 1」、「講者 2」開始重新流水號編號，無法將常客或固定講者綁定至真實名稱（例如：`Aimi`、`John`）。
3. **無聲紋防漂移機制（Voiceprint Drift）**：長期會議中單純使用 EMA (`0.85 * old + 0.15 * new`) 會因為環境底噪或短暫插話而導致聲紋特徵緩慢偏移。

---

## 2. 建議功能與架構方案 (Proposed Solution)

建立一套**本地永久聲紋庫（Persistent Voiceprint Database）與講者自訂身份辨識系統**：

### A. 磁碟持久化儲存 (`data/speakers/profiles.json`)
將註冊的語者資訊與 512 維特徵向量保存至本地檔案：
```json
{
  "version": 1,
  "model_type": "nemotron_512d",
  "profiles": [
    {
      "id": "spk_aimi_001",
      "name": "Aimi",
      "color": "#f472b6",
      "is_pinned": true,
      "sample_count": 24,
      "created_at": "2026-09-27 08:30:00",
      "anchor_emb": [0.0321, -0.0124, 0.0512, "... 512-dim normalized float vector ..."],
      "last_seen_at": "2026-09-27 08:45:00"
    }
  ]
}
```

### B. 雙層匹配策略與防漂移演算法 (Matching & Anti-Drift Engine)
1. **冷啟動自動加載（Zero-Shot Bootstrapping）**：
   - 後端啟動時自動加載 `profiles.json`。
   - 當已註冊講者（如 `Aimi`）首次發話時，餘弦相似度達標（$\ge 0.82$）即直接輸出字幕標籤 `Aimi`，無需再從「講者 1」重新分配。
2. **錨點保護（Anchor Protection）**：
   - 對於手動釘選（`is_pinned: true`）的永久聲紋，保留最初註冊時的「黃金錨點（Anchor Vector）」。
   - 僅在發話時長 $\ge 1.5$ 秒且置信度極高（$\ge 0.86$）時才微幅同化（如 `0.95 * anchor + 0.05 * new`），徹底杜絕聲紋漂移。

### C. 前端互動與講者重命名 (HUD Integration)
1. **桌面透明 HUD (PySide6)**：
   - 在 HUD 上的講者徽章（Badge）支援點擊或雙擊。
   - 彈出輕巧的重命名輸入框（例如：將「講者 1」改名為「Aimi」）。
   - 送出後自動釘選該講者聲紋並永久寫入磁碟，同時回溯更新畫面上的歷史字幕。
2. **REST API 支援**：
   - `GET /api/speakers`：取得現有已儲存聲紋列表。
   - `POST /api/speakers/{speaker_id}/rename`：修改名稱與顏色。
   - `DELETE /api/speakers/{speaker_id}`：刪除聲紋。

---

## 3. 預期效益 (Expected Benefits)
- **固定場合無感識別**：固定成員會議、Discord 語音、雙人或多人遊戲連線時，下次開機即可直接辨識出各發話者名稱。
- **大幅提升字幕可讀性**：不再受限於冷冰冰的「講者 1 / 講者 2」，直接呈現「Aimi」、「主持人」等自訂名稱。
