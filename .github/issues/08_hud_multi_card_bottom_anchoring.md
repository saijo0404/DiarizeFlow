# [UI/UX]: HUD 支援多卡片佇列與視窗底部對齊錨定 (Bottom-Anchoring)，避免多人連續/同步說話覆蓋

**Labels**: `ui`, `enhancement`, `frontend`

## 1. 需求背景與問題描述 (Problem Statement)
目前桌面原生懸浮視窗（PySide6, `src/diarizeflow/app/frontend/desktop_overlay.py`）在面對兩人連續說話或同步/重疊說話時，存在「**後發言者的字幕會立刻覆蓋掉前一位發言者**」的問題。

### 根因分析 (Root Cause Analysis)：
- 在 `TransparentSubtitleOverlay` 中，畫面上僅配置了單一組 `self.translated_text` (QLabel) 與 `self.speaker_label` (QLabel)。
- 當收到字幕事件呼叫 `show_subtitle()` 時，直接以 `self.translated_text.setText(...)` 覆蓋文字並重置 `self.fade_timer.start(...)`。
- 若兩位講者在極短間隔內接連說話（例如相隔 300ms ~ 800ms），講者 2 的結果一旦送達，講者 1 的字幕會被瞬間替換抹除，導致使用者根本來不及閱讀前一位講者的發言內容。

---

## 2. 建議解決方案 (Proposed Solution)

將桌面懸浮視窗由目前的「單一固定文字標籤」升級為「**多卡片對話堆疊佇列（Multi-Card Message Stack）**」，並引入「**視窗底部對齊錨定（Bottom-Anchoring）**」。

```
┌───────────────────────────────────────────────┐
│ 🎙️ DiarizeFlow  ● 已連線      [▶ 開始] [⚙ 設定] │  <-- 固定頂部控制列
├───────────────────────────────────────────────┤
│                                               │
│  [講者 1] (即將於 2.1s 後淡出)                 │  <-- 舊發言（被往上頂推）
│  "Hello, can everyone hear me clearly?"       │
│                                               │
│  [講者 2] (即將於 4.8s 後淡出)                 │  <-- 最新發言（自底部推入）
│  "Yes, I can hear you. Let's begin."          │
│                                               │
└───────────────────────────────────────────────┘  <-- 視窗底邊固定錨定 (Bottom-Anchored)
```

### A. 多卡片佇列結構 (Multi-Card Stack & Independent Lifecycle)
1. **多卡片容器**：
   - 替換單一 `QLabel`，改為支援垂直堆疊的卡片列表容器（最多保留 2 ~ 3 筆最新發言卡片）。
   - 每筆字幕獨立包裝為一個包含講者徽章（顏色 Badge）、譯文與原文的 Card Widget。
2. **每張卡片獨立淡出倒數（Per-Card Independent Fade Timer）**：
   - 每一張卡片綁定專屬的淡出 `QTimer`（例如 5 秒）。
   - 當講者 2 在 0.5 秒後發話時，講者 2 的新卡片自下方插入，而講者 1 的卡片依然會安穩停留在畫面上繼續倒數剩餘的 4.5 秒，雙方發言同時並存顯示。
   - 超過上限（如 3 筆）時，最頂端的舊卡片平滑淡出銷毀。

### B. 視窗底部對齊錨定 (Bottom-Anchoring)
1. **佈局對齊設定**：
   - 多卡片動態出現/消失時，佈局對齊應設為 `Qt.AlignmentFlag.AlignBottom`（或 `QVBoxLayout` 底端對齊）。
2. **防彈跳抖動（Anti-Jitter Geometry）**：
   - 新發言由底部推入、舊發言往上頂，保持視窗底邊位置（`anchor_bottom_y`）嚴格固定。
   - 避免因動態增減內容導致視窗外框或底部位置上下跳躍，確保全螢幕或遊戲覆蓋時的視覺穩定性。

---

## 3. 預期效益 (Expected Benefits)
- 雙人對談或多人會議時，所有人發言均能維持完整設定時間，不再互相覆蓋。
- 介面動態呈現平滑穩定，提升 HUD 的閱讀舒適度與專業質感。
