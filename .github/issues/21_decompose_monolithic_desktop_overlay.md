# [Refactor/Code Quality]: 拆解龐大模組 desktop_overlay.py：重構為模組化 Widgets、對話框與視窗元件

**Labels**: `refactor`, `frontend`

## 1. 需求背景與問題描述 (Problem Statement)
目前前端模組 `src/diarizeflow/app/frontend/desktop_overlay.py` 已膨脹至超過 1,514 行代碼，形成了典型的「單體巨型檔案 (Monolithic God File)」。

### 根因分析 (Root Cause Analysis)：
該單一檔案承擔了過多截然不同的職責：
- **設定 UI**：`SettingsDialog` 包含 400 多行的表單版面配置、裝置清單取得、日誌開啟與清空等。
- **自訂元件**：`SpeakerBadge`、`SubtitleCardWidget`（包含獨立淡出動畫、文字排版與重命名回調）。
- **視窗主體**：`TransparentSubtitleOverlay`（透明毛玻璃繪製、視窗拖曳、底部對齊錨定、事件分發）。
- **硬體與音訊**：音訊串流管理、即時音量 VU 波動計算、開始/停止收音狀態機。
- **網路通訊**：WebSocket 客戶端背景執行緒 (`_subtitles_worker`、`_audio_worker`)、REST API 請求發送。
- **主程式入口**：`run_overlay_app()` 與 `run_cli()`。

這導致：
1. 單元測試極其困難，修改一個微小的 UI 樣式可能無意影響網路或音訊狀態。
2. 代碼閱讀成本高，新加入的維護者不易理清職責邊界。

---

## 2. 建議解決方案 (Proposed Solution)

將 `desktop_overlay.py` 拆解重構至 `src/diarizeflow/app/frontend/` 套件目錄下：

```
src/diarizeflow/app/frontend/
├── __init__.py               # 導出 TransparentSubtitleOverlay, run_overlay_app
├── overlay_window.py         # 視窗核心：幾何控制、毛玻璃外框、底部錨定、拖曳
├── cards.py                  # SubtitleCardWidget, SpeakerBadge 與動畫生命週期
├── settings_dialog.py        # SettingsDialog 設定介面與裝置選單
├── widgets.py                # VU Meter、控制工具列按鈕與輔助視覺元件
└── network.py                # WebSocket 雙向通訊 worker 與 REST 客戶端包裝
```

同時在 `desktop_overlay.py` 中保留向下相容的導出，確保所有既有匯入路徑（如 `from diarizeflow.app.frontend.desktop_overlay import run_overlay_app`）完全不受影響。

---

## 3. 預期效益 (Expected Benefits)
- 符合單一職責原則（Single Responsibility Principle），使各元件可獨立進行 PySide6 單元測試。
- 模組層次清晰，後續擴展新功能（如自訂主題、快捷鍵設定、雙螢幕管理）更加靈活迅速。
