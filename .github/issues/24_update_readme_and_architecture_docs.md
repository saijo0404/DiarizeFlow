# [Documentation]: 同步更新 README.md 與架構文檔：反映模組化拆分、REST API 與雙軌路由

**Labels**: `documentation`

## 1. 需求背景與問題描述 (Problem Statement)
近期專案進行了多項核心架構演進（包含前後端解耦、模組化拆分、桌面懸浮窗重構、雙軌智慧路由等），但專案根目錄的 `README.md` 未能即時同步更新，導致文件與最新程式碼產生明顯脫節：

### 文件滯後現狀：
1. **目錄架構樹過時**：`README.md` 第 25-39 行仍呈現早期簡單結構（只有 `patches.py` 與 `export_onnx.py`），完全未提及現代化的 `src/diarizeflow/app/`、`audio/`、`backend/`、`frontend/` 等關鍵模組。
2. **缺乏前端模組化說明**：未介紹 `cards.py`、`settings_dialog.py`、`widgets.py`、`network.py` 與 `overlay_window.py` 的職責劃分。
3. **缺少 REST API 與前後端通訊規格**：未說明 `/api/config`（動態設定同步）、`/api/speakers`（聲紋資料庫管理）以及 `/ws/audio`、`/ws/subtitles` 的通訊機制。
4. **缺乏雙軌動態路由說明**：未在架構圖與說明中提及麥克風與系統音訊的 `SmartAudioRouter`（串音抑制與防爆音 cross-fading）。

---

## 2. 建議解決方案 (Proposed Solution)
全面更新 `README.md`：

1. **更新專案目錄架構樹**：
   詳細列出 `src/diarizeflow/` 下的各子目錄與關鍵檔案職責，特別強調 `app/` 架構。
2. **繪製最新前後端架構圖**：
   更新 ASCII 架構圖，呈現雙軌音訊輸入（WASAPI/Pulse）、`SmartAudioRouter`、FastAPI REST/WebSocket 端點、前端 PySide6 模組結構。
3. **擴充 API 與設定說明表格**：
   新增 REST API 路由清單（`/api/status`, `/api/config`, `/api/devices`, `/api/speakers`）與用途說明。
4. **補充常見操作指南**：
   更新快捷鍵清單（`Alt+Shift+H` / `Ctrl+Shift+T` 切換穿透）、系統托盤使用方式以及多螢幕視窗記憶特性的說明。

---

## 3. 預期效益 (Expected Benefits)
- 使開發者與新進使用者能迅速掌握當前專案的全貌與各模組職責。
- 消除文檔不實帶來的困擾，提升開源專案的專業度與易讀性。
