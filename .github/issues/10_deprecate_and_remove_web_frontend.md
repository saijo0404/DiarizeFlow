# [RFC/Refactor]: Deprecate and Remove Web Frontend in favor of Native PySide6 Desktop HUD (建議完全移除 Web 前端，專注於原生桌面體驗)

**Labels**: `refactoring`, `cleanup`, `ui`, `discussion`

## 1. 提案背景與現況 (Background & Context)
目前 DiarizeFlow 專案同時維護了兩套前端使用者介面：
1. **PySide6 原生透明懸浮視窗** (`src/diarizeflow/app/frontend/desktop_overlay.py`)：為專案預設與核心主力體驗，支援全螢幕滑鼠點擊穿透（Click-Through）、自帶 `sounddevice`/`soundcard` 雙軌音訊採集（麥克風 + 系統 Loopback）、完整的即時設定調整面板、動態調音 AGC、以及即時延遲診斷。與後端為同行程記憶體直連（In-Process Callback），零網路傳輸開銷。
2. **Web 瀏覽器懸浮 HUD** (`src/diarizeflow/app/frontend/web/`)：透過 FastAPI 託管的靜態頁面（`index.html`, `app.js`, `style.css`），透過瀏覽器 Web Audio API 錄音並由 WebSocket 傳輸音訊與字幕。

經過實務評估，Web 前端在目前架構中已非核心路徑，卻衍生出相當程度的技術負債與維護分歧。

---

## 2. 建議完全移除之理由 (Rationale for Removal)

### A. 消除程式碼與功能維護分歧 (Eliminate Dual-Stack Overhead)
- 目前專案需同時維護兩套 UI 代碼、兩套設定同步邏輯、以及兩套音訊擷取採樣邏輯（Python 底層 vs 瀏覽器 `ScriptProcessor`）。
- 許多新功能（例如滑鼠穿透、深層硬體校準狀態、即時延遲診斷、永久聲紋自訂）僅存在於桌面版，Web 前端長期處於功能不同步與次級維護狀態。

### B. 精簡軟體打包架構與體積 (Streamline PyInstaller Packaging)
- 移除 Web 前端後，PyInstaller 打包設定（`DiarizeFlow.spec` 與 `scripts/build_executable.py`）不再需要將 `frontend/web` 資源打包進二進位檔案。
- 未來可進一步將 FastAPI、Uvicorn、WebSockets 等肥厚網路層降級為選配，大幅縮減 Release 發行包體積與啟動時間。

### C. 擺脫瀏覽器限制與額外除錯成本 (Avoid Browser Sandboxing Issues)
- 免除瀏覽器 `navigator.mediaDevices` 權限阻擋、48kHz/44.1kHz 重採樣至 16kHz 的失真、WebSocket 斷線重連機制、以及跨網域（CORS）繁瑣問題。

---

## 3. 影響與評估 (Impact & Trade-offs)

完全移除 Web 前端僅會影響少數邊緣場景：
1. **OBS Studio 瀏覽器來源（Browser Source）**：
   - 替代方案：OBS 原生即可使用「視窗擷取（Window Capture）」直接捕獲 PySide6 懸浮視窗（具備原生 Alpha 透明通道），效果甚至優於瀏覽器疊加層。
2. **跨裝置 / 區網遠端檢視**：
   - 實務上極少有透過手機或第二台裝置查看本機即時遊戲/會議字幕的需求。

---

## 4. 具體重構與清理清單 (Proposed Cleanup Checklist)

- [ ] **刪除靜態檔案**：完全刪除 `src/diarizeflow/app/frontend/web/` 目錄。
- [ ] **清理伺服器路由**：
  - 移除 `src/diarizeflow/app/backend/server.py` 中的 `StaticFiles` 掛載、`@app.get("/overlay")` 與 `@app.get("/")`。
- [ ] **簡化應用啟動器**：
  - 清理 `src/diarizeflow/app/launcher.py`，移除 `--mode web` 參數、`webbrowser.open(url)` 以及視窗啟動失敗時自動開啟瀏覽器的降級代碼。
- [ ] **清理打包與構建腳本**：
  - 更新 `DiarizeFlow.spec` 與 `scripts/build_executable.py`，移除 `frontend/web` 的 `--add-data`。
- [ ] **清理輔助執行腳本與文件**：
  - 更新 `scripts/run_frontend.py`、`scripts/run_app.py`、`scripts/run_backend.py` 與 `README.md`，移除 Web HUD 相關說明。
