# [Refactor/Frontend]: 解耦 overlay_window.py 上帝類別：抽取 HotkeyManager、HUDTrayManager 與 GeometryManager

**Labels**: `refactor`, `ui`, `frontend`

## 1. 需求背景與問題描述 (Problem Statement)
目前前端桌面懸浮視窗主檔 `src/diarizeflow/app/frontend/overlay_window.py` 代碼量高達 1,270 行。雖然先前已將卡片、設定對話框與網路模組拆出，但主視窗類別 `TransparentSubtitleOverlay` 依然身兼多職，成為典型的上帝類別（God Class）：

### 違反單一職責原則 (SRP) 的具體表現：
1. **作業系統原生事件掛鉤**：在視窗類別內直接使用 `ctypes.windll.user32.RegisterHotKey` 註冊全域快捷鍵，並在 Qt 事件迴圈中處理原生事件過濾（`nativeEventFilter`）。
2. **系統托盤全套互動邏輯**：內部手動建立並維護 `QSystemTrayIcon`、托盤右鍵菜單與 Action 信號連結。
3. **視窗幾何動態運算與防跑版**：`_maintain_bottom_anchor`、多螢幕邊界安全防護、滑鼠拖曳定位與座標持久化邏輯與 UI 渲染程式碼深度混雜。
4. **音訊採集生命週期管理**：直接在 UI 內部呼叫 `AudioCaptureStream` 的啟動、停止與設備探測。

這種架構導致測試視窗佈局或文字格式化時，必須連帶拉起整套音訊流、系統托盤與 Win32 原生掛鉤，極難進行純粹的 UI 單元測試。

---

## 2. 建議解決方案 (Proposed Solution)
將輔助職責自 `overlay_window.py` 中抽取為專屬的管理器類別：

1. **抽取 `GlobalHotkeyManager` (`hotkey.py`)**：
   - 封裝 Windows `RegisterHotKey` / `UnregisterHotKey` 與 Linux 快捷鍵。
   - 僅透過標準 Qt Signal (`toggle_clickthrough_requested`) 與視窗解耦通信。
2. **抽取 `HUDTrayController` (`tray.py`)**：
   - 專門封裝 `QSystemTrayIcon` 的初始化、圖示動態繪製、右鍵選單邏輯。
3. **抽取 `HUDGeometryManager` (`geometry.py`)**：
   - 專責多螢幕座標邊界校驗、底部錨定（Bottom-Anchoring）高度維持、視窗幾何配置的讀取與儲存。
4. **精簡 `TransparentSubtitleOverlay`**：
   - 視窗本體僅保留 UI 元件的裝配、樣式設定以及信號調度（代碼量降至 300~400 行以內）。

---

## 3. 預期效益 (Expected Benefits)
- 大幅降低單一檔案的維護難度與認知負載。
- 提升跨平台（Windows / Linux Wayland）系統整合事件的單元測試便利性。
