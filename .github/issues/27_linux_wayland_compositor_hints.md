# [UI/UX]: Linux Wayland 環境感知提示：偵測顯示伺服器協議並指引托盤與視窗穿透操作

**Labels**: `ui`, `linux`, `enhancement`

## 1. 需求背景與問題描述 (Problem Statement)
目前 DiarizeFlow 桌面懸浮字幕視窗（PySide6, `src/diarizeflow/app/frontend/overlay_window.py`）具備無邊框半透明飄浮與滑鼠穿透（Click-Through）功能。
在 Windows 下透過 `ctypes.windll.user32.RegisterHotKey` 支援全域快捷鍵切換；而在 Linux 下：
- 若使用者運行在傳統 **X11** 環境，Qt 的視窗旗標（`WA_TransparentForMouseEvents`）與熱鍵可正常運作。
- 但現代主流 Linux 發行版（如 Ubuntu 22.04+、Fedora 36+、Debian 12、Arch 等）預設採用 **Wayland** 顯示協議。在 Wayland 嚴格的 Compositor 沙盒機制下：
  1. 應用程式通常預設不被允許攔截全域鍵盤熱鍵（如 `Alt+Shift+H`）。
  2. 部分 Wayland Compositor（如 GNOME Mutter）對無邊框視窗的滑鼠穿透支援程度有所差異，容易導致使用者誤以為穿透或熱鍵功能失效。

---

## 2. 建議解決方案 (Proposed Solution)

1. **啟動時環境感知偵測 (Display Server Detection)**：
   在 `launcher.py` 或 `overlay_window.py` 啟動階段檢查環境變數：
   ```python
   session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
   wayland_display = os.environ.get("WAYLAND_DISPLAY", "")
   is_wayland = (session_type == "wayland") or bool(wayland_display)
   ```
2. **終端與日誌友善指引提示**：
   若偵測到運行於 Wayland 環境下，主動印出提示訊息：
   ```text
   [*] 檢測到系統運行於 Linux Wayland 顯示環境。
       提示：Wayland 安全機制限制全域鍵盤熱鍵攔截。
       若需切換滑鼠穿透或控制字幕，請優先使用桌面系統托盤選單 (System Tray) 或頂部控制列操作。
   ```
3. **HUD 設定視窗提示增強**：
   在 `SettingsDialog` 或快捷鍵 Tooltip 中標明「（在 Wayland 環境下請使用系統托盤操作）」，避免使用者困惑。

---

## 3. 預期效益 (Expected Benefits)
- 提升 Linux 跨發行版的使用者體驗，清晰告知協議限制與最佳實踐方案。
- 避免使用者在 Wayland 下因無法使用全域熱鍵而誤報 Bug。
