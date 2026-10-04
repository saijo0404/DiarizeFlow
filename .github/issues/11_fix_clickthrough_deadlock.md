# [UI/UX]: 解決 HUD 滑鼠穿透模式 (Click-Through) 死鎖問題：支援系統托盤 (QSystemTrayIcon)、控制條獨立保護與全域快捷鍵

**Labels**: `ui`, `bug`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
目前桌面原生懸浮視窗（PySide6, `src/diarizeflow/app/frontend/desktop_overlay.py`）提供「`🛡️ 穿透`」按鈕，點擊後會呼叫 `self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, checked)`。

### 根因分析 (Root Cause Analysis)：
- HUD 視窗預設為無邊框且不常駐系統工作列的視窗屬性 (`Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool`)。
- 當使用者點擊「🛡️ 穿透」開啟 `WA_TransparentForMouseEvents` 之後，所有滑鼠事件（點擊、移動、右鍵）皆會直接穿透到底層視窗或遊戲。
- **致命缺陷**：因為滑鼠事件完全被忽略，使用者**再也無法點擊「🛡️ 穿透中」按鈕關閉穿透模式**，同時也無法點擊「設定」或「關閉視窗」。目前專案雖然匯入了 `QSystemTrayIcon`，但完全沒有初始化系統托盤，亦未註冊任何鍵盤快捷鍵。一旦進入穿透模式，使用者唯一的退出方法就是透過工作管理員或終端機強制殺死處理程序。

---

## 2. 建議解決方案 (Proposed Solution)

實作三重安全退出機制，徹底消除穿透死鎖：

### A. 實作系統托盤 (System Tray Icon)
1. 在 `desktop_overlay.py` 中初始化 `QSystemTrayIcon`，設定專屬麥克風/應用圖示。
2. 托盤右鍵選單提供：
   - 「切換滑鼠穿透 (Alt+Shift+H)」
   - 「開始/停止收音」
   - 「開啟設定視窗」
   - 「打開除錯日誌」
   - 「退出 DiarizeFlow」
3. 即便 HUD 處於完全穿透狀態，使用者隨時能透過右下角系統托盤一鍵解除穿透。

### B. 分區穿透保護 (Header Control Bar Non-Transparent)
- 在開啟穿透時，改為只讓「字幕卡片堆疊容器 (`subtitle_container`)」對滑鼠事件穿透，而**視窗頂部的操作工具列 (`header_widget`) 依然保持接受滑鼠事件**。
- 使用者仍可在視窗頂部正常點擊切換按鈕、設定與關閉。

### C. 全域快捷鍵切換 (Global Hotkey Toggle)
- 支援全域或焦點快捷鍵（例如 `Ctrl+Shift+T` 或 `Alt+Shift+H`），一鍵即時切換穿透狀態。

---

## 3. 預期效益 (Expected Benefits)
- 徹底消除使用者進入穿透模式後「無法返回、無法退出」的嚴重死鎖困境。
- 讓遊戲玩家在全螢幕或視窗化遊戲中能安心開啟穿透，隨時透過快捷鍵或托盤切換控制。
