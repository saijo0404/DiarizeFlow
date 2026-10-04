# [UI/UX]: HUD 支援自訂視窗幾何記憶 (Window Position & Geometry Persistence)

**Labels**: `ui`, `enhancement`

## 1. 需求背景與問題描述 (Problem Statement)
目前桌面原生懸浮視窗（PySide6, `src/diarizeflow/app/frontend/desktop_overlay.py`）雖然支援使用者隨意按住滑鼠左鍵拖曳移動視窗，但移動後的位置**從未被儲存至磁碟**。

### 根因分析 (Root Cause Analysis)：
- 每次啟動 `TransparentSubtitleOverlay` 時，皆會無條件執行 `self._center_at_bottom()`：
  ```python
  def _center_at_bottom(self):
      screen = QApplication.primaryScreen()
      if screen:
          geo = screen.availableGeometry()
          x = (geo.width() - self.width()) // 2
          y = geo.height() - self.height() - 60
          self.move(x, y)
          self.anchor_bottom_y = y + self.height()
  ```
- 當使用者使用**雙螢幕/多螢幕**，或是習慣將字幕放置於螢幕頂端、遊戲右側等特定位置時，每次重啟 DiarizeFlow 都會被強行重置回「主螢幕正下方」！
- 這導致使用者每次開機或重啟應用，都必須重新手動拖曳對齊，極度影響日常使用體驗。

---

## 2. 建議解決方案 (Proposed Solution)

將視窗位置與尺寸納入配置儲存體系：

1. **擴充 `UIConfig` 配置欄位**：
   在 `src/diarizeflow/app/config.py` 的 `UIConfig` 中新增：
   ```python
   window_x: Optional[int] = None
   window_y: Optional[int] = None
   ```
2. **拖曳放開時非同步保存**：
   在 `desktop_overlay.py` 的 `mouseReleaseEvent` 中，除了更新 `self.anchor_bottom_y` 外，同時更新 `self.config.ui.window_x = self.x()`, `self.config.ui.window_y = self.y()`，並防抖（Debounce）或於關閉視窗時呼叫 `self.config.save()`。
3. **啟動時恢復幾何並做好邊界防護（Screen Boundary Clamping）**：
   - 啟動時檢查若 `window_x` 與 `window_y` 存在：
     - 檢查該座標是否依然位於當前系統任何一個可用螢幕的視區內（防範拔除外接螢幕後視窗落在可視範圍外的問題）。
     - 若合法則直接 `self.move(x, y)`；若超出螢幕邊界則自動回退居中。

---

## 3. 預期效益 (Expected Benefits)
- 完美支援多螢幕與自訂桌面配置，重啟應用無縫還原個人化版面。
- 提升桌面應用的專業度與使用舒適感。
