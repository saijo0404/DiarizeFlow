# [Bug/Build]: build_executable.py 在 uv 虛擬環境中因缺失 pip 模組導致自動安裝 PyInstaller 崩潰

**Labels**: `bug`, `packaging`, `build`, `high-priority`

## 1. 需求背景與問題描述 (Problem Statement)
當使用者在標準 `uv` 管理的 Python 虛擬環境下執行打包桌面執行檔腳本：
```bash
uv run python scripts/build_executable.py
```
腳本在檢測到尚未安裝 `PyInstaller` 時，會觸發自動安裝回退邏輯，但隨即因環境中缺少 `pip` 模組而直接崩潰終止：

```text
=================================================================
  📦 打包 DiarizeFlow 原生桌面執行檔 (PyInstaller)
     模式: 完整標準版 (Standard, 包含預先量化多模型)
=================================================================
[*] 正在安裝 PyInstaller...
/home/yijun/Project/DiarizeFlow/.venv/bin/python3: No module named pip
Traceback (most recent call last):
  File "/home/yijun/Project/DiarizeFlow/scripts/build_executable.py", line 54, in build
    import PyInstaller
ModuleNotFoundError: No module named 'PyInstaller'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/yijun/Project/DiarizeFlow/scripts/build_executable.py", line 193, in <module>
    build()
  File "/home/yijun/Project/DiarizeFlow/scripts/build_executable.py", line 57, in build
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
  File "/usr/lib/python3.12/subprocess.py", line 413, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['/home/yijun/Project/DiarizeFlow/.venv/bin/python3', '-m', 'pip', 'install', 'pyinstaller']' returned non-zero exit status 1.
```

---

## 2. 根本原因分析 (Root Cause Analysis)

1. **`uv` 虛擬環境特徵**：
   - 本專案採用現代化套件管理工具 `uv`。由 `uv venv` 建立的虛擬環境預設是輕量極速的，**預設不包含 legacy 的 `pip` 模組**（除非建立時指定 `--seed`）。
2. **打包腳本硬編碼依賴 `python -m pip`**：
   - 在 `scripts/build_executable.py` 第 56-57 行：
     ```python
     try:
         import PyInstaller
     except ImportError:
         print("[*] 正在安裝 PyInstaller...")
         subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
     ```
     腳本未檢查環境中是否存在 `pip`，亦未考量目前正運行於 `uv` 環境內，導致 `python -m pip` 必定報出 `No module named pip`。
3. **缺少打包相關可選依賴項定義**：
   - `pyproject.toml` 中的 `[project.optional-dependencies]` 僅定義了 `export` 與 `dev`，缺少 `build`（例如 `pyinstaller>=6.0.0`），使使用者無法透過標準 `uv sync --extra build` 預先安裝打包工具。

---

## 3. 建議解決方案 (Proposed Solution)

### A. 完善 `pyproject.toml` 打包依賴宣告
在 `pyproject.toml` 的 `[project.optional-dependencies]` 新增 `build` 群組：
```toml
[project.optional-dependencies]
build = [
    "pyinstaller>=6.11.0",
]
```

### B. 重構 `scripts/build_executable.py` 中的安裝適配邏輯
改進 PyInstaller 檢測與自動安裝機制：
1. **優先使用 `uv`**：
   若環境中存在 `uv`（透過 `shutil.which("uv")` 偵測），優先執行：
   ```python
   subprocess.check_call(["uv", "pip", "install", "pyinstaller"])
   ```
2. **安全回退與友善指引**：
   若未找到 `uv`，再嘗試 `sys.executable -m pip install pyinstaller`；若兩者皆失敗或不存在，不拋出未捕獲的 `CalledProcessError`，而是印出清晰的引導訊息：
   ```text
   [!] 未檢測到 PyInstaller，且當前環境未安裝 pip。
       請透過以下任一指令安裝後重試：
       1. uv add --dev pyinstaller
       2. 或執行: uv run --with pyinstaller python scripts/build_executable.py
   ```

---

## 4. 預期效益 (Expected Benefits)
- 徹底消除在純 `uv` 虛擬環境下執行打包腳本時的崩潰錯誤。
- 遵循現代 Python 與 `uv` 依賴管理最佳實踐，提升開箱即用的打包體驗。
