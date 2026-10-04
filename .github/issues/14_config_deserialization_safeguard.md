# [Robustness/Config]: 增強 AppConfig.from_dict 反序列化防呆容錯：安全過濾未知欄位防止啟動崩潰

**Labels**: `bug`, `config`, `robustness`

## 1. 需求背景與問題描述 (Problem Statement)
目前在 `src/diarizeflow/app/config.py` 中，`AppConfig.from_dict()` 採用直接展開傳入字典的方式實例化各個 dataclass：
```python
return cls(
    audio=AudioConfig(**data.get("audio", {})),
    vad=VADConfig(**data.get("vad", {})),
    diarization=DiarizationConfig(**diar_dict),
    asr=ASRConfig(**data.get("asr", {})),
    llm=LLMConfig(**data.get("llm", {})),
    ui=UIConfig(**ui_dict),
    server=ServerConfig(**data.get("server", {})),
    hardware_calibrated=data.get("hardware_calibrated", False),
)
```

### 根因分析 (Root Cause Analysis)：
- Python 原生 `dataclass` 的 `__init__` 預設不支援多餘的關鍵字引數。
- 一旦使用者升級版本後殘留舊版屬性、手動編輯 `config.json` 添加自訂鍵值、或使用者拼錯參數名稱，Python 會直接拋出 `TypeError: AudioConfig.__init__() got an unexpected keyword argument '...'`。
- 該錯誤發生於應用程式最早期的設定載入階段，將導致整個應用程式（無論是 CLI、伺服器或桌面 HUD）完全崩潰且無法啟動。

---

## 2. 建議解決方案 (Proposed Solution)

在 `AppConfig.from_dict` 與各配置子模組建立通用的「欄位過濾輔助函式（Field Filtering Helper）」：

1. **實作安全的 dataclass 構建輔助函式**：
   ```python
   from dataclasses import fields

   def filter_dataclass_kwargs(cls, d: dict) -> dict:
       valid_fields = {f.name for f in fields(cls)}
       return {k: v for k, v in d.items() if k in valid_fields}
   ```
2. **在 `from_dict` 中全面過濾**：
   ```python
   return cls(
       audio=AudioConfig(**filter_dataclass_kwargs(AudioConfig, data.get("audio", {}))),
       vad=VADConfig(**filter_dataclass_kwargs(VADConfig, data.get("vad", {}))),
       diarization=DiarizationConfig(**filter_dataclass_kwargs(DiarizationConfig, diar_dict)),
       asr=ASRConfig(**filter_dataclass_kwargs(ASRConfig, data.get("asr", {}))),
       llm=LLMConfig(**filter_dataclass_kwargs(LLMConfig, data.get("llm", {}))),
       ui=UIConfig(**filter_dataclass_kwargs(UIConfig, ui_dict)),
       server=ServerConfig(**filter_dataclass_kwargs(ServerConfig, data.get("server", {}))),
       hardware_calibrated=data.get("hardware_calibrated", False),
   )
   ```
3. **安全載入回退**：
   若 `config.json` 語法錯誤或嚴重損毀，自動備份壞損檔案並還原至預設配置，並印出友善的警示訊息，而非拋出例外使程序崩潰。

---

## 3. 預期效益 (Expected Benefits)
- 保證向前與向後相容性，即使未來版號更迭或使用者誤增欄位，亦不會造成無法開機的致命中斷。
- 提升應用程式在跨平台與不同環境下的容錯健壯性。
