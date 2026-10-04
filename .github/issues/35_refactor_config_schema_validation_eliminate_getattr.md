# [Refactor/Config]: 健全配置中心 Schema 驗證：提供強型別保證並消除全域 getattr 防禦性代碼

**Labels**: `refactor`, `config`, `robustness`

## 1. 需求背景與問題描述 (Problem Statement)
目前在整個專案的程式庫中，存在**超過 100 處**防禦性的 `getattr(config.xxx, "key", default)` 調用：
- 例如：`getattr(self.config.audio, "agc_enabled", True)`
- `getattr(self.config.diarization, "sad_threshold", 0.50)`
- `getattr(self.config.vad, "pre_pad_ms", 150)`
- `getattr(self.config.asr, "model_path", "")`

### 核心問題：
1. **反映出配置物件缺乏型別保證**：呼叫端根本不敢直接使用 `self.config.audio.agc_enabled`，因為無法信任該屬性在經歷 `from_dict` 反序列化後必然存在。
2. **造成嚴重的程式碼噪音**：充斥各處的 `getattr(..., default)` 降低了代碼的可讀性，且預設值分散在各呼叫處，極易產生預設值不一致的隱蔽 Bug。
3. **缺乏型別與邊界驗證**：若配置檔中的數值型態不合法（例如應該是 float 卻填成非合法字串），缺少即時的 Schema 驗證報錯機制。

---

## 2. 建議解決方案 (Proposed Solution)

1. **引入強型別配置定義與建構子驗證**：
   - 在各 Config dataclass（`AudioConfig`, `DiarizationConfig`, `ASRConfig`, `LLMConfig` 等）中提供嚴格的預設值與型別標註。
   - 在 `from_dict` 反序列化時，進行嚴格的型態轉換與完整性填充，確保產出的 Config 物件**所有宣告的欄位必定存在且符合型態**。
2. **全面清理程式庫中的 `getattr`**：
   - 將業務代碼中的 `getattr(config.section, "key", default)` 全數重構成現代強型別屬性存取：
     ```python
     # 重構前:
     if getattr(self.config.audio, "agc_enabled", True): ...
     # 重構後:
     if self.config.audio.agc_enabled: ...
     ```
3. **增加配置綱要（Schema）單元測試**：
   - 確保缺失鍵值、額外未知鍵值或空字典輸入時，反序列化皆能穩定回退至合法完整的預設物件。

---

## 3. 預期效益 (Expected Benefits)
- 消除全專案上百處冗餘的 `getattr` 程式碼噪音，大幅提升代碼可讀性與 IDE 自動補全支援。
- 確保所有預設參數集中維護於 `config.py`，徹底杜絕分散預設值不一致的風險。
