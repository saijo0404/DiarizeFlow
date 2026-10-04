# [Performance/DSP]: Mel Spectrogram 濾波矩陣與窗函數快取：消除串流循環中重複構建開銷

**Labels**: `performance`, `audio`, `diarization`

## 1. 需求背景與問題描述 (Problem Statement)
在 NVIDIA Nemotron-3 (Sortformer) 語者分離推論管線中（`src/diarizeflow/app/backend/diarizer.py` 第 248-265 行）：
```python
def _extract_mel(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    n_fft = 512
    win_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=self._target_mel_bins,
        window="hamming",
        power=2.0,
    )
    log_mel = np.log(np.maximum(mel_spec, 1e-5)).T
    return log_mel.astype(np.float32)
```

### 根本原因與技術痛點：
1. **重複濾波矩陣計算**：`librosa.feature.melspectrogram` 在每次調用時，若未傳入預先計算好的 `mel_basis`，其底層皆會重新呼叫 `librosa.filters.mel` 來動態建立形狀為 `(n_mels, 1 + n_fft // 2)` 的三角濾波矩陣。
2. **重複窗函數權重構建**：底層 STFT 在未指定預先生成的窗函數時，每次都會重新生成長度為 `win_length` 的 Hamming Window 陣列。
3. **高頻率呼叫下的 CPU 開銷**：在即時串流語者活動偵測（SAD）與語者切分（每秒至少調用 1~2 次）的情境下，此靜態濾波矩陣的不必要重複運算會浪費 CPU 週期並產生額外的記憶體配置壓力。

---

## 2. 建議解決方案 (Proposed Solution)

1. **實例初始化時快取靜態濾波器與窗函數**：
   在 `NemotronDiarizer.__init__` 中預先計算並儲存靜態 DSP 結構：
   ```python
   import librosa
   import scipy.signal

   self._mel_basis = librosa.filters.mel(
       sr=16000,
       n_fft=512,
       n_mels=self._target_mel_bins,
   )
   self._mel_window = scipy.signal.windows.hamming(int(16000 * 0.025), sym=False)
   ```
2. **在 `_extract_mel` 中直接復用快取**：
   直接使用標準 STFT 投影矩陣：
   ```python
   # 或傳入 librosa.feature.melspectrogram(..., mel_basis=self._mel_basis, window=self._mel_window)
   # 或者透過自訂的純向量化乘法計算功率譜，徹底避免動態物件重建
   ```
3. **數值一致性驗證**：
   編寫測試比對優化前後 `_extract_mel` 的輸出張量，確保各維度數值絕對誤差小於 `1e-6`。

---

## 3. 預期效益 (Expected Benefits)
- 完全消除串流特徵萃取中每次重複建立濾波矩陣與窗函數的運算時間。
- 降低音訊前處理的 CPU 佔用率與 GC 頻率，縮短端到端辨識延遲。
