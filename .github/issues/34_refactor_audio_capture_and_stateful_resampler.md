# [Refactor/Audio]: 解耦 capture.py：獨立 audio/router.py 並引入狀態化 StreamingResampler

**Labels**: `refactor`, `audio`, `dsp`

## 1. 需求背景與問題描述 (Problem Statement)
目前音訊擷取核心檔案 `src/diarizeflow/app/audio/capture.py`（614 行）聚合了過多複雜度：
1. **業務邏輯與底層驅動混雜**：`SmartAudioRouter`（涉及音訊動態門控、串音抑制與平滑 Cross-fading 業務）與底層 Sounddevice WASAPI/Pulse 硬體串流管理擠在同一檔案中。
2. **無狀態重採樣帶來的邊界瑕疵**：
   - 每次音訊進來時，呼叫 `scipy.signal.resample_poly(audio, up, down)`。
   - `resample_poly` 預設為無狀態操作（Stateless），在連續小音訊 Chunk（如 100ms）邊界處，由於未能傳遞與保留多相濾波器內部延遲線狀態（Filter State），在頻譜上可能引入微小的相位跳躍或高頻雜音。

---

## 2. 建議解決方案 (Proposed Solution)

1. **獨立 `SmartAudioRouter`**：
   - 移動至 `src/diarizeflow/app/audio/router.py`，專注於多軌音訊排程、動態權重計算與 Cross-fading。
2. **實作狀態化串流重採樣器 (`StreamingResampler`)**：
   - 封裝多相濾波器內部狀態向量（Filter State），在 Chunk 邊界無縫傳遞濾波歷史：
     ```python
     class StreamingResampler:
         def __init__(self, orig_sr: int, target_sr: int = 16000):
             ...
         def process(self, audio: np.ndarray) -> np.ndarray:
             # 使用 scipy.signal.upfirdn 或維護 state 的 resample 邏輯
     ```
   - 徹底消除連續串流因分段造成的邊界相位不連續性。
3. **讓 `capture.py` 專注於硬體層抽象**：
   - 僅負責 sounddevice 的 InputStream 生命週期、錯誤重試與驅動端點探測。

---

## 3. 預期效益 (Expected Benefits)
- 提高音訊前處理的訊噪比（SNR），避免微小相角跳躍影響後續 ASR 辨識精度。
- 模組職責清晰，方便單獨針對路由演算法或重採樣器撰寫測試。
