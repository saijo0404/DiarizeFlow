# [Audio/DSP]: Redundant double AGC processing and aliasing distortion in chunk resampling

**Labels**: `audio`, `enhancement`

## Description
Two DSP issues in the audio processing chain degrade acoustic quality and ASR accuracy:

### 1. Cascaded Double AGC (Automatic Gain Control)
- Incoming audio chunks are first dynamically gained in [`pipeline.py:161`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/backend/pipeline.py#L161) via `StreamingInputAGC.process(chunk)` with default parameters `target_rms=0.06`, `max_gain=25.0`.
- After VAD collects the speech utterance segment, `_pipeline_worker` in [`pipeline.py:197`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/backend/pipeline.py#L197) executes `apply_speech_agc(audio_segment, ...)` a second time with different parameters `target_rms=0.08`, `max_gain=4.0`.
- Applying AGC twice amplifies background noise floor, distorts natural volume envelope, and leads to non-linear compression artifacts.

### 2. Naive Linear Resampling without Anti-Aliasing Filter
- In [`src/diarizeflow/app/audio/capture.py:135-147`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/audio/capture.py#L135-L147), downsampling audio from 48kHz / 44.1kHz to 16kHz is performed via `np.interp`:
  ```python
  x_orig = np.linspace(0.0, 1.0, len(audio), endpoint=False)
  x_target = np.linspace(0.0, 1.0, num_target_samples, endpoint=False)
  audio = np.interp(x_target, x_orig, audio)
  ```
- Because linear interpolation lacks an anti-aliasing low-pass filter, frequencies above Nyquist (8kHz) fold into the baseband (aliasing).
- Furthermore, normalizing `[0.0, 1.0]` per 250ms chunk independently breaks phase continuity across block boundaries, causing periodic clicks/clicks every 250ms.

## Proposed Fix
1. Remove the second AGC call in `_pipeline_worker` and maintain a single, coherent streaming AGC stage.
2. Replace `np.interp` with polyphase resampling (`scipy.signal.resample_poly`):
```python
def _resample_to_16k(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if orig_sr != self.target_sr and len(audio) > 0:
        import math
        import scipy.signal
        gcd = math.gcd(orig_sr, self.target_sr)
        up = self.target_sr // gcd
        down = orig_sr // gcd
        audio = scipy.signal.resample_poly(audio, up, down)
    return audio.astype(np.float32)
```
