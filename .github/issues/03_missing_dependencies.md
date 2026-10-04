# [Packaging]: Missing critical runtime dependencies in pyproject.toml (soundcard, faster-whisper, ml_dtypes)

**Labels**: `packaging`, `dependencies`

## Description
When installing the project on a clean system via the standard `uv sync` command specified in `README.md`, several essential runtime packages are missing from `pyproject.toml`, resulting in immediate `ModuleNotFoundError` during normal operation:

1. **`soundcard`**:
   - Required by [`src/diarizeflow/app/audio/capture.py:236`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/audio/capture.py#L236), [`src/diarizeflow/app/audio/devices.py:41`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/audio/devices.py#L41), and PyInstaller spec files for native Windows WASAPI loopback audio capture.
   - Without it, loopback audio capture fails or falls back to unsupported device IDs.

2. **`faster-whisper`**:
   - Highlighted in project description (`pyproject.toml:4`), `README.md`, and implemented in [`src/diarizeflow/app/backend/asr.py:294`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/backend/asr.py#L294) and [`scripts/benchmark_whisper.py`](file:///home/yijun/Project/DiarizeFlow/scripts/benchmark_whisper.py).
   - Missing from `dependencies`, preventing users from utilizing Faster-Whisper.

3. **`ml_dtypes`**:
   - Imported in [`src/diarizeflow/quantize.py:208`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/quantize.py#L208) and [`scripts/benchmark_sensevoice.py:121`](file:///home/yijun/Project/DiarizeFlow/scripts/benchmark_sensevoice.py#L121) for FP8 (`float8_e4m3fn`) tensor conversions.
   - Missing from `dependencies`.

## Steps to Reproduce
1. Clone repository on a fresh environment: `git clone https://github.com/saijo0404/DiarizeFlow.git && cd DiarizeFlow`.
2. Run `uv sync`.
3. Try running audio device enumeration or loopback capture on Windows (`uv run python -c "import soundcard"`).
4. Error: `ModuleNotFoundError: No module named 'soundcard'`.

## Proposed Fix
Add the missing packages to `dependencies` in `pyproject.toml`:

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -16,6 +16,9 @@ dependencies = [
     "sounddevice>=0.5.6",
+    "soundcard>=0.4.3",
+    "faster-whisper>=1.0.0",
+    "ml-dtypes>=0.4.0",
     "fastapi>=0.141.1",
     "uvicorn>=0.54.0",
     "websockets>=16.1.1",
```
