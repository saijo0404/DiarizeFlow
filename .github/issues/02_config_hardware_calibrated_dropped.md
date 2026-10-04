# [Bug]: AppConfig.from_dict drops hardware_calibrated causing calibration to re-run on every launch

**Labels**: `bug`, `config`

## Description
In `src/diarizeflow/app/config.py`, `AppConfig.from_dict` fails to pass `hardware_calibrated` to the `AppConfig` constructor.

When `ensure_calibrated_models` completes during first launch, it updates `config.hardware_calibrated = True` and saves the updated configuration to `config.json`. However, on all subsequent application launches, `AppConfig.load()` invokes `from_dict(data)`, which completely ignores the `"hardware_calibrated": true` key in `config.json` and defaults `hardware_calibrated` to `False`.

As a consequence, the condition in `calibration.py`:
```python
if getattr(config, "hardware_calibrated", False) and calib_marker.exists() and not force:
    return config
```
never evaluates to `True`, causing the application to re-run hardware detection and calibration on every single startup.

## Code Reference
Location: [`src/diarizeflow/app/config.py:168-184`](file:///home/yijun/Project/DiarizeFlow/src/diarizeflow/app/config.py#L168-L184)

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
    diar_dict = dict(data.get("diarization", {}))
    if diar_dict.get("speaker_threshold", 0) < 0.5:
        diar_dict["speaker_threshold"] = 0.82
    ui_dict = dict(data.get("ui", {}))
    if ui_dict.get("opacity", 0.5) > 0.85:
        ui_dict["opacity"] = 0.50
    return cls(
        audio=AudioConfig(**data.get("audio", {})),
        vad=VADConfig(**data.get("vad", {})),
        diarization=DiarizationConfig(**diar_dict),
        asr=ASRConfig(**data.get("asr", {})),
        llm=LLMConfig(**data.get("llm", {})),
        ui=UIConfig(**ui_dict),
        server=ServerConfig(**data.get("server", {})),
        # hardware_calibrated is omitted!
    )
```

## Steps to Reproduce
1. Start `scripts/run_app.py`. First-run calibration completes and writes `"hardware_calibrated": true` into `config.json`.
2. Inspect `config.json` to confirm `"hardware_calibrated": true` exists.
3. Restart the application.
4. Observe console output: the calibration banner is displayed again, rather than skipping with `[✓] 硬體加速與量化配置已完成校準，直接啟動最優推論模式（無等待秒開）`.

## Expected Behavior
`AppConfig.from_dict` must deserialize `hardware_calibrated` from the dictionary so cached calibration is respected.

## Proposed Fix
```diff
--- a/src/diarizeflow/app/config.py
+++ b/src/diarizeflow/app/config.py
@@ -181,6 +181,7 @@ class AppConfig:
             llm=LLMConfig(**data.get("llm", {})),
             ui=UIConfig(**ui_dict),
             server=ServerConfig(**data.get("server", {})),
+            hardware_calibrated=data.get("hardware_calibrated", False),
         )
```
