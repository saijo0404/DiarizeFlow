# DiarizeFlow GitHub Issues & Tracking Plan

This directory contains standardized GitHub Issue templates and issue drafts identifying critical bugs, packaging oversights, architecture improvements, and new feature RFCs in the DiarizeFlow project.

All active issues are tracked on GitHub:

| Issue | Title | Type | Status / Live Link |
|:---:|---|---|:---:|
| **#1** | Audio desynchronization and temporal expansion when mixing microphone and loopback audio | 🔴 Critical Bug | [Issue #1](https://github.com/saijo0404/DiarizeFlow/issues/1) (Closed) |
| **#2** | `AppConfig.from_dict` drops `hardware_calibrated` causing calibration to re-run on every launch | 🟠 Bug | [Issue #2](https://github.com/saijo0404/DiarizeFlow/issues/2) (Closed) |
| **#3** | Missing critical runtime dependencies in `pyproject.toml` (`soundcard`, `faster-whisper`, `ml_dtypes`) | 🟠 Packaging Bug | [Issue #3](https://github.com/saijo0404/DiarizeFlow/issues/3) (Closed) |
| **#4** | Redundant double AGC processing and aliasing distortion in chunk resampling | 🟡 Audio/DSP | [Issue #4](https://github.com/saijo0404/DiarizeFlow/issues/4) (Closed) |
| **#5** | Simulated weight-only quantizations (FP8/NVFP4/MXFP4) lack native ONNX operators and benchmark transparency | 🔵 Architecture | [Issue #5](https://github.com/saijo0404/DiarizeFlow/issues/5) (Closed) |
| **#6** | Persistent Voiceprint Profiles and Custom Speaker Identification (永久聲紋資料庫與自訂講者身份辨識) | ✨ Feature / RFC | [Issue #11](https://github.com/saijo0404/DiarizeFlow/issues/11) (Closed) |
| **#7** | End-to-End Streaming Diarization-Driven ASR Segmentation (移除傳統 VAD，由 Nemotron Sortformer SAD 驅動多軌音訊分流) | 🔵 Architecture | [Issue #13](https://github.com/saijo0404/DiarizeFlow/issues/13) (Closed) |
| **#8** | HUD 支援多卡片佇列與視窗底部對齊錨定 (Bottom-Anchoring)，避免多人連續/同步說話覆蓋 | 🎨 UI/UX | [Issue #14](https://github.com/saijo0404/DiarizeFlow/issues/14) (Closed) |
| **#9** | LLM 翻譯改用 asyncio.gather 並行非同步請求，降低多講者與多段落串聯延遲 | ⚡ Performance | [Issue #15](https://github.com/saijo0404/DiarizeFlow/issues/15) (Closed) |
| **#10** | Deprecate and Remove Web Frontend in favor of Native PySide6 Desktop HUD (建議完全移除 Web 前端，專注於原生桌面體驗) | 🧹 Refactor | [Issue #16](https://github.com/saijo0404/DiarizeFlow/issues/16) (Closed) |
| **#11** | 解決 HUD 滑鼠穿透模式 (Click-Through) 死鎖問題：支援系統托盤 (QSystemTrayIcon)、控制條獨立保護與全域快捷鍵 | 🔴 UI/UX Bug | [Issue #26](https://github.com/saijo0404/DiarizeFlow/issues/26) (Open) |
| **#12** | 統一應用程式啟動入口邏輯：將 DualLogger、AttachConsole 與硬體自適應校準下沉至 launcher.py | 🟠 Architecture | [Issue #27](https://github.com/saijo0404/DiarizeFlow/issues/27) (Open) |
| **#13** | 修正 WebSocket 事件處理遺漏 speaker_deleted 導致 HUD 渲染空白字幕卡片問題 | 🟠 Bug / Frontend | [Issue #28](https://github.com/saijo0404/DiarizeFlow/issues/28) (Open) |
| **#14** | 增強 AppConfig.from_dict 反序列化防呆容錯：安全過濾未知欄位防止啟動崩潰 | 🟡 Robustness | [Issue #29](https://github.com/saijo0404/DiarizeFlow/issues/29) (Open) |
| **#15** | 麥克風與系統音訊雙軌分流處理與智慧活動動態路由 (Dual-Track Decoupled Routing) | 🔵 Audio / DSP | [Issue #30](https://github.com/saijo0404/DiarizeFlow/issues/30) (Open) |
| **#16** | 消除 Sortformer 重複推論開銷：復用串流 SAD 聲軌資訊並優化 identify_speaker 嵌入萃取 | ⚡ Performance | [Issue #31](https://github.com/saijo0404/DiarizeFlow/issues/31) (Open) |
| **#17** | 補齊前後端分離模式下的設定同步機制：前端透過 REST API POST /api/config 即時更新後端狀態 | 🔵 Architecture | [Issue #32](https://github.com/saijo0404/DiarizeFlow/issues/32) (Open) |
| **#18** | LLMTranslator 採用持久化 aiohttp.ClientSession 連線池以降低 HTTP 連線與 TLS 握手延遲 | ⚡ Performance | [Issue #33](https://github.com/saijo0404/DiarizeFlow/issues/33) (Open) |
| **#19** | HUD 支援自訂視窗幾何記憶 (Window Position & Geometry Persistence) | 🎨 UI/UX | [Issue #34](https://github.com/saijo0404/DiarizeFlow/issues/34) (Open) |
| **#20** | 將 nemo-toolkit 與 onnxsim 移至可選依賴 [project.optional-dependencies] export 並修復 Pytest 測試路徑 | 📦 Packaging | [Issue #35](https://github.com/saijo0404/DiarizeFlow/issues/35) (Open) |
| **#21** | 拆解龐大模組 desktop_overlay.py：重構為模組化 Widgets、對話框與視窗元件 | 🧹 Refactor | [Issue #36](https://github.com/saijo0404/DiarizeFlow/issues/36) (Open) |

---

## Local Markdown Files
* [`01_audio_desync_mixing_bug.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/01_audio_desync_mixing_bug.md)
* [`02_config_hardware_calibrated_dropped.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/02_config_hardware_calibrated_dropped.md)
* [`03_missing_dependencies.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/03_missing_dependencies.md)
* [`04_double_agc_and_resampling.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/04_double_agc_and_resampling.md)
* [`05_quantization_benchmarks.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/05_quantization_benchmarks.md)
* [`06_persistent_voiceprint_profiles.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/06_persistent_voiceprint_profiles.md)
* [`07_diarization_driven_streaming_segmentation.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/07_diarization_driven_streaming_segmentation.md)
* [`08_hud_multi_card_bottom_anchoring.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/08_hud_multi_card_bottom_anchoring.md)
* [`09_concurrent_llm_translation_asyncio_gather.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/09_concurrent_llm_translation_asyncio_gather.md)
* [`10_deprecate_and_remove_web_frontend.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/10_deprecate_and_remove_web_frontend.md)
* [`11_fix_clickthrough_deadlock.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/11_fix_clickthrough_deadlock.md)
* [`12_unify_launcher_entrypoint.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/12_unify_launcher_entrypoint.md)
* [`13_fix_websocket_speaker_deleted_event.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/13_fix_websocket_speaker_deleted_event.md)
* [`14_config_deserialization_safeguard.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/14_config_deserialization_safeguard.md)
* [`15_dual_track_audio_routing.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/15_dual_track_audio_routing.md)
* [`16_eliminate_redundant_sortformer_inference.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/16_eliminate_redundant_sortformer_inference.md)
* [`17_standalone_frontend_config_sync.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/17_standalone_frontend_config_sync.md)
* [`18_persistent_llm_client_session.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/18_persistent_llm_client_session.md)
* [`19_window_geometry_persistence.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/19_window_geometry_persistence.md)
* [`20_decouple_nemo_dependency_and_fix_pytest.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/20_decouple_nemo_dependency_and_fix_pytest.md)
* [`21_decompose_monolithic_desktop_overlay.md`](file:///home/yijun/Project/DiarizeFlow/.github/issues/21_decompose_monolithic_desktop_overlay.md)
