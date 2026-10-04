#!/usr/bin/env python3
"""Script to publish DiarizeFlow GitHub Issues with automatic duplicate prevention."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import urllib.error

# Force unbuffered output immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

project_root = Path(__file__).resolve().parent.parent

ISSUES = [
    {
        "id": "1",
        "title": "[Bug]: Audio desynchronization and temporal expansion when mixing microphone and loopback audio",
        "file": project_root / ".github" / "issues" / "01_audio_desync_mixing_bug.md",
        "labels": ["bug", "audio", "high-priority"],
    },
    {
        "id": "2",
        "title": "[Bug]: AppConfig.from_dict drops hardware_calibrated causing calibration to re-run on every launch",
        "file": project_root / ".github" / "issues" / "02_config_hardware_calibrated_dropped.md",
        "labels": ["bug", "config"],
    },
    {
        "id": "3",
        "title": "[Packaging]: Missing critical runtime dependencies in pyproject.toml (soundcard, faster-whisper, ml_dtypes)",
        "file": project_root / ".github" / "issues" / "03_missing_dependencies.md",
        "labels": ["packaging", "dependencies"],
    },
    {
        "id": "4",
        "title": "[Audio/DSP]: Redundant double AGC processing and aliasing distortion in chunk resampling",
        "file": project_root / ".github" / "issues" / "04_double_agc_and_resampling.md",
        "labels": ["audio", "enhancement"],
    },
    {
        "id": "5",
        "title": "[Architecture/Benchmark]: Simulated weight-only quantizations (FP8/NVFP4/MXFP4) lack native ONNX operators and benchmark transparency",
        "file": project_root / ".github" / "issues" / "05_quantization_benchmarks.md",
        "labels": ["enhancement", "documentation"],
    },
    {
        "id": "6",
        "title": "[Feature]: Persistent Voiceprint Profiles and Custom Speaker Identification (永久聲紋資料庫與自訂講者身份辨識)",
        "file": project_root / ".github" / "issues" / "06_persistent_voiceprint_profiles.md",
        "labels": ["enhancement", "diarization", "ui"],
    },
    {
        "id": "7",
        "title": "[Architecture/Diarization]: End-to-End Streaming Diarization-Driven ASR Segmentation (移除傳統 VAD，由 Nemotron Sortformer SAD 驅動多軌音訊分流)",
        "file": project_root / ".github" / "issues" / "07_diarization_driven_streaming_segmentation.md",
        "labels": ["architecture", "diarization", "audio"],
    },
    {
        "id": "8",
        "title": "[UI/UX]: HUD 支援多卡片佇列與視窗底部對齊錨定 (Bottom-Anchoring)，避免多人連續/同步說話覆蓋",
        "file": project_root / ".github" / "issues" / "08_hud_multi_card_bottom_anchoring.md",
        "labels": ["ui", "enhancement", "frontend"],
    },
    {
        "id": "9",
        "title": "[Performance/Pipeline]: LLM 翻譯改用 asyncio.gather 並行非同步請求，降低多講者與多段落串聯延遲",
        "file": project_root / ".github" / "issues" / "09_concurrent_llm_translation_asyncio_gather.md",
        "labels": ["performance", "pipeline"],
    },
    {
        "id": "10",
        "title": "[RFC/Refactor]: Deprecate and Remove Web Frontend in favor of Native PySide6 Desktop HUD (建議完全移除 Web 前端，專注於原生桌面體驗)",
        "file": project_root / ".github" / "issues" / "10_deprecate_and_remove_web_frontend.md",
        "labels": ["refactor", "cleanup"],
    },
    {
        "id": "11",
        "title": "[UI/UX]: 解決 HUD 滑鼠穿透模式 (Click-Through) 死鎖問題：支援系統托盤 (QSystemTrayIcon)、控制條獨立保護與全域快捷鍵",
        "file": project_root / ".github" / "issues" / "11_fix_clickthrough_deadlock.md",
        "labels": ["ui", "bug", "high-priority"],
    },
    {
        "id": "12",
        "title": "[Architecture/Refactor]: 統一應用程式啟動入口邏輯：將 DualLogger、AttachConsole 與硬體自適應校準下沉至 launcher.py",
        "file": project_root / ".github" / "issues" / "12_unify_launcher_entrypoint.md",
        "labels": ["architecture", "refactor", "high-priority"],
    },
    {
        "id": "13",
        "title": "[Bug/Frontend]: 修正 WebSocket 事件處理遺漏 speaker_deleted 導致 HUD 渲染空白字幕卡片問題",
        "file": project_root / ".github" / "issues" / "13_fix_websocket_speaker_deleted_event.md",
        "labels": ["bug", "ui", "frontend"],
    },
    {
        "id": "14",
        "title": "[Robustness/Config]: 增強 AppConfig.from_dict 反序列化防呆容錯：安全過濾未知欄位防止啟動崩潰",
        "file": project_root / ".github" / "issues" / "14_config_deserialization_safeguard.md",
        "labels": ["bug", "config", "robustness"],
    },
    {
        "id": "15",
        "title": "[Audio/DSP]: 麥克風與系統音訊雙軌分流處理與智慧活動動態路由 (Dual-Track Decoupled Routing)",
        "file": project_root / ".github" / "issues" / "15_dual_track_audio_routing.md",
        "labels": ["audio", "enhancement", "architecture"],
    },
    {
        "id": "16",
        "title": "[Performance/Diarization]: 消除 Sortformer 重複推論開銷：復用串流 SAD 聲軌資訊並優化 identify_speaker 嵌入萃取",
        "file": project_root / ".github" / "issues" / "16_eliminate_redundant_sortformer_inference.md",
        "labels": ["performance", "diarization", "high-priority"],
    },
    {
        "id": "17",
        "title": "[Architecture/Feature]: 補齊前後端分離模式下的設定同步機制：前端透過 REST API POST /api/config 即時更新後端狀態",
        "file": project_root / ".github" / "issues" / "17_standalone_frontend_config_sync.md",
        "labels": ["architecture", "frontend", "api", "high-priority"],
    },
    {
        "id": "18",
        "title": "[Performance/Translation]: LLMTranslator 採用持久化 aiohttp.ClientSession 連線池以降低 HTTP 連線與 TLS 握手延遲",
        "file": project_root / ".github" / "issues" / "18_persistent_llm_client_session.md",
        "labels": ["performance", "translation"],
    },
    {
        "id": "19",
        "title": "[UI/UX]: HUD 支援自訂視窗幾何記憶 (Window Position & Geometry Persistence)",
        "file": project_root / ".github" / "issues" / "19_window_geometry_persistence.md",
        "labels": ["ui", "enhancement"],
    },
    {
        "id": "20",
        "title": "[Packaging/DevOps]: 將 nemo-toolkit 與 onnxsim 移至可選依賴 [project.optional-dependencies] export 並修復 Pytest 測試路徑",
        "file": project_root / ".github" / "issues" / "20_decouple_nemo_dependency_and_fix_pytest.md",
        "labels": ["packaging", "dependencies", "devops"],
    },
    {
        "id": "21",
        "title": "[Refactor/Code Quality]: 拆解龐大模組 desktop_overlay.py：重構為模組化 Widgets、對話框與視窗元件",
        "file": project_root / ".github" / "issues" / "21_decompose_monolithic_desktop_overlay.md",
        "labels": ["refactor", "frontend"],
    },
    {
        "id": "22",
        "title": "[Refactor/Backend]: FastAPI Lifespan 現代化升級：遷移 on_event 為 lifespan context manager 並消除 DeprecationWarning",
        "file": project_root / ".github" / "issues" / "22_fastapi_lifespan_modernization.md",
        "labels": ["refactor", "backend", "high-priority"],
    },
    {
        "id": "23",
        "title": "[Performance/DSP]: Mel Spectrogram 濾波矩陣與窗函數快取：消除串流循環中重複構建開銷",
        "file": project_root / ".github" / "issues" / "23_mel_spectrogram_filterbank_cache.md",
        "labels": ["performance", "audio", "diarization"],
    },
    {
        "id": "24",
        "title": "[Documentation]: 同步更新 README.md 與架構文檔：反映模組化拆分、REST API 與雙軌路由",
        "file": project_root / ".github" / "issues" / "24_update_readme_and_architecture_docs.md",
        "labels": ["documentation"],
    },
    {
        "id": "25",
        "title": "[Tooling/UX]: 實作一鍵模型下載與初次引導腳本 (scripts/download_models.py)",
        "file": project_root / ".github" / "issues" / "25_one_click_model_download_script.md",
        "labels": ["enhancement", "tooling", "ux"],
    },
    {
        "id": "26",
        "title": "[Architecture/Network]: 擴充 WebSocket 音訊串流協定 (/ws/audio)：支援軌道標籤與元資料標頭",
        "file": project_root / ".github" / "issues" / "26_websocket_audio_streaming_protocol_headers.md",
        "labels": ["architecture", "network", "audio"],
    },
    {
        "id": "27",
        "title": "[UI/UX]: Linux Wayland 環境感知提示：偵測顯示伺服器協議並指引托盤與視窗穿透操作",
        "file": project_root / ".github" / "issues" / "27_linux_wayland_compositor_hints.md",
        "labels": ["ui", "linux", "enhancement"],
    },
    {
        "id": "28",
        "title": "[Feature/Audio-DSP]: 引入目標講者語音提取 (Target-Speaker Extraction / TSE) 模型：解決單音軌多講者同步重疊說話 (Overlap Speech) 波形分離難題",
        "file": project_root / ".github" / "issues" / "28_target_speaker_extraction_tse.md",
        "labels": ["enhancement", "audio", "diarization", "architecture"],
    },
    {
        "id": "29",
        "title": "[Bug/Build]: build_executable.py 在 uv 虛擬環境中因缺失 pip 模組導致自動安裝 PyInstaller 崩潰",
        "file": project_root / ".github" / "issues" / "29_fix_build_executable_pip_missing_in_uv.md",
        "labels": ["bug", "packaging", "build", "high-priority"],
    },
]

REPO = "saijo0404/DiarizeFlow"


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy duplicate detection."""
    return "".join(c for c in title.lower() if c.isalnum())


def check_gh_cli():
    gh_bin = shutil.which("gh")
    if not gh_bin:
        return False, "gh CLI is not installed"
    
    try:
        res = subprocess.run([gh_bin, "auth", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8)
        if res.returncode != 0:
            return False, f"gh not logged in: {res.stderr.strip() or res.stdout.strip()}"
        return True, gh_bin
    except Exception as e:
        return False, f"gh error: {e}"


def fetch_existing_issues_gh(gh_bin: str) -> dict:
    """Fetch existing issues via gh CLI and map normalized titles to issue info."""
    cmd = [
        gh_bin, "issue", "list",
        "--repo", REPO,
        "--state", "all",
        "--limit", "100",
        "--json", "number,title,state,url",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            return {normalize_title(item["title"]): item for item in data}
    except Exception as e:
        print(f"[!] Warning: Failed to query existing issues: {e}", flush=True)
    return {}


def publish_via_gh(gh_bin: str, force: bool = False):
    existing_map = {} if force else fetch_existing_issues_gh(gh_bin)
    created = []
    skipped = []
    failed = []

    for issue in ISSUES:
        title = issue["title"]
        file_path = issue["file"]
        labels = ",".join(issue["labels"])
        norm = normalize_title(title)

        # Duplicate check
        if not force and norm in existing_map:
            ex = existing_map[norm]
            print(f"  [⏩ SKIP] Issue #{issue['id']} already exists: #{ex['number']} ({ex['state']}) -> {ex['url']}", flush=True)
            skipped.append((title, ex["url"]))
            continue

        if not file_path.exists():
            print(f"[-] Issue file not found: {file_path}", flush=True)
            failed.append((title, "File not found"))
            continue

        print(f"[*] Publishing Issue #{issue['id']}: {title}...", flush=True)
        
        # Try with labels
        cmd = [
            gh_bin, "issue", "create",
            "--repo", REPO,
            "--title", title,
            "--body-file", str(file_path),
            "--label", labels,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        except Exception as e:
            print(f"    [✗] Timed out or failed: {e}", flush=True)
            failed.append((title, str(e)))
            continue
        
        # If failed due to label, try without labels
        if res.returncode != 0 and "label" in res.stderr.lower():
            print("    [!] Label not found on repo, retrying without labels...", flush=True)
            cmd = [
                gh_bin, "issue", "create",
                "--repo", REPO,
                "--title", title,
                "--body-file", str(file_path),
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            except Exception as e:
                print(f"    [✗] Timed out or failed: {e}", flush=True)
                failed.append((title, str(e)))
                continue

        if res.returncode == 0:
            url = res.stdout.strip()
            print(f"    [✓] Created: {url}", flush=True)
            created.append((title, url))
        else:
            err = res.stderr.strip()
            print(f"    [✗] Failed: {err}", flush=True)
            failed.append((title, err))

    return created, skipped, failed


def fetch_existing_issues_api(token: str) -> dict:
    """Fetch existing issues via GitHub REST API."""
    url = f"https://api.github.com/repos/{REPO}/issues?state=all&per_page=100"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DiarizeFlow-Issue-Publisher",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {normalize_title(item["title"]): {"number": item["number"], "title": item["title"], "url": item.get("html_url", ""), "state": item.get("state", "")} for item in data}
    except Exception as e:
        print(f"[!] Warning: Failed to query existing issues via API: {e}", flush=True)
    return {}


def publish_via_api(token: str, force: bool = False):
    existing_map = {} if force else fetch_existing_issues_api(token)
    created = []
    skipped = []
    failed = []

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DiarizeFlow-Issue-Publisher",
    }
    url = f"https://api.github.com/repos/{REPO}/issues"

    for issue in ISSUES:
        title = issue["title"]
        file_path = issue["file"]
        norm = normalize_title(title)

        if not force and norm in existing_map:
            ex = existing_map[norm]
            print(f"  [⏩ SKIP] Issue #{issue['id']} already exists: #{ex['number']} ({ex['state']}) -> {ex['url']}", flush=True)
            skipped.append((title, ex["url"]))
            continue

        if not file_path.exists():
            continue
        
        body = file_path.read_text(encoding="utf-8")
        payload = {
            "title": title,
            "body": body,
            "labels": issue["labels"],
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                issue_url = data.get("html_url", "")
                print(f"    [✓] Created: {issue_url}", flush=True)
                created.append((title, issue_url))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")
            if e.code == 422:
                payload.pop("labels", None)
                req2 = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req2) as resp2:
                        data2 = json.loads(resp2.read().decode("utf-8"))
                        issue_url = data2.get("html_url", "")
                        print(f"    [✓] Created: {issue_url}", flush=True)
                        created.append((title, issue_url))
                        continue
                except Exception as ex:
                    err = str(ex)
            print(f"    [✗] Failed to create '{title}': HTTP {e.code} - {err}", flush=True)
            failed.append((title, err))
        except Exception as e:
            print(f"    [✗] Error: {e}", flush=True)
            failed.append((title, str(e)))

    return created, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Publish DiarizeFlow GitHub Issues with duplicate prevention")
    parser.add_argument("--force", action="store_true", help="Force publication without checking for duplicates")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f"🚀 Publishing DiarizeFlow Issues to GitHub ({REPO})", flush=True)
    if not args.force:
        print("🛡️  智慧去重模式已啟用 (已存在的 Issue 將自動略過)", flush=True)
    print("=" * 70, flush=True)

    # 1. Check gh CLI
    has_gh, info = check_gh_cli()
    if has_gh:
        print(f"[✓] Using GitHub CLI: {info}", flush=True)
        created, skipped, failed = publish_via_gh(info, force=args.force)
        print("\nSummary:", flush=True)
        print(f"  Created: {len(created)}")
        print(f"  Skipped (Already exists): {len(skipped)}")
        if failed:
            print(f"  Failed: {len(failed)}")
        return 0 if not failed else 1

    print(f"[-] gh CLI check: {info}", flush=True)

    # 2. Check GitHub Token in environment
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        print("[✓] Found GITHUB_TOKEN / GH_TOKEN in environment. Publishing via REST API...", flush=True)
        created, skipped, failed = publish_via_api(token, force=args.force)
        print("\nSummary:", flush=True)
        print(f"  Created: {len(created)}")
        print(f"  Skipped (Already exists): {len(skipped)}")
        if failed:
            print(f"  Failed: {len(failed)}")
        return 0 if not failed else 1

    print("\n[!] Neither authenticated `gh` CLI nor `GITHUB_TOKEN` is available in this environment.", flush=True)
    print("    To publish, please run one of the following:", flush=True)
    print("      1) `gh auth login` in your terminal, then run `python scripts/publish_issues.py`", flush=True)
    print("      2) Or export GITHUB_TOKEN=<your_token> and run `python scripts/publish_issues.py`", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
