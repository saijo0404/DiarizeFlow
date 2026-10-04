"""Model catalog, integrity validation, and automated downloading toolkit for DiarizeFlow.

Provides:
1. Model specifications for Nemotron-3 Diarization and SenseVoiceSmall ASR.
2. Hardware-aware precision recommendation (GPU -> FP16, CPU -> INT8).
3. Resumable downloading with HTTP Range headers, SHA256 integrity verification,
   and visual terminal progress bars.
4. Hugging Face and ModelScope mirrors with HF_ENDPOINT support.
5. Startup detection and friendly guidance (check_missing_models).
"""

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

try:
    import requests
except ImportError:
    requests = None

from diarizeflow.app.config import AppConfig, resolve_app_path
from diarizeflow.hardware import QuantPrecision, get_system_device_info


@dataclass
class ModelFileSpec:
    """Specification for a downloadable model file."""
    rel_path: str
    description: str
    category: str  # "sensevoice", "nemotron"
    precision: str  # "all", "fp32", "fp16", "int8"
    urls: Dict[str, str]  # source -> url
    sha256: Optional[str] = None
    estimated_size_mb: float = 0.0
    required: bool = True


# Comprehensive model registry for DiarizeFlow
MODEL_CATALOG: List[ModelFileSpec] = [
    # SenseVoiceSmall configs & assets (required for all precisions)
    ModelFileSpec(
        rel_path="sensevoice_small/am.mvn",
        description="SenseVoice CMVN 聲學特徵正規化檔",
        category="sensevoice",
        precision="all",
        urls={
            "huggingface": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/am.mvn",
            "modelscope": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/am.mvn",
        },
        sha256="29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
        estimated_size_mb=0.03,
        required=True,
    ),
    ModelFileSpec(
        rel_path="sensevoice_small/chn_jpn_yue_eng_ko_spectok.bpe.model",
        description="SenseVoice 多語言 BPE 分詞模型",
        category="sensevoice",
        precision="all",
        urls={
            "huggingface": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/chn_jpn_yue_eng_ko_spectok.bpe.model",
            "modelscope": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/chn_jpn_yue_eng_ko_spectok.bpe.model",
        },
        sha256="aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8",
        estimated_size_mb=0.45,
        required=True,
    ),
    ModelFileSpec(
        rel_path="sensevoice_small/config.yaml",
        description="SenseVoice 架構設定檔",
        category="sensevoice",
        precision="all",
        urls={
            "huggingface": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/config.yaml",
            "modelscope": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/config.yaml",
        },
        estimated_size_mb=0.01,
        required=False,
    ),

    # SenseVoiceSmall ONNX Models
    ModelFileSpec(
        rel_path="sensevoice_small/SenseVoiceSmall.onnx",
        description="SenseVoiceSmall 語音辨識模型 (FP32 原始精度)",
        category="sensevoice",
        precision="fp32",
        urls={
            "huggingface": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/model.onnx",
            "modelscope": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model.onnx",
        },
        estimated_size_mb=897.0,
        required=True,
    ),
    ModelFileSpec(
        rel_path="sensevoice_small/SenseVoiceSmall_fp16.onnx",
        description="SenseVoiceSmall 語音辨識模型 (FP16 GPU 加速)",
        category="sensevoice",
        precision="fp16",
        urls={
            "huggingface": "https://huggingface.co/saijo0404/DiarizeFlow-models/resolve/main/sensevoice_small/SenseVoiceSmall_fp16.onnx",
            "modelscope": "https://www.modelscope.cn/models/saijo0404/DiarizeFlow-models/resolve/master/sensevoice_small/SenseVoiceSmall_fp16.onnx",
        },
        estimated_size_mb=4.4,  # External weight ONNX graph or self-contained
        required=True,
    ),
    ModelFileSpec(
        rel_path="sensevoice_small/SenseVoiceSmall_int8.onnx",
        description="SenseVoiceSmall 語音辨識模型 (INT8 CPU 最速輕量化)",
        category="sensevoice",
        precision="int8",
        urls={
            "huggingface": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
            "modelscope": "https://www.modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model_quant.onnx",
        },
        estimated_size_mb=231.0,
        required=True,
    ),

    # Nemotron-3 Diarization ONNX Models
    ModelFileSpec(
        rel_path="nemotron_diarization/Nemotron-3-Diarization.onnx",
        description="NVIDIA Nemotron-3 語者分離模型 (FP32 基準)",
        category="nemotron",
        precision="fp32",
        urls={
            "huggingface": "https://huggingface.co/saijo0404/DiarizeFlow-models/resolve/main/nemotron_diarization/Nemotron-3-Diarization.onnx",
            "modelscope": "https://www.modelscope.cn/models/saijo0404/DiarizeFlow-models/resolve/master/nemotron_diarization/Nemotron-3-Diarization.onnx",
        },
        estimated_size_mb=380.0,
        required=True,
    ),
    ModelFileSpec(
        rel_path="nemotron_diarization/Nemotron-3-Diarization_fp16.onnx",
        description="NVIDIA Nemotron-3 語者分離模型 (FP16 GPU Tensor Core 加速)",
        category="nemotron",
        precision="fp16",
        urls={
            "huggingface": "https://huggingface.co/saijo0404/DiarizeFlow-models/resolve/main/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx",
            "modelscope": "https://www.modelscope.cn/models/saijo0404/DiarizeFlow-models/resolve/master/nemotron_diarization/Nemotron-3-Diarization_fp16.onnx",
        },
        estimated_size_mb=191.0,
        required=True,
    ),
    ModelFileSpec(
        rel_path="nemotron_diarization/Nemotron-3-Diarization_int8.onnx",
        description="NVIDIA Nemotron-3 語者分離模型 (INT8 CPU 極速輕量化)",
        category="nemotron",
        precision="int8",
        urls={
            "huggingface": "https://huggingface.co/saijo0404/DiarizeFlow-models/resolve/main/nemotron_diarization/Nemotron-3-Diarization_int8.onnx",
            "modelscope": "https://www.modelscope.cn/models/saijo0404/DiarizeFlow-models/resolve/master/nemotron_diarization/Nemotron-3-Diarization_int8.onnx",
        },
        estimated_size_mb=100.0,
        required=True,
    ),
]


def resolve_model_url(spec: ModelFileSpec, source: str = "huggingface", hf_mirror: Optional[str] = None) -> str:
    """Resolve download URL for a model specification based on mirror source and HF endpoint."""
    url = spec.urls.get(source) or spec.urls.get("huggingface") or list(spec.urls.values())[0]
    if source == "huggingface":
        mirror = hf_mirror or os.environ.get("HF_ENDPOINT")
        if mirror:
            mirror = mirror.rstrip("/")
            if url.startswith("https://huggingface.co"):
                url = url.replace("https://huggingface.co", mirror, 1)
    return url


def get_default_models_dir() -> Path:
    """Return default models directory path resolved against project root."""
    return resolve_app_path("models")


def format_size(bytes_val: int) -> str:
    """Format byte size into human readable string (KB, MB, GB)."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_recommended_precision() -> str:
    """Detect local hardware and return recommended precision ('fp16' for GPU, 'int8' for CPU)."""
    try:
        info = get_system_device_info()
        if info.has_gpu:
            return "fp16"
        return "int8"
    except Exception:
        return "int8"


def get_target_model_specs(
    models: str = "all",
    precision: str = "auto",
) -> List[ModelFileSpec]:
    """Filter model catalog specifications matching models category and precision."""
    actual_prec = detect_recommended_precision() if precision == "auto" else precision.lower()
    selected: List[ModelFileSpec] = []

    for spec in MODEL_CATALOG:
        # Check category
        if models != "all" and spec.category != models:
            continue

        # Check precision: "all" specs are always included; precision-specific specs match requested
        if spec.precision == "all":
            selected.append(spec)
        elif actual_prec == "all" or spec.precision == actual_prec:
            selected.append(spec)

    return selected


def check_missing_models(
    config: Optional[AppConfig] = None,
    target_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Check whether mandatory pre-trained models exist in target_dir.

    Returns:
        List of missing human-readable model descriptions (empty if all exist).
    """
    models_dir = Path(target_dir) if target_dir else get_default_models_dir()
    missing: List[str] = []

    # 1. SenseVoice CMVN and BPE tokenizer
    cmvn_file = models_dir / "sensevoice_small" / "am.mvn"
    bpe_file = models_dir / "sensevoice_small" / "chn_jpn_yue_eng_ko_spectok.bpe.model"
    if not cmvn_file.exists():
        missing.append("SenseVoice CMVN 特徵統計 (am.mvn)")
    if not bpe_file.exists():
        missing.append("SenseVoice 分詞模型 (bpe.model)")

    # 2. SenseVoice ONNX model (any precision)
    sense_dir = models_dir / "sensevoice_small"
    has_sense_model = any(
        (sense_dir / name).exists()
        for name in [
            "SenseVoiceSmall.onnx",
            "SenseVoiceSmall_fp16.onnx",
            "SenseVoiceSmall_int8.onnx",
            "model.onnx",
            "model_quant.onnx",
            "model.int8.onnx",
        ]
    )
    if not has_sense_model:
        missing.append("SenseVoiceSmall 語音辨識 ONNX 模型")

    # 3. Nemotron Diarization ONNX model (any precision)
    nemotron_dir = models_dir / "nemotron_diarization"
    has_nemotron_model = any(
        (nemotron_dir / name).exists()
        for name in [
            "Nemotron-3-Diarization.onnx",
            "Nemotron-3-Diarization_fp16.onnx",
            "Nemotron-3-Diarization_int8.onnx",
            "Nemotron-3-Diarization.nemo",
        ]
    )
    if not has_nemotron_model:
        missing.append("NVIDIA Nemotron-3 語者分離 ONNX 模型")

    return missing


def render_progress_bar(
    filename: str,
    downloaded_bytes: int,
    total_bytes: Optional[int],
    elapsed_time: float,
    finished: bool = False,
    bar_width: int = 25,
) -> str:
    """Render terminal progress bar with byte counter, speed, and ETA."""
    speed_bps = downloaded_bytes / max(elapsed_time, 0.001)
    speed_str = f"{format_size(int(speed_bps))}/s"

    if total_bytes and total_bytes > 0:
        percent = min(100.0, downloaded_bytes / total_bytes * 100.0)
        filled = int(round(bar_width * percent / 100.0))
        bar = "=" * filled + (">" if filled < bar_width else "")
        bar = bar.ljust(bar_width)

        if finished:
            return f"\r   {filename}: [{bar}] 100.0% ({format_size(total_bytes)}) [✓ 完成]"
        else:
            remaining_bytes = max(0, total_bytes - downloaded_bytes)
            eta_s = int(remaining_bytes / max(speed_bps, 1))
            eta_str = f"ETA {eta_s}s" if eta_s < 3600 else f"ETA {eta_s // 3600}h"
            return f"\r   {filename}: [{bar}] {percent:5.1f}% ({format_size(downloaded_bytes)} / {format_size(total_bytes)}) [{speed_str}, {eta_str}]"
    else:
        # Unknown total length
        spin = ["-", "\\", "|", "/"][int(elapsed_time * 4) % 4]
        return f"\r   {filename}: [{spin}] {format_size(downloaded_bytes)} 下載中... [{speed_str}]"


def download_file(
    url: str,
    target_path: Path,
    expected_sha256: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    session=None,
    chunk_size: int = 128 * 1024,
) -> bool:
    """Download a file with HTTP Range resumability and SHA256 integrity verification.

    Args:
        url: Direct download URL.
        target_path: Destination local file path.
        expected_sha256: Optional expected SHA256 checksum.
        force: Whether to overwrite existing destination file.
        dry_run: If True, check and report without network download.
        quiet: If True, suppress progress bar and verbose output.
        session: Optional persistent requests.Session.
        chunk_size: Chunk size in bytes for streaming.

    Returns:
        True if file exists or was successfully downloaded and verified, False otherwise.
    """
    target_path = Path(target_path)
    part_path = target_path.with_name(target_path.name + ".part")

    # 1. Existing file check
    if target_path.exists() and not force:
        if expected_sha256:
            calc_hash = calculate_sha256(target_path)
            if calc_hash.lower() == expected_sha256.lower():
                if not quiet:
                    print(f"[✓] 檔案已存在且校驗正確 (SHA256 OK): {target_path.name}")
                return True
            else:
                if not quiet:
                    print(f"[!] 檔案校驗不符，將重新下載: {target_path.name}")
        else:
            if target_path.stat().st_size > 0:
                if not quiet:
                    print(f"[✓] 檔案已存在 (跳過): {target_path.name} ({format_size(target_path.stat().st_size)})")
                return True

    if dry_run:
        if not quiet:
            print(f"[Dry-Run] 擬下載: {target_path.name} <- {url}")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Resumable download headers
    headers = {"User-Agent": "DiarizeFlow-ModelDownloader/1.0"}
    resume_bytes = 0
    if part_path.exists() and not force:
        resume_bytes = part_path.stat().st_size
        if resume_bytes > 0:
            headers["Range"] = f"bytes={resume_bytes}-"

    req_fn = session.get if session is not None else requests.get
    try:
        resp = req_fn(url, headers=headers, stream=True, timeout=(10, 60))
    except Exception as net_err:
        print(f"\n[!] 連線下載失敗 ({target_path.name}): {net_err}")
        return False

    # Handle Range response
    is_resumed = resp.status_code == 206
    if resp.status_code == 416:  # Range Not Satisfiable (already completed in .part)
        pass
    elif resp.status_code not in (200, 206):
        print(f"\n[!] 伺服器回應錯誤 (HTTP {resp.status_code}): {url}")
        return False

    total_bytes = None
    if "Content-Range" in resp.headers:
        # e.g. "bytes 100-999/1000"
        try:
            total_bytes = int(resp.headers["Content-Range"].split("/")[-1])
        except Exception:
            pass
    elif "Content-Length" in resp.headers:
        total_bytes = int(resp.headers["Content-Length"])
        if is_resumed:
            total_bytes += resume_bytes

    if not is_resumed:
        resume_bytes = 0

    mode = "ab" if is_resumed and resume_bytes > 0 else "wb"
    downloaded = resume_bytes
    t_start = time.perf_counter()

    if not quiet:
        resume_hint = f" (斷點續傳從 {format_size(resume_bytes)} 開始)" if is_resumed and resume_bytes > 0 else ""
        print(f"[*] 下載: {target_path.name}{resume_hint}")

    try:
        with open(part_path, mode) as f:
            if resp.status_code != 416:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if not quiet and sys.stdout.isatty():
                            elapsed = time.perf_counter() - t_start
                            sys.stdout.write(render_progress_bar(target_path.name, downloaded, total_bytes, elapsed))
                            sys.stdout.flush()

        if not quiet:
            elapsed = time.perf_counter() - t_start
            sys.stdout.write(render_progress_bar(target_path.name, downloaded, total_bytes, elapsed, finished=True) + "\n")
            sys.stdout.flush()

    except Exception as e:
        print(f"\n[!] 寫入或下載中斷: {e}")
        return False

    # 3. Checksum verification
    if expected_sha256:
        if not quiet:
            print(f"   [校驗] 正在比對 SHA256 完整性雜湊值...")
        calc_sha = calculate_sha256(part_path)
        if calc_sha.lower() != expected_sha256.lower():
            print(f"[!] 錯誤: SHA256 校驗失敗！\n    預期: {expected_sha256}\n    實得: {calc_sha}")
            try:
                part_path.unlink()
            except Exception:
                pass
            return False
        if not quiet:
            print("   [校驗] SHA256 完整性校驗通過 [✓]")

    # 4. Atomic rename from .part to final destination
    try:
        part_path.replace(target_path)
    except Exception as e:
        print(f"[!] 檔案移動失敗: {e}")
        return False

    return True


def download_models(
    models: str = "all",
    precision: str = "auto",
    source: str = "huggingface",
    hf_mirror: Optional[str] = None,
    target_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> bool:
    """Main workflow to download pre-trained speech models according to parameters."""
    base_dir = Path(target_dir) if target_dir else get_default_models_dir()
    specs = get_target_model_specs(models=models, precision=precision)
    actual_prec = detect_recommended_precision() if precision == "auto" else precision.lower()

    if not quiet:
        print("\n" + "=" * 75)
        print("📦 DiarizeFlow 語音模型一鍵下載與引導配置")
        print(f"   下載倉庫源: {source.upper()}" + (f" (鏡像: {hf_mirror})" if hf_mirror else ""))
        print(f"   目標硬體精度: {actual_prec.upper()}" + (" (硬體自動偵測推薦)" if precision == "auto" else " (手動指定)"))
        print(f"   模型儲存路徑: {base_dir}")
        print(f"   預計下載清單: 共 {len(specs)} 個檔案")
        print("=" * 75)

    if dry_run:
        print("\n[Dry-Run 預覽模式] 將處理以下檔案:")
        for idx, s in enumerate(specs, 1):
            target = base_dir / s.rel_path
            status = "已存在 [✓]" if target.exists() else "待下載 [⬇]"
            url = resolve_model_url(s, source=source, hf_mirror=hf_mirror)
            print(f" {idx:2d}. {s.description}")
            print(f"     路徑: {s.rel_path} ({status}, 約 {s.estimated_size_mb:.1f} MB)")
            print(f"     來源: {url}")
        print("\n[✓] Dry-run 完成，未變更任何本機檔案。")
        return True

    session = requests.Session() if requests is not None else None
    all_success = True

    for idx, spec in enumerate(specs, 1):
        target = base_dir / spec.rel_path
        url = resolve_model_url(spec, source=source, hf_mirror=hf_mirror)
        if not quiet:
            print(f"\n({idx}/{len(specs)}) 處理 [{spec.category}]: {spec.description}")

        ok = download_file(
            url=url,
            target_path=target,
            expected_sha256=spec.sha256,
            force=force,
            dry_run=dry_run,
            quiet=quiet,
            session=session,
        )
        if not ok:
            all_success = False
            print(f"[!] 下載失敗: {spec.rel_path}")

    if not quiet:
        print("\n" + "=" * 75)
        if all_success:
            print("🎉 所有必要語音模型已準備就緒！")
            print("   您現在可以直接執行以下指令啟動 DiarizeFlow：")
            print("       uv run diarizeflow-app")
        else:
            print("⚠️  部分模型下載失敗，請檢查網路連線或嘗試切換鏡像：")
            print("       uv run python scripts/download_models.py --source modelscope")
            print("       uv run python scripts/download_models.py --hf-mirror https://hf-mirror.com")
        print("=" * 75 + "\n")

    return all_success


def main():
    """CLI parser and entrypoint for scripts/download_models.py."""
    parser = argparse.ArgumentParser(
        description="DiarizeFlow 預訓練語音模型一鍵下載工具 (支援斷點續傳、SHA256 校驗與硬體自適應精度)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["all", "nemotron", "sensevoice"],
        help="欲下載的模型組合 (預設: all)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "int8", "fp32", "all"],
        help="目標量化精度: auto (自動偵測 GPU/CPU 推薦最優精度), fp16 (GPU), int8 (CPU 最速), fp32 (原始), all (全部下載)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="下載倉庫源 (預設: huggingface)",
    )
    parser.add_argument(
        "--hf-mirror",
        type=str,
        default=None,
        help="Hugging Face 國內鏡像站點 (例如 https://hf-mirror.com，預設自動讀取 $HF_ENDPOINT)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="模型儲存目標目錄 (預設: 專案 models/ 目錄)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新下載並覆蓋現有檔案",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅預覽下載清單與來源網址，不實際寫入任何檔案",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安靜模式，隱藏進度條與非錯誤提示",
    )

    args = parser.parse_args()
    success = download_models(
        models=args.models,
        precision=args.precision,
        source=args.source,
        hf_mirror=args.hf_mirror,
        target_dir=args.target_dir,
        force=args.force,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
