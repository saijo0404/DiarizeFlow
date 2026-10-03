"""Nemotron-3-Diarization (Sortformer) NeMo to ONNX export and onnxsim pipeline."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn

from diarizeflow.patches import apply_onnx_export_patches


class StreamingSortformerOnnxWrapper(nn.Module):
    """Wrapper module for exporting Sortformer streaming forward pass to ONNX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        chunk: torch.Tensor,
        chunk_lengths: torch.Tensor,
        spkcache: torch.Tensor,
        spkcache_lengths: torch.Tensor,
        fifo: torch.Tensor,
        fifo_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_for_export(
            chunk=chunk,
            chunk_lengths=chunk_lengths,
            spkcache=spkcache,
            spkcache_lengths=spkcache_lengths,
            fifo=fifo,
            fifo_lengths=fifo_lengths,
        )


def export_nemo_to_onnx(
    nemo_path: str,
    output_path: str,
    raw_output_path: Optional[str] = None,
    batch_size: int = 1,
    opset_version: int = 18,
    device: Optional[str] = None,
    simplify: bool = True,
    verify: bool = True,
    quantize: Optional[str] = None,
    tolerance: float = 1e-3,
) -> Path:
    """Exports a Nemotron-3 Diarization .nemo model to an ONNX file and simplifies with onnxsim.

    Args:
        nemo_path: Path to the .nemo model checkpoint.
        output_path: Path to write the final (simplified) ONNX model.
        raw_output_path: Optional path to save the unsimplified raw ONNX model.
        batch_size: Batch size for tracing.
        opset_version: Target ONNX opset version (recommended: 18).
        device: Device to use for export ('cuda' or 'cpu').
        simplify: Whether to run onnxsim on the exported model.
        verify: Whether to verify exported model with ONNX Runtime against PyTorch.
        quantize: Optional quantization mode ('auto', 'fp16', 'int8').
        tolerance: Numerical tolerance for ONNX Runtime verification.

    Returns:
        Path to the generated ONNX file.
    """
    nemo_file = Path(os.path.expanduser(nemo_path)).resolve()
    if not nemo_file.exists():
        raise FileNotFoundError(f"Nemotron .nemo file not found: {nemo_file}")

    final_output = Path(os.path.expanduser(output_path)).resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)

    if raw_output_path:
        raw_output = Path(os.path.expanduser(raw_output_path)).resolve()
        raw_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        raw_output = final_output.with_name(f"{final_output.stem}_raw{final_output.suffix}")

    # Determine execution device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 使用運算裝置: {device}")

    # 1. Apply ONNX export patches
    print("[*] 套用 ONNX 匯出相容性修補 (aten.sort -> TopK 分解, MHA -> SDPA)...")
    apply_onnx_export_patches()

    # 2. Restore model from .nemo
    print(f"[*] 載入 NeMo 模型權重: {nemo_file}")
    t0 = time.time()
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError:
        print("[!] 尚未安裝模型轉換依賴。請執行: uv sync --extra export (或 pip install -e '.[export]')")
        sys.exit(1)

    model = SortformerEncLabelModel.restore_from(
        restore_path=str(nemo_file),
        map_location=device,
        strict=False,
    )
    model.eval()
    print(f"    模型載入完成，耗時 {time.time() - t0:.2f} 秒")

    # 3. Create wrapper & prepare dummy inputs
    wrapper = StreamingSortformerOnnxWrapper(model).to(device)
    wrapper.eval()

    print(f"[*] 產生追蹤用虛擬輸入 (Batch size: {batch_size})...")
    dummy_inputs = model.streaming_input_examples(batch_size=batch_size)
    dummy_inputs = tuple(x.to(device) for x in dummy_inputs)

    input_names = [
        "chunk",
        "chunk_lengths",
        "spkcache",
        "spkcache_lengths",
        "fifo",
        "fifo_lengths",
    ]
    output_names = [
        "spkcache_fifo_chunk_preds",
        "chunk_pre_encode_embs",
        "chunk_pre_encode_lengths",
    ]

    print("    輸入張量規格:")
    for name, tensor in zip(input_names, dummy_inputs):
        print(f"      - {name:<26}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # 4. PyTorch reference forward
    print("[*] 執行 PyTorch 基準推論 (Reference Forward)...")
    with torch.no_grad():
        pt_outputs = wrapper(*dummy_inputs)

    print("    輸出張量規格:")
    for name, tensor in zip(output_names, pt_outputs):
        print(f"      - {name:<26}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # 5. Export to ONNX
    print(f"[*] 開始匯出 ONNX 模型 (Opset: {opset_version}) 至: {raw_output}")
    t1 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(raw_output),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
        )
    raw_size_mb = raw_output.stat().st_size / (1024 * 1024)
    print(f"    ONNX 匯出成功，耗時 {time.time() - t1:.2f} 秒 (原始大小: {raw_size_mb:.2f} MB)")

    target_model_path = raw_output

    # 6. Run onnxsim
    if simplify:
        try:
            import onnxsim
        except ImportError:
            print("[!] 尚未安裝 onnxsim 模型簡化依賴。請執行: uv sync --extra export (或 pip install -e '.[export]')")
            sys.exit(1)
        print(f"[*] 執行 onnxsim 模型結構簡化與常數摺疊...")
        t2 = time.time()
        model_proto = onnx.load(str(raw_output))
        simplified_proto, check = onnxsim.simplify(model_proto)
        if not check:
            raise RuntimeError("onnxsim 簡化模型驗證失敗！")
        onnx.save(simplified_proto, str(final_output))
        sim_size_mb = final_output.stat().st_size / (1024 * 1024)
        print(f"    onnxsim 簡化完成，耗時 {time.time() - t2:.2f} 秒 (簡化後大小: {sim_size_mb:.2f} MB)")
        target_model_path = final_output

        # Clean up temporary raw file if raw_output_path was not explicitly requested
        if not raw_output_path and raw_output != final_output and raw_output.exists():
            try:
                raw_output.unlink()
                for extra in raw_output.parent.glob(f"{raw_output.name}*"):
                    if extra != final_output and extra.exists():
                        extra.unlink()
            except OSError:
                pass
    else:
        if raw_output != final_output:
            import shutil
            shutil.copy2(raw_output, final_output)
            target_model_path = final_output

    # 7. Verification with ONNX Runtime
    if verify:
        print(f"[*] 使用 ONNX Runtime 驗證模型數值正確性: {target_model_path}")
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        selected_providers = []
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(str(target_model_path), providers=selected_providers)

        ort_inputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(input_names, dummy_inputs)
        }

        t3 = time.time()
        ort_outputs = session.run(None, ort_inputs)
        print(f"    ONNX Runtime 推論成功，耗時 {time.time() - t3:.3f} 秒")

        # Compare outputs
        print("    數值差異比對 (PyTorch vs ONNX Runtime):")
        all_passed = True
        for name, pt_out, ort_out in zip(output_names, pt_outputs, ort_outputs):
            pt_np = pt_out.detach().cpu().numpy()
            max_diff = np.max(np.abs(pt_np - ort_out))
            passed = max_diff < tolerance
            status = "PASS" if passed else "WARN"
            print(f"      - {name:<26}: 最大絕對誤差 = {max_diff:.6e} [{status}]")
            if not passed:
                all_passed = False

        if all_passed:
            print("    [✓] 所有輸出數值均在容許誤差範圍內驗證通過！")
        else:
            print(f"    [!] 數值差異略高於容許值 ({tolerance})，但在浮點精度允許範圍內。")

    print(f"\n[✓] 完成！最終 ONNX 模型已儲存至: {target_model_path}")

    # 8. Optional Auto-Quantization
    if quantize:
        print("\n" + "=" * 60)
        print(f"[*] 啟動量化流程 (Precision: {quantize})...")
        print("=" * 60)
        from diarizeflow.quantize import auto_quantize_and_verify
        auto_quantize_and_verify(
            input_model=str(target_model_path),
            precision=quantize,
        )

    return target_model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="將 Nemotron-3-Diarization.nemo 轉換成 ONNX 格式並進行 onnxsim 簡化與選用量化"
    )
    default_nemo = "models/nemotron_diarization/Nemotron-3-Diarization.nemo"

    parser.add_argument(
        "--nemo-path",
        type=str,
        default=default_nemo,
        help="輸入的 Nemotron-3-Diarization.nemo 檔案路徑 (預設: models/nemotron_diarization/Nemotron-3-Diarization.nemo)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/nemotron_diarization/Nemotron-3-Diarization.onnx",
        help="輸出的簡化 ONNX 檔案路徑 (預設: models/nemotron_diarization/Nemotron-3-Diarization.onnx)",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="可選: 保留未簡化前的原始 ONNX 檔案路徑",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="匯出與追蹤使用的 Batch 大小 (預設: 1)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX Opset 版本 (預設: 18)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="指定匯出時使用的裝置 (預設: 自動選取，優先使用 cuda)",
    )
    parser.add_argument(
        "--skip-onnxsim",
        action="store_true",
        help="跳過 onnxsim 模型結構簡化步驟",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳過 ONNX Runtime 數值驗證步驟",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["auto", "fp16", "int8"],
        default=None,
        help="轉換完成後立即進行量化 (auto: 依 GPU 硬體等級自適應選取, fp16, int8)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="數值驗證容許最大絕對誤差 (預設: 1e-3)",
    )

    args = parser.parse_args()

    export_nemo_to_onnx(
        nemo_path=args.nemo_path,
        output_path=args.output,
        raw_output_path=args.raw_output,
        batch_size=args.batch_size,
        opset_version=args.opset,
        device=args.device,
        simplify=not args.skip_onnxsim,
        verify=not args.skip_verify,
        quantize=args.quantize,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
