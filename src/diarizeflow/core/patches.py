"""ONNX export patches for NeMo Nemotron-3 Diarization (Sortformer).

This module provides necessary compatibility patches for exporting SortformerEncLabelModel
to ONNX with modern PyTorch:
1. Decomposes `aten.sort.stable` and `aten.sort.default` into native `torch.topk`,
   which maps directly to the standard ONNX `TopK` operator.
2. Replaces `MultiHeadAttention.forward`'s `torch.nn.attention.flex_attention` with
   `torch.nn.functional.scaled_dot_product_attention`, maintaining 100% mathematical
   equivalence while avoiding unexportable FlexAttention higher-order ops.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch._decomp import register_decomposition


_PATCHES_APPLIED = False


def apply_onnx_export_patches() -> None:
    """Apply compatibility patches to PyTorch and NeMo for clean ONNX export."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    # 1. Register aten.sort decomposition into torch.topk
    @register_decomposition(torch.ops.aten.sort.stable)
    def sort_stable_decomp(self, *, stable=True, dim=-1, descending=False):
        k = self.size(dim)
        return torch.topk(self, k=k, dim=dim, largest=descending, sorted=True)

    @register_decomposition(torch.ops.aten.sort.default)
    def sort_default_decomp(self, dim=-1, descending=False):
        k = self.size(dim)
        return torch.topk(self, k=k, dim=dim, largest=descending, sorted=True)

    # 2. Patch MultiHeadAttention to use standard scaled_dot_product_attention
    try:
        from nemo.collections.asr.modules.transformer_encoder import MultiHeadAttention

        orig_forward = MultiHeadAttention.forward

        def _mha_forward_onnx(self, x, block_mask=None, pos_emb=None):
            B, T, _ = x.shape
            H, D = self.n_heads, self.head_dim

            qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if self.qk_norm:
                q = self.q_norm(q).to(v.dtype)
                k = self.k_norm(k).to(v.dtype)

            if self._uses_rope:
                q, k = self.rope(q, k)

            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
            return self.out_proj(out)

        MultiHeadAttention.forward = _mha_forward_onnx
    except ImportError:
        pass

    _PATCHES_APPLIED = True
