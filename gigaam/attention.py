import torch
import torch.nn.functional as F
from torch import Tensor


def rtt_half(x: Tensor) -> Tensor:
    """
    Rotate tensor halves for rotary position embeddings.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=x1.ndim - 1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.
    """
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rtt_half(q) * sin), (k * cos) + (rtt_half(k) * sin)


def apply_masked_flash_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    h: int,
    d_k: int,
) -> Tensor:
    """
    Apply Flash Attention with padding masks.
    """
    from einops import rearrange
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
    from flash_attn.bert_padding import pad_input, unpad_input  # type: ignore[import-not-found]

    pad_mask = ~mask[:, 0, :]
    b, t = pad_mask.shape
    q = q.view(b, t, h * d_k)
    k = k.view(b, t, h * d_k)
    v = v.view(b, t, h * d_k)

    q_unpad, indices_q, _, max_seqlen_q = unpad_input(q, pad_mask)[:4]
    q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=h)

    k_unpad = unpad_input(k, pad_mask)[0]
    k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=h)

    v_unpad = unpad_input(v, pad_mask)[0]
    v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=h)

    lengths_q = pad_mask.sum(1).to(torch.int32).to(q.device)
    cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
    max_seqlen_q = torch.max(lengths_q)

    output_unpad = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_q,
    )

    scores = pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices_q,
        b,
        t,
    )

    return scores
