import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.ffn import FFN
from .embeddings import RoPE, apply_rope


class RoPEAttention(nn.Module):
    """Self-attention with RoPE applied to Q and K.

    Implements the same forward interface as nn.MultiheadAttention so it can
    be used as a drop-in replacement inside MultiheadAttentionBlock.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rope = RoPE(head_dim=self.head_dim)
        self._dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(max(T_q, T_k))
        q = apply_rope(q, cos[:T_q], sin[:T_q])
        k = apply_rope(k, cos[:T_k], sin[:T_k])

        merged_mask = None
        if attn_mask is not None:
            merged_mask = attn_mask.float() if attn_mask.dtype == torch.bool else attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].float() * -1e9
            merged_mask = kpm if merged_mask is None else merged_mask + kpm

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=merged_mask,
            dropout_p=self._dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T_q, self.embed_dim)
        return self.out_proj(out), None


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block.

    Applies pre-RMSNorm before each sublayer (attention and FFN). A separate
    KV norm is applied to the key/value inputs so cross-attention boundaries
    (PMA, decoder MABs) are scale-stable independently of the query path.

    Args:
        query_dim:        feature dim of Q input  (B, T_q, dim_Q)
        key_dim:          feature dim of K input  (B, T_k, dim_K)
        num_heads:        number of attention heads
        layer_norm:       whether to use pre-RMSNorm + KV norm
        dropout:          dropout rate for attention weights
        expansion_factor: FFN intermediate width multiplier (before 2/3 GLU scaling)
        glu_variant:      gated activation for the FFN sublayer; one of
                          "swiglu", "geglu", "reglu", "liglu"
        use_rope:         use RoPEAttention instead of nn.MultiheadAttention
    """
    def __init__(
        self,
        query_dim: int,
        key_dim: int = None,
        num_heads: int = 4,
        layer_norm: bool = False,
        dropout: float = 0.0,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_rope: bool = False,
    ):
        super().__init__()

        if key_dim is None:
            key_dim = query_dim

        if use_rope:
            self.attn = RoPEAttention(embed_dim=query_dim, num_heads=num_heads, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=query_dim,
                kdim=key_dim,
                vdim=key_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=False,
                batch_first=True,
            )

        if layer_norm:
            self.norm_q   = nn.RMSNorm(query_dim)
            self.norm_kv  = nn.RMSNorm(key_dim)
            self.norm_ffn = nn.RMSNorm(query_dim)
        else:
            self.norm_q = self.norm_kv = self.norm_ffn = None

        self.ffn = FFN(
            embed_dim=query_dim,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            dropout=dropout,
        )

    def forward(self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Q: (B, T_q, dim_Q)
        K: (B, T_k, dim_K) — used for both keys and values
        attn_mask: optional (T_q, T_k) or (B*num_heads, T_q, T_k)
        key_padding_mask: optional (B, T_k), True for PAD positions
        """
        q = self.norm_q(query) if self.norm_q is not None else query
        k = self.norm_kv(key)  if self.norm_kv is not None else key
        attn_out, _ = self.attn(
            query=q,
            key=k,
            value=k,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = query + attn_out

        normed = self.norm_ffn(out) if self.norm_ffn is not None else out
        out = out + self.ffn(normed)

        return out
