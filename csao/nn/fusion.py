"""
Cross-Attention Fusion Layer
================================
Fuses the cart embedding e_Φ, user embedding e_U,
context vector c, and meal gap vector g via cross-attention:

  e_fused = CrossAttn(query=e_Φ, key/value=[e_U; c; g])

The cart embedding acts as the query to dynamically pull
relevant aspects from the concatenated user history and context.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion layer.

    Projects heterogeneous inputs (user embedding, context, gap)
    into a shared space, then uses the cart embedding as query
    to attend over them.

    Args:
        dim: Model dimensionality (D).
        context_dim: Dimensionality of context vector c.
        gap_dim: Dimensionality of meal gap vector g.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 128,
        context_dim: int = 11,
        gap_dim: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # Project context and gap into the shared D-dimensional space
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.GELU(),
        )
        self.gap_proj = nn.Sequential(
            nn.Linear(gap_dim, dim),
            nn.GELU(),
        )

        # Cross-attention: cart (query) attends to [user; ctx; gap] (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(dim)

        # Post-fusion feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        e_cart: torch.Tensor,    # (B, D) — cart embedding from Set Transformer
        e_user: torch.Tensor,    # (B, D) — user embedding from GRU4Rec
        context: torch.Tensor,   # (B, context_dim) — contextual features
        gap: torch.Tensor,       # (B, gap_dim) — meal gap vector
    ) -> torch.Tensor:
        """
        Fuse all signals via cross-attention.

        e_fused = CrossAttn(query=e_Φ, key/value=[e_U; c; g])

        Args:
            e_cart:   Cart embedding (B, D).
            e_user:   User embedding (B, D).
            context:  Context vector (B, context_dim).
            gap:      Meal gap vector (B, gap_dim).

        Returns:
            e_fused: Fused embedding (B, D).
        """
        # Project context and gap to D-dimensional space
        c_proj = self.context_proj(context)  # (B, D)
        g_proj = self.gap_proj(gap)          # (B, D)

        # Stack key/value tokens: [e_U; c; g] → (B, 3, D)
        kv = torch.stack([e_user, c_proj, g_proj], dim=1)

        # Query: cart embedding → (B, 1, D)
        q = e_cart.unsqueeze(1)

        # Cross-attention: cart attends to [user; context; gap]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)

        # Residual + LayerNorm
        h = self.layer_norm(q + attn_out)  # (B, 1, D)

        # Feed-forward + residual
        out = self.layer_norm2(h + self.ff(h))  # (B, 1, D)

        return out.squeeze(1)  # (B, D)
