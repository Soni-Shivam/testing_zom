"""
Set Transformer — Cart Encoder
==================================
Permutation-invariant encoder for the cart set using:
  - MAB (Multihead Attention Block)
  - ISAB (Induced Set Attention Block) with m inducing points
  - PMA (Pooling by Multihead Attention) for fixed-size output

NO positional encodings are used — strictly permutation invariant.

References:
  Lee et al., "Set Transformer: A Framework for Attention-Based
  Permutation-Invariant Input" (ICML 2019)
"""

import torch
import torch.nn as nn
import math


class MAB(nn.Module):
    """
    Multihead Attention Block.

    MAB(X, Y) = LayerNorm(H + rFF(H))
    where H = X + MultiHeadAttn(X, Y, Y)

    Args:
        dim: Model dimensionality.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        # rFF: row-wise feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Query tensor (B, N, D).
            Y: Key/Value tensor (B, M, D).
            key_padding_mask: (B, M) bool mask, True = ignore.

        Returns:
            (B, N, D) output tensor.
        """
        # H = X + Attn(X, Y, Y)
        attn_out, _ = self.multihead_attn(
            query=X, key=Y, value=Y,
            key_padding_mask=key_padding_mask,
        )
        H = self.layer_norm1(X + attn_out)

        # LayerNorm(H + rFF(H))
        out = self.layer_norm2(H + self.ff(H))
        return out


class ISAB(nn.Module):
    """
    Induced Set Attention Block.

    ISAB(X) = MAB(X, MAB(I, X))

    Uses m learned inducing points I to reduce O(n²) attention
    to O(nm), where n is the set size and m << n.

    Args:
        dim: Model dimensionality.
        num_heads: Number of attention heads.
        num_inducing: Number of inducing points m.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_inducing: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learned inducing points: I ∈ R^{m × D}
        self.inducing_points = nn.Parameter(
            torch.randn(1, num_inducing, dim) * 0.02
        )

        # MAB(I, X): inducing points attend to input
        self.mab1 = MAB(dim, num_heads, ff_dim, dropout)

        # MAB(X, H): input attends to inducing-point summary
        self.mab2 = MAB(dim, num_heads, ff_dim, dropout)

    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Input set (B, N, D).
            key_padding_mask: (B, N) bool mask for padding.

        Returns:
            (B, N, D) output with induced attention.
        """
        B = X.size(0)

        # Expand inducing points to batch: (1, m, D) → (B, m, D)
        I = self.inducing_points.expand(B, -1, -1)

        # Step 1: MAB(I, X) — inducing points attend to input set
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)  # (B, m, D)

        # Step 2: MAB(X, H) — input attends to inducing-point summary
        out = self.mab2(X, H)  # (B, N, D)

        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    Aggregates a variable-size set into a fixed-size vector
    using a learned seed vector S and MAB:
      PMA(X) = MAB(S, X)

    Args:
        dim: Model dimensionality.
        num_heads: Number of attention heads.
        num_seeds: Number of output seed vectors (typically 1).
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Learned seed vector(s): S ∈ R^{k × D}
        self.seed_vectors = nn.Parameter(
            torch.randn(1, num_seeds, dim) * 0.02
        )

        self.mab = MAB(dim, num_heads, ff_dim, dropout)

    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            X: Input set (B, N, D).
            key_padding_mask: (B, N) bool mask for padding.

        Returns:
            (B, num_seeds, D) pooled output.
        """
        B = X.size(0)
        S = self.seed_vectors.expand(B, -1, -1)  # (B, k, D)
        return self.mab(S, X, key_padding_mask=key_padding_mask)


class SetTransformerEncoder(nn.Module):
    """
    Full Set Transformer cart encoder.

    Pipeline: X → ISAB → ISAB → PMA → e_Φ

    NO positional encodings — strictly permutation invariant.

    Args:
        dim: Model dimensionality.
        num_heads: Number of attention heads.
        num_inducing: Number of inducing points per ISAB.
        num_isab_layers: Number of stacked ISAB layers.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_inducing: int = 8,
        num_isab_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stack of ISAB layers
        self.isab_layers = nn.ModuleList([
            ISAB(dim, num_heads, num_inducing, ff_dim, dropout)
            for _ in range(num_isab_layers)
        ])

        # PMA: pool to single vector
        self.pma = PMA(dim, num_heads, num_seeds=1, ff_dim=ff_dim, dropout=dropout)

    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode a variable-size cart set into a fixed-size vector.

        Args:
            X: Item embeddings (B, N, D) — no positional encoding.
            key_padding_mask: (B, N) bool mask, True = padding.

        Returns:
            e_Φ: Cart embedding (B, D).
        """
        H = X
        for isab in self.isab_layers:
            H = isab(H, key_padding_mask=key_padding_mask)

        # PMA → (B, 1, D) → squeeze to (B, D)
        pooled = self.pma(H, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)
