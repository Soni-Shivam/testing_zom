"""
CSAOHybridModel — Main Hybrid Neural Architecture
=====================================================
Chains all components into a single nn.Module:

  QuantityAwareEmbedding → SetTransformer → GRU4Rec → CrossAttentionFusion → Score Head

Forward pass maps (cart, quantities, history, context, gap, candidates)
to per-candidate fusion scores for ranking.
"""

import torch
import torch.nn as nn

from csao.nn.embeddings import QuantityAwareEmbedding
from csao.nn.set_transformer import SetTransformerEncoder
from csao.nn.gru4rec import GRU4Rec
from csao.nn.fusion import CrossAttentionFusion


class CSAOHybridModel(nn.Module):
    """
    Hybrid neural model for Cart Super Add-On recommendation.

    Architecture:
      1. QuantityAwareEmbedding:  (item_ids, quantities) → (B, N, D)
      2. SetTransformerEncoder:   (B, N, D) → (B, D) cart embedding e_Φ
      3. GRU4Rec:                 (B, T, D) → (B, D) user embedding e_U
      4. CrossAttentionFusion:    (e_Φ, e_U, context, gap) → (B, D) e_fused
      5. Score Head:              dot(e_fused, e_candidate) → (B, K) scores

    Args:
        num_items:     Total number of unique items (vocab size).
        embedding_dim: Dimensionality of all embeddings (D).
        context_dim:   Dimensionality of contextual feature vector.
        gap_dim:       Dimensionality of meal gap vector.
        num_heads:     Number of attention heads.
        num_inducing:  Number of ISAB inducing points.
        num_isab_layers: Number of stacked ISAB layers.
        gru_layers:    Number of GRU layers.
        ff_dim:        Feed-forward hidden dimension.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        num_items: int = 200,
        embedding_dim: int = 128,
        slm_dim: int = 384,
        context_dim: int = 11,
        gap_dim: int = 5,
        num_heads: int = 4,
        num_inducing: int = 8,
        num_isab_layers: int = 2,
        gru_layers: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.slm_dim = slm_dim

        # ── Shared behavioral embedding for items ────────────────
        self.item_embed = QuantityAwareEmbedding(
            num_items=num_items,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # ── SLM Projection Layer (Asymmetric down-projection) ────
        self.slm_projector = nn.Sequential(
            nn.Linear(slm_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

        # ── Component 1: Set Transformer (cart encoder) ──────────
        self.cart_encoder = SetTransformerEncoder(
            dim=embedding_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_isab_layers=num_isab_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # ── Component 2: GRU4Rec (user history encoder) ──────────
        self.user_encoder = GRU4Rec(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )

        # ── Component 3: Cross-Attention Fusion ──────────────────
        self.fusion = CrossAttentionFusion(
            dim=embedding_dim,
            context_dim=context_dim,
            gap_dim=gap_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── Score head: project fused embedding to scoring space ─
        self.score_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _fuse_item_vectors(
        self,
        item_ids: torch.Tensor,
        quantities: torch.Tensor,
        slm_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Fused Item Representation:
          e_fused = (e_i + f_q * v_q) + projector(e_slm_384)
        """
        # Behavioral branch (B, S, 128)
        e_behav = self.item_embed(item_ids, quantities)

        # SLM branch down-projection (B, S, 128)
        e_slm_proj = self.slm_projector(slm_vectors)

        # Asymmetric Addition
        return e_behav + e_slm_proj

    def forward(
        self,
        cart_item_ids: torch.Tensor,       # (B, N)
        cart_quantities: torch.Tensor,     # (B, N)
        cart_slm_emb: torch.Tensor,        # (B, N, 384)
        cart_mask: torch.Tensor,           # (B, N)
        history_item_ids: torch.Tensor,    # (B, T)
        history_quantities: torch.Tensor,  # (B, T)
        history_slm_emb: torch.Tensor,     # (B, T, 384)
        history_lengths: torch.Tensor,     # (B,)
        context: torch.Tensor,             # (B, context_dim)
        gap: torch.Tensor,                 # (B, gap_dim)
        candidate_ids: torch.Tensor,       # (B, K)
        candidate_slm_emb: torch.Tensor,   # (B, K, 384)
    ) -> torch.Tensor:
        """
        Full forward pass producing per-candidate fusion scores.
        """
        # ── Step 1: Fusion ───────────────────────────────────────
        cart_emb = self._fuse_item_vectors(cart_item_ids, cart_quantities, cart_slm_emb)
        hist_emb = self._fuse_item_vectors(history_item_ids, history_quantities, history_slm_emb)

        cand_qty = torch.ones_like(candidate_ids, dtype=torch.float32)
        cand_emb = self._fuse_item_vectors(candidate_ids, cand_qty, candidate_slm_emb)

        # ── Step 2: Set Transformer → cart embedding e_Φ ────────
        e_cart = self.cart_encoder(cart_emb, key_padding_mask=cart_mask)  # (B, D)

        # ── Step 3: GRU4Rec → user embedding e_U ────────────────
        e_user = self.user_encoder(hist_emb, history_lengths)  # (B, D)

        # ── Step 4: Cross-Attention Fusion ───────────────────────
        e_fused = self.fusion(e_cart, e_user, context, gap)  # (B, D)

        # ── Step 5: Score candidates via dot product ─────────────
        e_scored = self.score_head(e_fused)  # (B, D)
        scores = torch.bmm(cand_emb, e_scored.unsqueeze(-1)).squeeze(-1)  # (B, K)

        return scores

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get raw item embeddings (for contrastive pre-training)."""
        qty = torch.ones_like(item_ids, dtype=torch.float32)
        return self.item_embed(item_ids, qty)
