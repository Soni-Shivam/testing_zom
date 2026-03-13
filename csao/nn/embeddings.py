"""
Quantity-Aware Item Tokenisation
===================================
Custom embedding module where each cart item (i, q) is represented
as a tuple embedding encoding both identity and quantity:

  e_{(i,q)} = e_i + f_q(log q) · v_q

Where:
  - e_i is the standard item embedding lookup
  - f_q(·) is a learned scalar function of log-quantity (small MLP)
  - v_q is a learned "quantity direction" vector in embedding space

This ensures a 5x order is treated differently than a 1x order.
"""

import torch
import torch.nn as nn
import math


class QuantityAwareEmbedding(nn.Module):
    """
    Quantity-aware item embedding module.

    For each (item_id, quantity) pair, computes:
      e_{(i,q)} = e_i + f_q(log q) · v_q

    Args:
        num_items: Vocabulary size (number of unique items).
        embedding_dim: Dimensionality of item embeddings.
        padding_idx: Index used for padding (default 0).
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Standard item embedding lookup: e_i
        self.item_embedding = nn.Embedding(
            num_items, embedding_dim, padding_idx=padding_idx
        )

        # Learned quantity direction vector: v_q ∈ R^D
        self.quantity_direction = nn.Parameter(
            torch.randn(embedding_dim) * 0.02
        )

        # Learned scalar function f_q(log q): R → R
        # Small 2-layer MLP mapping log(q) to a scalar
        self.quantity_scalar_fn = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for embeddings."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # skip padding
        for module in self.quantity_scalar_fn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, item_ids: torch.Tensor, quantities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantity-aware embeddings.

        Args:
            item_ids:    (batch, seq_len) long tensor of item indices.
            quantities:  (batch, seq_len) float tensor of quantities.

        Returns:
            (batch, seq_len, embedding_dim) tensor of embeddings.
        """
        # e_i: standard lookup → (B, S, D)
        e_i = self.item_embedding(item_ids)

        # log(q): (B, S) → (B, S, 1)
        # Clamping to 1.0 enforces log(1) = 0 for padded or unit items
        log_q = torch.log(quantities.float().clamp(min=1.0)).unsqueeze(-1)

        # f_q(log q): (B, S, 1) → scalar per item
        f_q = self.quantity_scalar_fn(log_q)  # (B, S, 1)

        # e_{(i,q)} = e_i + f_q(log q) · v_q
        # v_q: (D,) broadcast to (1, 1, D)
        quantity_offset = f_q * self.quantity_direction.unsqueeze(0).unsqueeze(0)

        return e_i + quantity_offset
