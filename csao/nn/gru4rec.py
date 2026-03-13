"""
GRU4Rec — User History Encoder
==================================
Recurrent encoder for sequential user history.

Processes ordered past sessions where temporal ordering matters
for preference drift detection:

  h_t = GRU(h_{t-1}, e_{i_t})
  e_U = h_{|history|}  (final hidden state)

Reference:
  Hidasi et al., "Session-based Recommendations with Recurrent
  Neural Networks" (ICLR 2016)
"""

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """
    GRU-based user history encoder.

    Processes the user's ordered history of item interactions
    and extracts the final hidden state as the user embedding.

    Args:
        embedding_dim: Dimensionality of input item embeddings.
        hidden_dim: GRU hidden state dimensionality.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Optional projection if hidden_dim != embedding_dim
        self.project = None
        if hidden_dim != embedding_dim:
            self.project = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode user history.

        h_t = GRU(h_{t-1}, e_{i_t})
        e_U = h_{|history|}

        Args:
            history_embeddings: (B, T, D) item embeddings in temporal order.
            history_lengths: (B,) actual lengths for packing (optional).

        Returns:
            e_U: User embedding (B, D) — final hidden state.
        """
        B = history_embeddings.size(0)

        if history_lengths is not None:
            # Pack padded sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                history_embeddings,
                history_lengths.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed)  # h_n: (num_layers, B, H)
        else:
            _, h_n = self.gru(history_embeddings)

        # Extract final layer's hidden state: e_U = h_{|history|}
        e_U = h_n[-1]  # (B, hidden_dim)

        # Optional projection back to embedding dim
        if self.project is not None:
            e_U = self.project(e_U)

        return e_U  # (B, D)
