"""
Contrastive Pre-Training Module
===================================
Pre-trains item embeddings using InfoNCE loss over
co-occurring pairs from the training corpus:

  L = -log [ exp(Sim(e_i, e_{j+}) / τ) /
             Σ_k exp(Sim(e_i, e_k) / τ) ]

Where:
  - j+ is a co-occurring item (positive pair)
  - k ranges over in-batch negatives
  - τ is the temperature parameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for item embedding pre-training.

    Uses in-batch negatives: for each anchor-positive pair,
    all other items in the batch serve as negatives.

    Args:
        temperature: Scaling temperature τ (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,   # (B, D)
        positive_emb: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        L = -log [ exp(Sim(e_i, e_{j+}) / τ) /
                   Σ_k exp(Sim(e_i, e_k) / τ) ]

        Args:
            anchor_emb:   (B, D) embeddings of anchor items.
            positive_emb: (B, D) embeddings of positive (co-occurring) items.

        Returns:
            Scalar loss averaged over the batch.
        """
        # L2-normalize for cosine similarity
        anchor = F.normalize(anchor_emb, dim=-1)
        positive = F.normalize(positive_emb, dim=-1)

        # Similarity matrix: (B, B) — each anchor vs all positives
        # Diagonal = positive pairs, off-diagonal = in-batch negatives
        logits = torch.mm(anchor, positive.t()) / self.temperature  # (B, B)

        # Labels: diagonal entries are the positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Cross-entropy over the similarity matrix
        loss = F.cross_entropy(logits, labels)

        return loss


def extract_cooccurrence_pairs(
    corpus_df,
    min_cooccurrence: int = 10,
) -> List[Tuple[str, str]]:
    """
    Extract co-occurring item pairs from the Stage 1 corpus.

    Two items co-occur if they appear in the same trajectory.

    Args:
        corpus_df: Stage 1 corpus DataFrame.
        min_cooccurrence: Minimum co-occurrence count to include.

    Returns:
        List of (item_a, item_b) co-occurring pairs.
    """
    from collections import Counter

    pair_counts = Counter()

    for traj_id, group in corpus_df.groupby("trajectory_id"):
        items = list(group["item_name"].unique())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = tuple(sorted([items[i], items[j]]))
                pair_counts[pair] += 1

    pairs = [
        pair for pair, count in pair_counts.items()
        if count >= min_cooccurrence
    ]
    return pairs


def contrastive_pretrain(
    model: nn.Module,
    item_to_idx: dict,
    pairs: List[Tuple[str, str]],
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.07,
    device: str = "cpu",
) -> List[float]:
    """
    Pre-train item embeddings using contrastive learning.

    Args:
        model: CSAOHybridModel (we train model.item_embed).
        item_to_idx: Mapping from item name to index.
        pairs: List of (item_a, item_b) co-occurring pairs.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        temperature: InfoNCE temperature τ.
        device: Device string.

    Returns:
        List of loss values per epoch.
    """
    criterion = InfoNCELoss(temperature=temperature).to(device)
    optimizer = torch.optim.Adam(model.item_embed.parameters(), lr=lr)

    # Convert pairs to index tensors
    pair_indices = []
    for a, b in pairs:
        if a in item_to_idx and b in item_to_idx:
            pair_indices.append((item_to_idx[a], item_to_idx[b]))

    if not pair_indices:
        print("[Contrastive] No valid pairs found.")
        return []

    pair_tensor = torch.tensor(pair_indices, dtype=torch.long)
    n_pairs = len(pair_tensor)

    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        # Shuffle pairs
        perm = torch.randperm(n_pairs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            batch_idx = perm[start:end]
            batch_pairs = pair_tensor[batch_idx].to(device)

            anchor_ids = batch_pairs[:, 0].unsqueeze(1)      # (B, 1)
            positive_ids = batch_pairs[:, 1].unsqueeze(1)     # (B, 1)

            # Get embeddings via the quantity-aware module (q=1)
            anchor_emb = model.get_item_embedding(anchor_ids).squeeze(1)    # (B, D)
            positive_emb = model.get_item_embedding(positive_ids).squeeze(1) # (B, D)

            loss = criterion(anchor_emb, positive_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [Contrastive] Epoch {epoch+1}/{epochs}  loss = {avg_loss:.4f}")

    return epoch_losses
