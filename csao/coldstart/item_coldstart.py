"""
Tier 1: Item Cold Start — Embedding Transfer
===============================================
Handles newly listed items (y_new) with zero interaction history.

Uses a mock SLM semantic embedding and nearest-neighbor lookup
in the shared item space to compute the initial representation
via weighted embedding transfer:

  e_{y_new} = Σ_{j ∈ NN(y_new)} Sim(y_new, j) · e_j
              ─────────────────────────────────────────
              Σ_{j ∈ NN(y_new)} Sim(y_new, j)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from csao.coldstart.config import ColdStartConfig, DEFAULT_CONFIG
from csao.features.item_features import CandidateItemFeatureGenerator


class ItemColdStart:
    """
    Manages the item embedding index and provides weighted
    embedding transfer for new items with zero interactions.
    """

    def __init__(
        self,
        item_feature_gen: CandidateItemFeatureGenerator,
        known_item_names: List[str],
        config: ColdStartConfig = DEFAULT_CONFIG,
    ):
        """
        Args:
            item_feature_gen: Stage 2 item feature generator
                              (provides SLM embeddings).
            known_item_names: List of all existing item names
                              with interaction history.
            config: Cold-start hyperparameters.
        """
        self.item_gen = item_feature_gen
        self.config = config

        # Build the embedding index for all known items
        self._embedding_index: Dict[str, np.ndarray] = {}
        self._index_names: List[str] = []
        self._index_matrix: Optional[np.ndarray] = None

        self._build_index(known_item_names)

    def _build_index(self, item_names: List[str]) -> None:
        """
        Build the item embedding index from known items.
        Stores an (N, D) matrix for efficient cosine similarity.
        """
        embeddings = []
        for name in item_names:
            emb = self.item_gen.slm_embedding(name)
            self._embedding_index[name] = emb
            self._index_names.append(name)
            embeddings.append(emb)

        if embeddings:
            self._index_matrix = np.stack(embeddings, axis=0)  # (N, 128)
        else:
            self._index_matrix = np.zeros((0, self.config.embedding_dim))

    def nearest_neighbor_lookup(
        self, y_new: str, k: Optional[int] = None
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Find the k nearest neighbors to y_new using cosine similarity.

        Args:
            y_new: Name of the new item.
            k: Number of neighbors. Defaults to config.nn_k.

        Returns:
            List of (item_name, cosine_similarity, embedding) tuples,
            sorted by descending similarity.
        """
        if k is None:
            k = self.config.nn_k

        if self._index_matrix is None or len(self._index_matrix) == 0:
            return []

        # Generate embedding for the new item
        e_new = self.item_gen.slm_embedding(y_new)

        # Cosine similarity: dot(e_new, e_j) / (||e_new|| * ||e_j||)
        # Since embeddings are L2-normalized, sim = dot product
        similarities = self._index_matrix @ e_new  # (N,)

        # Get top-k indices
        top_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            name = self._index_names[idx]
            sim = float(similarities[idx])
            emb = self._index_matrix[idx]
            results.append((name, sim, emb))

        return results

    def weighted_embedding_transfer(
        self, y_new: str
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Compute the transferred embedding for a new item using
        the weighted embedding transfer formula:

          e_{y_new} = Σ_{j ∈ NN(y_new)} Sim(y_new, j) · e_j
                      ─────────────────────────────────────────
                      Σ_{j ∈ NN(y_new)} Sim(y_new, j)

        Args:
            y_new: Name of the new item.

        Returns:
            Tuple of:
              - Transferred embedding (128-d), L2-normalized.
              - List of (neighbor_name, similarity) pairs used.
        """
        neighbors = self.nearest_neighbor_lookup(y_new)

        if not neighbors:
            # Fallback: use the item's own mock embedding
            return self.item_gen.slm_embedding(y_new), []

        # Weighted sum: Σ Sim(y_new, j) · e_j
        numerator = np.zeros(self.config.embedding_dim, dtype=np.float64)
        denominator = 0.0
        neighbor_info = []

        for name, sim, emb in neighbors:
            # Clamp similarity to non-negative (cosine can be negative)
            w = max(sim, 0.0)
            numerator += w * emb.astype(np.float64)
            denominator += w
            neighbor_info.append((name, float(sim)))

        if denominator > 0:
            transferred = numerator / denominator
        else:
            transferred = self.item_gen.slm_embedding(y_new).astype(np.float64)

        # L2-normalize the result
        norm = np.linalg.norm(transferred)
        if norm > 0:
            transferred = transferred / norm

        return transferred.astype(np.float32), neighbor_info

    def has_interactions(self, item_name: str) -> bool:
        """Check if an item exists in the known embedding index."""
        return item_name in self._embedding_index
