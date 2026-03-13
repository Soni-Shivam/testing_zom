"""
Feature Domain 2: Candidate Item Features
============================================
Feature generator for all candidate add-on items y:
  - SLM Semantic Embedding (mock 128-d)
  - Meal Gap Fill Score (GFS)
  - Delivery-Zone Velocity
  - Meal-Time Affinity P(cat(y) | t_h)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from csao.features.cart_features import CATEGORY_ORDER, CATEGORY_TO_IDX, N_CATEGORIES


class CandidateItemFeatureGenerator:
    """
    Generates feature vectors for candidate add-on items.

    Pre-computes corpus-derived statistics (meal-time affinity)
    and provides per-request feature generation.
    """

    def __init__(
        self,
        corpus_df: Optional[pd.DataFrame] = None,
        seed: int = 42,
    ):
        """
        Args:
            corpus_df: Stage 1 corpus DataFrame for computing
                       meal-time affinity statistics.
            seed: Random seed for reproducible mock embeddings.
        """
        self.rng = np.random.default_rng(seed)
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Pre-compute meal-time affinity from corpus
        self._meal_time_affinity = self._build_meal_time_affinity(corpus_df)

        # Pre-compute zone velocity (simulated)
        self._zone_velocity = self._build_zone_velocity(corpus_df)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def compute_candidate_features(
        self,
        item_name: str,
        item_category: str,
        meal_gap_vector: np.ndarray,
        hour_of_day: int,
        city: str,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for a single candidate item.

        Args:
            item_name: Name of the candidate item.
            item_category: Category of the candidate (main/side/etc).
            meal_gap_vector: Current meal gap vector from cart features.
            hour_of_day: Current hour (0-23).
            city: Current delivery zone / city.

        Returns:
            Dictionary of named feature arrays:
              - 'slm_embedding':       shape (128,)
              - 'gap_fill_score':      shape (1,)
              - 'zone_velocity':       shape (1,)
              - 'meal_time_affinity':  shape (1,)
        """
        return {
            "slm_embedding":      self.slm_embedding(item_name),
            "gap_fill_score":     self.gap_fill_score(item_category, meal_gap_vector),
            "zone_velocity":      self.delivery_zone_velocity(item_name, city),
            "meal_time_affinity": self.meal_time_affinity(item_category, hour_of_day),
        }

    # -----------------------------------------------------------------
    # Feature 1: SLM Semantic Embedding (mock)
    # -----------------------------------------------------------------

    def slm_embedding(self, item_name: str) -> np.ndarray:
        """
        Generate a deterministic mock 128-dimensional embedding
        for the item, simulating offline SLM text embedding.

        Uses a hash-based seed so the same item always gets the
        same embedding vector.

        Args:
            item_name: Name of the item.

        Returns:
            NumPy array of shape (128,) with values in [-1, 1].
        """
        if item_name in self._embedding_cache:
            return self._embedding_cache[item_name]

        # Deterministic seed from item name hash
        item_seed = abs(hash(item_name)) % (2**31)
        item_rng = np.random.default_rng(item_seed)

        # Sample from standard normal, then L2-normalize
        vec = item_rng.standard_normal(128).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._embedding_cache[item_name] = vec
        return vec

    # -----------------------------------------------------------------
    # Feature 2: Meal Gap Fill Score
    # -----------------------------------------------------------------

    @staticmethod
    def gap_fill_score(
        item_category: str, meal_gap_vector: np.ndarray
    ) -> np.ndarray:
        """
        Binary indicator: does item y fill an active gap?

        GFS(y) = 1 if cat(y) ∈ {c : g_c > 0}, else 0.

        Args:
            item_category: Category of the candidate item.
            meal_gap_vector: Current meal gap vector (5,).

        Returns:
            NumPy array of shape (1,) with value 0.0 or 1.0.
        """
        idx = CATEGORY_TO_IDX.get(item_category)
        if idx is not None and meal_gap_vector[idx] > 0:
            return np.array([1.0])
        return np.array([0.0])

    # -----------------------------------------------------------------
    # Feature 3: Delivery-Zone Velocity
    # -----------------------------------------------------------------

    def delivery_zone_velocity(
        self, item_name: str, city: str
    ) -> np.ndarray:
        """
        Simulated rolling 7-day order count for the item
        in the given geographic zone.

        Args:
            item_name: Item name.
            city: Delivery zone / city.

        Returns:
            NumPy array of shape (1,) with the velocity count.
        """
        key = (item_name, city)
        velocity = self._zone_velocity.get(key, 0.0)
        return np.array([velocity])

    # -----------------------------------------------------------------
    # Feature 4: Meal-Time Affinity
    # -----------------------------------------------------------------

    def meal_time_affinity(
        self, item_category: str, hour_of_day: int
    ) -> np.ndarray:
        """
        Compute P(cat(y) | t_h) — the historical probability of
        ordering category cat(y) given the current hour t_h.

        Args:
            item_category: Category of the candidate.
            hour_of_day: Current hour (0-23).

        Returns:
            NumPy array of shape (1,) with probability in [0, 1].
        """
        hour = hour_of_day % 24
        prob = self._meal_time_affinity.get((item_category, hour), 0.0)
        return np.array([prob])

    # -----------------------------------------------------------------
    # Pre-computation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_meal_time_affinity(
        corpus_df: Optional[pd.DataFrame],
    ) -> Dict[tuple, float]:
        """
        Compute P(category | hour) from the Stage 1 corpus.

        Returns dict mapping (category, hour) → probability.
        """
        affinity = {}

        if corpus_df is None or corpus_df.empty:
            # Fallback: uniform distribution
            for cat in CATEGORY_ORDER:
                for h in range(24):
                    affinity[(cat, h)] = 1.0 / N_CATEGORIES
            return affinity

        # Count category occurrences per hour
        grouped = corpus_df.groupby(["hour_of_day", "item_category"]).size()
        hour_totals = corpus_df.groupby("hour_of_day").size()

        for (hour, cat), count in grouped.items():
            total = hour_totals.get(hour, 1)
            affinity[(cat, int(hour))] = count / total

        return affinity

    @staticmethod
    def _build_zone_velocity(
        corpus_df: Optional[pd.DataFrame],
    ) -> Dict[tuple, float]:
        """
        Simulate rolling 7-day zone velocity from the corpus.

        Since the corpus doesn't have real dates, we use the full
        corpus item-city counts scaled to represent a 7-day window.

        Returns dict mapping (item_name, city) → velocity count.
        """
        velocity = {}

        if corpus_df is None or corpus_df.empty:
            return velocity

        # Aggregate item counts per city
        grouped = corpus_df.groupby(["item_name", "city"])["quantity"].sum()
        max_count = grouped.max() if len(grouped) > 0 else 1.0

        for (item, city), count in grouped.items():
            # Normalize to a simulated 7-day window (scale to ~0-500 range)
            velocity[(item, city)] = float(count * 7.0 / max(max_count, 1.0) * 100.0)

        return velocity
