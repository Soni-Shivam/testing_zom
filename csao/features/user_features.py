"""
Feature Domain 4: User History Features
==========================================
Processes the user's historical order log to compute:
  - RFM Triplet (Recency, Frequency, Monetary)
  - Cuisine Preference Vector (normalized frequency distribution)
  - Category Acceptance History (per-category acceptance rate)

Pre-computed offline from the Stage 1 corpus and stored in the
feature store for online retrieval.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from csao.config.taxonomies import ALL_CUISINES
from csao.features.cart_features import CATEGORY_ORDER, N_CATEGORIES


N_CUISINES = len(ALL_CUISINES)
CUISINE_TO_IDX = {c: i for i, c in enumerate(ALL_CUISINES)}


class UserHistoryFeatureExtractor:
    """
    Extracts user-level historical features from the Stage 1 corpus.

    Designed to be run in the NightlyOfflineJob tier.

    Total output per user: 17 features
      - RFM triplet:            3-dim
      - Cuisine preference:     9-dim (one per cuisine)
      - Category acceptance:    5-dim (one per category)
    """

    def __init__(self, corpus_df: Optional[pd.DataFrame] = None):
        """
        Args:
            corpus_df: Full Stage 1 corpus DataFrame.
        """
        self.corpus_df = corpus_df
        self._user_features_cache: Dict[int, Dict[str, np.ndarray]] = {}

        if corpus_df is not None and not corpus_df.empty:
            self._precompute_all_users()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_user_features(self, user_id: int) -> Dict[str, np.ndarray]:
        """
        Retrieve pre-computed features for a user.

        Args:
            user_id: User ID.

        Returns:
            Dictionary of named feature arrays:
              - 'rfm_triplet':           shape (3,)
              - 'cuisine_preference':    shape (9,)
              - 'category_acceptance':   shape (5,)
        """
        if user_id in self._user_features_cache:
            return self._user_features_cache[user_id]

        # Return zero vectors for unknown users
        return {
            "rfm_triplet":         np.zeros(3, dtype=np.float64),
            "cuisine_preference":  np.ones(N_CUISINES, dtype=np.float64) / N_CUISINES,
            "category_acceptance": np.full(N_CATEGORIES, 0.5, dtype=np.float64),
        }

    # -----------------------------------------------------------------
    # Offline pre-computation
    # -----------------------------------------------------------------

    def _precompute_all_users(self) -> None:
        """
        Pre-compute features for all users in the corpus.
        Called once during NightlyOfflineJob.
        """
        df = self.corpus_df
        user_ids = df["user_id"].unique()

        for uid in user_ids:
            user_df = df[df["user_id"] == uid]
            self._user_features_cache[uid] = {
                "rfm_triplet":         self._compute_rfm(user_df, uid),
                "cuisine_preference":  self._compute_cuisine_preference(user_df),
                "category_acceptance": self._compute_category_acceptance(user_df),
            }

    def _compute_rfm(self, user_df: pd.DataFrame, user_id: int) -> np.ndarray:
        """
        Compute RFM triplet:
          R = Recency (days since last order, simulated)
          F = Frequency (orders in last 30 days, simulated)
          M = Monetary (mean historical AOV)

        Since the corpus lacks real timestamps, we simulate:
          - Recency from trajectory ordering (last trajectory = most recent)
          - Frequency from number of distinct trajectories
        """
        trajectories = user_df["trajectory_id"].unique()
        n_orders = len(trajectories)

        # Recency: simulate as inverse of number of orders (more orders → more recent)
        # Scale to ~1-30 days range
        recency = max(1.0, 30.0 / max(n_orders, 1))

        # Frequency: number of orders (scaled to 30-day window)
        frequency = float(min(n_orders, 30))

        # Monetary: mean total price across trajectories
        monetary = user_df.groupby("trajectory_id")["total_price"].first().mean()

        return np.array([recency, frequency, monetary], dtype=np.float64)

    def _compute_cuisine_preference(self, user_df: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized cuisine frequency distribution.

        For each cuisine, count how many of the user's trajectories
        belong to that cuisine and normalize to a probability vector.
        """
        traj_cuisines = user_df.groupby("trajectory_id")["cuisine"].first()
        cuisine_counts = traj_cuisines.value_counts()

        vec = np.zeros(N_CUISINES, dtype=np.float64)
        for cuisine, count in cuisine_counts.items():
            idx = CUISINE_TO_IDX.get(cuisine)
            if idx is not None:
                vec[idx] = count

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            vec = np.ones(N_CUISINES, dtype=np.float64) / N_CUISINES

        return vec

    @staticmethod
    def _compute_category_acceptance(user_df: pd.DataFrame) -> np.ndarray:
        """
        Compute historical acceptance rate per category.

        This represents how often the user orders items from each
        category, normalized by total items ordered.
        Simulates: "how often does the user accept a beverage/dessert
        add-on when prompted?"
        """
        cat_idx = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        vec = np.zeros(N_CATEGORIES, dtype=np.float64)

        for _, row in user_df.iterrows():
            cat = row.get("item_category", "")
            idx = cat_idx.get(cat)
            if idx is not None:
                vec[idx] += row.get("quantity", 1)

        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            vec = np.full(N_CATEGORIES, 0.5, dtype=np.float64)

        return vec
