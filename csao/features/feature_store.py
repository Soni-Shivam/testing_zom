"""
Feature Store Architecture Simulation
========================================
Three operational tiers for the CSAO feature pipeline:

1. NightlyOfflineJob:  Pre-computes user embeddings, SLM item
                       embeddings, RFM triplets, cuisine preferences.
                       Stores results in simulated Redis (dict).

2. NearRealTimeJob:    Simulates 15-minute Spark streaming update
                       of delivery-zone velocity for all items.

3. OnlinePerRequestCalculator:  Takes current cart + timestamp + user,
                                retrieves offline features, computes
                                real-time cart & context features,
                                concatenates into a single vector.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from csao.features.cart_features import (
    CartFeatureCalculator, CATEGORY_ORDER, N_CATEGORIES,
)
from csao.features.item_features import CandidateItemFeatureGenerator
from csao.features.context_features import ContextEncoder
from csao.features.user_features import UserHistoryFeatureExtractor


class SimulatedRedisStore:
    """
    In-memory dict simulating a Redis key-value feature store.
    Supports namespaced get/set with string keys.
    """

    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, str] = {}

    def set(self, key: str, value: np.ndarray, namespace: str = "default") -> None:
        full_key = f"{namespace}:{key}"
        self._store[full_key] = value
        self._metadata[full_key] = namespace

    def get(self, key: str, namespace: str = "default") -> Optional[np.ndarray]:
        full_key = f"{namespace}:{key}"
        return self._store.get(full_key)

    def keys(self, namespace: str = "default") -> List[str]:
        prefix = f"{namespace}:"
        return [k[len(prefix):] for k in self._store if k.startswith(prefix)]

    @property
    def size(self) -> int:
        return len(self._store)


# =====================================================================
# Tier 1: Nightly Offline Job
# =====================================================================

class NightlyOfflineJob:
    """
    Simulates the nightly batch computation that pre-computes
    slow-changing features and persists them to the feature store.

    Computes:
      - User RFM triplets, cuisine preferences, category acceptance
      - SLM item embeddings (mock 128-d)
    """

    def __init__(self, store: SimulatedRedisStore, corpus_df: pd.DataFrame):
        self.store = store
        self.corpus_df = corpus_df

    def run(self) -> Dict[str, int]:
        """
        Execute the nightly job.

        Returns:
            Dict with counts of features stored per type.
        """
        print("[NightlyOfflineJob] Starting nightly feature computation...")
        t0 = time.time()

        # --- User history features ---
        user_extractor = UserHistoryFeatureExtractor(self.corpus_df)
        user_ids = self.corpus_df["user_id"].unique()
        n_users = 0

        for uid in user_ids:
            features = user_extractor.get_user_features(int(uid))
            # Flatten and store per-user
            for feat_name, feat_vec in features.items():
                self.store.set(
                    key=f"user:{uid}:{feat_name}",
                    value=feat_vec,
                    namespace="user_features",
                )
            n_users += 1

        # --- SLM item embeddings ---
        item_gen = CandidateItemFeatureGenerator(corpus_df=self.corpus_df)
        unique_items = self.corpus_df["item_name"].unique()
        n_items = 0

        for item_name in unique_items:
            embedding = item_gen.slm_embedding(item_name)
            self.store.set(
                key=f"item:{item_name}:slm_embedding",
                value=embedding,
                namespace="item_features",
            )
            n_items += 1

        elapsed = time.time() - t0
        print(f"[NightlyOfflineJob] Completed in {elapsed:.2f}s")
        print(f"  → {n_users} users × 3 feature sets = {n_users * 3} user feature vectors")
        print(f"  → {n_items} item SLM embeddings (128-d each)")

        return {"users": n_users, "items": n_items}


# =====================================================================
# Tier 2: Near-Real-Time Job
# =====================================================================

class NearRealTimeJob:
    """
    Simulates a 15-minute Spark streaming job that updates
    delivery-zone velocity for all items.
    """

    def __init__(self, store: SimulatedRedisStore, corpus_df: pd.DataFrame):
        self.store = store
        self.corpus_df = corpus_df

    def run(self) -> int:
        """
        Execute the near-real-time velocity update.

        Returns:
            Number of zone-velocity entries updated.
        """
        print("[NearRealTimeJob] Updating delivery-zone velocity...")
        t0 = time.time()

        # Compute item × city velocity
        item_gen = CandidateItemFeatureGenerator(corpus_df=self.corpus_df)

        grouped = self.corpus_df.groupby(["item_name", "city"])["quantity"].sum()
        max_count = grouped.max() if len(grouped) > 0 else 1.0
        n_entries = 0

        for (item_name, city), count in grouped.items():
            velocity = float(count * 7.0 / max(max_count, 1.0) * 100.0)
            self.store.set(
                key=f"velocity:{item_name}:{city}",
                value=np.array([velocity]),
                namespace="zone_velocity",
            )
            n_entries += 1

        elapsed = time.time() - t0
        print(f"[NearRealTimeJob] Completed in {elapsed:.2f}s")
        print(f"  → {n_entries} zone-velocity entries updated")

        return n_entries


# =====================================================================
# Tier 3: Online Per-Request Calculator
# =====================================================================

class OnlinePerRequestCalculator:
    """
    Highly optimized online feature calculator that takes a live
    request (current cart + timestamp + user) and produces the
    final concatenated feature vector for the neural network.

    Retrieves pre-computed features from the store and computes
    real-time cart-level and contextual features.
    """

    def __init__(self, store: SimulatedRedisStore, corpus_df: pd.DataFrame):
        self.store = store
        self.cart_calc = CartFeatureCalculator()
        self.context_enc = ContextEncoder()
        self.item_gen = CandidateItemFeatureGenerator(corpus_df=corpus_df)

    def compute_feature_vector(
        self,
        user_id: int,
        user_aov_ceiling: float,
        cart_items: List[dict],
        cart_total: float,
        cuisine: str,
        hour_of_day: int,
        day_of_week: int,
        is_weekend: bool,
        city: str,
        candidate_item_name: str,
        candidate_item_category: str,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Compute the full concatenated feature vector for a single
        (user, cart, candidate) request.

        Args:
            user_id: User ID for offline feature retrieval.
            user_aov_ceiling: User's AOV ceiling.
            cart_items: Current cart as list of dicts with 'category', 'quantity'.
            cart_total: Running cart total.
            cuisine: Current cuisine.
            hour_of_day: Current hour (0-23).
            day_of_week: Day of week (0-6).
            is_weekend: Weekend flag.
            city: Delivery city.
            candidate_item_name: Name of candidate add-on item.
            candidate_item_category: Category of candidate item.

        Returns:
            Tuple of:
              - Flat NumPy feature vector (all domains concatenated)
              - Dictionary mapping feature_name → (start_idx, end_idx) in vector
        """
        segments = {}
        vectors = []
        offset = 0

        # === Domain 1: Cart-Level Aggregates (online) ===
        cart_features = self.cart_calc.compute_all(
            cart_items=cart_items,
            cuisine=cuisine,
            cart_total=cart_total,
            user_aov_ceiling=user_aov_ceiling,
        )
        for name, vec in cart_features.items():
            segments[f"cart.{name}"] = (offset, offset + len(vec))
            vectors.append(vec)
            offset += len(vec)

        # === Domain 2: Candidate Item Features ===
        meal_gap = cart_features["meal_gap_vector"]
        item_features = self.item_gen.compute_candidate_features(
            item_name=candidate_item_name,
            item_category=candidate_item_category,
            meal_gap_vector=meal_gap,
            hour_of_day=hour_of_day,
            city=city,
        )
        for name, vec in item_features.items():
            segments[f"item.{name}"] = (offset, offset + len(vec))
            vectors.append(vec)
            offset += len(vec)

        # === Domain 3: Contextual Features (online) ===
        ctx_features = self.context_enc.compute_all(
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            city=city,
        )
        for name, vec in ctx_features.items():
            segments[f"ctx.{name}"] = (offset, offset + len(vec))
            vectors.append(vec)
            offset += len(vec)

        # === Domain 4: User History Features (from store) ===
        for feat_name in ["rfm_triplet", "cuisine_preference", "category_acceptance"]:
            stored = self.store.get(
                key=f"user:{user_id}:{feat_name}",
                namespace="user_features",
            )
            if stored is None:
                # Fallback: zero vector of expected size
                if feat_name == "rfm_triplet":
                    stored = np.zeros(3)
                elif feat_name == "cuisine_preference":
                    stored = np.ones(9) / 9
                else:
                    stored = np.full(N_CATEGORIES, 0.5)

            segments[f"user.{feat_name}"] = (offset, offset + len(stored))
            vectors.append(stored)
            offset += len(stored)

        # Concatenate all segments into a single flat vector
        feature_vector = np.concatenate(vectors).astype(np.float32)

        return feature_vector, segments
