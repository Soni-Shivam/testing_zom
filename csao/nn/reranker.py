"""
LightGBM Re-Ranker
=====================
Re-ranks the neural model's top candidates using a LightGBM
LambdaRank model that combines the neural fusion score with
sparse business signals.

Features per candidate:
  1. Neural fusion score (from the hybrid model)
  2. Meal Gap Fill Score (GFS)
  3. Item margin (simulated profit margin)
  4. Delivery-zone velocity
  5. Historical acceptance rate
  6. Price ratio: Price(y) / CartTotal
"""

import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


RERANKER_FEATURE_NAMES = [
    "neural_score",
    "gap_fill_score",
    "item_margin",
    "zone_velocity",
    "acceptance_rate",
    "price_ratio",
]


@dataclass
class RerankCandidate:
    """A single candidate item for re-ranking."""
    item_name: str
    neural_score: float
    gap_fill_score: float    # 0 or 1
    item_margin: float       # profit margin (0-1)
    zone_velocity: float     # rolling 7-day count
    acceptance_rate: float   # historical acceptance (0-1)
    price_ratio: float       # Price(y) / CartTotal


class LightGBMReranker:
    """
    LightGBM-based re-ranker using LambdaRank objective.

    Combines neural fusion scores with sparse business features
    to produce a calibrated ranked list of K=8 candidates.

    Args:
        k: Number of candidates to return (default 8).
        n_estimators: Number of boosting rounds.
        learning_rate: LightGBM learning rate.
        num_leaves: Maximum leaves per tree.
    """

    def __init__(
        self,
        k: int = 8,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ):
        self.k = k
        self.model: Optional[lgb.Booster] = None

        self.lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [k],
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "verbose": -1,
            "force_col_wise": True,
        }

    def train(
        self,
        features: np.ndarray,
        relevance_labels: np.ndarray,
        group_sizes: List[int],
    ) -> None:
        """
        Train the LambdaRank model.

        Args:
            features: (N_total, 6) feature matrix for all candidates
                      across all queries.
            relevance_labels: (N_total,) integer relevance grades.
            group_sizes: List of group sizes (candidates per query).
        """
        train_data = lgb.Dataset(
            data=features,
            label=relevance_labels,
            group=group_sizes,
            feature_name=RERANKER_FEATURE_NAMES,
        )

        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_data,
            num_boost_round=self.lgb_params["n_estimators"],
        )

        print(f"[LightGBM] Trained LambdaRank model with "
              f"{len(group_sizes)} queries, {len(relevance_labels)} candidates")

    def rerank(
        self, candidates: List[RerankCandidate]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using the trained LambdaRank model.

        Args:
            candidates: List of RerankCandidate objects.

        Returns:
            Top-K (item_name, rerank_score) tuples sorted by score desc.
        """
        if not candidates:
            return []

        # Build feature matrix
        features = np.array([
            [
                c.neural_score,
                c.gap_fill_score,
                c.item_margin,
                c.zone_velocity,
                c.acceptance_rate,
                c.price_ratio,
            ]
            for c in candidates
        ], dtype=np.float64)

        if self.model is not None:
            raw_scores = self.model.predict(features)
        else:
            # Fallback: weighted sum if model not trained
            weights = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
            raw_scores = features @ weights

        min_score = np.min(raw_scores)
        max_score = np.max(raw_scores)
        if max_score > min_score:
            scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            scores = np.zeros_like(raw_scores)

        # Sort by score descending, take top-K
        ranked_indices = np.argsort(scores)[::-1][:self.k]

        return [
            (candidates[i].item_name, float(scores[i]))
            for i in ranked_indices
        ]

    def generate_mock_training_data(
        self,
        n_queries: int = 500,
        candidates_per_query: int = 20,
        rng: np.random.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate mock training data for the LambdaRank model.

        Simulates queries with candidates, where relevance is
        correlated with neural score and gap fill score.

        Returns:
            (features, relevance_labels, group_sizes)
        """
        if rng is None:
            rng = np.random.default_rng(42)

        all_features = []
        all_labels = []
        group_sizes = []

        for _ in range(n_queries):
            n_cands = candidates_per_query
            group_sizes.append(n_cands)

            # Generate features
            neural_scores = rng.normal(0.5, 0.3, n_cands).clip(0, 1)
            gfs = rng.binomial(1, 0.4, n_cands).astype(float)
            margins = rng.uniform(0.1, 0.5, n_cands)
            velocities = rng.exponential(50, n_cands)
            acceptance = rng.beta(3, 2, n_cands)
            price_ratios = rng.uniform(0.05, 0.6, n_cands)

            features = np.column_stack([
                neural_scores, gfs, margins, velocities,
                acceptance, price_ratios,
            ])
            all_features.append(features)

            # Generate relevance labels (correlated with neural + gfs)
            rel_scores = (
                0.4 * neural_scores
                + 0.3 * gfs
                + 0.15 * acceptance
                + 0.15 * rng.random(n_cands)
            )
            # Discretize to 0-4 relevance grades
            labels = np.digitize(
                rel_scores,
                bins=[0.2, 0.4, 0.6, 0.8],
            ).astype(float)
            all_labels.append(labels)

        return (
            np.vstack(all_features),
            np.concatenate(all_labels),
            group_sizes,
        )
