"""
Tier 3: User Cold Start — Bayesian Archetype Assignment
==========================================================
Handles new users with fewer than τ_u past orders using
Bayesian updating over archetype distributions.

  P(archetype | obs) ∝ P(obs | archetype) × P(archetype)

Uses city-level popularity rankings as a secondary fallback
until the user accumulates sufficient order history.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from csao.config.taxonomies import (
    USER_ARCHETYPES, ARCHETYPE_NAMES, ALL_CUISINES,
    CUISINE_MENUS, CITY_NAMES,
)
from csao.coldstart.config import ColdStartConfig, DEFAULT_CONFIG


N_ARCHETYPES = len(ARCHETYPE_NAMES)
N_CUISINES = len(ALL_CUISINES)
CUISINE_TO_IDX = {c: i for i, c in enumerate(ALL_CUISINES)}


@dataclass
class UserObservation:
    """A single observation from a user's order history."""
    mean_aov: float          # Average order value of the session
    cuisine: str             # Cuisine ordered
    cart_size: int           # Number of items in the cart
    max_quantity: int        # Max item quantity in the order


class UserColdStart:
    """
    Bayesian archetype assignment for new users, with
    city-level popularity fallback.

    Uses population-level archetype distribution as the prior,
    updates the posterior with each new order observation.
    """

    def __init__(
        self,
        corpus_df: Optional[pd.DataFrame] = None,
        config: ColdStartConfig = DEFAULT_CONFIG,
    ):
        self.config = config

        # Population-level prior: P(archetype) from corpus
        self._prior = self._build_population_prior(corpus_df)

        # Archetype likelihood parameters (from taxonomies)
        self._archetype_params = self._build_likelihood_params()

        # City-level popularity rankings: city → cuisine → [(item, score)]
        self._city_popularity = self._build_city_popularity(corpus_df)

    # -----------------------------------------------------------------
    # Bayesian Update
    # -----------------------------------------------------------------

    def get_archetype_posterior(
        self, user_orders: List[UserObservation]
    ) -> Dict[str, float]:
        """
        Compute the posterior archetype distribution by accumulating
        Bayesian updates over observed orders.

        P(archetype | obs) ∝ P(obs | archetype) × P(archetype)

        Args:
            user_orders: List of UserObservation from the user's history.

        Returns:
            Dict mapping archetype name → posterior probability.
        """
        posterior = self._prior.copy()

        for obs in user_orders:
            posterior = self._bayesian_update(posterior, obs)

        return {
            name: float(posterior[i])
            for i, name in enumerate(ARCHETYPE_NAMES)
        }

    def _bayesian_update(
        self, prior: np.ndarray, obs: UserObservation
    ) -> np.ndarray:
        """
        Single Bayesian update step:
          posterior ∝ likelihood(obs | archetype) × prior

        Likelihood components:
          1. AOV likelihood: Gaussian with archetype's mean/std
          2. Cart size likelihood: Gaussian with archetype's session_length
          3. Quantity likelihood: High for FamilyOrder if max_q > 1
        """
        log_likelihood = np.zeros(N_ARCHETYPES, dtype=np.float64)

        for i, arch_name in enumerate(ARCHETYPE_NAMES):
            params = self._archetype_params[arch_name]

            # AOV likelihood: N(aov_mean, aov_std²)
            aov_ll = self._gaussian_log_likelihood(
                obs.mean_aov, params["aov_mean"], params["aov_std"]
            )

            # Cart size likelihood: N(session_length_mean, session_length_std²)
            cart_ll = self._gaussian_log_likelihood(
                obs.cart_size,
                params["session_length_mean"],
                params["session_length_std"],
            )

            # Quantity likelihood: bonus for FamilyOrder if bulk quantity
            qty_ll = 0.0
            if obs.max_quantity > 1:
                if arch_name == "FamilyOrder":
                    qty_ll = np.log(0.85)   # high likelihood
                else:
                    qty_ll = np.log(0.15)   # low likelihood
            else:
                if arch_name == "FamilyOrder":
                    qty_ll = np.log(0.20)   # unlikely for family
                else:
                    qty_ll = np.log(0.80)   # typical

            log_likelihood[i] = aov_ll + cart_ll + qty_ll

        # Combine with prior in log space
        log_posterior = np.log(prior + 1e-12) + log_likelihood

        # Normalize (softmax)
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        return posterior

    @staticmethod
    def _gaussian_log_likelihood(
        x: float, mean: float, std: float
    ) -> float:
        """Compute log N(x | mean, std²)."""
        std = max(std, 1e-6)
        return -0.5 * ((x - mean) / std) ** 2 - np.log(std)

    # -----------------------------------------------------------------
    # City-Level Popularity Fallback
    # -----------------------------------------------------------------

    def city_popularity_fallback(
        self,
        city: str,
        cuisine: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Return city-level popularity rankings for a cuisine.
        Used as secondary fallback when user has < τ_u orders.

        Args:
            city: Delivery city.
            cuisine: Cuisine type.
            top_k: Number of items to return.

        Returns:
            List of (item_name, popularity_score) tuples.
        """
        key = (city, cuisine)
        rankings = self._city_popularity.get(key, [])
        return rankings[:top_k]

    # -----------------------------------------------------------------
    # Pre-computation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_population_prior(
        corpus_df: Optional[pd.DataFrame],
    ) -> np.ndarray:
        """
        Build population-level archetype prior from the corpus.
        Falls back to configured probabilities if no corpus.
        """
        if corpus_df is not None and not corpus_df.empty:
            # Compute from actual corpus distribution
            traj_archetypes = corpus_df.groupby("trajectory_id")["archetype"].first()
            counts = traj_archetypes.value_counts()
            prior = np.zeros(N_ARCHETYPES, dtype=np.float64)
            for i, name in enumerate(ARCHETYPE_NAMES):
                prior[i] = counts.get(name, 0)
            total = prior.sum()
            if total > 0:
                prior /= total
            else:
                prior = np.array([
                    USER_ARCHETYPES[n]["probability"] for n in ARCHETYPE_NAMES
                ])
        else:
            prior = np.array([
                USER_ARCHETYPES[n]["probability"] for n in ARCHETYPE_NAMES
            ])

        return prior

    @staticmethod
    def _build_likelihood_params() -> Dict[str, Dict[str, float]]:
        """Extract archetype parameters for likelihood computation."""
        params = {}
        for name in ARCHETYPE_NAMES:
            arch = USER_ARCHETYPES[name]
            params[name] = {
                "aov_mean": arch["aov_mean"],
                "aov_std": arch["aov_std"],
                "session_length_mean": arch["session_length_mean"],
                "session_length_std": arch["session_length_std"],
                "min_quantity": arch["min_quantity"],
            }
        return params

    @staticmethod
    def _build_city_popularity(
        corpus_df: Optional[pd.DataFrame],
    ) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
        """
        Build city × cuisine popularity rankings from corpus.
        Returns dict mapping (city, cuisine) → [(item, score)].
        """
        popularity: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}

        if corpus_df is None or corpus_df.empty:
            # Fallback: use menu item prices as proxy for popularity
            for city in CITY_NAMES:
                for cuisine in ALL_CUISINES:
                    menu = CUISINE_MENUS.get(cuisine, {})
                    items = []
                    for cat_items in menu.values():
                        for item in cat_items:
                            items.append((item["name"], 1.0 / max(item["price"], 1)))
                    items.sort(key=lambda x: x[1], reverse=True)
                    popularity[(city, cuisine)] = items[:10]
            return popularity

        # Compute from corpus: count item × city × cuisine occurrences
        grouped = corpus_df.groupby(["city", "cuisine", "item_name"])["quantity"].sum()

        for (city, cuisine), group in grouped.groupby(level=[0, 1]):
            items = []
            max_count = group.max() if len(group) > 0 else 1.0
            for item_name, count in group.items():
                # item_name is a tuple (city, cuisine, name) at level 2
                name = item_name[2] if isinstance(item_name, tuple) else item_name
                score = float(count) / max(max_count, 1.0)
                items.append((str(name), round(score, 4)))
            items.sort(key=lambda x: x[1], reverse=True)
            popularity[(str(city), str(cuisine))] = items[:10]

        return popularity

    def get_user_order_count(self, user_id: int) -> int:
        """Get the number of orders for a user (stub for router)."""
        return 0  # New users always have 0 by default
