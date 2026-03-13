"""
Temporal and Geographic Realism Modifiers
==========================================
Three modifier functions applied during cart assembly to inject
realistic behavioral patterns:

1. Peak-hour urgency: 15-20% reduction in add-on probability
2. Price anchoring: Sharp drop when cart exceeds AOV ceiling by 20%
3. Geographic taste clusters: City-specific co-occurrence boosting
"""

import numpy as np
from typing import Dict, List, Optional

from csao.config.taxonomies import (
    PEAK_HOUR_PENALTY_RANGE, GEOGRAPHIC_COOCCURRENCE,
)


class RealismModifiers:
    """
    Applies three realism injections to item selection probabilities
    during the cart assembly loop.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)

    def apply_all_modifiers(
        self,
        item_probs: np.ndarray,
        item_names: List[str],
        item_prices: np.ndarray,
        is_peak_hour: bool,
        running_total: float,
        aov_ceiling: float,
        city: str,
        cart_item_names: List[str],
    ) -> np.ndarray:
        """
        Apply all three modifiers sequentially to the item probability
        vector. Returns modified probabilities (renormalized).

        Args:
            item_probs: Base probability vector for candidate items.
            item_names: Names of candidate items (aligned with probs).
            item_prices: Prices of candidate items (aligned with probs).
            is_peak_hour: Whether current session is in peak hours.
            running_total: Current cart running total.
            aov_ceiling: User's AOV ceiling.
            city: User's city for geographic co-occurrence.
            cart_item_names: Names of items already in the cart.

        Returns:
            Modified and renormalized probability vector.
        """
        probs = item_probs.copy()

        # --- Modifier 1: Peak-hour urgency ---
        probs = self.peak_hour_modifier(probs, is_peak_hour)

        # --- Modifier 2: Price anchoring ---
        probs = self.price_anchor_modifier(
            probs, item_prices, running_total, aov_ceiling
        )

        # --- Modifier 3: Geographic taste clusters ---
        probs = self.geographic_taste_modifier(
            probs, item_names, city, cart_item_names
        )

        # Renormalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # Fallback to uniform if all probs collapsed
            probs = np.ones_like(probs) / len(probs)

        return probs

    def peak_hour_modifier(
        self, probs: np.ndarray, is_peak_hour: bool
    ) -> np.ndarray:
        """
        Modifier 1: Peak-hour urgency.

        During peak hours (12-2PM, 7-10PM), sessions are faster with
        fewer add-ons. Decrease all item probabilities by 15-20%.

        This effectively models users making quicker decisions and
        being less likely to browse and add extra items.
        """
        if not is_peak_hour:
            return probs

        # Sample a penalty factor in [0.80, 0.85] (i.e., 15-20% reduction)
        penalty = self.rng.uniform(*PEAK_HOUR_PENALTY_RANGE)
        return probs * penalty

    def price_anchor_modifier(
        self,
        probs: np.ndarray,
        item_prices: np.ndarray,
        running_total: float,
        aov_ceiling: float,
    ) -> np.ndarray:
        """
        Modifier 2: Price anchoring.

        Track the running cart total. If it exceeds the user's AOV
        ceiling by 20%, sharply drop the probability of adding
        high-price items (top 30% by price).

        This models budget-conscious behavior where users become
        reluctant to add expensive items once the cart is "full".
        """
        threshold = aov_ceiling * 1.20

        if running_total < threshold:
            return probs

        # Identify high-price items (above 70th percentile of candidates)
        if len(item_prices) == 0:
            return probs

        price_threshold = np.percentile(item_prices, 70)
        high_price_mask = item_prices >= price_threshold

        # Apply 90% reduction to high-price items
        probs[high_price_mask] *= 0.10

        return probs

    def geographic_taste_modifier(
        self,
        probs: np.ndarray,
        item_names: List[str],
        city: str,
        cart_item_names: List[str],
    ) -> np.ndarray:
        """
        Modifier 3: Geographic taste clusters.

        Use city-specific item affinity matrices. When a trigger item
        is already in the cart, boost the probability of its known
        companion items by the configured factor.

        Example: Hyderabadi users with Biryani in cart → 5× boost
        for Salan and Mirchi Ka Salan.
        """
        if city not in GEOGRAPHIC_COOCCURRENCE:
            return probs

        rules = GEOGRAPHIC_COOCCURRENCE[city]

        # Build a lookup for fast name → index mapping
        name_to_idx = {name: i for i, name in enumerate(item_names)}

        for rule in rules:
            trigger = rule["trigger"]
            if trigger not in cart_item_names:
                continue

            # Trigger item is in the cart → boost companion items
            for boost_item in rule["boost_items"]:
                if boost_item in name_to_idx:
                    idx = name_to_idx[boost_item]
                    probs[idx] *= rule["factor"]

        return probs
