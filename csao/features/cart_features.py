"""
Feature Domain 1: Cart-Level Aggregates
=========================================
Online feature calculator that processes the current cart state Φ_t
and computes:
  - Meal Gap Feature Vector (g_c)
  - Quantity-Weighted Category Histogram (h_c)
  - Cart Diversity Index (D) via Herfindahl
  - Price Anchor Ratio (PAR)

All computations are vectorized with NumPy for sub-millisecond latency.
"""

import numpy as np
from typing import Dict, List, Tuple

from csao.config.taxonomies import MEAL_TEMPLATES, ALL_CUISINES

# Canonical category ordering for all vector outputs
CATEGORY_ORDER = ["main", "side", "beverage", "dessert", "snack"]
N_CATEGORIES = len(CATEGORY_ORDER)
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_ORDER)}


class CartFeatureCalculator:
    """
    Computes real-time cart-level aggregate features from the current
    cart state. Designed for the OnlinePerRequestCalculator tier.

    All outputs are NumPy arrays aligned to CATEGORY_ORDER.
    """

    def __init__(self):
        self._cat_idx = CATEGORY_TO_IDX

    def compute_all(
        self,
        cart_items: List[dict],
        cuisine: str,
        cart_total: float,
        user_aov_ceiling: float,
    ) -> Dict[str, np.ndarray]:
        """
        Compute all cart-level features in a single pass.

        Args:
            cart_items: List of dicts with keys 'category' and 'quantity'.
            cuisine: Current cuisine for meal template lookup.
            cart_total: Running total price of the cart.
            user_aov_ceiling: User's AOV ceiling from archetype profile.

        Returns:
            Dictionary of named feature arrays:
              - 'meal_gap_vector':   shape (5,)
              - 'category_histogram': shape (5,)
              - 'cart_diversity':     shape (1,)
              - 'price_anchor_ratio': shape (1,)
        """
        histogram = self.quantity_weighted_histogram(cart_items)
        gap = self.meal_gap_vector(histogram, cuisine)
        diversity = self.cart_diversity_index(histogram)
        par = self.price_anchor_ratio(cart_total, user_aov_ceiling)

        return {
            "meal_gap_vector":    gap,
            "category_histogram": histogram,
            "cart_diversity":     diversity,
            "price_anchor_ratio": par,
        }

    def quantity_weighted_histogram(
        self, cart_items: List[dict]
    ) -> np.ndarray:
        """
        Compute h_c: sum of quantities per category.

        h_c = Σ_{k: cat(i_k)=c} q_k

        Args:
            cart_items: List of dicts with 'category' and 'quantity'.

        Returns:
            NumPy array of shape (5,) aligned to CATEGORY_ORDER.
        """
        h = np.zeros(N_CATEGORIES, dtype=np.float64)
        for item in cart_items:
            cat = item.get("category", "")
            qty = item.get("quantity", 1)
            idx = self._cat_idx.get(cat)
            if idx is not None:
                h[idx] += qty
        return h

    def meal_gap_vector(
        self, histogram: np.ndarray, cuisine: str
    ) -> np.ndarray:
        """
        Compute g_c = max(0, T_{m*}(c) - h_c) for each category c.

        Where T_{m*}(c) is the target count for category c in the
        meal template for the given cuisine.

        Args:
            histogram: Quantity-weighted category histogram (5,).
            cuisine: Cuisine name for template lookup.

        Returns:
            NumPy array of shape (5,) — the meal gap vector.
        """
        template = MEAL_TEMPLATES.get(cuisine, {"main": 1, "side": 1, "beverage": 1})

        target = np.zeros(N_CATEGORIES, dtype=np.float64)
        for cat, count in template.items():
            idx = self._cat_idx.get(cat)
            if idx is not None:
                target[idx] = count

        gap = np.maximum(0.0, target - histogram)
        return gap

    def cart_diversity_index(self, histogram: np.ndarray) -> np.ndarray:
        """
        Compute the Herfindahl diversity index:
          D = 1 - Σ_c (h_c / Σ_{c'} h_{c'})²

        D=0 means all items are in one category (no diversity).
        D approaches 1 as items spread evenly across categories.

        Args:
            histogram: Quantity-weighted category histogram (5,).

        Returns:
            NumPy array of shape (1,) with D in [0, 1].
        """
        total = histogram.sum()
        if total == 0:
            return np.array([0.0])

        shares = histogram / total
        herfindahl = np.sum(shares ** 2)
        diversity = 1.0 - herfindahl
        return np.array([diversity])

    def price_anchor_ratio(
        self, cart_total: float, user_aov_ceiling: float
    ) -> np.ndarray:
        """
        Compute PAR = CartTotal / UserAOVTier.

        Args:
            cart_total: Running total price of the cart.
            user_aov_ceiling: User's AOV ceiling.

        Returns:
            NumPy array of shape (1,) with PAR >= 0.
        """
        if user_aov_ceiling <= 0:
            return np.array([0.0])
        return np.array([cart_total / user_aov_ceiling])
