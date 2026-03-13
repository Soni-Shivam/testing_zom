"""
Level 3: Restaurant and Cuisine Selection
===========================================
Samples restaurants from a geographically stratified distribution
reflecting real-world cuisine concentrations.

Required geographic constraints:
- Delhi-NCR: North Indian dominant
- Chennai: South Indian dominant
- Mumbai: Coastal Seafood dominant
- Hyderabad: Biryani dominant

The cuisine is selected by combining city-level affinity weights
with the user's personal cuisine affinity vector via element-wise
product and renormalization.
"""

import numpy as np
from typing import Tuple

from csao.config.taxonomies import (
    CITIES, ALL_CUISINES, CUISINE_MENUS, RESTAURANT_POOLS,
)
from csao.models.schema import User


class RestaurantGenerator:
    """
    Generates restaurant and cuisine selections conditioned on
    the user's city and cuisine affinity vector.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)

    def select_restaurant(self, user: User) -> Tuple[str, str]:
        """
        Select a cuisine and restaurant for the user's session.

        Selection follows:
        1. Compute P(cuisine | city, user) ∝ city_affinity ⊙ user_affinity
        2. Sample cuisine from this combined distribution.
        3. Sample restaurant from the cuisine's restaurant pool.

        Args:
            user: User with city and cuisine_affinity attributes.

        Returns:
            Tuple of (cuisine_name, restaurant_name).
        """
        # --- Step 1: Get city-level cuisine affinities ---
        city_affinities = CITIES[user.city]["cuisine_affinity"]

        # --- Step 2: Build combined probability vector ---
        # Element-wise product of city weights and user's personal affinity
        combined = np.array([
            city_affinities.get(c, 0.005) * user.cuisine_affinity.get(c, 0.005)
            for c in ALL_CUISINES
        ], dtype=np.float64)

        # Ensure no zero probabilities (additive smoothing)
        combined = combined + 1e-6

        # Normalize to probability distribution
        combined /= combined.sum()

        # --- Step 3: Sample cuisine ---
        cuisine = self.rng.choice(ALL_CUISINES, p=combined)

        # --- Step 4: Sample restaurant from cuisine pool ---
        restaurant_pool = RESTAURANT_POOLS.get(cuisine, ["Generic Restaurant"])
        restaurant_name = self.rng.choice(restaurant_pool)

        return cuisine, restaurant_name

    def get_menu(self, cuisine: str) -> dict:
        """
        Retrieve the full menu catalog for a cuisine.

        Args:
            cuisine: Cuisine name.

        Returns:
            Dictionary of {category: [item_dicts]}.
        """
        return CUISINE_MENUS.get(cuisine, {})
