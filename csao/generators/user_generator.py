"""
Level 1: User Archetype Generator
===================================
Generates synthetic users drawn from a Categorical distribution
over archetypes {Budget, Premium, Occasional, FamilyOrder}.

Each user is assigned:
- AOV ceiling from archetype-specific Normal(μ, σ)
- Cuisine affinity vector: Dirichlet-sampled, biased by city
- Session length, ordering frequency, and quantity constraints
"""

import numpy as np
from typing import List

from csao.config.taxonomies import (
    USER_ARCHETYPES, ARCHETYPE_NAMES, ARCHETYPE_PROBS,
    CITIES, CITY_NAMES, CITY_WEIGHTS, ALL_CUISINES,
)
from csao.models.schema import User


class UserGenerator:
    """
    Generates users from a four-archetype Categorical distribution.
    
    Each user's attributes (AOV, cuisine affinity, city, etc.) are
    sampled from archetype-specific distributions, producing realistic
    heterogeneity across the user base.
    """

    def __init__(self, rng: np.random.Generator = None):
        """
        Args:
            rng: NumPy random generator for reproducibility.
        """
        self.rng = rng or np.random.default_rng(42)

    def generate_users(self, n_users: int) -> List[User]:
        """
        Generate n_users synthetic users.

        Args:
            n_users: Number of users to generate.

        Returns:
            List of User dataclass instances.
        """
        users = []
        for uid in range(n_users):
            user = self._generate_single_user(uid)
            users.append(user)
        return users

    def _generate_single_user(self, user_id: int) -> User:
        """Generate a single user with archetype-conditioned attributes."""

        # --- Step 1: Sample archetype from Categorical distribution ---
        archetype = self.rng.choice(ARCHETYPE_NAMES, p=ARCHETYPE_PROBS)
        arch_config = USER_ARCHETYPES[archetype]

        # --- Step 2: Sample city from population-weighted distribution ---
        city = self.rng.choice(CITY_NAMES, p=CITY_WEIGHTS)

        # --- Step 3: Sample AOV ceiling from Normal(μ, σ), clipped ---
        aov_ceiling = float(np.clip(
            self.rng.normal(arch_config["aov_mean"], arch_config["aov_std"]),
            arch_config["aov_mean"] * 0.5,  # lower bound
            arch_config["aov_ceiling"],       # strict upper bound
        ))

        # --- Step 4: Generate cuisine affinity vector ---
        # Start from the city's geographic cuisine distribution,
        # then add Dirichlet noise scaled by the archetype's diversity.
        cuisine_affinity = self._sample_cuisine_affinity(city, arch_config)

        # --- Step 5: Sample ordering frequency with some noise ---
        order_frequency = max(0.5, self.rng.normal(
            arch_config["order_frequency_weekly"],
            arch_config["order_frequency_weekly"] * 0.2,
        ))

        # --- Step 6: Session length mean ---
        session_length_mean = max(2.0, self.rng.normal(
            arch_config["session_length_mean"],
            arch_config["session_length_std"],
        ))

        return User(
            user_id=user_id,
            archetype=archetype,
            city=city,
            aov_ceiling=round(aov_ceiling, 2),
            cuisine_affinity=cuisine_affinity,
            order_frequency=round(order_frequency, 2),
            session_length_mean=round(session_length_mean, 2),
            min_quantity=arch_config["min_quantity"],
        )

    def _sample_cuisine_affinity(
        self, city: str, arch_config: dict
    ) -> dict:
        """
        Sample a cuisine affinity vector for a user.

        The vector is a mixture of the city's geographic distribution
        and Dirichlet noise. Higher archetype diversity → more uniform
        exploration; lower diversity → stronger geographic bias.

        Args:
            city: User's city.
            arch_config: Archetype configuration dict.

        Returns:
            Dictionary mapping cuisine name → affinity weight (sums to 1).
        """
        city_affinities = CITIES[city]["cuisine_affinity"]
        diversity = arch_config["cuisine_diversity"]

        # Build the base vector aligned with ALL_CUISINES
        base = np.array([
            city_affinities.get(c, 0.01) for c in ALL_CUISINES
        ], dtype=np.float64)

        # Dirichlet concentration: low diversity → peaked (high alpha on
        # city-dominant cuisines); high diversity → flatter
        alpha = base * (1.0 / max(diversity, 0.05)) + 0.1
        sampled = self.rng.dirichlet(alpha)

        return {c: round(float(w), 4) for c, w in zip(ALL_CUISINES, sampled)}
