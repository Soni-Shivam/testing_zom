"""
Level 2: Session Intent Sampler
=================================
Generates sessions conditioned on the user archetype and hour of day.

Samples intents from {Solo Lunch, Family Dinner, Late-Night Snack,
Weekend Brunch} with archetype-conditioned probabilities, then assigns
a realistic hour of day from intent-specific Gaussian distributions.

Peak-hour dynamics (12-2PM, 7-10PM) are tagged for downstream modifiers.
"""

import numpy as np
from typing import Tuple

from csao.config.taxonomies import (
    SESSION_INTENTS, INTENT_NAMES, PEAK_HOURS,
)
from csao.models.schema import User, Session


class SessionGenerator:
    """
    Generates ordering sessions conditioned on user archetype
    and temporal context.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)
        self._session_counter = 0

    def generate_session(self, user: User) -> Session:
        """
        Generate a single session for the given user.

        The generation follows:
        1. Sample intent from archetype-conditioned Categorical.
        2. Sample hour_of_day from intent's Gaussian time window.
        3. Determine peak-hour flag and weekend flag.

        Args:
            user: The User for whom to generate a session.

        Returns:
            Populated Session dataclass.
        """
        # --- Step 1: Sample intent conditioned on archetype ---
        intent = self._sample_intent(user.archetype)
        intent_config = SESSION_INTENTS[intent]

        # --- Step 2: Sample hour of day from intent time window ---
        hour = self._sample_hour(intent_config)

        # --- Step 3: Determine peak hour flag ---
        is_peak = self._is_peak_hour(hour)

        # --- Step 4: Weekend flag (roughly 2/7 of sessions) ---
        is_weekend = self.rng.random() < (2.0 / 7.0)

        # Weekend brunch intent override: if weekend and hour is morning,
        # increase probability it's weekend brunch
        if is_weekend and intent != "Weekend Brunch" and 9 <= hour <= 13:
            if self.rng.random() < 0.4:
                intent = "Weekend Brunch"

        self._session_counter += 1

        return Session(
            session_id=self._session_counter,
            user_id=user.user_id,
            intent=intent,
            hour_of_day=hour,
            is_peak_hour=is_peak,
            is_weekend=is_weekend,
        )

    def _sample_intent(self, archetype: str) -> str:
        """
        Sample session intent from Categorical distribution
        conditioned on user archetype.

        Each intent defines an archetype_affinity dict specifying
        how likely each archetype is to trigger that intent.
        We compute P(intent | archetype) ∝ affinity[archetype]
        across all intents.
        """
        # Build probability vector: P(intent_j | archetype)
        probs = np.array([
            SESSION_INTENTS[intent]["archetype_affinity"].get(archetype, 0.1)
            for intent in INTENT_NAMES
        ], dtype=np.float64)

        # Normalize to valid probability distribution
        probs /= probs.sum()

        return self.rng.choice(INTENT_NAMES, p=probs)

    def _sample_hour(self, intent_config: dict) -> int:
        """
        Sample hour of day from a Gaussian centered on the intent's
        peak hour with intent-specific spread.

        Returns:
            Integer hour in [0, 23].
        """
        raw = self.rng.normal(
            intent_config["hour_peak"],
            intent_config["hour_spread"],
        )
        # Wrap around midnight and clip to valid range
        hour = int(round(raw)) % 24
        return max(0, min(23, hour))

    @staticmethod
    def _is_peak_hour(hour: int) -> bool:
        """
        Check if the given hour falls within defined peak windows.

        Peak windows:
        - Lunch:  12:00 – 14:00
        - Dinner: 19:00 – 22:00
        """
        for _, (start, end) in PEAK_HOURS.items():
            if start <= hour <= end:
                return True
        return False
