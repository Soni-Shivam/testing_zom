"""
Cold-Start Decision Tree Router
==================================
Online routing function that evaluates incoming requests against
hyperparameter thresholds τ_u and τ_r, triggering the appropriate
cold-start tier(s).

Decision Tree:
  1. user_history < τ_u  → Bayesian archetype prior fallback
  2. restaurant_interactions < τ_r → Seed from cuisine KG
  3. item_interactions == 0 → Weighted embedding transfer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from csao.coldstart.config import ColdStartConfig, DEFAULT_CONFIG
from csao.coldstart.item_coldstart import ItemColdStart
from csao.coldstart.restaurant_coldstart import RestaurantColdStart
from csao.coldstart.user_coldstart import UserColdStart, UserObservation


@dataclass
class ColdStartRequest:
    """Input request to the cold-start router."""
    user_id: int
    user_order_count: int
    user_orders: List[UserObservation] = field(default_factory=list)

    restaurant_name: str = ""
    restaurant_interaction_count: int = 0
    restaurant_menu: List[str] = field(default_factory=list)

    candidate_item_name: str = ""
    candidate_item_has_interactions: bool = True

    cart_main_items: List[str] = field(default_factory=list)
    cuisine: str = ""
    city: str = ""


@dataclass
class ColdStartResult:
    """Output from the cold-start router."""
    # Flags indicating which tiers were triggered
    user_cold_start_triggered: bool = False
    restaurant_cold_start_triggered: bool = False
    item_cold_start_triggered: bool = False

    # Tier 3: User cold start
    archetype_posterior: Optional[Dict[str, float]] = None
    city_popularity_recs: Optional[List[Tuple[str, float]]] = None

    # Tier 2: Restaurant cold start
    kg_seeded_recs: Optional[List[Tuple[str, float]]] = None

    # Tier 1: Item cold start
    transferred_embedding: Optional[np.ndarray] = None
    embedding_neighbors: Optional[List[Tuple[str, float]]] = None


class ColdStartRouter:
    """
    Online decision tree router that evaluates three binary flags
    at request time and triggers the appropriate cold-start tier(s).

    Multiple tiers can fire simultaneously (e.g., a new user at a
    new restaurant ordering a new item triggers all three).
    """

    def __init__(
        self,
        item_cs: ItemColdStart,
        restaurant_cs: RestaurantColdStart,
        user_cs: UserColdStart,
        config: ColdStartConfig = DEFAULT_CONFIG,
    ):
        self.item_cs = item_cs
        self.restaurant_cs = restaurant_cs
        self.user_cs = user_cs
        self.config = config

    def route(self, request: ColdStartRequest) -> ColdStartResult:
        """
        Evaluate the cold-start decision tree for an incoming request.

        Checks three conditions and triggers the corresponding tier(s):

          if user_history < tau_u:
              → Bayesian archetype prior fallback (Tier 3)
          if restaurant_interactions < tau_r:
              → Seed candidates from KG (Tier 2)
          if item_interactions == 0:
              → Weighted embedding transfer (Tier 1)

        Args:
            request: ColdStartRequest with all context fields.

        Returns:
            ColdStartResult with flags and fallback data.
        """
        result = ColdStartResult()

        # ── Flag 1: User Cold Start ──────────────────────────────
        if request.user_order_count < self.config.tau_u:
            result.user_cold_start_triggered = True

            # Bayesian archetype assignment
            result.archetype_posterior = self.user_cs.get_archetype_posterior(
                request.user_orders
            )

            # City-level popularity fallback
            result.city_popularity_recs = self.user_cs.city_popularity_fallback(
                city=request.city,
                cuisine=request.cuisine,
            )

        # ── Flag 2: Restaurant Cold Start ────────────────────────
        if request.restaurant_interaction_count < self.config.tau_r:
            result.restaurant_cold_start_triggered = True

            # Seed from cuisine knowledge graph G
            result.kg_seeded_recs = self.restaurant_cs.get_seeded_recommendations(
                restaurant_menu=request.restaurant_menu,
                cart_main_items=request.cart_main_items,
                cuisine=request.cuisine,
                city=request.city,
            )

        # ── Flag 3: Item Cold Start ──────────────────────────────
        if not request.candidate_item_has_interactions:
            result.item_cold_start_triggered = True

            # Weighted embedding transfer
            emb, neighbors = self.item_cs.weighted_embedding_transfer(
                request.candidate_item_name
            )
            result.transferred_embedding = emb
            result.embedding_neighbors = neighbors

        return result
