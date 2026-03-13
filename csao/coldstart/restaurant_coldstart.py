"""
Tier 2: Restaurant Cold Start — Cuisine Knowledge Graph
==========================================================
Handles new restaurants with no user-interaction data by leveraging
an offline knowledge graph G.

CRITICAL CONSTRAINT: All LLM/graph construction logic runs strictly
offline (nightly job). The online path is a dict lookup to meet
the 200-300ms SLA.

The knowledge graph stores triples: (i_main, i_addon, w) where
w ∈ [0,1] is the association weight.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

from csao.config.taxonomies import (
    CUISINE_MENUS, ALL_CUISINES, CITIES, CITY_NAMES,
    GEOGRAPHIC_COOCCURRENCE, GROUND_TRUTH_PAIRINGS,
)
from csao.coldstart.config import ColdStartConfig, DEFAULT_CONFIG


@dataclass
class KGTriple:
    """A single knowledge graph triple: (main_item, addon_item, weight)."""
    main_item: str
    addon_item: str
    weight: float  # w ∈ [0, 1]


class RestaurantColdStart:
    """
    Manages the cuisine knowledge graph G for seeding recommendations
    for new restaurants with zero user interactions.

    The KG is built offline by mocking an LLM extraction pipeline,
    then stored as a dict for O(1) online lookup.
    """

    def __init__(
        self,
        config: ColdStartConfig = DEFAULT_CONFIG,
        rng: np.random.Generator = None,
    ):
        self.config = config
        self.rng = rng or np.random.default_rng(42)

        # Knowledge graph: main_item → [(addon_item, weight), ...]
        self._graph: Dict[str, List[Tuple[str, float]]] = {}

        # Restaurant interaction counts (simulated)
        self._restaurant_interactions: Dict[str, int] = {}

        # Build the offline KG
        self.build_knowledge_graph()

    # -----------------------------------------------------------------
    # Offline: Knowledge Graph Construction (Nightly Job)
    # -----------------------------------------------------------------

    def build_knowledge_graph(self) -> None:
        """
        Build the cuisine knowledge graph offline.

        Mocks an LLM extraction pipeline using the prompt structure:
          "List the 5 most commonly ordered add-on items when a
           customer orders [MAIN ITEM] from a [CUISINE TYPE]
           restaurant in [CITY]."

        Parses the mocked output into structured triples
        (i_main, i_addon, w) where w ∈ [0,1].
        """
        print("[RestaurantColdStart] Building cuisine knowledge graph (offline)...")

        total_triples = 0

        for cuisine in ALL_CUISINES:
            menu = CUISINE_MENUS.get(cuisine, {})
            main_items = menu.get("main", [])
            side_items = menu.get("side", [])
            beverage_items = menu.get("beverage", [])
            dessert_items = menu.get("dessert", [])

            addon_pool = side_items + beverage_items + dessert_items

            for main in main_items:
                main_name = main["name"]

                # For each city, mock the LLM extraction
                for city in CITY_NAMES:
                    triples = self._mock_llm_extraction(
                        main_item=main_name,
                        cuisine=cuisine,
                        city=city,
                        addon_pool=addon_pool,
                    )

                    for triple in triples:
                        if main_name not in self._graph:
                            self._graph[main_name] = []

                        # Merge: keep highest weight if duplicate
                        existing = {
                            t[0]: t[1] for t in self._graph[main_name]
                        }
                        if triple.addon_item in existing:
                            existing[triple.addon_item] = max(
                                existing[triple.addon_item], triple.weight
                            )
                        else:
                            existing[triple.addon_item] = triple.weight

                        self._graph[main_name] = [
                            (addon, w) for addon, w in existing.items()
                        ]
                        total_triples += 1

        print(f"  → {len(self._graph)} main items indexed")
        print(f"  → {total_triples} total triples stored")

    def _mock_llm_extraction(
        self,
        main_item: str,
        cuisine: str,
        city: str,
        addon_pool: List[dict],
    ) -> List[KGTriple]:
        """
        Mock the LLM extraction pipeline.

        Prompt (simulated):
          "List the 5 most commonly ordered add-on items when a
           customer orders [MAIN ITEM] from a [CUISINE TYPE]
           restaurant in [CITY]."

        Returns structured triples with weights derived from:
        - Known ground-truth co-occurrences (high weight)
        - Geographic co-occurrence boosts (high weight)
        - Random pairing with menu items (lower weight)
        """
        if not addon_pool:
            return []

        triples = []
        used_addons: Set[str] = set()

        # --- Priority 1: Ground-truth co-occurrence pairings ---
        for item_a, item_b in GROUND_TRUTH_PAIRINGS:
            if item_a == main_item:
                addon_names = {x["name"] for x in addon_pool}
                if item_b in addon_names and item_b not in used_addons:
                    triples.append(KGTriple(
                        main_item=main_item,
                        addon_item=item_b,
                        weight=round(self.rng.uniform(0.75, 0.95), 3),
                    ))
                    used_addons.add(item_b)

        # --- Priority 2: Geographic co-occurrence rules ---
        if city in GEOGRAPHIC_COOCCURRENCE:
            for rule in GEOGRAPHIC_COOCCURRENCE[city]:
                if rule["trigger"] == main_item:
                    addon_names = {x["name"] for x in addon_pool}
                    for boost_item in rule["boost_items"]:
                        if (boost_item in addon_names
                                and boost_item not in used_addons):
                            triples.append(KGTriple(
                                main_item=main_item,
                                addon_item=boost_item,
                                weight=round(self.rng.uniform(0.70, 0.90), 3),
                            ))
                            used_addons.add(boost_item)

        # --- Priority 3: Random pairing from menu ---
        remaining_slots = self.config.kg_top_k - len(triples)
        if remaining_slots > 0:
            available = [
                a for a in addon_pool if a["name"] not in used_addons
            ]
            n_pick = min(remaining_slots, len(available))
            if n_pick > 0:
                chosen = self.rng.choice(
                    len(available), size=n_pick, replace=False
                )
                for idx in chosen:
                    triples.append(KGTriple(
                        main_item=main_item,
                        addon_item=available[idx]["name"],
                        weight=round(self.rng.uniform(0.30, 0.60), 3),
                    ))

        return triples[:self.config.kg_top_k]

    # -----------------------------------------------------------------
    # Online: Seeded Recommendations for New Restaurants
    # -----------------------------------------------------------------

    def get_seeded_recommendations(
        self,
        restaurant_menu: List[str],
        cart_main_items: List[str],
        cuisine: str,
        city: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Query the knowledge graph G to seed recommendations for
        a new restaurant. Results are filtered to items actually
        present on the restaurant's menu.

        Args:
            restaurant_menu: List of item names on the restaurant's menu.
            cart_main_items: Main items currently in the user's cart.
            cuisine: Cuisine type of the restaurant.
            city: City / delivery zone.
            top_k: Number of recommendations to return.

        Returns:
            List of (addon_item, weight) tuples, sorted by weight desc.
        """
        menu_set = set(restaurant_menu)
        candidates: Dict[str, float] = {}

        # Look up each cart main item in the KG
        for main_item in cart_main_items:
            kg_entries = self._graph.get(main_item, [])

            for addon, weight in kg_entries:
                # CRITICAL: Only return items on this restaurant's menu
                if addon in menu_set:
                    candidates[addon] = max(candidates.get(addon, 0.0), weight)

        # Sort by weight descending
        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_candidates[:top_k]

    def set_restaurant_interactions(
        self, restaurant_name: str, count: int
    ) -> None:
        """Set the interaction count for a restaurant (simulated)."""
        self._restaurant_interactions[restaurant_name] = count

    def get_restaurant_interactions(self, restaurant_name: str) -> int:
        """Get the interaction count for a restaurant."""
        return self._restaurant_interactions.get(restaurant_name, 0)
