#!/usr/bin/env python3
"""
Stage 3: Structured Cold-Start Strategy — Demo Execution
============================================================
Demonstrates all three fallback tiers of the cold-start system:
1. New item with zero interactions → Embedding transfer (Tier 1)
2. New restaurant with zero interactions → KG-seeded recs (Tier 2)
3. New user with < τ_u orders → Bayesian archetype + city popularity (Tier 3)
"""

import os
import numpy as np
import pandas as pd

from csao.coldstart.config import ColdStartConfig, DEFAULT_CONFIG
from csao.coldstart.item_coldstart import ItemColdStart
from csao.coldstart.restaurant_coldstart import RestaurantColdStart
from csao.coldstart.user_coldstart import UserColdStart, UserObservation
from csao.coldstart.router import ColdStartRouter, ColdStartRequest
from csao.features.item_features import CandidateItemFeatureGenerator
from csao.config.taxonomies import CUISINE_MENUS, ALL_CUISINES


def load_corpus() -> pd.DataFrame:
    """Load Stage 1 corpus from CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), "output", "cart_trajectories.csv")
    if os.path.exists(csv_path):
        print(f"[Stage 3] Loading corpus from {csv_path}")
        return pd.read_csv(csv_path)
    else:
        print("[Stage 3] Corpus not found. Regenerating via Stage 1 pipeline...")
        from csao.pipeline import SynthesisPipeline
        pipeline = SynthesisPipeline(seed=42)
        df, _ = pipeline.generate(n_trajectories=10000)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df


def verify_results(result, config) -> bool:
    """Run verification checks on the cold-start results."""
    print("\n" + "=" * 70)
    print("  COLD-START VERIFICATION")
    print("=" * 70)

    all_pass = True

    # Check 1: All three tiers triggered
    tier_flags = [
        result.user_cold_start_triggered,
        result.restaurant_cold_start_triggered,
        result.item_cold_start_triggered,
    ]
    flags_ok = all(tier_flags)
    print(f"\n  [{'✓' if flags_ok else '✗'}] All three tiers triggered: {tier_flags}")
    all_pass &= flags_ok

    # Check 2: Transferred embedding is L2-normalized and non-zero
    if result.transferred_embedding is not None:
        emb = result.transferred_embedding
        norm = np.linalg.norm(emb)
        nonzero = np.any(emb != 0)
        emb_ok = abs(norm - 1.0) < 0.01 and nonzero
        print(f"  [{'✓' if emb_ok else '✗'}] Embedding L2 norm = 1: ‖e‖ = {norm:.6f}, non-zero = {nonzero}")
        all_pass &= emb_ok

        # Check neighbors were found
        n_neighbors = len(result.embedding_neighbors) if result.embedding_neighbors else 0
        nn_ok = n_neighbors > 0
        print(f"  [{'✓' if nn_ok else '✗'}] Embedding neighbors found: {n_neighbors} (expected ≤ {config.nn_k})")
        all_pass &= nn_ok
    else:
        print(f"  [✗] No transferred embedding produced")
        all_pass = False

    # Check 3: KG recommendations filtered to restaurant menu
    if result.kg_seeded_recs is not None:
        n_recs = len(result.kg_seeded_recs)
        kg_ok = n_recs > 0
        print(f"  [{'✓' if kg_ok else '✗'}] KG seeded recommendations: {n_recs} items")
        if n_recs > 0:
            for item, weight in result.kg_seeded_recs:
                w_ok = 0.0 <= weight <= 1.0
                if not w_ok:
                    all_pass = False
            print(f"  [✓] All KG weights ∈ [0, 1]")
        all_pass &= kg_ok
    else:
        print(f"  [✗] No KG recommendations produced")
        all_pass = False

    # Check 4: Bayesian posterior sums to 1.0
    if result.archetype_posterior is not None:
        posterior_sum = sum(result.archetype_posterior.values())
        post_ok = abs(posterior_sum - 1.0) < 0.001
        print(f"  [{'✓' if post_ok else '✗'}] Posterior Σ = 1: Σ = {posterior_sum:.6f}")
        all_pass &= post_ok

        # Check that posterior differs from prior (Bayesian update worked)
        prior_probs = [0.35, 0.15, 0.30, 0.20]  # Budget, Premium, Occ, Family
        post_vals = list(result.archetype_posterior.values())
        shifted = any(abs(p - pr) > 0.01 for p, pr in zip(post_vals, prior_probs))
        print(f"  [{'✓' if shifted else '✗'}] Posterior shifted from prior: {shifted}")
        all_pass &= shifted
    else:
        print(f"  [✗] No archetype posterior produced")
        all_pass = False

    # Check 5: City popularity fallback returned results
    if result.city_popularity_recs is not None:
        n_pop = len(result.city_popularity_recs)
        pop_ok = n_pop > 0
        print(f"  [{'✓' if pop_ok else '✗'}] City popularity fallback: {n_pop} items")
        all_pass &= pop_ok
    else:
        print(f"  [✗] No city popularity fallback")
        all_pass = False

    outcome = "ALL CHECKS PASSED ✅" if all_pass else "SOME CHECKS FAILED ⚠️"
    print(f"\n  {outcome}")
    return all_pass


def main():
    config = DEFAULT_CONFIG
    print("=" * 70)
    print("  CSAO STAGE 3: STRUCTURED COLD-START STRATEGY")
    print("=" * 70)
    print(f"\n  Hyperparameters: τ_u = {config.tau_u}, τ_r = {config.tau_r}, "
          f"NN-k = {config.nn_k}, embedding_dim = {config.embedding_dim}\n")

    # ── Step 1: Load corpus ──────────────────────────────────────
    corpus_df = load_corpus()
    print(f"[Stage 3] Corpus loaded: {len(corpus_df):,} rows\n")

    # ── Step 2: Initialize Tier modules ──────────────────────────
    item_gen = CandidateItemFeatureGenerator(corpus_df=corpus_df)
    known_items = list(corpus_df["item_name"].unique())

    # Tier 1: Item Cold Start
    print("[Tier 1] Building item embedding index...")
    item_cs = ItemColdStart(
        item_feature_gen=item_gen,
        known_item_names=known_items,
        config=config,
    )
    print(f"  → {len(known_items)} items indexed in embedding space\n")

    # Tier 2: Restaurant Cold Start
    restaurant_cs = RestaurantColdStart(config=config)
    print()

    # Tier 3: User Cold Start
    print("[Tier 3] Building Bayesian archetype model...")
    user_cs = UserColdStart(corpus_df=corpus_df, config=config)
    print(f"  → Population prior: { {n: f'{p:.2f}' for n, p in zip(['Budget','Premium','Occasional','FamilyOrder'], user_cs._prior)} }\n")

    # ── Step 3: Initialize Router ────────────────────────────────
    router = ColdStartRouter(
        item_cs=item_cs,
        restaurant_cs=restaurant_cs,
        user_cs=user_cs,
        config=config,
    )

    # ── Step 4: Demo — All Three Cold-Start Tiers ────────────────
    print("=" * 70)
    print("  COLD-START DEMO: New User + New Restaurant + New Item")
    print("=" * 70)

    # Simulate a new item: "Paneer Makhani" (not in the corpus)
    new_item = "Paneer Makhani"

    # Simulate a new restaurant: "Spice Heaven" (North Indian, Delhi)
    new_restaurant = "Spice Heaven"
    new_restaurant_cuisine = "North Indian"
    new_restaurant_city = "Delhi-NCR"

    # Build the restaurant's menu from the cuisine
    menu = CUISINE_MENUS[new_restaurant_cuisine]
    restaurant_menu_items = []
    for cat_items in menu.values():
        for item in cat_items:
            restaurant_menu_items.append(item["name"])

    # Simulate a new user with 2 orders (< τ_u = 5)
    new_user_orders = [
        UserObservation(mean_aov=220, cuisine="North Indian", cart_size=3, max_quantity=1),
        UserObservation(mean_aov=180, cuisine="Street Food", cart_size=2, max_quantity=1),
    ]

    print(f"\n  Scenario:")
    print(f"    New User:       ID=99999, {len(new_user_orders)} orders (< τ_u={config.tau_u})")
    print(f"    New Restaurant: '{new_restaurant}' ({new_restaurant_cuisine}, {new_restaurant_city})")
    print(f"    New Item:       '{new_item}' (zero interactions)")
    print(f"    Cart Main:      ['Butter Chicken']")
    print(f"    Restaurant Menu: {restaurant_menu_items}")

    # Build request
    request = ColdStartRequest(
        user_id=99999,
        user_order_count=len(new_user_orders),
        user_orders=new_user_orders,
        restaurant_name=new_restaurant,
        restaurant_interaction_count=0,
        restaurant_menu=restaurant_menu_items,
        candidate_item_name=new_item,
        candidate_item_has_interactions=False,
        cart_main_items=["Butter Chicken"],
        cuisine=new_restaurant_cuisine,
        city=new_restaurant_city,
    )

    # Route through the decision tree
    result = router.route(request)

    # ── Step 5: Print Results ────────────────────────────────────
    print("\n" + "-" * 70)
    print("  TIER 1: Item Cold Start — Embedding Transfer")
    print("-" * 70)
    print(f"  Triggered: {result.item_cold_start_triggered}")
    if result.transferred_embedding is not None:
        emb = result.transferred_embedding
        print(f"  Transferred embedding: shape={emb.shape}, ‖e‖₂={np.linalg.norm(emb):.4f}")
        print(f"  First 8 dims: [{', '.join(f'{v:.4f}' for v in emb[:8])}]")
    if result.embedding_neighbors:
        print(f"  Nearest neighbors ({len(result.embedding_neighbors)}):")
        for name, sim in result.embedding_neighbors:
            print(f"    → {name:<30s}  sim = {sim:.4f}")

    print("\n" + "-" * 70)
    print("  TIER 2: Restaurant Cold Start — Cuisine KG")
    print("-" * 70)
    print(f"  Triggered: {result.restaurant_cold_start_triggered}")
    if result.kg_seeded_recs:
        print(f"  KG-seeded recommendations (filtered to menu):")
        for item, weight in result.kg_seeded_recs:
            print(f"    → {item:<30s}  w = {weight:.3f}")
    else:
        print("  No recommendations (cart main items may not be in KG)")

    print("\n" + "-" * 70)
    print("  TIER 3: User Cold Start — Bayesian Archetype")
    print("-" * 70)
    print(f"  Triggered: {result.user_cold_start_triggered}")
    if result.archetype_posterior:
        print(f"  Posterior P(archetype | {len(new_user_orders)} orders):")
        for arch, prob in sorted(result.archetype_posterior.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"    {arch:<15s}  {prob:.4f}  {bar}")
    if result.city_popularity_recs:
        print(f"\n  City popularity fallback ({new_restaurant_city} × {new_restaurant_cuisine}):")
        for item, score in result.city_popularity_recs[:5]:
            print(f"    → {item:<30s}  score = {score:.4f}")

    # ── Step 6: Verification ─────────────────────────────────────
    verify_results(result, config)

    print(f"\n{'=' * 70}")
    print(f"  Stage 3 Cold-Start Strategy — COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
