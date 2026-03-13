#!/usr/bin/env python3
"""
Stage 2: Advanced Feature Engineering — Demo Execution
========================================================
Demonstrates the full feature pipeline by:
1. Loading the Stage 1 corpus
2. Populating the feature store via NightlyOfflineJob + NearRealTimeJob
3. Running the OnlinePerRequestCalculator on a sample request
4. Printing the labeled, concatenated feature vector
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from csao.features.feature_store import (
    SimulatedRedisStore,
    NightlyOfflineJob,
    NearRealTimeJob,
    OnlinePerRequestCalculator,
)
from csao.features.cart_features import CATEGORY_ORDER
from csao.features.context_features import ContextEncoder


def load_corpus() -> pd.DataFrame:
    """Load Stage 1 corpus from CSV or regenerate if missing."""
    csv_path = os.path.join(os.path.dirname(__file__), "output", "cart_trajectories.csv")
    if os.path.exists(csv_path):
        print(f"[Stage 2] Loading corpus from {csv_path}")
        return pd.read_csv(csv_path)
    else:
        print("[Stage 2] Corpus not found. Regenerating via Stage 1 pipeline...")
        from csao.pipeline import SynthesisPipeline
        pipeline = SynthesisPipeline(seed=42)
        df, _ = pipeline.generate(n_trajectories=10000)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        return df


def verify_features(feature_vector: np.ndarray, segments: dict) -> None:
    """Run sanity checks on the computed feature vector."""
    print("\n" + "=" * 70)
    print("  FEATURE VECTOR VERIFICATION")
    print("=" * 70)

    all_pass = True

    # Check 1: Herfindahl diversity D ∈ [0, 1]
    d_start, d_end = segments["cart.cart_diversity"]
    diversity = feature_vector[d_start:d_end][0]
    d_ok = 0.0 <= diversity <= 1.0
    print(f"\n  [{'✓' if d_ok else '✗'}] Cart Diversity D ∈ [0,1]: D = {diversity:.4f}")
    all_pass &= d_ok

    # Check 2: Cyclical encoding sin²+cos² = 1
    h_start, h_end = segments["ctx.cyclical_hour"]
    hour_vec = feature_vector[h_start:h_end]
    sin_cos_sum = hour_vec[0]**2 + hour_vec[1]**2
    cyc_ok = abs(sin_cos_sum - 1.0) < 1e-5
    print(f"  [{'✓' if cyc_ok else '✗'}] Cyclical hour sin²+cos² = 1: {sin_cos_sum:.6f}")
    all_pass &= cyc_ok

    d_start, d_end = segments["ctx.cyclical_day"]
    day_vec = feature_vector[d_start:d_end]
    sin_cos_sum_d = day_vec[0]**2 + day_vec[1]**2
    cyc_d_ok = abs(sin_cos_sum_d - 1.0) < 1e-5
    print(f"  [{'✓' if cyc_d_ok else '✗'}] Cyclical day sin²+cos² = 1:  {sin_cos_sum_d:.6f}")
    all_pass &= cyc_d_ok

    # Check 3: SLM embedding is L2-normalized
    e_start, e_end = segments["item.slm_embedding"]
    emb = feature_vector[e_start:e_end]
    emb_norm = np.linalg.norm(emb)
    emb_ok = abs(emb_norm - 1.0) < 0.01
    print(f"  [{'✓' if emb_ok else '✗'}] SLM embedding L2 norm = 1:  {emb_norm:.6f}")
    all_pass &= emb_ok

    # Check 4: PAR >= 0
    par_start, par_end = segments["cart.price_anchor_ratio"]
    par = feature_vector[par_start:par_end][0]
    par_ok = par >= 0.0
    print(f"  [{'✓' if par_ok else '✗'}] Price Anchor Ratio ≥ 0:    PAR = {par:.4f}")
    all_pass &= par_ok

    # Check 5: Cuisine preference sums to ~1
    cp_start, cp_end = segments["user.cuisine_preference"]
    cp = feature_vector[cp_start:cp_end]
    cp_sum = cp.sum()
    cp_ok = abs(cp_sum - 1.0) < 0.01
    print(f"  [{'✓' if cp_ok else '✗'}] Cuisine preference Σ = 1:  Σ = {cp_sum:.6f}")
    all_pass &= cp_ok

    overall = "ALL CHECKS PASSED ✅" if all_pass else "SOME CHECKS FAILED ⚠️"
    print(f"\n  {overall}")


def print_feature_breakdown(feature_vector: np.ndarray, segments: dict) -> None:
    """Print a labeled breakdown of the feature vector."""
    print("\n" + "=" * 70)
    print("  FEATURE VECTOR BREAKDOWN")
    print("=" * 70)
    print(f"\n  Total vector length: {len(feature_vector)} dimensions\n")

    print(f"  {'Feature Segment':<35s} {'Range':<15s} {'Dims':>5s}  {'Values (first 5)'}")
    print(f"  {'─' * 35} {'─' * 15} {'─' * 5}  {'─' * 40}")

    for name, (start, end) in segments.items():
        dims = end - start
        vals = feature_vector[start:end]
        val_str = ", ".join(f"{v:.3f}" for v in vals[:5])
        if dims > 5:
            val_str += f", ... ({dims - 5} more)"
        print(f"  {name:<35s} [{start:3d}:{end:3d}]       {dims:>3d}   [{val_str}]")

    # Domain subtotals
    print(f"\n  {'─' * 70}")
    domain_totals = {}
    for name, (start, end) in segments.items():
        domain = name.split(".")[0]
        domain_totals[domain] = domain_totals.get(domain, 0) + (end - start)

    domain_labels = {"cart": "Cart-Level Aggregates", "item": "Candidate Item",
                     "ctx": "Contextual", "user": "User History"}
    for domain, total in domain_totals.items():
        label = domain_labels.get(domain, domain)
        print(f"  Domain: {label:<30s}  {total:>3d} dims")
    print(f"  {'─' * 42}  {'─' * 3}")
    print(f"  {'TOTAL':<42s}  {sum(domain_totals.values()):>3d} dims")


def main():
    print("=" * 70)
    print("  CSAO STAGE 2: ADVANCED FEATURE ENGINEERING")
    print("=" * 70)

    # ── Step 1: Load corpus ──────────────────────────────────────────
    corpus_df = load_corpus()
    print(f"[Stage 2] Corpus loaded: {len(corpus_df):,} rows, "
          f"{corpus_df['trajectory_id'].nunique():,} trajectories\n")

    # ── Step 2: Initialize feature store ─────────────────────────────
    store = SimulatedRedisStore()

    # ── Step 3: Tier 1 — Nightly Offline Job ─────────────────────────
    nightly = NightlyOfflineJob(store=store, corpus_df=corpus_df)
    nightly_stats = nightly.run()

    # ── Step 4: Tier 2 — Near-Real-Time Job ──────────────────────────
    nrt = NearRealTimeJob(store=store, corpus_df=corpus_df)
    nrt_entries = nrt.run()

    print(f"\n[Store] Total keys in store: {store.size:,}\n")

    # ── Step 5: Tier 3 — Online Per-Request Calculator ───────────────
    print("=" * 70)
    print("  ONLINE PER-REQUEST DEMO")
    print("=" * 70)

    calculator = OnlinePerRequestCalculator(store=store, corpus_df=corpus_df)

    # Build a sample request from the corpus
    sample_traj = corpus_df[corpus_df["trajectory_id"] == 0]
    sample_user_id = int(sample_traj["user_id"].iloc[0])
    sample_aov = float(sample_traj["aov_ceiling"].iloc[0])
    sample_city = str(sample_traj["city"].iloc[0])
    sample_cuisine = str(sample_traj["cuisine"].iloc[0])
    sample_hour = int(sample_traj["hour_of_day"].iloc[0])
    sample_weekend = bool(sample_traj["is_weekend"].iloc[0])

    # Cart items from the trajectory
    cart_items = [
        {"category": row["item_category"], "quantity": int(row["quantity"])}
        for _, row in sample_traj.iterrows()
    ]
    cart_total = float(sample_traj["total_price"].iloc[0])

    # Pick a candidate add-on item (something NOT in the current cart)
    all_items = corpus_df[corpus_df["cuisine"] == sample_cuisine][
        ["item_name", "item_category"]
    ].drop_duplicates()
    cart_item_names = set(sample_traj["item_name"])
    candidate_items = all_items[~all_items["item_name"].isin(cart_item_names)]

    if len(candidate_items) == 0:
        candidate_items = all_items  # fallback

    candidate = candidate_items.iloc[0]
    cand_name = str(candidate["item_name"])
    cand_cat = str(candidate["item_category"])

    print(f"\n  Sample Request:")
    print(f"    User ID:      {sample_user_id}")
    print(f"    City:         {sample_city}")
    print(f"    Cuisine:      {sample_cuisine}")
    print(f"    Hour:         {sample_hour}:00")
    print(f"    Cart items:   {len(cart_items)} items, ₹{cart_total:.0f} total")
    print(f"    Cart:         {list(sample_traj['item_name'])}")
    print(f"    Candidate:    {cand_name} ({cand_cat})")

    # Run the online calculator with timing
    t0 = time.perf_counter_ns()
    feature_vector, segments = calculator.compute_feature_vector(
        user_id=sample_user_id,
        user_aov_ceiling=sample_aov,
        cart_items=cart_items,
        cart_total=cart_total,
        cuisine=sample_cuisine,
        hour_of_day=sample_hour,
        day_of_week=2,  # Wednesday
        is_weekend=sample_weekend,
        city=sample_city,
        candidate_item_name=cand_name,
        candidate_item_category=cand_cat,
    )
    elapsed_ns = time.perf_counter_ns() - t0
    elapsed_us = elapsed_ns / 1000.0

    print(f"\n  ⚡ Online computation time: {elapsed_us:.1f} µs ({elapsed_us / 1000:.3f} ms)")

    # ── Step 6: Print feature breakdown ──────────────────────────────
    print_feature_breakdown(feature_vector, segments)

    # ── Step 7: Verify features ──────────────────────────────────────
    verify_features(feature_vector, segments)

    print(f"\n{'=' * 70}")
    print("  Stage 2 Feature Engineering Pipeline — COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
