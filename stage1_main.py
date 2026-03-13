#!/usr/bin/env python3
"""
CSAO Hierarchical Generative Data Synthesis
=============================================
Entry point for generating 10,000 synthetic cart trajectories
using the four-level hierarchical generative model.

Usage:
    python main.py
"""

import os
import time
import numpy as np
import pandas as pd

from csao.pipeline import SynthesisPipeline
from csao.validation.validator import CorpusValidator


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print comprehensive summary statistics of the generated corpus."""

    print("\n" + "=" * 70)
    print("  CORPUS SUMMARY STATISTICS")
    print("=" * 70)

    n_trajectories = df["trajectory_id"].nunique()
    n_items_total = len(df)
    n_users = df["user_id"].nunique()

    print(f"\n  Total trajectories:    {n_trajectories:,}")
    print(f"  Total item rows:       {n_items_total:,}")
    print(f"  Unique users:          {n_users:,}")

    # --- Archetype distribution ---
    print("\n  ARCHETYPE DISTRIBUTION:")
    arch_dist = (
        df.groupby("trajectory_id")["archetype"]
        .first()
        .value_counts(normalize=True)
    )
    for arch, pct in arch_dist.items():
        print(f"    {arch:15s}  {pct:.1%}")

    # --- City distribution ---
    print("\n  CITY DISTRIBUTION:")
    city_dist = (
        df.groupby("trajectory_id")["city"]
        .first()
        .value_counts(normalize=True)
    )
    for city, pct in city_dist.items():
        print(f"    {city:15s}  {pct:.1%}")

    # --- Cuisine distribution ---
    print("\n  CUISINE DISTRIBUTION:")
    cuisine_dist = (
        df.groupby("trajectory_id")["cuisine"]
        .first()
        .value_counts(normalize=True)
    )
    for cuisine, pct in cuisine_dist.head(10).items():
        print(f"    {cuisine:20s}  {pct:.1%}")

    # --- Session intent distribution ---
    print("\n  SESSION INTENT DISTRIBUTION:")
    intent_dist = (
        df.groupby("trajectory_id")["intent"]
        .first()
        .value_counts(normalize=True)
    )
    for intent, pct in intent_dist.items():
        print(f"    {intent:20s}  {pct:.1%}")

    # --- Cart metrics ---
    cart_sizes = df.groupby("trajectory_id").size()
    cart_totals = df.groupby("trajectory_id")["total_price"].first()

    print(f"\n  CART METRICS:")
    print(f"    Mean cart size:      {cart_sizes.mean():.2f} items")
    print(f"    Median cart size:    {cart_sizes.median():.1f} items")
    print(f"    Mean cart value:     ₹{cart_totals.mean():.0f}")
    print(f"    Median cart value:   ₹{cart_totals.median():.0f}")

    # --- Peak hour stats ---
    peak_col = df.groupby("trajectory_id")["is_peak_hour"].first()
    peak_pct = peak_col.mean()
    print(f"\n  Peak-hour sessions:    {peak_pct:.1%}")

    # --- FamilyOrder quantity check ---
    family_rows = df[df["archetype"] == "FamilyOrder"]
    if len(family_rows) > 0:
        min_qty_family = family_rows["quantity"].min()
        mean_qty_family = family_rows["quantity"].mean()
        print(f"\n  FAMILYORDER QUANTITY CHECK:")
        print(f"    Min quantity:        {min_qty_family}")
        print(f"    Mean quantity:       {mean_qty_family:.2f}")
        pct_gt1 = (family_rows["quantity"] > 1).mean()
        print(f"    % with q > 1:        {pct_gt1:.1%}")


def main():
    """Main entry point."""
    seed = 42
    n_trajectories = 10_000

    print("=" * 70)
    print("  CSAO HIERARCHICAL GENERATIVE DATA SYNTHESIS")
    print("  Generating synthetic cart trajectories...")
    print("=" * 70)

    t0 = time.time()

    # --- Run pipeline ---
    pipeline = SynthesisPipeline(seed=seed)
    df, trajectories = pipeline.generate(n_trajectories=n_trajectories)

    elapsed = time.time() - t0
    print(f"\n[Pipeline] Generation completed in {elapsed:.1f}s")

    # --- Print summary statistics ---
    print_summary_statistics(df)

    # --- Run validation ---
    validator = CorpusValidator(trajectories=trajectories, df=df)
    validation_results = validator.run_all_validations()

    # --- Save output ---
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "cart_trajectories.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[Output] Saved {len(df):,} rows to {output_path}")
    print(f"[Output] {df['trajectory_id'].nunique():,} trajectories × "
          f"{len(df) / df['trajectory_id'].nunique():.1f} avg items")


if __name__ == "__main__":
    main()
