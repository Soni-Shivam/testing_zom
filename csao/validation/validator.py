"""
Generator Validation Module
==============================
Tests the generated corpus against statistical benchmarks:

1. Co-occurrence χ² test: Validates that known food pairings
   (e.g., Biryani-Salan) appear significantly in the corpus.
2. Session length KL-divergence: Checks that generated session
   length distribution matches industry benchmarks.
3. Meal template fill rate: Verifies per-cuisine template
   completion rates.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from collections import Counter

from csao.config.taxonomies import (
    GROUND_TRUTH_PAIRINGS,
    SESSION_LENGTH_BENCHMARK,
    ALL_CUISINES,
)
from csao.models.schema import CartTrajectory


class CorpusValidator:
    """
    Statistical validation of the generated synthetic corpus.

    Runs three automated tests and produces a structured report
    with PASS/FAIL verdicts.
    """

    def __init__(self, trajectories: List[CartTrajectory], df: pd.DataFrame):
        """
        Args:
            trajectories: List of CartTrajectory objects.
            df: DataFrame representation of the corpus.
        """
        self.trajectories = trajectories
        self.df = df

    def run_all_validations(self) -> Dict[str, dict]:
        """
        Run all three validation tests.

        Returns:
            Dictionary with test names as keys and result dicts as values.
        """
        results = {}

        print("\n" + "=" * 70)
        print("  CORPUS VALIDATION REPORT")
        print("=" * 70)

        # Test 1: Co-occurrence χ²
        results["cooccurrence_chi2"] = self.test_cooccurrence_chi2()

        # Test 2: Session length KL-divergence
        results["session_length_kl"] = self.test_session_length_kl_divergence()

        # Test 3: Template fill rate
        results["template_fill_rate"] = self.test_template_fill_rate()

        # Summary
        n_pass = sum(1 for r in results.values() if r.get("verdict") == "PASS")
        n_total = len(results)
        print(f"\n{'=' * 70}")
        print(f"  OVERALL: {n_pass}/{n_total} tests passed")
        print(f"{'=' * 70}\n")

        return results

    def test_cooccurrence_chi2(self) -> dict:
        """
        Test 1: Co-occurrence χ² test.

        For each known ground-truth pairing (A, B), count:
        - n_both: sessions containing both A and B
        - n_A_only: sessions with A but not B
        - n_B_only: sessions with B but not A
        - n_neither: sessions with neither

        Then run a χ² test of independence. If the pairing is
        statistically significant (p < 0.05), it passes.

        A pairing PASSES if:
        - The χ² test shows significant association (p < 0.05), OR
        - Item A appears and B co-occurs in > 30% of A's sessions.
        """
        print("\n[Test 1] Co-occurrence χ² Test")
        print("-" * 50)

        # Build per-trajectory item sets
        traj_items = {}
        for traj in self.trajectories:
            traj_items[traj.trajectory_id] = set(traj.item_names)

        n_total = len(traj_items)
        results_per_pair = []
        n_significant = 0

        for item_a, item_b in GROUND_TRUTH_PAIRINGS:
            n_both = 0
            n_a_only = 0
            n_b_only = 0
            n_neither = 0

            for items in traj_items.values():
                has_a = item_a in items
                has_b = item_b in items
                if has_a and has_b:
                    n_both += 1
                elif has_a:
                    n_a_only += 1
                elif has_b:
                    n_b_only += 1
                else:
                    n_neither += 1

            # Build 2x2 contingency table
            contingency = np.array([
                [n_both, n_a_only],
                [n_b_only, n_neither],
            ])

            # Run χ² test (handle edge cases with small counts)
            n_a_total = n_both + n_a_only
            co_occurrence_rate = n_both / max(n_a_total, 1)

            if contingency.sum() > 0 and contingency.min() >= 0:
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(
                        contingency, correction=True
                    )
                except ValueError:
                    chi2, p_value = 0.0, 1.0
            else:
                chi2, p_value = 0.0, 1.0

            # Verdict: significant association or high co-occurrence rate
            significant = (p_value < 0.05) or (co_occurrence_rate > 0.30)
            if significant:
                n_significant += 1

            pair_result = {
                "pair": f"{item_a} ↔ {item_b}",
                "n_both": n_both,
                "n_a_total": n_a_total,
                "co_occurrence_rate": round(co_occurrence_rate, 3),
                "chi2": round(chi2, 2),
                "p_value": round(p_value, 4),
                "significant": significant,
            }
            results_per_pair.append(pair_result)

            status = "✓" if significant else "✗"
            print(
                f"  {status} {item_a:35s} ↔ {item_b:20s}  "
                f"co-occ={co_occurrence_rate:.1%}  χ²={chi2:8.1f}  "
                f"p={p_value:.4f}"
            )

        # Overall verdict: at least 60% of known pairings should be significant
        pass_threshold = 0.60
        fraction_significant = n_significant / max(len(GROUND_TRUTH_PAIRINGS), 1)
        verdict = "PASS" if fraction_significant >= pass_threshold else "FAIL"

        print(
            f"\n  {verdict}: {n_significant}/{len(GROUND_TRUTH_PAIRINGS)} "
            f"pairings significant ({fraction_significant:.0%} ≥ {pass_threshold:.0%})"
        )

        return {
            "verdict": verdict,
            "fraction_significant": round(fraction_significant, 3),
            "details": results_per_pair,
        }

    def test_session_length_kl_divergence(self) -> dict:
        """
        Test 2: Session length KL-divergence.

        Computes the distribution of session lengths (number of distinct
        items per trajectory) and measures KL-divergence against the
        benchmark distribution.

        PASSES if KL-divergence < 0.3 (relaxed threshold for synthetic data).
        """
        print("\n[Test 2] Session Length KL-Divergence")
        print("-" * 50)

        # Compute generated session length distribution
        session_lengths = [traj.num_items for traj in self.trajectories]
        length_counts = Counter(session_lengths)
        max_len = max(
            max(length_counts.keys()),
            max(SESSION_LENGTH_BENCHMARK.keys()),
        )

        # Build aligned probability vectors
        generated_dist = np.array([
            length_counts.get(i, 0) for i in range(1, max_len + 1)
        ], dtype=np.float64)
        generated_dist = generated_dist / generated_dist.sum()

        benchmark_dist = np.array([
            SESSION_LENGTH_BENCHMARK.get(i, 0.001) for i in range(1, max_len + 1)
        ], dtype=np.float64)
        benchmark_dist = benchmark_dist / benchmark_dist.sum()

        # Add small epsilon to avoid log(0) in KL calculation
        eps = 1e-10
        generated_dist = generated_dist + eps
        benchmark_dist = benchmark_dist + eps
        generated_dist /= generated_dist.sum()
        benchmark_dist /= benchmark_dist.sum()

        # KL(generated || benchmark)
        kl_div = float(stats.entropy(generated_dist, benchmark_dist))

        # Print distribution comparison
        print(f"  {'Length':>8}  {'Generated':>12}  {'Benchmark':>12}")
        for i in range(1, min(max_len + 1, 10)):
            gen_p = generated_dist[i - 1] if i - 1 < len(generated_dist) else 0
            bench_p = benchmark_dist[i - 1] if i - 1 < len(benchmark_dist) else 0
            print(f"  {i:>8d}  {gen_p:>12.3f}  {bench_p:>12.3f}")

        threshold = 0.3
        verdict = "PASS" if kl_div < threshold else "FAIL"

        print(f"\n  KL-divergence: {kl_div:.4f}")
        print(f"  {verdict}: KL = {kl_div:.4f} {'<' if kl_div < threshold else '>='} {threshold}")

        return {
            "verdict": verdict,
            "kl_divergence": round(kl_div, 4),
            "threshold": threshold,
            "generated_mean": round(np.mean(session_lengths), 2),
            "generated_std": round(np.std(session_lengths), 2),
        }

    def test_template_fill_rate(self) -> dict:
        """
        Test 3: Meal template fill rate.

        For each cuisine, compute the fraction of trajectories that
        correctly filled their meal template. Reports per-cuisine
        fill rates.

        PASSES if the overall weighted fill rate >= 0.50.
        """
        print("\n[Test 3] Meal Template Fill Rate")
        print("-" * 50)

        cuisine_stats = {}
        for traj in self.trajectories:
            cuisine = traj.cuisine
            if cuisine not in cuisine_stats:
                cuisine_stats[cuisine] = {"total": 0, "filled": 0}
            cuisine_stats[cuisine]["total"] += 1
            if traj.template_filled:
                cuisine_stats[cuisine]["filled"] += 1

        total_filled = 0
        total_all = 0
        cuisine_results = {}

        for cuisine in sorted(cuisine_stats.keys()):
            s = cuisine_stats[cuisine]
            rate = s["filled"] / max(s["total"], 1)
            cuisine_results[cuisine] = round(rate, 3)
            total_filled += s["filled"]
            total_all += s["total"]

            status = "✓" if rate >= 0.50 else "△"
            print(f"  {status} {cuisine:20s}  {rate:.1%}  ({s['filled']}/{s['total']})")

        overall_rate = total_filled / max(total_all, 1)
        threshold = 0.50
        verdict = "PASS" if overall_rate >= threshold else "FAIL"

        print(f"\n  Overall fill rate: {overall_rate:.1%}")
        print(f"  {verdict}: {overall_rate:.1%} {'≥' if overall_rate >= threshold else '<'} {threshold:.0%}")

        return {
            "verdict": verdict,
            "overall_fill_rate": round(overall_rate, 3),
            "per_cuisine": cuisine_results,
            "threshold": threshold,
        }
