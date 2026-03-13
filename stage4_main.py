#!/usr/bin/env python3
"""
Stage 4: Hybrid Neural Architecture — Demo Execution
========================================================
Demonstrates the full neural pipeline:
1. CSAOHybridModel forward pass with dummy tensors
2. Permutation invariance verification of Set Transformer
3. Contrastive pre-training of item embeddings
4. LightGBM LambdaRank re-ranking to produce K=8 candidates
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from csao.nn.model import CSAOHybridModel
from csao.nn.contrastive import (
    InfoNCELoss, contrastive_pretrain, extract_cooccurrence_pairs,
)
from csao.nn.reranker import LightGBMReranker, RerankCandidate


def demo_forward_pass(model, device):
    """Demo: forward pass with dummy tensors."""
    print("=" * 70)
    print("  COMPONENT TEST 1: CSAOHybridModel Forward Pass")
    print("=" * 70)

    B, N, T, K = 4, 6, 10, 8  # batch, cart_size, hist_len, candidates
    context_dim, gap_dim = 11, 5

    # Dummy inputs
    cart_ids = torch.randint(1, 100, (B, N), device=device)
    cart_qty = torch.randint(1, 4, (B, N), device=device).float()
    cart_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    cart_mask[:, -2:] = True  # last 2 positions are padding

    hist_ids = torch.randint(1, 100, (B, T), device=device)
    hist_qty = torch.ones(B, T, device=device)
    hist_lens = torch.tensor([8, 10, 5, 7], device=device)

    context = torch.randn(B, context_dim, device=device)
    gap = torch.randn(B, gap_dim, device=device).abs()

    cand_ids = torch.randint(1, 100, (B, K), device=device)

    # Forward pass with timing
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter_ns()
        scores = model(
            cart_ids, cart_qty, cart_mask,
            hist_ids, hist_qty, hist_lens,
            context, gap, cand_ids,
        )
        elapsed = (time.perf_counter_ns() - t0) / 1e6

    print(f"\n  Input shapes:")
    print(f"    cart_item_ids:      {tuple(cart_ids.shape)}")
    print(f"    cart_quantities:    {tuple(cart_qty.shape)}")
    print(f"    history_item_ids:   {tuple(hist_ids.shape)}")
    print(f"    context:            {tuple(context.shape)}")
    print(f"    gap:                {tuple(gap.shape)}")
    print(f"    candidate_ids:      {tuple(cand_ids.shape)}")
    print(f"\n  Output scores shape: {tuple(scores.shape)}")
    print(f"  Sample scores (batch 0): [{', '.join(f'{s:.4f}' for s in scores[0])}]")
    print(f"  ⚡ Forward pass: {elapsed:.2f} ms")

    # Verify output shape
    shape_ok = scores.shape == (B, K)
    print(f"\n  [{'✓' if shape_ok else '✗'}] Output shape (B, K) = ({B}, {K}): {shape_ok}")

    return scores, shape_ok


def demo_permutation_invariance(model, device):
    """Demo: Set Transformer permutation invariance."""
    print("\n" + "=" * 70)
    print("  COMPONENT TEST 2: Permutation Invariance")
    print("=" * 70)

    B, N, T, K = 1, 5, 5, 4

    # Create a cart with specific items
    cart_ids = torch.tensor([[10, 20, 30, 40, 50]], device=device)
    cart_qty = torch.ones(B, N, device=device)
    cart_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    hist_ids = torch.randint(1, 100, (B, T), device=device)
    hist_qty = torch.ones(B, T, device=device)
    hist_lens = torch.tensor([T], device=device)
    context = torch.randn(B, 11, device=device)
    gap = torch.randn(B, 5, device=device).abs()
    cand_ids = torch.randint(1, 100, (B, K), device=device)

    # Forward pass with original order
    model.eval()
    with torch.no_grad():
        scores_original = model(
            cart_ids, cart_qty, cart_mask,
            hist_ids, hist_qty, hist_lens,
            context, gap, cand_ids,
        )

    # Permute cart items: [10,20,30,40,50] → [50,30,10,40,20]
    perm = torch.tensor([[4, 2, 0, 3, 1]], device=device)
    cart_ids_perm = cart_ids.gather(1, perm)
    cart_qty_perm = cart_qty.gather(1, perm)

    with torch.no_grad():
        scores_permuted = model(
            cart_ids_perm, cart_qty_perm, cart_mask,
            hist_ids, hist_qty, hist_lens,
            context, gap, cand_ids,
        )

    diff = (scores_original - scores_permuted).abs().max().item()
    perm_ok = diff < 1e-4
    print(f"\n  Original order:  {cart_ids[0].tolist()}")
    print(f"  Permuted order:  {cart_ids_perm[0].tolist()}")
    print(f"  Score diff (max): {diff:.8f}")
    print(f"  [{'✓' if perm_ok else '✗'}] Permutation invariant (diff < 1e-4): {perm_ok}")

    return perm_ok


def demo_contrastive_pretrain(model, device):
    """Demo: contrastive pre-training."""
    print("\n" + "=" * 70)
    print("  COMPONENT TEST 3: Contrastive Pre-Training (InfoNCE)")
    print("=" * 70)

    # Load corpus for co-occurrence pairs
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), "output", "cart_trajectories.csv")
    if os.path.exists(csv_path):
        corpus_df = pd.read_csv(csv_path)
        pairs = extract_cooccurrence_pairs(corpus_df, min_cooccurrence=20)
        print(f"\n  Co-occurrence pairs extracted: {len(pairs)}")
    else:
        # Mock pairs
        pairs = [(f"item_{i}", f"item_{i+1}") for i in range(50)]
        print(f"\n  Using mock pairs: {len(pairs)}")

    # Build item-to-index mapping
    all_items = set()
    for a, b in pairs:
        all_items.add(a)
        all_items.add(b)
    item_to_idx = {name: idx + 1 for idx, name in enumerate(sorted(all_items))}

    print(f"  Unique items in pairs: {len(item_to_idx)}")
    print(f"\n  Training InfoNCE (10 epochs)...")

    losses = contrastive_pretrain(
        model=model,
        item_to_idx=item_to_idx,
        pairs=pairs,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        temperature=0.07,
        device=str(device),
    )

    if len(losses) >= 2:
        decreased = losses[-1] < losses[0]
        print(f"\n  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")
        print(f"  [{'✓' if decreased else '✗'}] Loss decreased: {decreased}")
        return decreased
    return False


def demo_reranker():
    """Demo: LightGBM LambdaRank re-ranker."""
    print("\n" + "=" * 70)
    print("  COMPONENT TEST 4: LightGBM LambdaRank Re-Ranker")
    print("=" * 70)

    reranker = LightGBMReranker(k=8)

    # Generate mock training data
    rng = np.random.default_rng(42)
    features, labels, groups = reranker.generate_mock_training_data(
        n_queries=500,
        candidates_per_query=20,
        rng=rng,
    )
    print(f"\n  Training data: {features.shape[0]} candidates across {len(groups)} queries")
    
    from csao.nn.reranker import RERANKER_FEATURE_NAMES
    print(f"  Features per candidate: {features.shape[1]} ({', '.join(RERANKER_FEATURE_NAMES)})")

    # Train
    reranker.train(features, labels, groups)

    # Create mock candidates for re-ranking
    candidates = [
        RerankCandidate("Garlic Naan",    neural_score=0.85, gap_fill_score=1.0, item_margin=0.35, zone_velocity=120, acceptance_rate=0.7, price_ratio=0.08),
        RerankCandidate("Raita",          neural_score=0.72, gap_fill_score=1.0, item_margin=0.40, zone_velocity=90,  acceptance_rate=0.6, price_ratio=0.07),
        RerankCandidate("Gulab Jamun",    neural_score=0.65, gap_fill_score=0.0, item_margin=0.50, zone_velocity=60,  acceptance_rate=0.4, price_ratio=0.12),
        RerankCandidate("Lassi",          neural_score=0.78, gap_fill_score=1.0, item_margin=0.30, zone_velocity=100, acceptance_rate=0.8, price_ratio=0.11),
        RerankCandidate("Butter Roti",    neural_score=0.60, gap_fill_score=0.0, item_margin=0.60, zone_velocity=150, acceptance_rate=0.5, price_ratio=0.04),
        RerankCandidate("Rasmalai",       neural_score=0.55, gap_fill_score=0.0, item_margin=0.45, zone_velocity=40,  acceptance_rate=0.3, price_ratio=0.14),
        RerankCandidate("Masala Chai",    neural_score=0.68, gap_fill_score=1.0, item_margin=0.35, zone_velocity=80,  acceptance_rate=0.65, price_ratio=0.06),
        RerankCandidate("Papad",          neural_score=0.45, gap_fill_score=0.0, item_margin=0.70, zone_velocity=70,  acceptance_rate=0.55, price_ratio=0.03),
        RerankCandidate("Onion Salad",    neural_score=0.40, gap_fill_score=0.0, item_margin=0.65, zone_velocity=50,  acceptance_rate=0.45, price_ratio=0.04),
        RerankCandidate("Kheer",          neural_score=0.50, gap_fill_score=0.0, item_margin=0.42, zone_velocity=35,  acceptance_rate=0.35, price_ratio=0.13),
    ]

    print(f"\n  Re-ranking {len(candidates)} candidates → top K={reranker.k}")
    ranked = reranker.rerank(candidates)

    print(f"\n  {'Rank':<6s} {'Item':<25s} {'Score':>8s}")
    print(f"  {'─'*6} {'─'*25} {'─'*8}")
    for i, (name, score) in enumerate(ranked):
        print(f"  {i+1:<6d} {name:<25s} {score:>8.4f}")

    k_ok = len(ranked) == 8
    print(f"\n  [{'✓' if k_ok else '✗'}] Returned K={len(ranked)} candidates (expected 8)")

    return k_ok


def main():
    print("=" * 70)
    print("  CSAO STAGE 4: HYBRID NEURAL ARCHITECTURE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Initialize model ─────────────────────────────────────────
    model = CSAOHybridModel(
        num_items=200,
        embedding_dim=128,
        context_dim=11,
        gap_dim=5,
        num_heads=4,
        num_inducing=8,
        num_isab_layers=2,
        gru_layers=1,
        ff_dim=256,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Run all component tests ──────────────────────────────────
    results = {}

    scores, results["forward_pass"] = demo_forward_pass(model, device)
    results["permutation_invariance"] = demo_permutation_invariance(model, device)
    results["contrastive_loss_decreased"] = demo_contrastive_pretrain(model, device)
    results["reranker_k8"] = demo_reranker()

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}")
        all_pass &= passed

    outcome = "ALL CHECKS PASSED ✅" if all_pass else "SOME CHECKS FAILED ⚠️"
    print(f"\n  {outcome}")

    print(f"\n{'=' * 70}")
    print(f"  Stage 4 Hybrid Neural Architecture — COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
