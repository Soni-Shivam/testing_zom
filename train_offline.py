#!/usr/bin/env python3
"""
train_offline.py
================
Offline training script for the CSAO recommendation system.

Run this ONCE before launching the Streamlit app:

    python train_offline.py

This script:
  1. Instantiates CSAOEngine
  2. Runs the full offline pipeline (data synthesis + feature store)
  3. Trains the PyTorch neural backbone and LightGBM re-ranker
  4. Serializes ALL required artifacts to ./artifacts/ so that
     app.py can boot instantly without any training.

Artifacts produced
------------------
  artifacts/neural_model.pt     - PyTorch state_dict
  artifacts/temperature.pt      - Learned temperature scalar
  artifacts/lgbm_model.txt      - LightGBM booster (text format)
  artifacts/slm_cache.pt        - SLM embedding bytes dict {key: bytes}
  artifacts/item_mappings.pkl   - item_to_idx, idx_to_item, item_prices
  artifacts/item_meta.pkl       - item_meta list (for ColdStartRouter)
  artifacts/corpus_df.parquet   - Synthetic corpus (for feature store re-hydration)
"""

import os
import sys
import time
import pickle
import torch

# ── Resolve project root so this script works from any cwd ──────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Ensure project root is on sys.path so csao.* imports are available
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine import CSAOEngine  # noqa: E402 — after sys.path fix


def main() -> None:
    t_start = time.time()

    print("=" * 70)
    print("  CSAO OFFLINE TRAINING PIPELINE")
    print("=" * 70)

    # ── Step 1: Instantiate engine ───────────────────────────────────────────
    print("\n[train_offline] Instantiating CSAOEngine...")
    
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"  [Info] Training on device: {device.upper()}")
    engine = CSAOEngine(seed=42, device=device)

    # ── Step 2: Run offline pipeline (data + features + cold-start) ─────────
    print("\n[train_offline] Running offline pipeline (n_trajectories=10000)...")
    engine.run_offline_pipeline(n_trajectories=10000)

    # ── Step 3: Train neural backbone + LightGBM re-ranker ──────────────────
    print("\n[train_offline] Training system (epochs=5, limit_batches=None)...")
    engine.train_system(epochs=5, limit_batches=None)

    # ── Step 4: Persist artifacts ────────────────────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"\n[train_offline] Saving artifacts to: {ARTIFACT_DIR}")

    # 4a. PyTorch neural model weights
    neural_model_path = os.path.join(ARTIFACT_DIR, "neural_model.pt")
    torch.save(engine.neural_model.state_dict(), neural_model_path)
    print(f"   neural_model.pt  ({os.path.getsize(neural_model_path) / 1024:.1f} KB)")

    # 4b. Learned temperature scalar (nn.Parameter)
    temperature_path = os.path.join(ARTIFACT_DIR, "temperature.pt")
    torch.save(engine.temperature.data, temperature_path)
    print(f"   temperature.pt")

    # 4c. LightGBM booster (text format — portable, human-readable)
    lgbm_path = os.path.join(ARTIFACT_DIR, "lgbm_model.txt")
    if engine.reranker is not None and engine.reranker.model is not None:
        engine.reranker.model.save_model(lgbm_path)
        print(f"   lgbm_model.txt   ({os.path.getsize(lgbm_path) / 1024:.1f} KB)")
    else:
        print("   LightGBM model is None — skipping lgbm_model.txt")

    # 4d. SLM embedding cache
    # The SimulatedRedisStore holds SLM vectors as raw bytes at keys "slm:<item_name>".
    # We extract them into a plain dict so they can be replayed cheaply at serve time.
    slm_cache: dict = {}
    for full_key, value in engine.feature_store._store.items():
        # Only grab keys in the "default" namespace that are SLM embeddings
        # SimulatedRedisStore prefixes keys: "default:slm:<name>"
        if full_key.startswith("default:slm:"):
            bare_key = full_key[len("default:"):]   # strip namespace prefix
            slm_cache[bare_key] = value              # value is bytes
    slm_cache_path = os.path.join(ARTIFACT_DIR, "slm_cache.pt")
    torch.save(slm_cache, slm_cache_path)
    print(f"   slm_cache.pt     ({len(slm_cache)} entries, "
          f"{os.path.getsize(slm_cache_path) / 1024:.1f} KB)")

    # 4e. item_to_idx / idx_to_item / item_prices
    mappings = {
        "item_to_idx": engine.item_to_idx,
        "idx_to_item": engine.idx_to_item,
        "item_prices": engine.item_prices,
    }
    mappings_path = os.path.join(ARTIFACT_DIR, "item_mappings.pkl")
    with open(mappings_path, "wb") as f:
        pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   item_mappings.pkl ({len(engine.item_to_idx)} items, "
          f"{os.path.getsize(mappings_path) / 1024:.1f} KB)")

    # 4f. item_meta list (needed to re-build ColdStartRouter at serve time)
    item_meta_path = os.path.join(ARTIFACT_DIR, "item_meta.pkl")
    with open(item_meta_path, "wb") as f:
        pickle.dump(engine.item_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   item_meta.pkl    ({len(engine.item_meta)} entries)")

    # 4g. Corpus DataFrame (Parquet — fast, columnar, small)
    corpus_path = os.path.join(ARTIFACT_DIR, "corpus_df.parquet")
    engine.corpus_df.to_parquet(corpus_path, index=False)
    print(f"   corpus_df.parquet ({len(engine.corpus_df)} rows, "
          f"{os.path.getsize(corpus_path) / 1024:.1f} KB)")

    # 4h. Synthetic User DB containing order history / behavior profiles
    user_db_path = os.path.join(ARTIFACT_DIR, "user_db.pkl")
    with open(user_db_path, "wb") as f:
        pickle.dump(engine.user_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   user_db.pkl      ({len(engine.user_db)} users, "
          f"{os.path.getsize(user_db_path) / 1024:.1f} KB)")

    elapsed = time.time() - t_start
    print(f"\n[train_offline]  All artifacts saved in {elapsed:.1f}s → {ARTIFACT_DIR}")
    print("  You can now launch: streamlit run app.py")


if __name__ == "__main__":
    main()
