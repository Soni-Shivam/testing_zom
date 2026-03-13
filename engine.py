#!/usr/bin/env python3
"""
CSAO End-to-End Orchestration Engine
====================================
Integrates the four stages of the Cart Super Add-On recommendation system:
1. Stage 1: Data Synthesis
2. Stage 2: Feature Engineering Store
3. Stage 3: Cold-Start Strategy
4. Stage 4: Hybrid Neural Architecture

This engine serves as the unified class containing:
- run_offline_pipeline()
- train_system()
- predict_addon()
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

# --- Stage 1 Imports ---
from csao.pipeline import SynthesisPipeline

# --- Stage 2 Imports ---
from csao.features.feature_store import (
    SimulatedRedisStore,
    NightlyOfflineJob,
    NearRealTimeJob,
    OnlinePerRequestCalculator,
)
from csao.features.item_features import CandidateItemFeatureGenerator
from csao.config.taxonomies import CUISINE_MENUS, ALL_CUISINES

# --- Stage 3 Imports ---
from csao.coldstart.config import DEFAULT_CONFIG as COLDSTART_CONFIG
from csao.coldstart.item_coldstart import ItemColdStart
from csao.coldstart.restaurant_coldstart import RestaurantColdStart
from csao.coldstart.user_coldstart import UserColdStart, UserObservation
from csao.coldstart.router import ColdStartRouter, ColdStartRequest

# --- Stage 4 Imports ---
from csao.nn.model import CSAOHybridModel
from csao.nn.slm import SLMEmbedder
from csao.nn.contrastive import (
    contrastive_pretrain,
    extract_cooccurrence_pairs,
)
from csao.nn.reranker import LightGBMReranker, RerankCandidate


class CSAOEngine:
    """End-to-end orchestration engine for the CSAO recommendation system."""

    def __init__(self, seed: int = 42, device: str = "cpu"):
        self.seed = seed
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        # Storage
        self.corpus_df: Optional[pd.DataFrame] = None
        self.feature_store: Optional[SimulatedRedisStore] = None
        
        # Stage 3 Routers
        self.cold_start_router: Optional[ColdStartRouter] = None

        # Stage 4 Models
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.neural_model: Optional[CSAOHybridModel] = None
        self.reranker: Optional[LightGBMReranker] = None

        # Stage 2 Calculator
        self.online_calculator: Optional[OnlinePerRequestCalculator] = None
        self.item_gen: Optional[CandidateItemFeatureGenerator] = None

        # SLM Storage
        self.slm_embedder: Optional[SLMEmbedder] = None
        self.item_slm_cache: Dict[str, torch.Tensor] = {}

    def run_offline_pipeline(self, n_trajectories: int = 1000) -> None:
        """
        1. Data & Feature Initialization
        - Generate Stage 1 data corpus
        - Build Stage 2 feature store
        - Build Stage 3 knowledge graph
        """
        print("\n" + "="*70)
        print("  STEP 1: OFFLINE PIPELINE INITIALIZATION")
        print("="*70)

        # Generate Data
        print(f"[Engine] Generating synthetic corpus ({n_trajectories} trajectories)...")
        pipeline = SynthesisPipeline(seed=self.seed)
        df, _ = pipeline.generate(n_trajectories=n_trajectories)
        self.corpus_df = df
        
        known_items = list(self.corpus_df["item_name"].unique())
        self.item_to_idx = {name: idx + 1 for idx, name in enumerate(sorted(known_items))}
        self.idx_to_item = {idx: name for name, idx in self.item_to_idx.items()}
        print(f"[Engine] Data generation complete. {len(df)} interactions across {len(known_items)} unique items.")

        # Build Feature Store
        print("[Engine] Initializing feature store and running nightly/NRT jobs...")
        self.feature_store = SimulatedRedisStore()
        
        nightly_job = NightlyOfflineJob(store=self.feature_store, corpus_df=self.corpus_df)
        nightly_job.run()
        
        nrt_job = NearRealTimeJob(store=self.feature_store, corpus_df=self.corpus_df)
        nrt_job.run()
        
        self.online_calculator = OnlinePerRequestCalculator(store=self.feature_store, corpus_df=self.corpus_df)
        self.item_gen = CandidateItemFeatureGenerator(corpus_df=self.corpus_df)

        # Build Knowledge Graph and Cold Start
        print("[Engine] Initializing Stage 3 Cold-Start Knowledge Graph & Models...")
        item_cs = ItemColdStart(
            item_feature_gen=self.item_gen,
            known_item_names=known_items,
            config=COLDSTART_CONFIG,
        )
        restaurant_cs = RestaurantColdStart(config=COLDSTART_CONFIG)
        user_cs = UserColdStart(corpus_df=self.corpus_df, config=COLDSTART_CONFIG)
        
        self.cold_start_router = ColdStartRouter(
            item_cs=item_cs,
            restaurant_cs=restaurant_cs,
            user_cs=user_cs,
            config=COLDSTART_CONFIG,
        )

        # Initialize SLM Embedder and pre-compute embeddings
        print("[Engine] Initializing SLM Embedder and pre-computing item embeddings...")
        self.slm_embedder = SLMEmbedder(device=str(self.device))
        
        # Create a unique list of items with cuisine/desc for SLM
        # In this simulation, we'll just use the item_name and its cuisine from the corpus
        item_meta_df = self.corpus_df[["item_name", "cuisine"]].drop_duplicates("item_name")
        # In a real system, we'd have descriptions; here we mock a simple one
        item_meta_df["description"] = item_meta_df["item_name"] + " from the menu."
        
        slm_embs = self.slm_embedder.generate_embeddings(item_meta_df)
        for i, row in enumerate(item_meta_df.itertuples()):
            self.item_slm_cache[row.item_name] = torch.tensor(
                slm_embs[i], dtype=torch.float32, device=self.device
            )
        
        # Add a zero vector for padding/unknown
        self.item_slm_cache["<PAD>"] = torch.zeros(384, dtype=torch.float32, device=self.device)

        print("[Engine] Offline initialization complete.")

    def train_system(self, epochs: int = 1, limit_batches: int = 50) -> None:
        """
        2. Model Training Orchestration
        - Contrastive Pre-training
        - Neural Backbone Training (Looping over trajectories)
        - LightGBM LambdaRank Training
        """
        if self.corpus_df is None:
            raise ValueError("Corpus not initialized. Call run_offline_pipeline() first.")

        print("\n" + "="*70)
        print("  STEP 2: MODEL TRAINING ORCHESTRATION")
        print("="*70)

        num_items = len(self.item_to_idx) + 1  # 1-indexed, 0 is padding
        self.neural_model = CSAOHybridModel(
            num_items=num_items,
            embedding_dim=128,
            slm_dim=384,      # Updated for all-MiniLM-L6-v2
            context_dim=11,
            gap_dim=5,
            num_heads=4,
            num_inducing=8,
            num_isab_layers=2,
            gru_layers=1,
            ff_dim=256,
            dropout=0.1,
        ).to(self.device)

        # Contrastive Pre-training
        print("[Engine] Running contrastive pre-training on item co-occurrences...")
        pairs = extract_cooccurrence_pairs(self.corpus_df, min_cooccurrence=5)
        contrastive_pretrain(
            model=self.neural_model,
            item_to_idx=self.item_to_idx,
            pairs=pairs,
            epochs=2,
            batch_size=32,
            device=str(self.device),
        )

        # Neural Backbone Training Loop
        print("[Engine] Training Neural Backbone on Cart Trajectories...")
        self.neural_model.train()
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trajectory_ids = self.corpus_df["trajectory_id"].unique()
        self.rng.shuffle(trajectory_ids)
        
        batches_processed = 0
        total_loss = 0.0

        for traj_id in trajectory_ids[:limit_batches]:
            traj_df = self.corpus_df[self.corpus_df["trajectory_id"] == traj_id].sort_values("step_index")
            if len(traj_df) < 2:
                continue

            # Fake User History (Since we simulate single trajectories per user for simplicity here)
            # We'll just pad zero history
            hist_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            hist_qty = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            hist_slm = torch.zeros((1, 1, 384), dtype=torch.float32, device=self.device)
            hist_lengths = torch.ones(1, dtype=torch.long, device=self.device)

            cart_items = []
            cart_total = 0.0
            
            # Step through the trajectory
            for i in range(len(traj_df) - 1):
                curr_row = traj_df.iloc[i]
                next_row = traj_df.iloc[i + 1]

                cart_items.append({
                    "name": curr_row["item_name"],
                    "category": curr_row["item_category"],
                    "quantity": curr_row["quantity"],
                    "unit_price": curr_row["item_price"],
                })
                cart_total += curr_row["quantity"] * curr_row["item_price"]

                # We need context & gap from OnlinePerRequestCalculator
                f_vec, segments = self.online_calculator.compute_feature_vector(
                    user_id=int(curr_row["user_id"]),
                    user_aov_ceiling=float(curr_row["aov_ceiling"]),
                    cart_items=cart_items,
                    cart_total=cart_total,
                    cuisine=str(curr_row["cuisine"]),
                    hour_of_day=int(curr_row["hour_of_day"]),
                    day_of_week=0,
                    is_weekend=bool(curr_row["is_weekend"]),
                    city=str(curr_row["city"]),
                    candidate_item_name=str(next_row["item_name"]),
                    candidate_item_category=str(next_row["item_category"]),
                )

                # Extract context
                ctx_cyc_h = f_vec[segments["ctx.cyclical_hour"][0]:segments["ctx.cyclical_hour"][1]]
                ctx_cyc_d = f_vec[segments["ctx.cyclical_day"][0]:segments["ctx.cyclical_day"][1]]
                ctx_type = f_vec[segments["ctx.day_type"][0]:segments["ctx.day_type"][1]]
                ctx_wth = f_vec[segments["ctx.weather_proxy"][0]:segments["ctx.weather_proxy"][1]]
                context = np.concatenate([ctx_cyc_h, ctx_cyc_d, ctx_type, ctx_wth])
                context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Extract gap
                gap_start, gap_end = segments["cart.meal_gap_vector"]
                gap = f_vec[gap_start:gap_end]
                gap_t = torch.tensor(gap, dtype=torch.float32, device=self.device).unsqueeze(0)

                target_item_name = next_row["item_name"]
                target_idx = self.item_to_idx.get(target_item_name, 0)
                
                # We will pick 1 target and 7 negative samples
                cands = [target_item_name]
                while len(cands) < 8:
                    neg = self.rng.choice(list(self.item_to_idx.keys()))
                    if neg not in cands:
                        cands.append(neg)
                
                cand_ids = torch.tensor([self.item_to_idx[c] for c in cands], dtype=torch.long, device=self.device).unsqueeze(0)
                cand_slm = torch.stack([self.item_slm_cache.get(c, self.item_slm_cache["<PAD>"]) for c in cands]).unsqueeze(0)

                # Prepare Cart
                cart_idx_seq = [self.item_to_idx.get(c["name"], 0) for c in cart_items]
                cart_qty_seq = [c["quantity"] for c in cart_items]
                
                cart_t = torch.tensor(cart_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                qty_t = torch.tensor(cart_qty_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.zeros(1, len(cart_idx_seq), dtype=torch.bool, device=self.device)

                # Prepare Cart SLM
                cart_slm = torch.stack([self.item_slm_cache.get(c["name"], self.item_slm_cache["<PAD>"]) for c in cart_items]).unsqueeze(0)

                # Forward pass
                scores = self.neural_model(
                    cart_t, qty_t, cart_slm, mask_t,
                    hist_ids, hist_qty, hist_slm, hist_lengths,
                    context_t, gap_t, cand_ids, cand_slm
                )

                # Target is index 0
                labels = torch.zeros(1, dtype=torch.long, device=self.device)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches_processed += 1

        print(f"[Engine] Backbone Training Complete. Processed {batches_processed} steps, Avg Loss: {total_loss/max(1, batches_processed):.4f}")

        # LightGBM Re-Ranker Training
        print("[Engine] Generating mock tabular features mixed with neural scores to train LightGBM...")
        self.neural_model.eval()
        self.reranker = LightGBMReranker(k=8)
        
        # Simple mock feature generation mimicking the loop output
        features, labels, groups = self.reranker.generate_mock_training_data(
            n_queries=100, candidates_per_query=20, rng=self.rng
        )
        self.reranker.train(features, labels, groups)
        print("[Engine] Model Training Orchestration Complete.")

    def predict_addon(
        self,
        user_id: int,
        cart_items: List[Dict[str, any]],
        restaurant_id: str,
        restaurant_name: str,
        restaurant_cuisine: str,
        city: str,
        hour_of_day: int,
        day_of_week: int,
        is_weekend: bool,
    ) -> List[Tuple[str, float]]:
        """
        3. Online Inference Endpoint
        Takes a live request and returns top-K predictions under 65ms latency constraint.
        """
        t0 = time.perf_counter_ns()
        
        # Determine candidate items from restaurant menu
        if restaurant_cuisine in CUISINE_MENUS:
            menu = CUISINE_MENUS[restaurant_cuisine]
            menu_items = []
            for cat_items in menu.values():
                for item in cat_items:
                    if item["name"] not in [c["name"] for c in cart_items]:
                        menu_items.append(item["name"])
        else:
            menu_items = list(self.item_to_idx.keys())[:20]

        cart_names = [c["name"] for c in cart_items]
        
        # Step A: Cold-Start Routing
        request = ColdStartRequest(
            user_id=user_id,
            user_order_count=0, # Assume 0 to trigger user cold start
            user_orders=[],
            restaurant_name=restaurant_name,
            restaurant_interaction_count=0, # Assume 0 to trigger rest cold start
            restaurant_menu=menu_items,
            candidate_item_name=menu_items[0] if menu_items else "",
            candidate_item_has_interactions=True,
            cart_main_items=cart_names,
            cuisine=restaurant_cuisine,
            city=city,
        )
        cs_result = self.cold_start_router.route(request)
        
        # Gather final candidates (Combine CS candidates with Menu)
        candidates = set(menu_items)
        if cs_result.city_popularity_recs:
            for item, _ in cs_result.city_popularity_recs[:5]:
                candidates.add(item)
        if cs_result.kg_seeded_recs:
            for item, _ in cs_result.kg_seeded_recs[:5]:
                candidates.add(item)
                
        candidates = list(candidates)[:20] # Take at most 20 candidates for scoring

        # --- Batch Neural Scoring ---
        # 1. Compute context and gap once (they are independent of candidates)
        cart_total = sum(c.get("unit_price", 100) * c.get("quantity", 1) for c in cart_items)
        user_aov = 500.0
        
        f_vec_base, segments_base = self.online_calculator.compute_feature_vector(
            user_id=user_id,
            user_aov_ceiling=user_aov,
            cart_items=cart_items,
            cart_total=cart_total,
            cuisine=restaurant_cuisine,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            city=city,
            candidate_item_name=candidates[0] if candidates else "",
            candidate_item_category="Unknown",
        )
        
        ctx_cyc_h = f_vec_base[segments_base["ctx.cyclical_hour"][0]:segments_base["ctx.cyclical_hour"][1]]
        ctx_cyc_d = f_vec_base[segments_base["ctx.cyclical_day"][0]:segments_base["ctx.cyclical_day"][1]]
        ctx_type = f_vec_base[segments_base["ctx.day_type"][0]:segments_base["ctx.day_type"][1]]
        ctx_wth = f_vec_base[segments_base["ctx.weather_proxy"][0]:segments_base["ctx.weather_proxy"][1]]
        context = np.concatenate([ctx_cyc_h, ctx_cyc_d, ctx_type, ctx_wth])
        
        context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Meal gap vector
        gap_start, gap_end = segments_base["cart.meal_gap_vector"]
        gap = f_vec_base[gap_start:gap_end]
        gap_t = torch.tensor(gap, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Cart tensors
        cart_idx_seq = [self.item_to_idx.get(c["name"], 0) for c in cart_items]
        cart_qty_seq = [c.get("quantity", 1) for c in cart_items]
        
        cart_t = torch.tensor(cart_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)
        qty_t = torch.tensor(cart_qty_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.zeros(1, len(cart_idx_seq), dtype=torch.bool, device=self.device)
        
        # History (empty)
        hist_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        hist_qty = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        hist_lengths = torch.ones(1, dtype=torch.long, device=self.device)
        
        # Candidate tensors: process all at once
        cand_idx_seq = [self.item_to_idx.get(cand, 0) for cand in candidates]
        cand_t = torch.tensor(cand_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, K)
        cand_slm = torch.stack([self.item_slm_cache.get(cand, self.item_slm_cache["<PAD>"]) for cand in candidates]).unsqueeze(0)
        
        # Additional tensors for SLM
        cart_slm = torch.stack([self.item_slm_cache.get(c["name"], self.item_slm_cache["<PAD>"]) for c in cart_items]).unsqueeze(0)
        hist_slm = torch.zeros((1, 1, 384), dtype=torch.float32, device=self.device)

        self.neural_model.eval()
        with torch.no_grad():
            scores = self.neural_model(
                cart_t, qty_t, cart_slm, mask_t,
                hist_ids, hist_qty, hist_slm, hist_lengths,
                context_t, gap_t, cand_t, cand_slm
            )  # (1, K)
        
        # --- Form Rerank Candidates ---
        rerank_candidates = []
        for i, cand in enumerate(candidates):
            neural_score = scores[0, i].item()
            
            # Fast sparse feature lookup
            # Since generating full vector is slow, we lookup the components for tabular features directly
            gfs_features = self.item_gen.compute_candidate_features(cand, "Unknown", gap, hour_of_day, city)
            
            gfs = float(gfs_features.get("gap_fill_score", [0.0])[0])
            velocity = float(gfs_features.get("zone_velocity", [0.0])[0])
            
            rc = RerankCandidate(
                item_name=cand,
                neural_score=neural_score,
                gap_fill_score=gfs,
                item_margin=0.3,
                zone_velocity=velocity,
                acceptance_rate=0.4,
                price_ratio=100.0 / max(cart_total, 1.0)
            )
            rerank_candidates.append(rc)
                
        # Step D: Re-Ranking
        ranked = self.reranker.rerank(rerank_candidates)
        
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        print(f"[Inference] predict_addon completed in {elapsed_ms:.2f} ms")
        return ranked


if __name__ == "__main__":
    print("="*70)
    print("  CSAO END-TO-END ENGINE INITIALIZATION")
    print("="*70)
    engine = CSAOEngine()
    
    # 1. Run Offline Pipeline
    engine.run_offline_pipeline(n_trajectories=100)
    
    # 2. Train System
    engine.train_system(epochs=1, limit_batches=50)
    
    # 3. Online Prediction Call
    print("\n" + "="*70)
    print("  STEP 3: ONLINE PREDICTION ENDPOINT")
    print("="*70)
    
    live_cart = [
        {"name": "Butter Chicken", "category": "Mains", "quantity": 1, "unit_price": 400},
        {"name": "Garlic Naan", "category": "Breads", "quantity": 2, "unit_price": 60}
    ]
    
    ranked_candidates = engine.predict_addon(
        user_id=12345,
        cart_items=live_cart,
        restaurant_id="rest_999",
        restaurant_name="Spice Heaven",
        restaurant_cuisine="North Indian",
        city="Delhi-NCR",
        hour_of_day=20,
        day_of_week=5,
        is_weekend=True
    )
    
    print("\n  Final Ranked Recommendations (K=8):")
    for i, (item, score) in enumerate(ranked_candidates):
        print(f"    {i+1}: {item:<20s} [Score: {score:.4f}]")
