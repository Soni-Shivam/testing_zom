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
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
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
        
        # In-Memory Backend User Database
        self.user_db = {
            1: {"order_count": 0, "total_spend": 0.0, "mean_aov": 0.0, "past_ordered_items": [], "cuisine_counts": {}},
            2: {"order_count": 4, "total_spend": 1200.0, "mean_aov": 300.0, "past_ordered_items": [
                {"name": "Masala Dosa", "quantity": 1}, {"name": "Idli", "quantity": 2},
                {"name": "Vada", "quantity": 1}, {"name": "Filter Coffee", "quantity": 2}
            ], "cuisine_counts": {"South Indian": 4}},
            3: {"order_count": 15, "total_spend": 7500.0, "mean_aov": 500.0, "past_ordered_items": [
                {"name": "Butter Chicken", "quantity": 1}, {"name": "Garlic Naan", "quantity": 3},
                {"name": "Dal Makhani", "quantity": 1}, {"name": "Jeera Rice", "quantity": 1},
                {"name": "Paneer Tikka", "quantity": 1}, {"name": "Lassi", "quantity": 2},
                {"name": "Tandoori Roti", "quantity": 4}, {"name": "Gulab Jamun", "quantity": 2},
                {"name": "Chicken Biryani", "quantity": 1}, {"name": "Raita", "quantity": 1}
            ], "cuisine_counts": {"North Indian": 10, "Mughlai": 5}}
        }
        self.rng = np.random.default_rng(seed)

        # Storage
        self.corpus_df: Optional[pd.DataFrame] = None
        self.feature_store: Optional[SimulatedRedisStore] = None
        
        # Stage 3 Routers
        self.cold_start_router: Optional[ColdStartRouter] = None

        # Stage 4 Models
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.item_meta: List[Dict] = []          # Rich item metadata (loaded from taxonomy)
        self.item_prices: Dict[str, float] = {}  # Global catalog prices
        self.neural_model: Optional[CSAOHybridModel] = None
        self.reranker: Optional[LightGBMReranker] = None

        # Stage 2 Calculator
        self.online_calculator: Optional[OnlinePerRequestCalculator] = None
        self.item_gen: Optional[CandidateItemFeatureGenerator] = None

        # SLM Storage (Fix 5)
        self.slm_embedder: Optional[SLMEmbedder] = None
        # We will use self.feature_store for SLM (Fix 5)
        
        # Temperature Calibration
        self.temperature: nn.Parameter = nn.Parameter(torch.ones(1, device=self.device) * 1.5)

        # Fix 5: Prefetch cache keyed by cuisine name
        self._prefetch_cache: Dict[str, List[str]] = {}


    def place_order(self, user_id: int, cart_items: List[Dict[str, any]], cart_total: float, cuisine: str = "Unknown") -> None:
        """Saves a completed order to the user's history in the in-memory database."""
        if user_id not in self.user_db:
            self.user_db[user_id] = {"order_count": 0, "total_spend": 0.0, "mean_aov": 0.0, "past_ordered_items": [], "cuisine_counts": {}}
            
        profile = self.user_db[user_id]
        profile["order_count"] += 1
        profile["total_spend"] += cart_total
        profile["mean_aov"] = profile["total_spend"] / profile["order_count"]
        
        for item in cart_items:
            profile["past_ordered_items"].append({"name": item["name"], "quantity": item.get("quantity", 1)})
            
        profile["cuisine_counts"][cuisine] = profile["cuisine_counts"].get(cuisine, 0) + 1
        print(f"[Engine] Order placed for User {user_id}. Total Orders: {profile['order_count']}, AOV: ₹{profile['mean_aov']:.2f}")

    def get_user_analytics(self, user_id: int) -> dict:
        """
        Extracts high-level behavioral metrics and aggregates purchasing
        patterns for user profiling and CRM display.
        """
        if user_id not in self.user_db:
            return {"status": "not_found"}

        profile = self.user_db[user_id]
        if profile["order_count"] == 0:
            return {"status": "cold_start", "message": "No order history available."}

        # Aggregate item frequencies across all past orders
        item_counts: Dict[str, int] = {}
        for item in profile["past_ordered_items"]:
            name = item["name"]
            item_counts[name] = item_counts.get(name, 0) + item.get("quantity", 1)

        top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        top_cuisines = sorted(profile["cuisine_counts"].items(), key=lambda x: x[1], reverse=True)

        return {
            "status": "active",
            "total_orders": profile["order_count"],
            "lifetime_value": profile["total_spend"],
            "aov": profile["mean_aov"],
            "top_cuisines": top_cuisines[:3],
            "favorite_items": top_items[:5],
            "distinct_items_tried": len(item_counts),
        }

    def get_homepage_recommendations(self, user_id: int, k: int = 5) -> List[tuple]:
        """
        Generates global catalog recommendations by pooling the user's
        historical SLM embeddings (latent user-profile vector) and scoring
        un-purchased inventory via cosine similarity.

        This is pure Content-Based Filtering — no extra training needed.
        """
        analytics = self.get_user_analytics(user_id)
        if analytics["status"] != "active":
            # Cold-start fallback: global taxonomy bestsellers
            return [("Butter Chicken", 0.99), ("Masala Dosa", 0.95), ("Margherita Pizza", 0.90)]

        # 1. Identify the user's historical favorites
        favorite_item_names = [item[0] for item in analytics["favorite_items"]]

        # 2. Build the Latent User Profile Vector via mean-pooled SLM embeddings
        fav_tensors = self._get_slm_batch(favorite_item_names)   # (Num_Favs, 384)
        user_profile_vector = fav_tensors.mean(dim=0, keepdim=True)  # (1, 384)

        # 3. Exploration: restrict candidates to items the user has NEVER ordered
        past_item_set = {item["name"] for item in self.user_db[user_id]["past_ordered_items"]}
        candidate_catalog = [
            name for name in self.item_to_idx.keys()
            if name not in past_item_set
        ]

        if not candidate_catalog:
            return []  # User has tried every item in the catalog

        # 4. Score via Cosine Similarity in the 384-d semantic space
        catalog_tensors = self._get_slm_batch(candidate_catalog)  # (N, 384)
        similarities = torch.nn.functional.cosine_similarity(user_profile_vector, catalog_tensors)

        # 5. Rank and return top-K
        sim_scores = similarities.cpu().numpy()
        ranked_indices = np.argsort(sim_scores)[::-1][:k]

        return [
            (candidate_catalog[i], float(sim_scores[i]))
            for i in ranked_indices
        ]



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

        # Generate or Load Data
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "cart_trajectories.csv")
        if os.path.exists(csv_path):
            print(f"[Engine] Loading full canonical corpus from {csv_path} (10k+ entries)...")
            self.corpus_df = pd.read_csv(csv_path)
        else:
            print(f"[Engine] Generating synthetic corpus (10,000 trajectories)...")
            pipeline = SynthesisPipeline(seed=self.seed)
            df, _ = pipeline.generate(n_trajectories=10000)
            self.corpus_df = df
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            self.corpus_df.to_csv(csv_path, index=False)
        
        # [Taxonomy Fix 1]: Global Vocabulary Initialization
        print("[Engine] Establishing global vocabulary, semantic index, and price maps from taxonomies...")
        canonical_items = set()
        self.item_meta = [] # Store rich metadata for SLM
        self.item_prices = {} # TRUE Global Catalog Prices
        
        for cuisine, categories in CUISINE_MENUS.items():
            for category, items in categories.items():
                for item in items:
                    name = item["name"]
                    self.item_prices[name] = float(item.get("price", 150.0))
                    if name not in canonical_items:
                        canonical_items.add(name)
                        self.item_meta.append({
                            "item_name": name,
                            "cuisine": cuisine,
                            "category": category,
                            "description": f"A delicious {category} dish from {cuisine} cuisine named {name}."
                        })
        
        known_items = sorted(list(canonical_items))
        self.item_to_idx = {name: idx + 1 for idx, name in enumerate(known_items)}
        self.idx_to_item = {idx: name for name, idx in self.item_to_idx.items()}
        print(f"[Engine] Data loading complete. {len(self.corpus_df)} simulated interactions mapped across {len(known_items)} canonical items.")

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
        print("[Engine] Initializing SLM Embedder and pre-computing semantic embeddings into simulated Redis...")
        self.slm_embedder = SLMEmbedder(device=str(self.device))
        
        # [Taxonomy Fix 2]: Complete SLM Semantic Indexing
        item_meta_df = pd.DataFrame(self.item_meta)
        slm_embs = self.slm_embedder.generate_embeddings(item_meta_df)
        
        # Store in SimulatedRedisStore
        for i, row in enumerate(item_meta_df.itertuples()):
            key = f"slm:{row.item_name}"
            # Serialize numpy array to bytes
            self.feature_store.set(key, slm_embs[i].tobytes())
            
        # Add PAD vector
        self.feature_store.set("slm:<PAD>", np.zeros(384, dtype=np.float32).tobytes())

        print("[Engine] Offline initialization complete.")
        
    # Fix 3: _mock_item_feature() deleted — replaced by real feature store lookups.

    def _get_slm_batch(self, item_names: List[str]) -> torch.Tensor:
        """Helper to fetch 384-d SLM vectors from Redis (Fix 5)."""
        keys = [f"slm:{name}" for name in item_names]
        # mget returns list of bytes or None
        raw_values = []
        for key in keys:
            val = self.feature_store.get(key)
            if val is not None:
                raw_values.append(val)
            else:
                raw_values.append(self.feature_store.get("slm:<PAD>"))
                
        # Decode bytes back to numpy array
        arrays = [np.frombuffer(val, dtype=np.float32) for val in raw_values]
        tensors = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in arrays]
        return torch.stack(tensors)

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

        from tqdm import tqdm
        target_trajectories = trajectory_ids[:limit_batches] if limit_batches else trajectory_ids
        for traj_id in tqdm(target_trajectories, desc="Backbone Training", dynamic_ncols=True):
            traj_df = self.corpus_df[self.corpus_df["trajectory_id"] == traj_id].sort_values("step_index")
            if len(traj_df) < 2:
                continue

            # Define User History extraction (Fix 1)
            # Find all prior trajectories for this user
            curr_user_id = traj_df["user_id"].iloc[0]
            curr_time_step = traj_df["time_step"].iloc[0] if "time_step" in traj_df else 0 # Assuming ordered
            # We assume trajectory_ids are chronological
            past_orders = self.corpus_df[
                (self.corpus_df["user_id"] == curr_user_id) & 
                (self.corpus_df["trajectory_id"] < traj_id)
            ].sort_values("trajectory_id")

            if len(past_orders) > 0:
                # Take up to T=10 most recent items
                recent_items = past_orders.tail(10)
                hist_names = recent_items["item_name"].tolist()
                hist_qtys = recent_items["quantity"].tolist()
                
                # Dynamic Length
                actual_len = len(hist_names)
                h_lens = torch.tensor([actual_len], dtype=torch.long, device=self.device)
                
                # Pad sequence to T=10
                pad_len = 10 - actual_len
                hist_names = ["<PAD>"] * pad_len + hist_names
                hist_qtys = [0.0] * pad_len + hist_qtys
                
                h_ids = torch.tensor([self.item_to_idx.get(n, 0) for n in hist_names], dtype=torch.long, device=self.device).unsqueeze(0)
                h_qty = torch.tensor(hist_qtys, dtype=torch.float32, device=self.device).unsqueeze(0)
                h_slm = self._get_slm_batch(hist_names).unsqueeze(0)
            else:
                # New user, no history
                h_ids = torch.zeros((1, 10), dtype=torch.long, device=self.device)
                h_qty = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
                h_slm = torch.zeros((1, 10, 384), dtype=torch.float32, device=self.device)
                h_lens = torch.ones(1, dtype=torch.long, device=self.device)

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
                all_taxonomic_items = list(self.item_to_idx.keys())
                while len(cands) < 8:
                    neg = self.rng.choice(all_taxonomic_items)
                    if neg not in cands:
                        cands.append(neg)
                
                # [Taxonomy Fix 3]: Fallback Safety using .get()
                cand_ids = torch.tensor([self.item_to_idx.get(c, 0) for c in cands], dtype=torch.long, device=self.device).unsqueeze(0)
                cand_slm = self._get_slm_batch(cands).unsqueeze(0)

                # Prepare Cart
                cart_idx_seq = [self.item_to_idx.get(c["name"], 0) for c in cart_items]
                cart_qty_seq = [c["quantity"] for c in cart_items]
                cart_names = [c["name"] for c in cart_items]
                
                cart_t = torch.tensor(cart_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                qty_t = torch.tensor(cart_qty_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.zeros(1, len(cart_idx_seq), dtype=torch.bool, device=self.device)

                # Prepare Cart SLM
                cart_slm = self._get_slm_batch(cart_names).unsqueeze(0)

                # Forward pass
                scores = self.neural_model(
                    cart_t, qty_t, cart_slm, mask_t,
                    h_ids, h_qty, h_slm, h_lens,
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
        
        # print("[Engine] Generating mock tabular features mixed with neural scores to train LightGBM...")
        self.neural_model.eval()
        self.reranker = LightGBMReranker(k=10) # Set to 10 for validation extraction
        
        # Call Temperature Calibration (Fix 3)
        self.calibrate_temperature(limit_batches=10)
        
        # --------------------------------------------------------------------------------
        # [Fix 2]: Train LightGBM on True Neural Features, Not Mock Data
        # --------------------------------------------------------------------------------
        print("[Engine] Extracting true neural scores on validation set for LightGBM...")
        
        lgb_features = []
        lgb_labels = []
        lgb_groups = []
        
        # We will use the remaining trajectories as validation
        # If limit_batches is None, we trained on all of them. To get validation data, we'll
        # just pick the last 20 trajectories from the full list.
        if limit_batches is None:
            val_start_idx = len(trajectory_ids) - 20
            val_end_idx = len(trajectory_ids)
        else:
            val_start_idx = limit_batches
            val_end_idx = limit_batches + 20
            
        val_trajectory_ids = trajectory_ids[val_start_idx : val_end_idx]
        
        for traj_id in val_trajectory_ids:
            traj_df = self.corpus_df[self.corpus_df["trajectory_id"] == traj_id].sort_values("step_index")
            if len(traj_df) < 2:
                continue
                
            curr_user_id = traj_df["user_id"].iloc[0]
            past_orders = self.corpus_df[
                (self.corpus_df["user_id"] == curr_user_id) & 
                (self.corpus_df["trajectory_id"] < traj_id)
            ].sort_values("trajectory_id")

            if len(past_orders) > 0:
                recent_items = past_orders.tail(10)
                hist_names = recent_items["item_name"].tolist()
                hist_qtys = recent_items["quantity"].tolist()
                actual_len = len(hist_names)
                h_lens = torch.tensor([actual_len], dtype=torch.long, device=self.device)
                pad_len = 10 - actual_len
                hist_names = ["<PAD>"] * pad_len + hist_names
                hist_qtys = [0.0] * pad_len + hist_qtys
                h_ids = torch.tensor([self.item_to_idx.get(n, 0) for n in hist_names], dtype=torch.long, device=self.device).unsqueeze(0)
                h_qty = torch.tensor(hist_qtys, dtype=torch.float32, device=self.device).unsqueeze(0)
                h_slm = self._get_slm_batch(hist_names).unsqueeze(0)
            else:
                h_ids = torch.zeros((1, 10), dtype=torch.long, device=self.device)
                h_qty = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
                h_slm = torch.zeros((1, 10, 384), dtype=torch.float32, device=self.device)
                h_lens = torch.ones(1, dtype=torch.long, device=self.device)

            cart_items = []
            cart_total = 0.0
            
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

                ctx_cyc_h = f_vec[segments["ctx.cyclical_hour"][0]:segments["ctx.cyclical_hour"][1]]
                ctx_cyc_d = f_vec[segments["ctx.cyclical_day"][0]:segments["ctx.cyclical_day"][1]]
                ctx_type = f_vec[segments["ctx.day_type"][0]:segments["ctx.day_type"][1]]
                ctx_wth = f_vec[segments["ctx.weather_proxy"][0]:segments["ctx.weather_proxy"][1]]
                context = np.concatenate([ctx_cyc_h, ctx_cyc_d, ctx_type, ctx_wth])
                context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)

                gap_start, gap_end = segments["cart.meal_gap_vector"]
                gap = f_vec[gap_start:gap_end]
                gap_t = torch.tensor(gap, dtype=torch.float32, device=self.device).unsqueeze(0)

                target_item_name = next_row["item_name"]
                
                cands = [target_item_name]
                all_taxonomic_items = list(self.item_to_idx.keys())
                while len(cands) < 10:
                    neg = self.rng.choice(all_taxonomic_items)
                    if neg not in cands:
                        cands.append(neg)
                
                # [Taxonomy Fix 3]: Fallback Safety using .get()
                cand_ids = torch.tensor([self.item_to_idx.get(c, 0) for c in cands], dtype=torch.long, device=self.device).unsqueeze(0)
                cand_slm = self._get_slm_batch(cands).unsqueeze(0)

                cart_idx_seq = [self.item_to_idx.get(c["name"], 0) for c in cart_items]
                cart_qty_seq = [c["quantity"] for c in cart_items]
                cart_names = [c["name"] for c in cart_items]
                
                cart_t = torch.tensor(cart_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                qty_t = torch.tensor(cart_qty_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.zeros(1, len(cart_idx_seq), dtype=torch.bool, device=self.device)
                cart_slm = self._get_slm_batch(cart_names).unsqueeze(0)

                with torch.no_grad():
                    scores = self.neural_model(
                        cart_t, qty_t, cart_slm, mask_t,
                        h_ids, h_qty, h_slm, h_lens,
                        context_t, gap_t, cand_ids, cand_slm
                    )
                    scores = scores / self.temperature # Apply Calibration
                    
                # Form LightGBM row
                query_group_size = 0
                for c_idx, cand_name in enumerate(cands):
                    gfs_features = self.item_gen.compute_candidate_features(
                        cand_name, "Unknown", gap, int(curr_row["hour_of_day"]), str(curr_row["city"])
                    )

                    gfs = float(gfs_features.get("gap_fill_score", [0.0])[0])
                    velocity = float(gfs_features.get("zone_velocity", [0.0])[0])

                    # Fix 3: Real business signals (replaces _mock_item_feature)
                    price = self.item_prices.get(cand_name, 150.0)
                    raw_acc = self.feature_store.get(f"item:{cand_name}:acceptance_rate")
                    acc_rate = float(raw_acc) if raw_acc is not None else 0.5
                    raw_margin = self.feature_store.get(f"item:{cand_name}:margin")
                    margin = float(raw_margin) if raw_margin is not None else 0.2
                    price_ratio = price / max(cart_total, 1.0)

                    n_score = scores[0, c_idx].item()

                    row = [n_score, gfs, margin, velocity, acc_rate, price_ratio]
                    lgb_features.append(row)
                    
                    # Label: Graded relevance
                    is_target = (c_idx == 0)
                    if is_target and gfs > 0.5:
                        lgb_labels.append(4)
                    elif is_target and gfs <= 0.5:
                        lgb_labels.append(3)
                    elif not is_target and gfs > 0.5:
                        lgb_labels.append(1)
                    else:
                        lgb_labels.append(0)
                    query_group_size += 1
                
                lgb_groups.append(query_group_size)

        print(f"[Engine] Training LightGBM on {len(lgb_features)} True Validations Samples...")
        X = np.array(lgb_features)
        y = np.array(lgb_labels)
        groups = np.array(lgb_groups)
        self.reranker.train(X, y, groups)
        
        print("[Engine] Model Training Orchestration Complete.")

    def calibrate_temperature(self, limit_batches: int = 10) -> None:
        """
        [Fix 3]: Temperature Scaling Calibration
        Optimize a single parameter T using L-BFGS to calibrate the logits.
        """
        print("\n[Engine] Starting Temperature Calibration (Stage 5) via L-BFGS...")
        self.neural_model.eval()
        
        logits_list = []
        labels_list = []
        
        trajectory_ids = self.corpus_df["trajectory_id"].unique()
        self.rng.shuffle(trajectory_ids)
        
        for traj_id in trajectory_ids[:limit_batches]:
            traj_df = self.corpus_df[self.corpus_df["trajectory_id"] == traj_id].sort_values("step_index")
            if len(traj_df) < 2: continue
            
            curr_user_id = traj_df["user_id"].iloc[0]
            past_orders = self.corpus_df[(self.corpus_df["user_id"] == curr_user_id) & (self.corpus_df["trajectory_id"] < traj_id)].sort_values("trajectory_id")

            if len(past_orders) > 0:
                recent_items = past_orders.tail(10)
                hist_names = recent_items["item_name"].tolist()
                hist_qtys = recent_items["quantity"].tolist()
                actual_len = len(hist_names)
                h_lens = torch.tensor([actual_len], dtype=torch.long, device=self.device)
                pad_len = 10 - actual_len
                hist_names = ["<PAD>"] * pad_len + hist_names
                hist_qtys = [0.0] * pad_len + hist_qtys
                h_ids = torch.tensor([self.item_to_idx.get(n, 0) for n in hist_names], dtype=torch.long, device=self.device).unsqueeze(0)
                h_qty = torch.tensor(hist_qtys, dtype=torch.float32, device=self.device).unsqueeze(0)
                h_slm = self._get_slm_batch(hist_names).unsqueeze(0)
            else:
                h_ids = torch.zeros((1, 10), dtype=torch.long, device=self.device)
                h_qty = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
                h_slm = torch.zeros((1, 10, 384), dtype=torch.float32, device=self.device)
                h_lens = torch.ones(1, dtype=torch.long, device=self.device)

            cart_items = []
            cart_total = 0.0
            
            for i in range(len(traj_df) - 1):
                curr_row = traj_df.iloc[i]
                next_row = traj_df.iloc[i + 1]
                cart_items.append({"name": curr_row["item_name"], "category": curr_row["item_category"], "quantity": curr_row["quantity"], "unit_price": curr_row["item_price"]})
                cart_total += curr_row["quantity"] * curr_row["item_price"]

                f_vec, segments = self.online_calculator.compute_feature_vector(
                    user_id=int(curr_row["user_id"]), user_aov_ceiling=float(curr_row["aov_ceiling"]),
                    cart_items=cart_items, cart_total=cart_total, cuisine=str(curr_row["cuisine"]),
                    hour_of_day=int(curr_row["hour_of_day"]), day_of_week=0, is_weekend=bool(curr_row["is_weekend"]),
                    city=str(curr_row["city"]), candidate_item_name=str(next_row["item_name"]), candidate_item_category=str(next_row["item_category"])
                )

                ctx_cyc_h = f_vec[segments["ctx.cyclical_hour"][0]:segments["ctx.cyclical_hour"][1]]
                ctx_cyc_d = f_vec[segments["ctx.cyclical_day"][0]:segments["ctx.cyclical_day"][1]]
                ctx_type = f_vec[segments["ctx.day_type"][0]:segments["ctx.day_type"][1]]
                ctx_wth = f_vec[segments["ctx.weather_proxy"][0]:segments["ctx.weather_proxy"][1]]
                context = np.concatenate([ctx_cyc_h, ctx_cyc_d, ctx_type, ctx_wth])
                context_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
                gap_t = torch.tensor(f_vec[segments["cart.meal_gap_vector"][0]:segments["cart.meal_gap_vector"][1]], dtype=torch.float32, device=self.device).unsqueeze(0)

                target_item_name = next_row["item_name"]
                cands = [target_item_name]
                all_taxonomic_items = list(self.item_to_idx.keys())
                while len(cands) < 8:
                    neg = self.rng.choice(all_taxonomic_items)
                    if neg not in cands: cands.append(neg)
                
                cand_ids = torch.tensor([self.item_to_idx.get(c, 0) for c in cands], dtype=torch.long, device=self.device).unsqueeze(0)
                cand_slm = self._get_slm_batch(cands).unsqueeze(0)
                cart_idx_seq = [self.item_to_idx.get(c["name"], 0) for c in cart_items]
                cart_qty_seq = [c["quantity"] for c in cart_items]
                cart_t = torch.tensor(cart_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                qty_t = torch.tensor(cart_qty_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.zeros(1, len(cart_idx_seq), dtype=torch.bool, device=self.device)
                cart_slm = self._get_slm_batch([c["name"] for c in cart_items]).unsqueeze(0)

                with torch.no_grad():
                    scores = self.neural_model(
                        cart_t, qty_t, cart_slm, mask_t,
                        h_ids, h_qty, h_slm, h_lens,
                        context_t, gap_t, cand_ids, cand_slm
                    )
                logits_list.append(scores)
                # target is always 0 in this construction
                labels_list.append(torch.zeros(1, dtype=torch.long, device=self.device))
                
        if not logits_list:
            print("[Engine] Calibration skipped (insufficient validation data).")
            return
            
        all_logits = torch.cat(logits_list, dim=0).to(self.device)
        all_labels = torch.cat(labels_list, dim=0).to(self.device)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def eval():
            optimizer.zero_grad()
            loss = criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        print(f"[Engine] Temperature calibration finished. T_opt = {self.temperature.item():.4f}")

    # =========================================================================
    # SERVING MODE: Load pre-computed artifacts from disk
    # =========================================================================

    def load_pretrained_artifacts(self, artifact_dir: str) -> None:
        """
        Serving-mode initializer — bypasses all data generation and training.

        Loads the 7 artifacts produced by train_offline.py and wires up every
        component that predict_addon() needs:
          - item_to_idx / idx_to_item / item_prices
          - item_meta (for ColdStartRouter)
          - corpus_df (for feature store re-hydration)
          - SimulatedRedisStore (NightlyOfflineJob + NearRealTimeJob)
          - SLM embedding cache (raw bytes, replayed into feature_store)
          - CSAOHybridModel weights + temperature scalar
          - LightGBMReranker booster

        Args:
            artifact_dir: Path to the directory containing artifacts.
                          Typically 'artifacts/' relative to project root.
        """
        artifact_dir = os.path.abspath(artifact_dir)
        print(f"\n[Engine] Loading pre-trained artifacts from: {artifact_dir}")

        # ── 1. Item mappings & prices ─────────────────────────────────────────
        mappings_path = os.path.join(artifact_dir, "item_mappings.pkl")
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
        self.item_to_idx = mappings["item_to_idx"]
        self.idx_to_item = mappings["idx_to_item"]
        self.item_prices = mappings["item_prices"]
        print(f"   item_mappings loaded  ({len(self.item_to_idx)} items)")

        # ── 2. Item meta (for ColdStartRouter knowledge graph) ────────────────
        item_meta_path = os.path.join(artifact_dir, "item_meta.pkl")
        with open(item_meta_path, "rb") as f:
            self.item_meta = pickle.load(f)
        print(f"   item_meta loaded      ({len(self.item_meta)} entries)")

        # ── 3. Corpus DataFrame (for feature store re-hydration) ──────────────
        corpus_path = os.path.join(artifact_dir, "corpus_df.parquet")
        self.corpus_df = pd.read_parquet(corpus_path)
        print(f"   corpus_df loaded      ({len(self.corpus_df)} rows)")
        # ── 3b. User Database re-hydration ────────────────────────────────────
        user_db_path = os.path.join(artifact_dir, "user_db.pkl")
        if os.path.exists(user_db_path):
            with open(user_db_path, "rb") as f:
                self.user_db = pickle.load(f)
            print(f"   user_db loaded       ({len(self.user_db)} users)")
        else:
            print("   user_db.pkl not found. Falling back to default user dictionary.")

        # ── 4. Feature store re-hydration (NightlyOfflineJob + NearRealTimeJob) ─
        # These are fast in-memory operations (~1-3s total) and are REQUIRED because
        # OnlinePerRequestCalculator reads user RFM triplets and zone-velocities
        # from the SimulatedRedisStore at inference time.
        print("  [Engine] Re-hydrating SimulatedRedisStore (NightlyJob + NRT)...")
        self.feature_store = SimulatedRedisStore()

        nightly_job = NightlyOfflineJob(store=self.feature_store, corpus_df=self.corpus_df)
        nightly_job.run()

        nrt_job = NearRealTimeJob(store=self.feature_store, corpus_df=self.corpus_df)
        nrt_job.run()

        self.online_calculator = OnlinePerRequestCalculator(
            store=self.feature_store, corpus_df=self.corpus_df
        )
        self.item_gen = CandidateItemFeatureGenerator(corpus_df=self.corpus_df)
        print("   feature store hydrated")

        # ── 5. ColdStartRouter re-build ───────────────────────────────────────
        print("  [Engine] Re-building ColdStartRouter from saved corpus...")
        known_items = sorted(list(self.item_to_idx.keys()))
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
        print("   ColdStartRouter ready")

        # ── 6. SLM cache — replay bytes into feature_store ───────────────────
        slm_cache_path = os.path.join(artifact_dir, "slm_cache.pt")
        slm_cache: dict = torch.load(slm_cache_path, map_location="cpu", weights_only=False)
        for key, value in slm_cache.items():
            # key is "slm:<item_name>"; set() prepends "default:" namespace
            self.feature_store.set(key, value)
        print(f"   SLM cache replayed    ({len(slm_cache)} embeddings)")

        # ── 7. Neural model — instantiate architecture then load weights ──────
        num_items = len(self.item_to_idx) + 1  # 1-indexed, 0 = padding
        self.neural_model = CSAOHybridModel(
            num_items=num_items,
            embedding_dim=128,
            slm_dim=384,
            context_dim=11,
            gap_dim=5,
            num_heads=4,
            num_inducing=8,
            num_isab_layers=2,
            gru_layers=1,
            ff_dim=256,
            dropout=0.1,
        ).to(self.device)

        neural_model_path = os.path.join(artifact_dir, "neural_model.pt")
        state_dict = torch.load(neural_model_path, map_location=self.device, weights_only=True)
        self.neural_model.load_state_dict(state_dict)
        self.neural_model.eval()
        print("   neural_model loaded and set to eval()")

        # ── 8. Temperature scalar ─────────────────────────────────────────────
        temperature_path = os.path.join(artifact_dir, "temperature.pt")
        temp_tensor = torch.load(temperature_path, map_location=self.device, weights_only=True)
        self.temperature = nn.Parameter(temp_tensor.to(self.device))
        print(f"   temperature loaded    (T = {self.temperature.item():.4f})")

        # ── 9. LightGBM Reranker ──────────────────────────────────────────────
        lgbm_path = os.path.join(artifact_dir, "lgbm_model.txt")
        self.reranker = LightGBMReranker(k=10)
        self.reranker.model = lgb.Booster(model_file=lgbm_path)
        print("   LightGBM reranker loaded")

        print("[Engine]  All artifacts loaded. Engine is inference-ready.\n")

    # =========================================================================
    # Fix 5: Candidate Pre-Fetching
    # =========================================================================

    def prefetch_candidates(self, restaurant_cuisine: str, cart_names: set) -> List[str]:
        """
        Fix 5: Asynchronously pre-fetch the top-100 candidate item names for a given
        cuisine, minus anything already in the cart.  The result is stored in
        self._prefetch_cache[restaurant_cuisine] and returned directly.

        Call this when the cuisine / city context is locked in (e.g. on cart init)
        so that predict_addon() can skip the expensive dynamic CUISINE_MENUS loop.

        Args:
            restaurant_cuisine: Cuisine key from CUISINE_MENUS.
            cart_names:         Set of normalised item names currently in the cart.

        Returns:
            List of up to 100 candidate item name strings.
        """
        cart_names_norm = {c.lower().strip() for c in cart_names}

        if restaurant_cuisine in CUISINE_MENUS:
            menu = CUISINE_MENUS[restaurant_cuisine]
            candidates = [
                item["name"]
                for cat_items in menu.values()
                for item in cat_items
                if item["name"].lower().strip() not in cart_names_norm
            ]
        else:
            # Fallback: sample top items from the full catalog
            all_known = list(self.item_to_idx.keys())
            candidates = [
                i for i in all_known if i.lower().strip() not in cart_names_norm
            ]

        # Cap at 100, inject Beverages & Desserts for cross-selling coverage
        candidate_set = set(candidates[:100])
        for item in self.item_meta:
            if item.get("category") in ["Beverages", "Desserts"]:
                if item["item_name"].lower().strip() not in cart_names_norm:
                    candidate_set.add(item["item_name"])

        result = list(candidate_set)[:100]
        self._prefetch_cache[restaurant_cuisine] = result
        print(f"[Engine] prefetch_candidates: cached {len(result)} items for '{restaurant_cuisine}'")
        return result

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
        # Fix 5: optional pre-fetched candidate list; if supplied, skips the
        # dynamic CUISINE_MENUS scan to meet the 65 ms SLA.
        prefetched_candidates: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, any]]:
        """
        3. Online Inference Endpoint
        Takes a live request and returns top-K predictions under 65ms latency constraint.
        """
        t0 = time.perf_counter_ns()
        
        cart_names = [c["name"] for c in cart_items]
        # Robust normalization for filtering
        cart_names_norm = {c.lower().strip() for c in cart_names}

        # Fix 5: Skip expensive CUISINE_MENUS scan if a pre-fetched list was provided.
        if prefetched_candidates is not None:
            menu_items = [
                c for c in prefetched_candidates
                if c.lower().strip() not in cart_names_norm
            ]
            print(f"[Inference] predict_addon: Using {len(menu_items)} pre-fetched candidates.")
        elif restaurant_cuisine in CUISINE_MENUS:
            # Dynamic candidate generation (original path)
            menu = CUISINE_MENUS[restaurant_cuisine]
            menu_items = []
            for cat_items in menu.values():
                for item in cat_items:
                    if item["name"].lower().strip() not in cart_names_norm:
                        menu_items.append(item["name"])
        else:
            # If cuisine unknown, grab top 40 items from catalog minus cart
            all_known = list(self.item_to_idx.keys())
            menu_items = [i for i in all_known if i.lower().strip() not in cart_names_norm][:40]

        # Fetch matching user profile
        if user_id not in self.user_db:
            self.user_db[user_id] = {"order_count": 0, "total_spend": 0.0, "mean_aov": 0.0, "past_ordered_items": [], "cuisine_counts": {}}
        profile = self.user_db[user_id]

        # Step A: Cold-Start Routing
        # Fix 4: Hydrate user_orders so UserColdStart can compute Bayesian archetype posteriors.
        # UserObservation requires: mean_aov, cuisine, cart_size, max_quantity
        past_items = profile.get("past_ordered_items", [])
        if past_items:
            dominant_cuisine = (
                max(profile.get("cuisine_counts", {}), key=profile["cuisine_counts"].get)
                if profile.get("cuisine_counts") else restaurant_cuisine
            )
            user_orders = [
                UserObservation(
                    mean_aov=float(profile["mean_aov"]),
                    cuisine=dominant_cuisine,
                    cart_size=len(past_items),
                    max_quantity=int(max((o.get("quantity", 1) for o in past_items), default=1)),
                )
            ]
        else:
            user_orders = []  # Genuinely new user — no observations available

        request = ColdStartRequest(
            user_id=user_id,
            user_order_count=profile["order_count"],
            user_orders=user_orders,  # Fix 4: hydrated with real history
            restaurant_name=restaurant_name,
            restaurant_interaction_count=0,
            restaurant_menu=menu_items,
            candidate_item_name=menu_items[0] if menu_items else "Dummy",
            candidate_item_has_interactions=True,
            cart_main_items=cart_names,
            cuisine=restaurant_cuisine,
            city=city,
        )
        cs_result = self.cold_start_router.route(request)
        
        # Gather final candidates (Combine CS candidates with Menu)
        candidates_set = set(menu_items)
        if cs_result.city_popularity_recs:
            for item, _ in cs_result.city_popularity_recs[:5]:
                candidates_set.add(item)
        if cs_result.kg_seeded_recs:
            for item, _ in cs_result.kg_seeded_recs[:5]:
                candidates_set.add(item)
                
        # [Taxonomy Fix 4]: Taxonomy-Driven Candidate Generation (NO MAX CAP)
        # Score the entire eligible taxonomic pool; do NOT truncate to 20.
        
        # Inject Beverages and Desserts explicitly to guarantee cross-selling entropy
        for item in self.item_meta:
            if item.get("category") in ["Beverages", "Desserts"]:
                candidates_set.add(item["item_name"])
                
        # Mask out any items that are already in the cart (Final Pass)
        candidates = [c for c in candidates_set if c.lower().strip() not in cart_names_norm]
        if not candidates:
            # Fallback just in case they added the entire menu to their cart
            candidates = list(candidates_set)
        
        print(f"[Inference] predict_addon: Modeling {len(candidates)} candidates. Cart has: {cart_names}")

        # --- Batch Neural Scoring ---
        # 1. Compute context and gap once (they are independent of candidates)
        cart_total = sum(c.get("unit_price", 100) * c.get("quantity", 1) for c in cart_items)
        user_aov = profile["mean_aov"] if profile["order_count"] > 0 else 500.0
        
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
        
        # Candidate tensors: process all at once
        cand_idx_seq = [self.item_to_idx.get(cand, 0) for cand in candidates]
        cand_t = torch.tensor(cand_idx_seq, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, K)
        cand_slm = self._get_slm_batch(candidates).unsqueeze(0)
        
        # Additional tensors for SLM
        cart_slm = self._get_slm_batch(cart_names).unsqueeze(0)
        
        # [Task 2]: Dynamic Neural History (GRU4Rec Hydration)
        hist_ids = torch.zeros((1, 10), dtype=torch.long, device=self.device)
        hist_qty = torch.zeros((1, 10), dtype=torch.float32, device=self.device)
        hist_slm = torch.zeros((1, 10, 384), dtype=torch.float32, device=self.device)
        hist_lengths = torch.ones(1, dtype=torch.long, device=self.device)
        gru_status = "Bypassed (Zero-Tensors)"
        
        if profile["order_count"] >= 3 and len(profile["past_ordered_items"]) > 0:
            past_items = profile["past_ordered_items"][-10:] # Take last 10
            gru_status = f"Hydrated ({len(past_items)} items)"
            seq_len = len(past_items)
            hist_lengths[0] = seq_len
            
            p_ids = [self.item_to_idx.get(item["name"], 0) for item in past_items]
            p_qty = [item["quantity"] for item in past_items]
            p_names = [item["name"] for item in past_items]
            
            p_ids_t = torch.tensor(p_ids, dtype=torch.long, device=self.device)
            p_qty_t = torch.tensor(p_qty, dtype=torch.float32, device=self.device)
            p_slm_t = self._get_slm_batch(p_names)
            
            # Place at the end (right-aligned or left-aligned? Usually GRU4Rec takes left-aligned with lengths)
            hist_ids[0, :seq_len] = p_ids_t
            hist_qty[0, :seq_len] = p_qty_t
            hist_slm[0, :seq_len, :] = p_slm_t

        self.neural_model.eval()
        with torch.no_grad():
            scores = self.neural_model(
                cart_t, qty_t, cart_slm, mask_t,
                hist_ids, hist_qty, hist_slm, hist_lengths,
                context_t, gap_t, cand_t, cand_slm
            )  # (1, K)

            # Apply Temperature scaling to logits
            raw_logits = scores / self.temperature
            # Fix 1: Softmax over candidate dimension (dim=1) so probabilities sum to 1,
            # which is required for Temperature Scaling calibration to be meaningful.
            probs = torch.softmax(raw_logits, dim=1)[0].cpu().numpy()
        
        # Fix 2: Entropy-based popularity fallback.
        # If the model has very low confidence (high entropy), bypass LightGBM and
        # return city-level popularity rankings instead.
        H_max = 2.5  # Maximum acceptable entropy (~uniform over ~12 items)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        print(f"[Inference] Softmax entropy: {entropy:.4f} (H_max={H_max})")

        if entropy > H_max:
            print("[Inference] High entropy detected — triggering popularity fallback.")
            popularity_recs = cs_result.city_popularity_recs or []
            fallback_items = [
                (item, score) for item, score in popularity_recs
                if item.lower().strip() not in cart_names_norm
            ][:8]
            elapsed_ms_fb = (time.perf_counter_ns() - t0) / 1e6
            debug_payload_fb = {
                "latency": elapsed_ms_fb,
                "entropy_fallback": True,   # Fix 2 indicator
                "entropy_value": float(entropy),
                "cold_start_path": "Entropy Fallback → City Popularity",
                "gru4rec_status": gru_status,
                "feature_state": {
                    "meal_gap_vector": gap.tolist() if isinstance(gap, np.ndarray) else list(gap),
                    "cart_total": cart_total,
                    "user_aov": user_aov,
                    "price_anchor_ratio": (cart_total / max(user_aov, 1.0))
                },
                "scoring_breakdown": []
            }
            return fallback_items, debug_payload_fb

        # --- Form Rerank Candidates ---
        # [Fix 4A]: Wallet Cap Filter
        margin_allowance = 50.0 # Define a reasonable buffer limit
        valid_candidates = []
        
        for i, cand in enumerate(candidates):
            gfs_features = self.item_gen.compute_candidate_features(
                cand, "Unknown", gap, hour_of_day, city
            )

            # Fix 3: Fetch real price from global taxonomy catalog (was _mock_item_feature)
            price = self.item_prices.get(cand, 150.0)

            # Fix 3: Fetch acceptance_rate and margin from feature store; default if absent
            raw_acc = self.feature_store.get(f"item:{cand}:acceptance_rate")
            acceptance_rate = float(raw_acc) if raw_acc is not None else 0.5

            raw_margin = self.feature_store.get(f"item:{cand}:margin")
            item_margin = float(raw_margin) if raw_margin is not None else 0.2

            # Dynamic Wallet Cap: Ensure we don't completely lock out users making large orders.
            effective_wallet = max(user_aov, cart_total)
            margin_allowance = max(150.0, effective_wallet * 0.3)

            if cart_total + price > effective_wallet + margin_allowance:
                continue  # Discard VERY expensive items outside allowed bounds

            neural_prob = probs[i].item()
            gfs = float(gfs_features.get("gap_fill_score", [0.0])[0])
            velocity = float(gfs_features.get("zone_velocity", [0.0])[0])

            rc = RerankCandidate(
                item_name=cand,
                neural_score=neural_prob,
                gap_fill_score=gfs,
                item_margin=item_margin,        # Fix 3: real margin from feature store
                zone_velocity=velocity,
                acceptance_rate=acceptance_rate, # Fix 3: real acc_rate from feature store
                price_ratio=price / max(cart_total, 1.0)
            )
            valid_candidates.append(rc)
                
        # Step D: LightGBM Re-Ranking (base score generation)
        if not valid_candidates:
            print("[Inference] predict_addon completed: 0 candidates passed Wallet Cap.")
            t1 = time.perf_counter_ns()
            elapsed_ms = (t1 - t0) / 1e6
            cold_start_path = []
            if cs_result.user_cold_start_triggered: cold_start_path.append("User Tier 3")
            if cs_result.restaurant_cold_start_triggered: cold_start_path.append("Restaurant Tier 2")
            if cs_result.item_cold_start_triggered: cold_start_path.append("Item Tier 1")
            
            debug_payload = {
                "latency": elapsed_ms,
                "cold_start_path": " | ".join(cold_start_path) if cold_start_path else "None (Warm Flow)",
                "gru4rec_status": gru_status,
                "feature_state": {
                    "meal_gap_vector": gap.tolist() if isinstance(gap, np.ndarray) else list(gap),
                    "cart_total": cart_total,
                    "user_aov": user_aov,
                    "price_anchor_ratio": (cart_total / max(user_aov, 1.0))
                },
                "scoring_breakdown": []
            }
            return [], debug_payload

        ranked_base = self.reranker.rerank(valid_candidates)
        
        # [Fix 4B]: Maximal Marginal Relevance (MMR)
        # i* = argmax[ lambda * P(i) * (1 + alpha * GFS(i)) - (1 - lambda) * max_j_in_R Sim(i, j) ]
        lambda_param = 0.7
        alpha_param = 0.2
        k_target = min(8, len(ranked_base))
        
        # Prepare data for MMR math
        cand_names = [item for item, _ in ranked_base]
        cand_probs = {item: score for item, score in ranked_base}
        cand_gfs = {c.item_name: c.gap_fill_score for c in valid_candidates}
        cand_lgbm_scores = {item: score for item, score in ranked_base}
        
        # Pre-fetch all 384-d SLM vectors for candidate similarity computations
        cand_slm_pool = self._get_slm_batch(cand_names) # (Num_Cands, 384)
        cand_slm_dict = {name: cand_slm_pool[idx] for idx, name in enumerate(cand_names)}
        
        selected_items = []
        selected_scores = []
        unselected_items = list(cand_names)
        
        for _ in range(k_target):
            best_item = None
            best_mmr_score = -float('inf')
            
            for item in unselected_items:
                # Part 1: Relevance Trade-off
                p_i = cand_probs[item]
                gfs_i = cand_gfs[item]
                lgbm_score_i = cand_lgbm_scores[item]
                
                # Correctly merge the calibrated LightGBM score as the primary relevance driver
                relevance = lambda_param * lgbm_score_i * (1.0 + alpha_param * gfs_i)
                
                # Part 2: Diversity Penalty
                max_sim = 0.0
                if selected_items:
                    vec_i = cand_slm_dict[item]
                    # Compute cosine similarity against all selected items
                    sims = [torch.nn.functional.cosine_similarity(vec_i.unsqueeze(0), cand_slm_dict[j].unsqueeze(0)).item() for j in selected_items]
                    max_sim = max(sims)
                    
                penalty = (1.0 - lambda_param) * max_sim
                
                # Final MMR Score for this iteration
                mmr_score = relevance - penalty
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_item = item
            
            if best_item is not None:
                selected_items.append(best_item)
                selected_scores.append(best_mmr_score)
                unselected_items.remove(best_item)
        
        if selected_scores:
            min_mmr = min(selected_scores)
            max_mmr = max(selected_scores)
            if max_mmr > min_mmr:
                selected_scores = [(s - min_mmr) / (max_mmr - min_mmr) for s in selected_scores]
            else:
                selected_scores = [1.0 for _ in selected_scores]
        
        final_ranked = list(zip(selected_items, selected_scores))
        
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        
        # --- Build Verbose Debug Payload ---
        cold_start_path = []
        if cs_result.user_cold_start_triggered: cold_start_path.append("User Tier 3")
        if cs_result.restaurant_cold_start_triggered: cold_start_path.append("Restaurant Tier 2")
        if cs_result.item_cold_start_triggered: cold_start_path.append("Item Tier 1")
        
        scoring_breakdown = []
        for i, item_name in enumerate(selected_items):
            # Fetch original features used
            c_idx = candidates.index(item_name)
            neural_prob = probs[c_idx].item()
            gfs = cand_gfs[item_name]
            # Use velocity from the original valid candidates list ideally, but we can recompute or leave it out.
            velocity = 0.0 # Just for structure if needed, or omit if not explicitly required by user. User asked for: "[Item Name | Neural Prob | Gap Fill | Velocity | Final Scaled Score]"
            # Hmm, let's grab velocity from valid_candidates
            for vc in valid_candidates:
                if vc.item_name == item_name:
                    velocity = vc.zone_velocity
                    break

            scoring_breakdown.append({
                "Rank": i + 1,
                "Item Name": item_name,
                "Neural Prob": float(neural_prob),
                "Gap Fill": float(gfs),
                "Velocity": float(velocity),
                "Final Scaled Score": float(selected_scores[i])
            })
            
        debug_payload = {
            "latency": elapsed_ms,
            "cold_start_path": " | ".join(cold_start_path) if cold_start_path else "None (Warm Flow)",
            "gru4rec_status": gru_status,
            "feature_state": {
                "meal_gap_vector": gap.tolist() if isinstance(gap, np.ndarray) else list(gap),
                "cart_total": cart_total,
                "user_aov": user_aov,
                "price_anchor_ratio": (cart_total / max(user_aov, 1.0))
            },
            "scoring_breakdown": scoring_breakdown
        }
        
        print(f"[Inference] predict_addon completed in {elapsed_ms:.2f} ms")
        return final_ranked, debug_payload

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
