#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class SLMEmbedder:
    """
    Offline model for extracting 384-dimensional native text embeddings
    using all-MiniLM-L6-v2, and performing Tier 1 Cold-Start Transfer.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers.")
            
        print("[SLM] Loading all-MiniLM-L6-v2...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        self.model.eval()

    def generate_embeddings(self, items_df: pd.DataFrame) -> np.ndarray:
        """
        Generate 384-d L2-normalized embeddings for a dataframe of items.
        
        Args:
            items_df: DataFrame with at least 'item_name', 'description', 'cuisine'.
            
        Returns:
            (N, 384) numpy array.
        """
        texts = []
        for _, row in items_df.iterrows():
            text = f"{row['item_name']}. {row.get('description', '')}. Cuisine: {row.get('cuisine', '')}."
            texts.append(text)
            
        print(f"[SLM] Generating {len(texts)} embeddings (384-d)...")
        # all-MiniLM-L6-v2 generates 384-dimensional vectors
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def cold_start_transfer(
        self,
        new_item_slm_emb: np.ndarray,      # (384,)
        known_items_slm_emb: np.ndarray,   # (N, 384)
        known_items_behav_emb: np.ndarray, # (N, 128)
        k: int = 5
    ) -> np.ndarray:
        """
        Tier 1 Cold-Start: E_{y_new} = sum(Sim * e_j) / sum(Sim)
        """
        if len(new_item_slm_emb.shape) == 1:
            new_item_slm_emb = new_item_slm_emb.reshape(1, -1)
            
        # 1. Cosine similarity (dot product for normalized vectors)
        sim_scores = np.dot(known_items_slm_emb, new_item_slm_emb.T).flatten()
        
        # 2. Top-K
        top_k_indices = np.argsort(sim_scores)[::-1][:k]
        top_k_sims = sim_scores[top_k_indices]
        top_k_behav = known_items_behav_emb[top_k_indices]
        
        top_k_sims = np.clip(top_k_sims, a_min=1e-6, a_max=None)
        
        # 3. Weighted sum
        weights = top_k_sims / np.sum(top_k_sims)
        transferred_emb = np.sum(weights[:, np.newaxis] * top_k_behav, axis=0)
        
        # L2-Normalize
        norm = np.linalg.norm(transferred_emb)
        if norm > 0:
            transferred_emb = transferred_emb / norm
        return transferred_emb
