#!/usr/bin/env python3
"""
CSAO NLP Text Embedding Pipeline
==================================
Demonstrates embedding new restaurant items using all-MiniLM-L6-v2,
applying Tier 1 nearest-neighbor cold-start embedding transfer,
and fusing 384-d semantic vectors with 128-d behavioral vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Base CSAO Blocks
from csao.nn.embeddings import QuantityAwareEmbedding
from csao.nn.set_transformer import SetTransformerEncoder
from csao.nn.gru4rec import GRU4Rec
from csao.nn.fusion import CrossAttentionFusion


# ============================================================================
# PART 1: Offline Embedding & Cold-Start (Stage 3 Update)
# ============================================================================

class SLM_Embedder:
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
        
        Args:
            new_item_slm_emb: (384,) SLM vector of the new item.
            known_items_slm_emb: (N, 384) SLM vectors of known items.
            known_items_behav_emb: (N, 128) Collaborative behavioral embeddings.
            k: number of nearest neighbors to consider.
            
        Returns:
            (128,) Transferred behavioral embedding.
        """
        if len(new_item_slm_emb.shape) == 1:
            new_item_slm_emb = new_item_slm_emb.reshape(1, -1)
            
        # 1. Cosine similarity in 384-d space (since vectors are L2 normalized, dot product = cos sim)
        sim_scores = np.dot(known_items_slm_emb, new_item_slm_emb.T).flatten()  # (N,)
        
        # 2. Find Top-K nearest neighbors
        top_k_indices = np.argsort(sim_scores)[::-1][:k]
        top_k_sims = sim_scores[top_k_indices]
        top_k_behav = known_items_behav_emb[top_k_indices]  # (K, 128)
        
        # Prevent zero-division if similarities are very small or negative
        top_k_sims = np.clip(top_k_sims, a_min=1e-6, a_max=None)
        
        # 3. Weighted sum transfer
        weights = top_k_sims / np.sum(top_k_sims)
        transferred_emb = np.sum(weights[:, np.newaxis] * top_k_behav, axis=0)  # (128,)
        
        # L2-Normalize
        transferred_emb = transferred_emb / np.linalg.norm(transferred_emb)
        return transferred_emb


# ============================================================================
# PART 2: Asymmetric PyTorch Architecture (Stage 4 Update)
# ============================================================================

class CSAOHybridModel(nn.Module):
    """
    Hybrid neural model supporting dual-inputs: 
      1. Trainable Behavioral Token IDs -> 128-d
      2. Pre-computed SLM Text Vectors -> 384-d
      
    Fuses them asymmetrically to maintain 128-d network speed.
    """

    def __init__(
        self,
        num_items: int = 200,
        behav_dim: int = 128,
        slm_dim: int = 384,
        context_dim: int = 11,
        gap_dim: int = 5,
        num_heads: int = 4,
        num_inducing: int = 8,
        num_isab_layers: int = 2,
        gru_layers: int = 1,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.behav_dim = behav_dim
        self.slm_dim = slm_dim

        # Behavioral branch
        self.item_embed = QuantityAwareEmbedding(
            num_items=num_items,
            embedding_dim=behav_dim,
            padding_idx=0,
        )

        # SLM Projection Layer (Asymmetric down-projection)
        self.slm_projector = nn.Sequential(
            nn.Linear(slm_dim, behav_dim),
            nn.GELU(),
            nn.LayerNorm(behav_dim),
        )

        # Backbone
        self.cart_encoder = SetTransformerEncoder(
            dim=behav_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_isab_layers=num_isab_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.user_encoder = GRU4Rec(
            embedding_dim=behav_dim,
            hidden_dim=behav_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )
        self.fusion = CrossAttentionFusion(
            dim=behav_dim,
            context_dim=context_dim,
            gap_dim=gap_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Score Head
        self.score_head = nn.Sequential(
            nn.Linear(behav_dim, behav_dim),
            nn.GELU(),
            nn.Linear(behav_dim, behav_dim),
        )

    def _fuse_item_vectors(
        self, 
        item_ids: torch.Tensor, 
        quantities: torch.Tensor, 
        slm_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantity-Aware Fusion:
          e_fused = (e_i + f_q * v_q) + projector(e_slm_384)
        """
        # Behavioral branch (B, S, 128)
        e_behav = self.item_embed(item_ids, quantities)
        
        # SLM branch down-projection (B, S, 128)
        e_slm_proj = self.slm_projector(slm_vectors)
        
        # Asymmetric Addition
        return e_behav + e_slm_proj

    def forward(
        self,
        # Cart branch
        cart_item_ids: torch.Tensor,       # (B, N) format
        cart_quantities: torch.Tensor,     # (B, N)
        cart_slm_emb: torch.Tensor,        # (B, N, 384) native SLM vectors
        cart_mask: torch.Tensor,           # (B, N)
        
        # History branch
        history_item_ids: torch.Tensor,    # (B, T)
        history_quantities: torch.Tensor,  # (B, T)
        history_slm_emb: torch.Tensor,     # (B, T, 384) native SLM vectors
        history_lengths: torch.Tensor,     # (B,)
        
        # Context
        context: torch.Tensor,             # (B, context_dim)
        gap: torch.Tensor,                 # (B, gap_dim)
        
        # Candidates
        candidate_ids: torch.Tensor,       # (B, K)
        candidate_slm_emb: torch.Tensor,   # (B, K, 384)
    ) -> torch.Tensor:
        """Full forward pass."""
        
        # 1. Duel-Embedding Fusion
        e_cart_items = self._fuse_item_vectors(cart_item_ids, cart_quantities, cart_slm_emb)
        e_hist_items = self._fuse_item_vectors(history_item_ids, history_quantities, history_slm_emb)
        
        cand_qty = torch.ones_like(candidate_ids, dtype=torch.float32)
        e_cand_items = self._fuse_item_vectors(candidate_ids, cand_qty, candidate_slm_emb)

        # 2. Shared Backbone (128-d latent space)
        e_cart = self.cart_encoder(e_cart_items, key_padding_mask=cart_mask)  # (B, 128)
        e_user = self.user_encoder(e_hist_items, history_lengths)             # (B, 128)
        e_fused = self.fusion(e_cart, e_user, context, gap)                   # (B, 128)

        # 3. Final Scoring
        e_scored = self.score_head(e_fused)  # (B, 128)
        scores = torch.bmm(e_cand_items, e_scored.unsqueeze(-1)).squeeze(-1)  # (B, K)

        return scores


# ============================================================================
# PART 3: Mock Execution Block
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  CSAO SLM ARCHITECTURE INTEGRATION TEST")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n[SLM] Instantiating Stage 3 Offline Embedder...")
    try:
        embedder = SLM_Embedder(device=device)
        
        # Mock Menu DataFrame
        mock_df = pd.DataFrame([
            {"item_name": "Paneer Tikka Masala", "description": "Cottage cheese in rich tomato gravy", "cuisine": "North Indian"},
            {"item_name": "Garlic Naan", "description": "Flatbread with butter and garlic", "cuisine": "North Indian"},
            {"item_name": "Mint Chutney", "description": "Fresh mint dip", "cuisine": "North Indian"}
        ])
        
        slm_vectors = embedder.generate_embeddings(mock_df)
        print(f"  → Generated Native Vector Shape: {slm_vectors.shape} [L2 norm: {np.linalg.norm(slm_vectors[0]):.4f}]")
        
        print("\n[Cold Start] Demonstrating Tier 1 Transfer...")
        known_behav = np.random.randn(3, 128).astype(np.float32)
        transferred = embedder.cold_start_transfer(slm_vectors[0], slm_vectors, known_behav, k=2)
        print(f"  → Resulting Transferred Embedding: {transferred.shape} [L2 norm: {np.linalg.norm(transferred):.4f}]")
        
    except Exception as e:
        print(f"  → SLM Failed to load (Missing sentence-transformers or network?): {e}")

    print("\n" + "=" * 70)
    print("\n[PyTorch] Instantiating Asymmetric CSAOHybridModel...")
    model = CSAOHybridModel(
        num_items=100,
        behav_dim=128,
        slm_dim=384,
        context_dim=11,
        gap_dim=5
    ).to(device)
    
    B, N, T, K = 2, 4, 3, 5  # Batch=2, Cart=4, Hist=3, Cands=5
    
    # ── Mock Neural Tensors ──
    # Cart
    c_ids = torch.randint(1, 100, (B, N), device=device)
    c_qty = torch.randint(1, 4, (B, N), device=device).float()
    c_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    c_slm = torch.randn(B, N, 384, device=device) # 384-d!
    
    # History
    h_ids = torch.randint(1, 100, (B, T), device=device)
    h_qty = torch.ones(B, T, device=device)
    h_lens = torch.tensor([3, 2], device=device)
    h_slm = torch.randn(B, T, 384, device=device) # 384-d!
    
    # Context
    ctx = torch.randn(B, 11, device=device)
    gap = torch.randn(B, 5, device=device)
    
    # Candidates
    cand_ids = torch.randint(1, 100, (B, K), device=device)
    cand_slm = torch.randn(B, K, 384, device=device) # 384-d!

    print(f"  → Cart Tensor Shape:      {tuple(c_ids.shape)}")
    print(f"  → Cart SLM Text Features: {tuple(c_slm.shape)}")
    print(f"  → Target Vocabulary Size: K={K} candidates")
    
    print("\n[PyTorch] Forward Pass...")
    target_shape = (B, K)
    
    model.eval()
    with torch.no_grad():
        scores = model(
            c_ids, c_qty, c_slm, c_mask,
            h_ids, h_qty, h_slm, h_lens,
            ctx, gap,
            cand_ids, cand_slm
        )
    
    print(f"  → Expected Output Shape: {target_shape}")
    print(f"  → Actual Output Shape:   {tuple(scores.shape)}")
    
    if scores.shape == target_shape:
        print("\n  [✓] Forward Pass Successfully completed matching Target Vocabulary without Shape Mismatches! ✅")
    else:
        print("\n  [✗] Neural Projection Failed. Shape mismatch.")
    print("=" * 70)
