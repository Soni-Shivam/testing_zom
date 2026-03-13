"""
Cold-Start Configuration
==========================
Hyperparameters and thresholds for the three-tier cold-start system.
"""

from dataclasses import dataclass


@dataclass
class ColdStartConfig:
    """
    Hyperparameters for the cold-start decision tree.

    Attributes:
        tau_u: User order threshold. Users with fewer than tau_u
               past orders trigger User Cold Start (Tier 3).
        tau_r: Restaurant interaction threshold. Restaurants with
               fewer than tau_r total interactions trigger
               Restaurant Cold Start (Tier 2).
        nn_k:  Number of nearest neighbors for item embedding
               transfer (Tier 1).
        embedding_dim: Dimensionality of item embeddings.
        kg_top_k: Number of add-on triples returned per main item
                  from the knowledge graph.
    """
    tau_u: int = 5
    tau_r: int = 20
    nn_k: int = 5
    embedding_dim: int = 128
    kg_top_k: int = 5


# Default config singleton
DEFAULT_CONFIG = ColdStartConfig()
