"""
Synthesis Pipeline Orchestrator
=================================
Chains all four generation levels into a single pipeline that
produces N cart trajectories as a pandas DataFrame.

Pipeline flow per trajectory:
  Level 1: UserGenerator → User
  Level 2: SessionGenerator → Session
  Level 3: RestaurantGenerator → (Cuisine, Restaurant)
  Level 4: CartAssembler → CartTrajectory
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from csao.generators.user_generator import UserGenerator
from csao.generators.session_generator import SessionGenerator
from csao.generators.restaurant_generator import RestaurantGenerator
from csao.generators.cart_assembler import CartAssembler
from csao.models.schema import CartTrajectory


class SynthesisPipeline:
    """
    End-to-end hierarchical generative data synthesis pipeline.

    Orchestrates four generation levels to produce a corpus of
    cart trajectories with full-sequence dependencies.
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize all generators with shared RNG lineage
        self.user_gen = UserGenerator(rng=np.random.default_rng(seed))
        self.session_gen = SessionGenerator(rng=np.random.default_rng(seed + 1))
        self.restaurant_gen = RestaurantGenerator(rng=np.random.default_rng(seed + 2))
        self.cart_assembler = CartAssembler(rng=np.random.default_rng(seed + 3))

    def generate(
        self,
        n_trajectories: int = 10000,
        n_users: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate n_trajectories cart trajectories.

        Args:
            n_trajectories: Number of trajectories to generate.
            n_users: Number of unique users. Defaults to n_trajectories // 5
                     (each user has ~5 sessions on average).

        Returns:
            DataFrame with one row per cart item, grouped by trajectory_id.
        """
        if n_users is None:
            n_users = max(100, n_trajectories // 5)

        print(f"[Pipeline] Generating {n_users} users...")
        users = self.user_gen.generate_users(n_users)

        print(f"[Pipeline] Generating {n_trajectories} cart trajectories...")
        trajectories: List[CartTrajectory] = []

        for tid in range(n_trajectories):
            if (tid + 1) % 2000 == 0:
                print(f"  ... {tid + 1}/{n_trajectories} trajectories generated")

            # --- Level 1: Pick a user (round-robin with some randomness) ---
            user = users[self.rng.integers(0, len(users))]

            # --- Level 2: Generate session ---
            session = self.session_gen.generate_session(user)

            # --- Level 3: Select restaurant and cuisine ---
            cuisine, restaurant_name = self.restaurant_gen.select_restaurant(user)

            # --- Level 4: Assemble cart ---
            trajectory = self.cart_assembler.assemble_cart(
                user=user,
                session=session,
                cuisine=cuisine,
                restaurant_name=restaurant_name,
                trajectory_id=tid,
            )

            trajectories.append(trajectory)

        print(f"[Pipeline] All {n_trajectories} trajectories generated.")

        # Convert to DataFrame
        return self._trajectories_to_dataframe(trajectories), trajectories

    @staticmethod
    def _trajectories_to_dataframe(
        trajectories: List[CartTrajectory],
    ) -> pd.DataFrame:
        """
        Flatten list of CartTrajectory objects into a pandas DataFrame.

        Output schema:
        - trajectory_id, user_id, archetype, city
        - session_id, intent, hour_of_day, is_peak_hour, is_weekend
        - restaurant, cuisine
        - step_index, item_name, item_category, item_price, quantity
        - slot_filled, total_price, template_filled
        """
        rows = []
        for traj in trajectories:
            for ci in traj.cart_items:
                rows.append({
                    "trajectory_id":   traj.trajectory_id,
                    "user_id":         traj.user.user_id,
                    "archetype":       traj.user.archetype,
                    "city":            traj.city,
                    "session_id":      traj.session.session_id,
                    "intent":          traj.session.intent,
                    "hour_of_day":     traj.session.hour_of_day,
                    "is_peak_hour":    traj.session.is_peak_hour,
                    "is_weekend":      traj.session.is_weekend,
                    "restaurant":      traj.restaurant_name,
                    "cuisine":         traj.cuisine,
                    "step_index":      ci.step_index,
                    "item_name":       ci.item.name,
                    "item_category":   ci.item.category,
                    "item_price":      ci.item.price,
                    "quantity":        ci.quantity,
                    "slot_filled":     ci.slot_filled,
                    "total_price":     traj.total_price,
                    "template_filled": traj.template_filled,
                    "aov_ceiling":     traj.user.aov_ceiling,
                    "min_quantity":    traj.user.min_quantity,
                })

        df = pd.DataFrame(rows)
        return df
