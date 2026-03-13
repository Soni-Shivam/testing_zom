"""
Data Models for CSAO Hierarchical Generative Data Synthesis
============================================================
Dataclass models representing core entities in the synthesis pipeline:
MenuItem, User, Session, CartItem, and CartTrajectory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MenuItem:
    """A single item from a cuisine menu."""
    name: str
    category: str        # 'main', 'side', 'beverage', 'dessert'
    price: float
    cuisine: str


@dataclass
class User:
    """A synthetic user with archetype-driven attributes."""
    user_id: int
    archetype: str       # Budget, Premium, Occasional, FamilyOrder
    city: str
    aov_ceiling: float   # Maximum Average Order Value for this user
    cuisine_affinity: Dict[str, float]   # cuisine → affinity weight
    order_frequency: float               # orders per week
    session_length_mean: float           # expected items per session
    min_quantity: int = 1                # min item quantity (>1 for FamilyOrder)


@dataclass
class Session:
    """A single ordering session conditioned on user and time."""
    session_id: int
    user_id: int
    intent: str          # Solo Lunch, Family Dinner, etc.
    hour_of_day: int     # 0-23
    is_peak_hour: bool
    is_weekend: bool = False


@dataclass
class CartItem:
    """A single item added to the cart during sequential assembly."""
    item: MenuItem
    quantity: int
    step_index: int      # position in the sequential assembly
    slot_filled: str     # which template slot this fills


@dataclass
class CartTrajectory:
    """A complete cart trajectory encoding the full generation path."""
    trajectory_id: int
    session: Session
    user: User
    restaurant_name: str
    cuisine: str
    city: str
    cart_items: List[CartItem] = field(default_factory=list)
    total_price: float = 0.0
    template_filled: bool = False
    template_slots_required: Dict[str, int] = field(default_factory=dict)
    template_slots_filled: Dict[str, int] = field(default_factory=dict)

    def compute_total(self) -> float:
        """Recalculate total price from cart items."""
        self.total_price = sum(
            ci.item.price * ci.quantity for ci in self.cart_items
        )
        return self.total_price

    def check_template_complete(self) -> bool:
        """Check if all template slots are filled."""
        self.template_filled = all(
            self.template_slots_filled.get(cat, 0) >= req
            for cat, req in self.template_slots_required.items()
        )
        return self.template_filled

    @property
    def num_items(self) -> int:
        """Number of distinct items in the cart."""
        return len(self.cart_items)

    @property
    def item_names(self) -> List[str]:
        """List of item names in the cart (order preserved)."""
        return [ci.item.name for ci in self.cart_items]
