"""
Level 4: Cart Assembly via Meal Template Completion
=====================================================
Assembles cart items sequentially by progressively filling meal
template slots. Uses a meal gap vector g_{k-1} to drive the
probability of the next added item.

The assembly implements full-sequence dependencies: P(item_k)
depends on the complete set of previous items Φ_{k-1} through:
- Meal gap vector (which slots remain unfilled)
- Running cart total (price anchoring)
- Geographic co-occurrence boosts
- Peak-hour urgency penalties
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

from csao.config.taxonomies import (
    CUISINE_MENUS, MEAL_TEMPLATES, SESSION_INTENTS,
)
from csao.models.schema import (
    User, Session, MenuItem, CartItem, CartTrajectory,
)
from csao.modifiers.realism import RealismModifiers


class CartAssembler:
    """
    Assembles cart items sequentially using meal template completion.

    At each step k, the assembler:
    1. Computes the meal gap vector g_{k-1} (unfilled slots).
    2. Samples the next slot category proportional to remaining gaps.
    3. Builds candidate item list for that category.
    4. Applies all realism modifiers to candidate probabilities.
    5. Samples the item, assigns quantity, and updates the gap vector.

    Terminates when the template is filled, the session length limit
    is reached, or the price ceiling is hit.
    """

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng(42)
        self.modifiers = RealismModifiers(rng=self.rng)

    def assemble_cart(
        self,
        user: User,
        session: Session,
        cuisine: str,
        restaurant_name: str,
        trajectory_id: int,
    ) -> CartTrajectory:
        """
        Assemble a complete cart trajectory for a single session.

        Args:
            user: The user placing the order.
            session: The session context (intent, time, etc.).
            cuisine: Selected cuisine type.
            restaurant_name: Name of the selected restaurant.
            trajectory_id: Unique ID for this trajectory.

        Returns:
            Populated CartTrajectory with sequential item additions.
        """
        # --- Initialize meal template and gap vector ---
        base_template = MEAL_TEMPLATES.get(
            cuisine, {"main": 1, "side": 1, "beverage": 1}
        )
        template = self._adjust_template_for_intent(
            base_template, session.intent, user.archetype
        )

        # Gap vector: tracks remaining slots per category
        gap_vector = deepcopy(template)
        filled_slots = {cat: 0 for cat in template}

        # Get the cuisine menu
        menu = CUISINE_MENUS.get(cuisine, {})
        if not menu:
            menu = CUISINE_MENUS.get("Continental", {})

        # Compute max items for this session (must be >= template size)
        intent_config = SESSION_INTENTS.get(session.intent, {})
        template_size = sum(template.values())
        max_items = self._compute_max_items(
            user, session, intent_config, template_size
        )

        # Initialize trajectory
        trajectory = CartTrajectory(
            trajectory_id=trajectory_id,
            session=session,
            user=user,
            restaurant_name=restaurant_name,
            cuisine=cuisine,
            city=user.city,
            template_slots_required=deepcopy(template),
            template_slots_filled=filled_slots,
        )

        running_total = 0.0
        step = 0
        cart_item_names = []

        # =================================================================
        # SEQUENTIAL ASSEMBLY LOOP
        # Each iteration depends on ALL previous items (full-sequence dep.)
        # =================================================================
        while step < max_items:
            # --- Stochastic early-termination ---
            # Models users who make quick 1-2 item orders.
            # Probability decreases with step count, and is higher
            # for Budget/Late-Night scenarios.
            if step >= 1 and step <= 2:
                is_quick_scenario = (
                    user.archetype == "Budget"
                    or session.intent == "Late-Night Snack"
                    or session.intent == "Solo Lunch"
                )
                if is_quick_scenario:
                    # step 1: ~5% chance of stopping (1-item cart)
                    # step 2: ~12% chance of stopping (2-item cart)
                    early_exit_prob = 0.05 if step == 1 else 0.12
                    if self.rng.random() < early_exit_prob:
                        break
            # --- Check if template is fully filled ---
            remaining_gaps = {
                cat: gap_vector[cat] for cat in gap_vector if gap_vector[cat] > 0
            }

            if not remaining_gaps:
                # Template filled. Small chance of an extra add-on
                # (dessert or extra beverage), reduced during peak hours.
                extra_prob = 0.25
                if session.is_peak_hour:
                    extra_prob *= self.rng.uniform(0.80, 0.85)

                if self.rng.random() > extra_prob or step >= max_items:
                    break

                # Try to add a dessert or extra beverage
                remaining_gaps = {}
                if "dessert" in menu and menu["dessert"]:
                    remaining_gaps["dessert"] = 1
                elif "beverage" in menu and menu["beverage"]:
                    remaining_gaps["beverage"] = 1
                else:
                    break

            # --- Step 1: Sample next slot category from gap vector ---
            category = self._sample_category_from_gaps(remaining_gaps)

            # --- Step 2: Get candidate items for this category ---
            candidates = menu.get(category, [])
            if not candidates:
                # If no items in this category, skip and reduce gap
                gap_vector[category] = 0
                continue

            # Build MenuItem objects
            candidate_items = [
                MenuItem(
                    name=item["name"],
                    category=category,
                    price=item["price"],
                    cuisine=cuisine,
                )
                for item in candidates
            ]

            # --- Step 3: Compute base probabilities (uniform) ---
            n_candidates = len(candidate_items)
            base_probs = np.ones(n_candidates, dtype=np.float64) / n_candidates

            # --- Step 4: Apply realism modifiers ---
            # This is where full-sequence dependencies are encoded:
            # the modifier uses cart_item_names (all previous items)
            # and running_total (cumulative price) to adjust P(item_k)
            item_names = [ci.name for ci in candidate_items]
            item_prices = np.array([ci.price for ci in candidate_items])

            modified_probs = self.modifiers.apply_all_modifiers(
                item_probs=base_probs,
                item_names=item_names,
                item_prices=item_prices,
                is_peak_hour=session.is_peak_hour,
                running_total=running_total,
                aov_ceiling=user.aov_ceiling,
                city=user.city,
                cart_item_names=cart_item_names,
            )

            # --- Step 5: Sample item ---
            idx = self.rng.choice(n_candidates, p=modified_probs)
            selected_item = candidate_items[idx]

            # --- Step 6: Determine quantity ---
            quantity = self._determine_quantity(user)

            # --- Step 7: Create CartItem or update state ---
            existing_item = next((ci for ci in trajectory.cart_items if ci.item.name == selected_item.name), None)
            if existing_item:
                existing_item.quantity += quantity
            else:
                cart_item = CartItem(
                    item=selected_item,
                    quantity=quantity,
                    step_index=step,
                    slot_filled=category,
                )
                trajectory.cart_items.append(cart_item)

            cart_item_names.append(selected_item.name)
            running_total += selected_item.price * quantity

            # Update gap vector
            if category in gap_vector and gap_vector[category] > 0:
                gap_vector[category] -= 1
            if category in filled_slots:
                filled_slots[category] = filled_slots.get(category, 0) + 1

            step += 1

            # --- Hard stop: price ceiling breach ---
            # Only hard-stop AFTER template is filled, or if way over budget
            if running_total > user.aov_ceiling * 2.0 and not remaining_gaps:
                break

        # Finalize trajectory
        trajectory.total_price = running_total
        trajectory.template_slots_filled = filled_slots
        trajectory.check_template_complete()

        return trajectory

    def _adjust_template_for_intent(
        self,
        base_template: Dict[str, int],
        intent: str,
        archetype: str,
    ) -> Dict[str, int]:
        """
        Adjust meal template size based on session intent and archetype.

        Family Dinner / FamilyOrder → larger templates.
        Late-Night Snack → smaller templates.
        """
        template = deepcopy(base_template)

        if intent == "Family Dinner" or archetype == "FamilyOrder":
            # Expand: add extra side and dessert (moderate expansion)
            template["side"] = template.get("side", 1) + 1
            template["dessert"] = template.get("dessert", 0) + 1

        elif intent == "Late-Night Snack":
            # Lighter template but still 3 items: main + side + beverage
            template = {"main": 1, "side": 1, "beverage": 1}

        elif intent == "Weekend Brunch":
            # Add dessert
            template["dessert"] = template.get("dessert", 0) + 1

        return template

    def _compute_max_items(
        self, user: User, session: Session, intent_config: dict,
        template_size: int = 3,
    ) -> int:
        """
        Compute the maximum number of items for this session.

        Combines user's session length preference with intent hints,
        then applies peak-hour reduction. Guarantees at least
        template_size items so the template can be filled.
        """
        intent_hint = intent_config.get("template_size_hint", 3)
        base = max(template_size, int(self.rng.normal(
            (user.session_length_mean + intent_hint) / 2.0,
            0.8,
        )))

        # Peak-hour sessions are faster → fewer items
        # But never below the template size
        if session.is_peak_hour:
            penalty = self.rng.uniform(0.80, 0.85)
            base = max(template_size, int(base * penalty))

        return min(base, 10)  # hard cap at 10 items

    def _sample_category_from_gaps(
        self, remaining_gaps: Dict[str, int]
    ) -> str:
        """
        Sample the next category to fill, weighted by remaining gap counts.

        Categories with more unfilled slots have higher probability.
        """
        categories = list(remaining_gaps.keys())
        weights = np.array(
            [remaining_gaps[c] for c in categories], dtype=np.float64
        )
        weights /= weights.sum()
        return self.rng.choice(categories, p=weights)

    def _determine_quantity(self, user: User) -> int:
        """
        Determine item quantity.

        CRUCIAL: FamilyOrder users MUST have quantity > 1.
        Uses Poisson(λ=2) + 1 for family orders, 1 for others.
        """
        if user.min_quantity > 1:
            # FamilyOrder: q ~ Poisson(2) + 1, guaranteeing q >= 2
            return int(self.rng.poisson(2)) + 1
        else:
            # Non-family: mostly q=1, with small chance of q=2
            if self.rng.random() < 0.10:
                return 2
            return 1
