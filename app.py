import os
import streamlit as st
import pandas as pd
import numpy as np

from engine import CSAOEngine
from csao.config.taxonomies import CUISINE_MENUS

st.set_page_config(page_title="Zomato CSAO Super Add-On", layout="wide", page_icon="")

# ==========================================
# 1. Model Initialization
# ==========================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


@st.cache_resource(show_spinner=False)
def load_engine():
    """
    Serving-mode boot: loads pre-computed artifacts from disk.
    Run `python train_offline.py` once to generate the artifacts/ directory.
    """
    if not os.path.isdir(ARTIFACT_DIR):
        st.error(
            " **Artifacts not found.** "
            "The model has not been trained yet. "
            "Please run the following command in your terminal first:\n\n"
            "```bash\n"
            "python train_offline.py\n"
            "```\n\n"
            "This will train the models and save all required artifacts to `artifacts/`. "
            "Once complete, restart the app."
        )
        st.stop()

    engine = CSAOEngine()
    engine.load_pretrained_artifacts(ARTIFACT_DIR)
    return engine

with st.spinner("Booting up backend Hybrid Engine..."):
    engine = load_engine()

# ==========================================
# 2. Session State Management
# ==========================================
import copy

if "cart" not in st.session_state:
    st.session_state.cart = []
if "debug_payload" not in st.session_state:
    st.session_state.debug_payload = None
if "ranked_candidates" not in st.session_state:
    st.session_state.ranked_candidates = []
if "needs_addon_refresh" not in st.session_state:
    st.session_state.needs_addon_refresh = False
if "user_db" not in st.session_state:
    # Deepcopy the cached global user_db into this specific browser session
    st.session_state.user_db = copy.deepcopy(engine.user_db)
# Fix 5: Pre-fetched candidate cache — populated whenever the inferred cuisine is known.
if "prefetched_candidates" not in st.session_state:
    st.session_state.prefetched_candidates = None

def add_to_cart(item_name: str, category: str, price: float, cuisine: str = "Unknown"):
    for item in st.session_state.cart:
        if item["name"] == item_name:
            item["quantity"] += 1
            st.session_state.debug_payload = None
            st.session_state.ranked_candidates = []
            st.session_state.needs_addon_refresh = True
            # Fix 5: Re-prefetch with updated cart so the inference path skips CUISINE_MENUS scan
            cart_name_set = {c["name"] for c in st.session_state.cart}
            st.session_state.prefetched_candidates = engine.prefetch_candidates(cuisine, cart_name_set)
            return
    st.session_state.cart.append({
        "name": item_name,
        "category": category,
        "quantity": 1,
        "unit_price": float(price)
    })
    st.session_state.debug_payload = None
    st.session_state.ranked_candidates = []
    st.session_state.needs_addon_refresh = True
    # Fix 5: Pre-fetch candidates for the detected cuisine so predict_addon can skip scanning
    cart_name_set = {c["name"] for c in st.session_state.cart}
    st.session_state.prefetched_candidates = engine.prefetch_candidates(cuisine, cart_name_set)

def clear_cart():
    st.session_state.cart = []
    st.session_state.debug_payload = None
    st.session_state.ranked_candidates = []
    st.session_state.needs_addon_refresh = False
    st.session_state.prefetched_candidates = None  # Clear prefetch on cart reset

# ==========================================
# 3. Sidebar User Switcher & Dashboard
# ==========================================
with st.sidebar:
    st.header(" Active User Simulator")
    user_options = {
        1: "New User (0 Orders)",
        2: "Regular User (4 Orders)",
        3: "Power User (15 Orders)"
    }
    sel_user_id = st.selectbox("Select Active User", options=list(user_options.keys()), format_func=lambda x: user_options[x], index=0)

    st.divider()
    st.subheader(" User CRM Profile")

    # Always sync engine's view with the latest session-state user_db
    engine.user_db = st.session_state.user_db
    analytics = engine.get_user_analytics(sel_user_id)

    if analytics["status"] == "active":
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Orders", analytics["total_orders"])
            st.metric("AOV", f"₹{analytics['aov']:.0f}")
        with col_b:
            st.metric("Lifetime ₹", f"₹{analytics['lifetime_value']:.0f}")
            st.metric("Items Tried", analytics["distinct_items_tried"])

        if analytics["top_cuisines"]:
            st.caption(f"**Top Cuisine:** {analytics['top_cuisines'][0][0]}")

        st.write("**Most Ordered:**")
        for item_name, count in analytics["favorite_items"][:3]:
            st.caption(f"- {item_name} (×{count})")

        st.divider()
        st.subheader(" Recommended For You")
        st.caption("Based on your semantic taste profile")

        homepage_recs = engine.get_homepage_recommendations(sel_user_id, k=3)
        for rec_item, score in homepage_recs:
            rec_price = engine.item_prices.get(rec_item, 150.0)
            with st.container(border=True):
                st.markdown(f"**{rec_item}**")
                st.caption(f"Match: {score * 100:.1f}%  |  ₹{rec_price:.0f}")
                st.button(
                    f"Add for ₹{rec_price:.0f}",
                    key=f"home_rec_{rec_item}_{sel_user_id}",
                    on_click=add_to_cart,
                    args=(rec_item, "Recommendation", rec_price),
                    use_container_width=True,
                )
    else:
        st.info("New user — place an order to generate analytics and personalized recommendations.")

# ==========================================
# 4. Global Context
# ==========================================
st.title(" Zomato Smart Cart & Add-Ons")

with st.expander(" Shopping Environment Settings (City, Time)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_city = st.selectbox("City", ["Delhi-NCR", "Mumbai", "Bangalore"])
    with c2:
        sel_hour = st.slider("Hour of Day", 0, 23, 20)
    with c3:
        days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        sel_day = st.selectbox("Day of Week", options=list(days.keys()), format_func=lambda x: days[x], index=5)
        sel_is_weekend = st.checkbox("Is Weekend", value=(sel_day >= 5))

# ==========================================
# 4. Main UI Layout
# ==========================================
st.divider()
col1, col2 = st.columns([2, 1])

# --- LEFT COLUMN: Restaurant Menu ---
with col1:
    st.header(" Complete Menu Catalog")
    st.caption("Showing items from all available restaurants and cuisines")
    
    for cuisine_name, menu_data in CUISINE_MENUS.items():
        st.subheader(f" {cuisine_name} Restaurant")
        for category, items in menu_data.items():
            with st.expander(f" {cuisine_name} - {category.capitalize()}", expanded=False):
                cols_per_row = 3
                for i in range(0, len(items), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(items):
                            item = items[i + j]
                            name = item["name"]
                            price = float(item["price"])
                            
                            with row_cols[j]:
                                with st.container(border=True):
                                    st.markdown(f"**{name}**")
                                    st.markdown(f"₹{price:.2f}")
                                    st.button(
                                        "Add to Cart",
                                        key=f"menu_{cuisine_name}_{name}_{category}_{i}_{j}",
                                        on_click=add_to_cart,
                                        # Fix 5: pass cuisine so prefetch knows which menu to scan
                                        args=(name, category, price, cuisine_name),
                                        width='stretch'
                                    )

# --- RIGHT COLUMN: Live Cart ---
with col2:
    st.header(" Current Cart")
    
    total_value = sum(item["quantity"] * item["unit_price"] for item in st.session_state.cart)
    
    if len(st.session_state.cart) == 0:
        st.info("Your cart is empty. Please add items from the menu.")
    else:
        for item in st.session_state.cart:
            qty = item["quantity"]
            name = item["name"]
            line_cost = qty * item["unit_price"]
            st.markdown(f"({qty}x) **{name}** — ₹{line_cost:.2f}")
            
        st.divider()
        st.markdown(f"### Total: ₹{total_value:.2f}")
        
        # Determine dominant cuisine based on first cart item, or default
        inferred_cuisine = "North Indian"
        inferred_rest_name = "Zomato Global Hub"
        if len(st.session_state.cart) > 0:
            first_item = st.session_state.cart[0]["name"]
            for c_name, m_data in CUISINE_MENUS.items():
                for cat, its in m_data.items():
                    if any(it["name"] == first_item for it in its):
                        inferred_cuisine = c_name
                        inferred_rest_name = f"{c_name} Restaurant"
                        break

        # Fix 4: Use a valid canonical item name instead of "" to prevent KeyError
        dummy_item = list(engine.item_to_idx.keys())[0] if engine.item_to_idx else "Unknown Item"
        
        f_vec, segments = engine.online_calculator.compute_feature_vector(
            user_id=sel_user_id,
            user_aov_ceiling=800.0,
            cart_items=st.session_state.cart,
            cart_total=total_value,
            cuisine=inferred_cuisine,
            hour_of_day=sel_hour,
            day_of_week=sel_day,
            is_weekend=sel_is_weekend,
            city=sel_city,
            candidate_item_name=dummy_item,
            candidate_item_category="Dummy"
        )
        gap_start, gap_end = segments["cart.meal_gap_vector"]
        gap = f_vec[gap_start:gap_end]
        st.caption(f"**Inferred Meal Gap Vector (M/S/B/D):** {gap.tolist()}")
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.button(" Clear Cart", on_click=clear_cart, use_container_width=True)
        with c_btn2:
            def place_checkout():
                if len(st.session_state.cart) > 0:
                    # Fix 3: Temporarily inject session explicitly before mutation
                    engine.user_db = st.session_state.user_db
                    engine.place_order(sel_user_id, st.session_state.cart, total_value, inferred_cuisine)
                    clear_cart()
                    
            if st.button(" Checkout & Place Order", type="primary", use_container_width=True, on_click=place_checkout):
                st.toast("Order Placed Successfully! User History updated.", icon="")
        
        st.divider()
        if len(st.session_state.cart) > 0:
            if st.session_state.needs_addon_refresh:
                with st.spinner("AI actively analyzing cart and user history..."):
                    # Fix 3: Inject session state DB for read paths too
                    engine.user_db = st.session_state.user_db
                    ranked, debug = engine.predict_addon(
                        user_id=sel_user_id,
                        cart_items=st.session_state.cart,
                        restaurant_id="rest_ui_global",
                        restaurant_name=inferred_rest_name,
                        restaurant_cuisine=inferred_cuisine,
                        city=sel_city,
                        hour_of_day=sel_hour,
                        day_of_week=sel_day,
                        is_weekend=sel_is_weekend,
                        # Fix 5: pass cached candidates to skip CUISINE_MENUS scan at inference time
                        prefetched_candidates=st.session_state.prefetched_candidates,
                    )
                    st.session_state.ranked_candidates = ranked
                    st.session_state.debug_payload = debug
                    st.session_state.needs_addon_refresh = False

            if st.session_state.ranked_candidates:
                st.subheader(" You might also like...")
            for idx, (rec_item, score) in enumerate(st.session_state.ranked_candidates[:5]):
                with st.container(border=True):
                    rec_price = 150.0 
                    for m_data in CUISINE_MENUS.values():
                        for cat, its in m_data.items():
                            for it in its:
                                if it["name"] == rec_item:
                                    rec_price = float(it["price"])
                                    break
                    
                    st.markdown(f"**{rec_item}**  \n`Score: {score:.3f}` | ₹{rec_price:.2f}")
                    st.button(
                        f"Add {rec_item}", 
                        key=f"rec_{rec_item}_{idx}", 
                        on_click=add_to_cart, 
                        args=(rec_item, "Add-On", rec_price),
                        type="secondary",
                        width='stretch'
                    )

# ==========================================
# 5. Verbose Logging View
# ==========================================
st.divider()
st.header(" Pipeline Execution Logs")
if st.session_state.debug_payload:
    dbg = st.session_state.debug_payload
    
    lc1, lc2 = st.columns([1, 1])
    with lc1:
        st.metric("Total Inference Latency (ms)", f"{dbg['latency']:.2f}")
        
    with lc2:
        st.info(f"**Cold-Start Routing Path:** {dbg['cold_start_path']}")
        st.info(f"**GRU4Rec Status:** {dbg.get('gru4rec_status', 'Unknown')}")
        
    with st.expander("Raw Feature State Dict", expanded=False):
        st.json(dbg["feature_state"])
        
    st.subheader("Top 8 Candidates Scoring Breakdown")
    if dbg["scoring_breakdown"]:
        df_breakdown = pd.DataFrame(dbg["scoring_breakdown"])
        st.dataframe(df_breakdown, width='stretch', hide_index=True)
else:
    st.caption("Awaiting inference... Add items to cart and generate recommendations to see logs.")
