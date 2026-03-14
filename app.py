import os
import streamlit as st
import pandas as pd
import numpy as np

from engine import CSAOEngine
from csao.config.taxonomies import CUISINE_MENUS

st.set_page_config(page_title="Zomato CSAO Super Add-On", layout="wide", page_icon="🛒")

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
            "❌ **Artifacts not found.** "
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
if "cart" not in st.session_state:
    st.session_state.cart = []
if "debug_payload" not in st.session_state:
    st.session_state.debug_payload = None
if "ranked_candidates" not in st.session_state:
    st.session_state.ranked_candidates = []

def add_to_cart(item_name: str, category: str, price: float):
    for item in st.session_state.cart:
        if item["name"] == item_name:
            item["quantity"] += 1
            st.session_state.debug_payload = None
            st.session_state.ranked_candidates = []
            return
    st.session_state.cart.append({
        "name": item_name,
        "category": category,
        "quantity": 1,
        "unit_price": float(price)
    })
    st.session_state.debug_payload = None
    st.session_state.ranked_candidates = []

def clear_cart():
    st.session_state.cart = []
    st.session_state.debug_payload = None
    st.session_state.ranked_candidates = []

# ==========================================
# 3. Sidebar User Switcher & Dashboard
# ==========================================
with st.sidebar:
    st.header("👤 Active User Simulator")
    user_options = {
        1: "New User (0 Orders)",
        2: "Regular User (4 Orders)",
        3: "Power User (15 Orders)"
    }
    sel_user_id = st.selectbox("Select Active User", options=list(user_options.keys()), format_func=lambda x: user_options[x], index=0)
    
    profile = engine.user_db.get(sel_user_id, {"order_count": 0, "mean_aov": 0.0, "cuisine_counts": {}})
    top_cuisine = "None"
    if profile["cuisine_counts"]:
        top_cuisine = max(profile["cuisine_counts"], key=profile["cuisine_counts"].get)
        
    st.divider()
    st.subheader("📊 User Profile")
    st.metric("Total Orders", profile["order_count"])
    st.metric("Average Order Value (AOV)", f"₹{profile['mean_aov']:.2f}")
    st.metric("Top Cuisine", top_cuisine)

# ==========================================
# 4. Global Context
# ==========================================
st.title("🛒 Zomato Smart Cart & Add-Ons")

with st.expander("⚙️ Shopping Environment Settings (City, Time)", expanded=False):
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
    st.header("🍽️ Complete Menu Catalog")
    st.caption("Showing items from all available restaurants and cuisines")
    
    for cuisine_name, menu_data in CUISINE_MENUS.items():
        st.subheader(f"🏪 {cuisine_name} Restaurant")
        for category, items in menu_data.items():
            with st.expander(f"🥘 {cuisine_name} - {category.capitalize()}", expanded=False):
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
                                        args=(name, category, price),
                                        width='stretch'
                                    )

# --- RIGHT COLUMN: Live Cart ---
with col2:
    st.header("🛒 Current Cart")
    
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

        # Calculate inferred Meal Template categories
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
            candidate_item_name="",
            candidate_item_category=""
        )
        gap_start, gap_end = segments["cart.meal_gap_vector"]
        gap = f_vec[gap_start:gap_end]
        st.caption(f"**Inferred Meal Gap Vector (M/S/B/D):** {gap.tolist()}")
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.button("🗑️ Clear Cart", on_click=clear_cart, use_container_width=True)
        with c_btn2:
            def place_checkout():
                if len(st.session_state.cart) > 0:
                    engine.place_order(sel_user_id, st.session_state.cart, total_value, inferred_cuisine)
                    clear_cart()
                    
            if st.button("🛒 Checkout & Place Order", type="primary", use_container_width=True, on_click=place_checkout):
                st.toast("Order Placed Successfully! User History updated.", icon="🎉")
        
        st.divider()
        if len(st.session_state.cart) > 0:
            with st.spinner("AI actively analyzing cart and user history..."):
                ranked, debug = engine.predict_addon(
                    user_id=sel_user_id,
                    cart_items=st.session_state.cart,
                    restaurant_id="rest_ui_global",
                    restaurant_name=inferred_rest_name,
                    restaurant_cuisine=inferred_cuisine,
                    city=sel_city,
                    hour_of_day=sel_hour,
                    day_of_week=sel_day,
                    is_weekend=sel_is_weekend
                )
                st.session_state.ranked_candidates = ranked
                st.session_state.debug_payload = debug

            st.subheader("✨ You might also like...")
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
st.header("🔬 Pipeline Execution Logs")
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
