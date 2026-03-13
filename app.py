import streamlit as st
import pandas as pd
import numpy as np

from engine import CSAOEngine
from csao.config.taxonomies import CUISINE_MENUS

st.set_page_config(page_title="Zomato CSAO Super Add-On", layout="wide", page_icon="🛒")

# ==========================================
# 1. Model Initialization
# ==========================================
@st.cache_resource(show_spinner=False)
def load_engine():
    engine = CSAOEngine()
    engine.run_offline_pipeline(n_trajectories=100)
    engine.train_system(epochs=1, limit_batches=50)
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
# 3. Global Context
# ==========================================
st.title("🛒 Zomato Smart Cart & Add-Ons")

with st.expander("⚙️ Shopping Environment Settings (User, City, Time)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_user_id = st.number_input("User ID", value=12345, step=1)
        sel_city = st.selectbox("City", ["Delhi-NCR", "Mumbai", "Bangalore"])
    with c2:
        cuisines = list(CUISINE_MENUS.keys())
        sel_cuisine = st.selectbox("Cuisine Context", cuisines, index=0)
        sel_rest_name = st.text_input("Restaurant Name", value="Spice Heaven")
    with c3:
        sel_hour = st.slider("Hour of Day", 0, 23, 20)
    with c4:
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
    st.header(f"🍽️ Menu: {sel_rest_name}")
    st.caption(f"Showing complete catalog for Cuisine: **{sel_cuisine}**")
    
    menu_data = CUISINE_MENUS.get(sel_cuisine, {})
    for category, items in menu_data.items():
        with st.expander(f"🥘 {category.capitalize()}", expanded=True):
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
                                    key=f"menu_{name}_{category}_{i}_{j}", 
                                    on_click=add_to_cart, 
                                    args=(name, category, price),
                                    use_container_width=True
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
        
        # Calculate inferred Meal Template categories
        f_vec, segments = engine.online_calculator.compute_feature_vector(
            user_id=sel_user_id,
            user_aov_ceiling=800.0,
            cart_items=st.session_state.cart,
            cart_total=total_value,
            cuisine=sel_cuisine,
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
        
        st.button("🗑️ Clear Cart", on_click=clear_cart, use_container_width=True)
        
        st.divider()
        if st.button("Generate Add-On Recommendations", type="primary", use_container_width=True):
            with st.spinner("AI thinking..."):
                ranked, debug = engine.predict_addon(
                    user_id=sel_user_id,
                    cart_items=st.session_state.cart,
                    restaurant_id="rest_ui_1",
                    restaurant_name=sel_rest_name,
                    restaurant_cuisine=sel_cuisine,
                    city=sel_city,
                    hour_of_day=sel_hour,
                    day_of_week=sel_day,
                    is_weekend=sel_is_weekend
                )
                st.session_state.ranked_candidates = ranked
                st.session_state.debug_payload = debug

        if st.session_state.ranked_candidates:
            st.subheader("✨ You might also like...")
            for idx, (rec_item, score) in enumerate(st.session_state.ranked_candidates[:5]):
                with st.container(border=True):
                    rec_price = 150.0 
                    for cat, its in CUISINE_MENUS.get(sel_cuisine, {}).items():
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
                        use_container_width=True
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
        
    with st.expander("Raw Feature State Dict", expanded=False):
        st.json(dbg["feature_state"])
        
    st.subheader("Top 8 Candidates Scoring Breakdown")
    if dbg["scoring_breakdown"]:
        df_breakdown = pd.DataFrame(dbg["scoring_breakdown"])
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
else:
    st.caption("Awaiting inference... Add items to cart and generate recommendations to see logs.")
