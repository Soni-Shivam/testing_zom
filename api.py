import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import the engine
from engine import CSAOEngine

app = FastAPI(title="Spice Heaven API", description="Backend for Zomato Clone UI")

# Initialize and load the engine
engine = CSAOEngine()

# Ensure we are loading artifacts from the proper directory
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
engine.load_pretrained_artifacts(ARTIFACT_DIR)

# Mount static files for frontend assets (HTML, CSS, JS, Images)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

# Make sure frontend and static dirs exist
os.makedirs(FRONTEND_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount product images directory
FOOD_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visuals", "generated_food_images")
print(f"Mounting product images from: {FOOD_IMAGES_DIR}")
if os.path.exists(FOOD_IMAGES_DIR):
    app.mount("/product-images", StaticFiles(directory=FOOD_IMAGES_DIR), name="product_images")
    print(f"Successfully mounted /product-images. Contains {len(os.listdir(FOOD_IMAGES_DIR))} files.")
else:
    print(f"Error: {FOOD_IMAGES_DIR} does not exist!")

# Setup models for request validation
class CartItem(BaseModel):
    name: str
    category: str
    quantity: int
    unit_price: float

class RecommendRequest(BaseModel):
    user_id: int
    cart: List[CartItem]
    restaurant_id: str = "rest_999"
    restaurant_name: str = "Spice Heaven"
    restaurant_cuisine: Optional[str] = None  # Require frontend to pass actual cuisine; None triggers dynamic catalog fallback
    city: str = "Delhi-NCR"
    hour_of_day: int = 20
    day_of_week: int = 5
    is_weekend: bool = True

class CheckoutRequest(BaseModel):
    user_id: int
    cart: List[CartItem]

class SimulateRequest(BaseModel):
    mean_aov: float
    order_count: int
    persona_type: str

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{view_name}.html")
def serve_html_view(view_name: str):
    """Serves the HTML template fragments for the SPA router."""
    view_path = os.path.join(FRONTEND_DIR, f"{view_name}.html")
    if os.path.exists(view_path):
        return FileResponse(view_path)
    raise HTTPException(status_code=404, detail="View not found")

@app.get("/restaurants")
def get_restaurants():
    """
    Returns all restaurants grouped by cuisine.
    """
    from csao.config.taxonomies import RESTAURANT_POOLS, ALL_CUISINES
    
    restaurants = []
    for cuisine in ALL_CUISINES:
        pool = RESTAURANT_POOLS.get(cuisine, [])
        for name in pool:
            restaurants.append({
                "name": name,
                "cuisine": cuisine,
                "rating": round(3.8 + (pool.index(name) % 10) * 0.1, 1), # Deterministic fake rating
                "deliveryTime": "25-35 min"
            })
    return {"restaurants": restaurants}

@app.get("/menu")
def get_menu(cuisine: str = "North Indian"):
    """
    Returns the restaurant menu categorized for a specific cuisine.
    """
    from csao.config.taxonomies import CUISINE_MENUS
    
    if cuisine not in CUISINE_MENUS:
        raise HTTPException(status_code=404, detail=f"Cuisine '{cuisine}' menu not found")
        
    menu_data = CUISINE_MENUS[cuisine]
    enriched_menu = {}
    
    for category, items in menu_data.items():
        enriched_category = []
        for item in items:
            name = item["name"]
            # Fetch actual global catalog price if it exists in engine, fallback to taxonomy price
            price = engine.item_prices.get(name, item.get("price", 150.0))
            enriched_category.append({
                "name": name,
                "price": price,
                "isVeg": item.get("isVeg", True),
                "description": item.get("description", f"A delicious {name} from our {cuisine} kitchen.")
            })
        enriched_menu[category] = enriched_category
            
    return {"cuisine": cuisine, "menu": enriched_menu}

@app.post("/cart/recommend")
def get_recommendations(req: RecommendRequest):
    """
    Accepts current cart context and calls engine.predict_addon().
    """
    try:
        cart_dicts = [item.dict() for item in req.cart]
        ranked_candidates, debug_payload = engine.predict_addon(
            user_id=req.user_id,
            cart_items=cart_dicts,
            restaurant_id=req.restaurant_id,
            restaurant_name=req.restaurant_name,
            restaurant_cuisine=req.restaurant_cuisine,
            city=req.city,
            hour_of_day=req.hour_of_day,
            day_of_week=req.day_of_week,
            is_weekend=req.is_weekend
        )
        
        # Enrich ranked candidates with prices and categories
        enriched_recs = []
        for cand, score in ranked_candidates:
            price = engine.item_prices.get(cand, 150.0)
            
            # Find category from metadata if possible
            category = "Recommended"
            for m in engine.item_meta:
                if m["item_name"] == cand:
                    category = m.get("category", category)
                    break
                    
            enriched_recs.append({
                "name": cand,
                "price": price,
                "score": score,
                "category": category
            })
            
        return {
            "recommendations": enriched_recs,
            "debug": debug_payload
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/checkout")
def place_order(req: CheckoutRequest):
    """
    Simulates placing an order and updates the user's history in memory.
    """
    try:
        cart_dicts = [item.dict() for item in req.cart]
        cart_total = sum(c["unit_price"] * c["quantity"] for c in cart_dicts)
        
        engine.place_order(
            user_id=req.user_id,
            cart_items=cart_dicts,
            cart_total=cart_total,
            cuisine="North Indian"
        )
        
        return {"status": "success", "message": "Order placed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/analytics")
def get_user_analytics(user_id: int):
    """
    Returns user profile analytics and aggregated recommendations for Home Page.
    """
    try:
        analytics = engine.get_user_analytics(user_id)
        recs = engine.get_homepage_recommendations(user_id, k=5)
        
        enriched_recs = []
        for cand, score in recs:
            price = engine.item_prices.get(cand, 150.0)
            enriched_recs.append({
                "name": cand,
                "price": price,
                "score": score
            })
            
        return {
            "analytics": analytics,
            "homepage_recommendations": enriched_recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/{user_id}/simulate")
def simulate_persona(user_id: int, req: SimulateRequest):
    """
    Forcefully overrides the user's historical profile for demo purposes.
    """
    try:
        if user_id not in engine.user_db:
            engine.user_db[user_id] = {
                "past_ordered_items": [],
                "past_order_totals": [],
                "favorite_cuisines": {}
            }
            
        profile = engine.user_db[user_id]
        
        # Override basic stats
        # We simulate this by adjusting the totals directly
        # The engine infers `mean_aov` by doing sum(past_order_totals) / len(past_order_totals)
        # So we force the list arrays to match our requested AOV and counts
        profile["past_order_totals"] = [req.mean_aov] * req.order_count
        
        # Inject exact history based on persona
        if req.persona_type == "The Health Nut":
            profile["past_ordered_items"] = (
                [{"name": "Greek Salad", "quantity": 1}] * 5 +
                [{"name": "Diet Coke", "quantity": 1}] * 5 +
                [{"name": "Quinoa Bowl", "quantity": 1}] * 5
            )
        elif req.persona_type == "The Budget Student":
            profile["past_ordered_items"] = [{"name": "Masala Dosa", "quantity": 1}] * 5
        elif req.persona_type == "The Family/Bulk Orderer":
            profile["past_ordered_items"] = (
                [{"name": "Chicken Biryani", "quantity": 4}] * 8 +
                [{"name": "Garlic Naan", "quantity": 6}] * 8
            )
        elif req.persona_type == "The Brand New User":
            profile["past_ordered_items"] = []
            profile["past_order_totals"] = []
            
        # Clear out cuisine counts and let it rebuild if needed, though engine doesn't actively use it yet for homepage
        profile["favorite_cuisines"] = {}
            
        print(f"User {user_id} simulated as {req.persona_type}: AOV={req.mean_aov}, Orders={req.order_count}")
        return {"status": "success", "message": f"Persona {req.persona_type} applied"}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
