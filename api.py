import os
from typing import List, Dict, Any
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
    restaurant_cuisine: str = "North Indian"
    city: str = "Delhi-NCR"
    hour_of_day: int = 20
    day_of_week: int = 5
    is_weekend: bool = True

class CheckoutRequest(BaseModel):
    user_id: int
    cart: List[CartItem]

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/menu")
def get_menu():
    """
    Returns the restaurant menu categorized, with prices.
    Uses the canonical data loaded from taxonomy to form the menu.
    """
    from csao.config.taxonomies import CUISINE_MENUS
    
    # Using 'North Indian' as our default Spice Heaven cuisine
    cuisine = "North Indian"
    if cuisine not in CUISINE_MENUS:
        raise HTTPException(status_code=404, detail="Cuisine menu not found")
        
    menu_data = CUISINE_MENUS[cuisine]
    enriched_menu = {}
    
    for category, items in menu_data.items():
        enriched_category = []
        for item in items:
            name = item["name"]
            # Fetch actual global catalog price if differs, fallback to 150
            price = engine.item_prices.get(name, item.get("price", 150.0))
            enriched_category.append({
                "name": name,
                "price": price,
                "isVeg": item.get("isVeg", True),
                "description": item.get("description", f"A delicious {name}")
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

if __name__ == "__main__":
    print("Starting API Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
