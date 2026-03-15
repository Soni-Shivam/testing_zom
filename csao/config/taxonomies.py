"""
Core Taxonomies for CSAO Hierarchical Generative Data Synthesis
================================================================
Defines all domain knowledge as structured dictionaries:
cities, cuisine menus, archetypes, session intents, meal templates,
and ground-truth co-occurrence pairings.
"""

import numpy as np

# =============================================================================
# 1. CITY DEFINITIONS & GEOGRAPHIC CUISINE AFFINITY MATRICES
# =============================================================================
# Each city maps to a dictionary of cuisine → probability weight.
# These weights encode real-world geographic taste clusters so the
# generator never learns geographically unrepresentative co-occurrences.

CITIES = {
    "Delhi-NCR": {
        "population_weight": 0.30,
        "cuisine_affinity": {
            "North Indian":   0.40,
            "Mughlai":        0.20,
            "Chinese":        0.10,
            "Street Food":    0.10,
            "Continental":    0.05,
            "South Indian":   0.05,
            "Italian":        0.05,
            "Biryani":        0.05,
        }
    },
    "Mumbai": {
        "population_weight": 0.25,
        "cuisine_affinity": {
            "Coastal Seafood": 0.25,
            "Street Food":     0.20,
            "North Indian":    0.15,
            "Chinese":         0.10,
            "South Indian":    0.08,
            "Italian":         0.07,
            "Continental":     0.08,
            "Mughlai":         0.07,
        }
    },
    "Chennai": {
        "population_weight": 0.15,
        "cuisine_affinity": {
            "South Indian":    0.45,
            "Chinese":         0.12,
            "North Indian":    0.10,
            "Biryani":         0.10,
            "Street Food":     0.08,
            "Continental":     0.08,
            "Italian":         0.04,
            "Coastal Seafood": 0.03,
        }
    },
    "Hyderabad": {
        "population_weight": 0.15,
        "cuisine_affinity": {
            "Biryani":         0.40,
            "South Indian":    0.15,
            "North Indian":    0.10,
            "Street Food":     0.10,
            "Chinese":         0.10,
            "Mughlai":         0.08,
            "Continental":     0.04,
            "Italian":         0.03,
        }
    },
    "Bangalore": {
        "population_weight": 0.15,
        "cuisine_affinity": {
            "South Indian":    0.25,
            "North Indian":    0.15,
            "Chinese":         0.15,
            "Continental":     0.12,
            "Italian":         0.10,
            "Biryani":         0.08,
            "Street Food":     0.08,
            "Coastal Seafood": 0.07,
        }
    },
}

# Extract ordered list for sampling
CITY_NAMES = list(CITIES.keys())
CITY_WEIGHTS = np.array([CITIES[c]["population_weight"] for c in CITY_NAMES])


# =============================================================================
# 2. CUISINE MENU CATALOGS
# =============================================================================
# Each cuisine has items tagged by category (main, side, beverage, dessert)
# with realistic price ranges in INR.

CUISINE_MENUS = {
    "North Indian": {
        "main":     [
            {"name": "Butter Chicken",      "price": 280},
            {"name": "Paneer Tikka Masala",  "price": 250},
            {"name": "Dal Makhani",          "price": 200},
            {"name": "Chole Bhature",        "price": 180},
            {"name": "Rajma Chawal",         "price": 170},
            {"name": "Kadai Paneer",         "price": 240},
        ],
        "side":     [
            {"name": "Naan",                 "price": 40},
            {"name": "Garlic Naan",          "price": 55},
            {"name": "Butter Roti",          "price": 30},
            {"name": "Raita",                "price": 50},
            {"name": "Onion Salad",          "price": 30},
            {"name": "Papad",                "price": 25},
        ],
        "beverage": [
            {"name": "Lassi",                "price": 80},
            {"name": "Masala Chai",          "price": 40},
            {"name": "Jaljeera",             "price": 50},
            {"name": "Buttermilk",           "price": 45},
        ],
        "dessert":  [
            {"name": "Gulab Jamun",          "price": 80},
            {"name": "Rasmalai",             "price": 100},
            {"name": "Kheer",                "price": 90},
        ],
    },
    "Mughlai": {
        "main":     [
            {"name": "Mutton Korma",         "price": 350},
            {"name": "Chicken Nihari",       "price": 320},
            {"name": "Shahi Paneer",         "price": 260},
            {"name": "Mutton Rogan Josh",    "price": 380},
        ],
        "side":     [
            {"name": "Sheermal",             "price": 60},
            {"name": "Roomali Roti",         "price": 45},
            {"name": "Mughlai Paratha",      "price": 70},
        ],
        "beverage": [
            {"name": "Rooh Afza Sharbat",    "price": 60},
            {"name": "Masala Chai",          "price": 40},
        ],
        "dessert":  [
            {"name": "Shahi Tukda",          "price": 110},
            {"name": "Phirni",               "price": 90},
        ],
    },
    "South Indian": {
        "main":     [
            {"name": "Masala Dosa",          "price": 120},
            {"name": "Idli Sambar",          "price": 90},
            {"name": "Uttapam",              "price": 110},
            {"name": "Medu Vada",            "price": 80},
            {"name": "Pongal",               "price": 100},
            {"name": "Rava Dosa",            "price": 130},
        ],
        "side":     [
            {"name": "Coconut Chutney",      "price": 30},
            {"name": "Sambar",               "price": 40},
            {"name": "Tomato Chutney",       "price": 30},
            {"name": "Podi",                 "price": 25},
        ],
        "beverage": [
            {"name": "Filter Coffee",        "price": 50},
            {"name": "Buttermilk",           "price": 40},
            {"name": "Rasam",                "price": 35},
        ],
        "dessert":  [
            {"name": "Payasam",              "price": 70},
            {"name": "Mysore Pak",           "price": 80},
        ],
    },
    "Chinese": {
        "main":     [
            {"name": "Veg Manchurian",       "price": 180},
            {"name": "Chicken Fried Rice",   "price": 200},
            {"name": "Veg Hakka Noodles",    "price": 170},
            {"name": "Chilli Chicken",       "price": 220},
            {"name": "Schezwan Noodles",     "price": 190},
        ],
        "side":     [
            {"name": "Spring Roll",          "price": 120},
            {"name": "Dim Sum",              "price": 150},
            {"name": "Crispy Corn",          "price": 140},
        ],
        "beverage": [
            {"name": "Lemon Iced Tea",       "price": 70},
            {"name": "Cold Coffee",          "price": 90},
        ],
        "dessert":  [
            {"name": "Honey Noodles",        "price": 110},
            {"name": "Date Pancake",         "price": 100},
        ],
    },
    "Street Food": {
        "main":     [
            {"name": "Pav Bhaji",            "price": 120},
            {"name": "Vada Pav",             "price": 50},
            {"name": "Pani Puri",            "price": 60},
            {"name": "Kathi Roll",           "price": 100},
            {"name": "Aloo Tikki",           "price": 70},
            {"name": "Samosa",               "price": 40},
        ],
        "side":     [
            {"name": "Green Chutney",        "price": 15},
            {"name": "Tamarind Chutney",     "price": 15},
            {"name": "Sev",                  "price": 20},
        ],
        "beverage": [
            {"name": "Masala Soda",          "price": 30},
            {"name": "Sugarcane Juice",      "price": 40},
            {"name": "Nimbu Pani",           "price": 25},
        ],
        "dessert":  [
            {"name": "Jalebi",               "price": 60},
            {"name": "Rabri",                "price": 70},
        ],
    },
    "Coastal Seafood": {
        "main":     [
            {"name": "Fish Curry",           "price": 300},
            {"name": "Prawn Masala",         "price": 350},
            {"name": "Bombil Fry",           "price": 280},
            {"name": "Crab Masala",          "price": 400},
            {"name": "Fish Thali",           "price": 320},
        ],
        "side":     [
            {"name": "Steamed Rice",         "price": 60},
            {"name": "Sol Kadhi",            "price": 50},
            {"name": "Fried Papad",          "price": 30},
        ],
        "beverage": [
            {"name": "Kokum Sharbat",        "price": 50},
            {"name": "Buttermilk",           "price": 40},
        ],
        "dessert":  [
            {"name": "Modak",                "price": 80},
            {"name": "Puran Poli",           "price": 70},
        ],
    },
    "Biryani": {
        "main":     [
            {"name": "Hyderabadi Chicken Biryani", "price": 280},
            {"name": "Hyderabadi Mutton Biryani",  "price": 350},
            {"name": "Veg Biryani",                "price": 200},
            {"name": "Egg Biryani",                "price": 220},
            {"name": "Lucknowi Biryani",           "price": 300},
        ],
        "side":     [
            {"name": "Mirchi Ka Salan",      "price": 80},
            {"name": "Raita",                "price": 50},
            {"name": "Salan",                "price": 70},
            {"name": "Onion Raita",          "price": 45},
        ],
        "beverage": [
            {"name": "Rooh Afza",            "price": 50},
            {"name": "Lassi",                "price": 70},
        ],
        "dessert":  [
            {"name": "Double Ka Meetha",     "price": 90},
            {"name": "Qubani Ka Meetha",     "price": 100},
        ],
    },
    "Italian": {
        "main":     [
            {"name": "Margherita Pizza",     "price": 250},
            {"name": "Pasta Alfredo",        "price": 230},
            {"name": "Pepperoni Pizza",      "price": 300},
            {"name": "Penne Arrabiata",      "price": 220},
        ],
        "side":     [
            {"name": "Garlic Bread",         "price": 120},
            {"name": "Bruschetta",           "price": 140},
            {"name": "Caesar Salad",         "price": 160},
        ],
        "beverage": [
            {"name": "Virgin Mojito",        "price": 120},
            {"name": "Cold Coffee",          "price": 100},
            {"name": "Lemonade",             "price": 80},
        ],
        "dessert":  [
            {"name": "Tiramisu",             "price": 180},
            {"name": "Panna Cotta",          "price": 160},
        ],
    },
    "Continental": {
        "main":     [
            {"name": "Grilled Chicken Burger","price": 220},
            {"name": "Veg Burger",            "price": 180},
            {"name": "Club Sandwich",         "price": 200},
            {"name": "Fish and Chips",        "price": 280},
            {"name": "Chicken Wrap",          "price": 190},
        ],
        "side":     [
            {"name": "French Fries",         "price": 100},
            {"name": "Coleslaw",             "price": 60},
            {"name": "Onion Rings",          "price": 110},
            {"name": "Potato Wedges",        "price": 120},
        ],
        "beverage": [
            {"name": "Coca Cola",            "price": 50},
            {"name": "Milkshake",            "price": 130},
            {"name": "Iced Tea",             "price": 70},
        ],
        "dessert":  [
            {"name": "Brownie",              "price": 120},
            {"name": "Cheesecake",           "price": 180},
        ],
    },
}

# All cuisine names for indexing
ALL_CUISINES = list(CUISINE_MENUS.keys())


# =============================================================================
# 3. RESTAURANT NAME POOLS (per cuisine)
# =============================================================================

RESTAURANT_POOLS = {
    "North Indian":    ["Punjab Grill", "Dhaba Express", "Tandoori Nights", "Pind Balluchi",
                        "Haveli", "Saffron Kitchen", "Dilli 32", "Amritsari Zaika"],
    "Mughlai":         ["Karim's", "Al Jawahar", "Tunday Kababi", "Lucknow Central",
                        "Nawab's Kitchen", "Mughal Darbar"],
    "South Indian":    ["Saravana Bhavan", "Murugan Idli", "Adyar Ananda Bhavan",
                        "Ratna Cafe", "Dakshin", "Vasudev Adigas"],
    "Chinese":         ["Chung Wah", "Mainland China", "Wok Express", "Golden Dragon",
                        "Chopstick", "Sichuan House"],
    "Street Food":     ["Chaat Corner", "Bikanervala", "Haldiram's", "Jumbo King",
                        "Shiv Sagar", "Sardarji Pav Bhaji"],
    "Coastal Seafood": ["Gajalee", "Trishna", "Mahesh Lunch Home", "Konkan Cafe",
                        "Coastal Pearl", "Fish Curry Rice"],
    "Biryani":         ["Paradise Biryani", "Bawarchi", "Shah Ghouse", "Cafe Bahar",
                        "Pista House", "Shadab"],
    "Italian":         ["Pizza Hut", "Domino's", "La Pino'z", "Fat Lulu's",
                        "Imperfecto", "Olive Bar & Kitchen"],
    "Continental":     ["McDonald's", "Burger King", "The Burger Company",
                        "Wendy's", "Hard Rock Cafe", "TGIF"],
}


# =============================================================================
# 4. USER ARCHETYPES
# =============================================================================

USER_ARCHETYPES = {
    "Budget": {
        "probability": 0.35,
        "aov_mean": 250,
        "aov_std": 60,
        "aov_ceiling": 400,
        "session_length_mean": 3.0,  # avg items per session
        "session_length_std": 1.0,
        "order_frequency_weekly": 4.0,
        "cuisine_diversity": 0.3,     # low → tends to stick to fewer cuisines
        "min_quantity": 1,
    },
    "Premium": {
        "probability": 0.15,
        "aov_mean": 650,
        "aov_std": 120,
        "aov_ceiling": 900,
        "session_length_mean": 5.0,
        "session_length_std": 1.5,
        "order_frequency_weekly": 3.0,
        "cuisine_diversity": 0.8,
        "min_quantity": 1,
    },
    "Occasional": {
        "probability": 0.30,
        "aov_mean": 350,
        "aov_std": 80,
        "aov_ceiling": 500,
        "session_length_mean": 3.5,
        "session_length_std": 1.2,
        "order_frequency_weekly": 1.5,
        "cuisine_diversity": 0.5,
        "min_quantity": 1,
    },
    "FamilyOrder": {
        "probability": 0.20,
        "aov_mean": 500,
        "aov_std": 100,
        "aov_ceiling": 700,
        "session_length_mean": 6.0,
        "session_length_std": 1.5,
        "order_frequency_weekly": 2.5,
        "cuisine_diversity": 0.4,
        "min_quantity": 2,  # CRUCIAL: family orders always have q > 1
    },
}

ARCHETYPE_NAMES = list(USER_ARCHETYPES.keys())
ARCHETYPE_PROBS = np.array([USER_ARCHETYPES[a]["probability"] for a in ARCHETYPE_NAMES])


# =============================================================================
# 5. SESSION INTENTS
# =============================================================================

SESSION_INTENTS = {
    "Solo Lunch": {
        "hour_peak":   13,
        "hour_spread": 1.5,
        "template_size_hint": 3,    # main + side + beverage
        "archetype_affinity": {
            "Budget": 0.40, "Premium": 0.25, "Occasional": 0.35, "FamilyOrder": 0.05,
        },
    },
    "Family Dinner": {
        "hour_peak":   20,
        "hour_spread": 1.5,
        "template_size_hint": 5,    # multiple mains + sides + beverages + dessert
        "archetype_affinity": {
            "Budget": 0.10, "Premium": 0.25, "Occasional": 0.15, "FamilyOrder": 0.55,
        },
    },
    "Late-Night Snack": {
        "hour_peak":   23,
        "hour_spread": 1.0,
        "template_size_hint": 2,    # light: main + beverage
        "archetype_affinity": {
            "Budget": 0.30, "Premium": 0.20, "Occasional": 0.40, "FamilyOrder": 0.05,
        },
    },
    "Weekend Brunch": {
        "hour_peak":   11,
        "hour_spread": 1.5,
        "template_size_hint": 4,    # main + side + beverage + dessert
        "archetype_affinity": {
            "Budget": 0.20, "Premium": 0.35, "Occasional": 0.30, "FamilyOrder": 0.20,
        },
    },
}

INTENT_NAMES = list(SESSION_INTENTS.keys())


# =============================================================================
# 6. MEAL TEMPLATES (per cuisine)
# =============================================================================
# Each template defines the required slot counts: {category: required_count}
# These are used to build the meal gap vector g_k.

MEAL_TEMPLATES = {
    "North Indian":    {"main": 1, "side": 1, "beverage": 1},
    "Mughlai":         {"main": 1, "side": 1, "beverage": 1},
    "South Indian":    {"main": 1, "side": 2, "beverage": 1},
    "Chinese":         {"main": 1, "side": 1, "beverage": 1},
    "Street Food":     {"main": 2, "side": 1, "beverage": 1},
    "Coastal Seafood": {"main": 1, "side": 1, "beverage": 1},
    "Biryani":         {"main": 1, "side": 2, "beverage": 1},
    "Italian":         {"main": 1, "side": 1, "beverage": 1},
    "Continental":     {"main": 1, "side": 1, "beverage": 1},
}


# =============================================================================
# 7. PEAK HOUR DEFINITIONS
# =============================================================================

PEAK_HOURS = {
    "lunch":  (12, 14),   # 12:00 PM – 2:00 PM
    "dinner": (19, 22),   # 7:00 PM – 10:00 PM
}

PEAK_HOUR_PENALTY_RANGE = (0.80, 0.85)  # 15-20% reduction in add-on probability


# =============================================================================
# 8. GEOGRAPHIC CO-OCCURRENCE BOOSTS
# =============================================================================
# City-specific item affinity overrides. When a trigger item is already
# in the cart, the boost items get their probability multiplied by the factor.

GEOGRAPHIC_COOCCURRENCE = {
    "Hyderabad": [
        {"trigger": "Hyderabadi Chicken Biryani", "boost_items": ["Mirchi Ka Salan", "Salan", "Raita"], "factor": 5.0},
        {"trigger": "Hyderabadi Mutton Biryani",  "boost_items": ["Mirchi Ka Salan", "Salan", "Raita"], "factor": 5.0},
    ],
    "Delhi-NCR": [
        {"trigger": "Butter Chicken",   "boost_items": ["Garlic Naan", "Naan", "Lassi"],  "factor": 1.5},
        {"trigger": "Chole Bhature",    "boost_items": ["Lassi", "Onion Salad"],            "factor": 1.5},
    ],
    "Chennai": [
        {"trigger": "Masala Dosa",      "boost_items": ["Coconut Chutney", "Sambar", "Filter Coffee"], "factor": 4.0},
        {"trigger": "Idli Sambar",      "boost_items": ["Coconut Chutney", "Filter Coffee"],            "factor": 4.0},
    ],
    "Mumbai": [
        {"trigger": "Pav Bhaji",        "boost_items": ["Masala Soda", "Lassi"],            "factor": 1.5},
        {"trigger": "Fish Curry",       "boost_items": ["Steamed Rice", "Sol Kadhi"],       "factor": 1.5},
    ],
    "Bangalore": [
        {"trigger": "Masala Dosa",      "boost_items": ["Coconut Chutney", "Filter Coffee"],"factor": 3.5},
    ],
}


# =============================================================================
# 9. GROUND TRUTH CO-OCCURRENCE PAIRS (for χ² validation)
# =============================================================================
# These are known real-world pairings. The validator checks that the
# synthetic corpus exhibits statistically significant co-occurrence for these.

GROUND_TRUTH_PAIRINGS = [
    ("Hyderabadi Chicken Biryani", "Mirchi Ka Salan"),
    ("Hyderabadi Chicken Biryani", "Salan"),
    ("Hyderabadi Mutton Biryani",  "Mirchi Ka Salan"),
    ("Butter Chicken",             "Garlic Naan"),
    ("Butter Chicken",             "Naan"),
    ("Masala Dosa",                "Coconut Chutney"),
    ("Masala Dosa",                "Sambar"),
    ("Idli Sambar",                "Coconut Chutney"),
    ("Grilled Chicken Burger",     "French Fries"),
    ("Veg Burger",                 "French Fries"),
    ("Fish Curry",                 "Steamed Rice"),
    ("Margherita Pizza",           "Garlic Bread"),
    ("Pav Bhaji",                  "Masala Soda"),
]


# =============================================================================
# 10. SESSION LENGTH BENCHMARK DISTRIBUTION
# =============================================================================
# Expected distribution of cart sizes (number of distinct items).
# Keys = item count, Values = probability mass.
# Derived from industry benchmarks for food delivery platforms.

SESSION_LENGTH_BENCHMARK = {
    1: 0.05,
    2: 0.15,
    3: 0.30,
    4: 0.25,
    5: 0.15,
    6: 0.06,
    7: 0.03,
    8: 0.01,
}
