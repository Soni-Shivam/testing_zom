# CS-AO: Cart Super Add-On Recommendation System
## Stage 1: Hierarchical Generative Data Synthesis

This repository implements **Stage 1** of a production-grade recommendation engine designed to predict "Super Add-Ons" for food delivery carts. This module focuses on **Hierarchical Generative Data Synthesis**, creating a synthetic training corpus of 10,000 cart trajectories that faithfully encode complex real-world variables, including peak-hour dynamics, geographic taste clusters, and archetype-specific ordering behaviors.

---

## 🚀 Overview

Unlike naive random data generators, this system uses a **four-level hierarchical model** to ensure that every generated session exhibits "Full-Sequence Dependency." This means the probability of adding an item $i_k$ depends on the complete set of previous items $\Phi_{k-1}$ and the contextual state of the session.

### Core Features
- **Object-Oriented Design**: Built with modular Python, using `numpy` and `pandas` for vectorized operations.
- **Geographic Stratification**: Realistic cuisine concentrations across 5 major Indian cities (Delhi, Mumbai, Chennai, Hyderabad, Bangalore).
- **Temporal Realism**: Models "Peak-Hour Urgency" (12-2 PM, 7-10 PM) where session length and add-on probabilities drop.
- **Archetype Sampling**: Users are drawn from discrete categories: *Budget, Premium, Occasional,* and *FamilyOrder*.
- **Statistical Validation**: A built-in validation suite using $\chi^2$ tests and KL-Divergence to ensure the corpus matches ground-truth food pairings and industry benchmarks.

---

## 🏗️ Architecture: The 4-Level Hierarchy

### Level 1: User Archetype Assignment
Users are drawn from a Categorical distribution. Each archetype (e.g., *FamilyOrder*) enforces specific constraints, such as high AOV ceilings and mandatory item quantities $q_k > 1$ for bulk orders.

### Level 2: Session Intent Sampling
Sessions are conditioned on the user and the hour of the day. Intents include `Solo Lunch`, `Family Dinner`, `Late-Night Snack`, and `Weekend Brunch`.

### Level 3: Restaurant & Cuisine Selection
Uses a geographically stratified distribution. For example, Simulated users in Hyderabad will overwhelmingly favor Biryani, while Mumbai users show higher affinities for Coastal Seafood.

### Level 4: Cart Assembly via Meal Templates
Carts are filled progressively using **Meal Gap Vectors** (e.g., `{1 main, 1 side, 1 beverage}`). The system ensures that if a user adds a "Biryani" in Hyderabad, the probability of "Salan" or "Mirchi Ka Salan" being predicted as the next add-on increases significantly.

---

## 💹 Validation Suite

The system includes a `CorpusValidator` that runs three critical tests after generation:
1. **Co-occurrence $\chi^2$ Test**: Compares generated item pairings (e.g., Burger-Fries) against a ground-truth matrix.
2. **Session Length KL-Divergence**: Measures the "distance" between the generated cart size distribution and standard industry benchmarks.
3. **Template Fill Rate**: Verifies the fraction of orders that successfully completed a meal template for their specific cuisine.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd zoma-thon
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

To generate the synthetic training corpus and run the validation suite:

```bash
python main.py
```

The system will:
1. Generate 10,000 cart trajectories.
2. Print summary statistics.
3. Run the 3-point validation test.
4. Save the results to `output/cart_trajectories.csv`.

---

## 📂 Project Structure

```text
zoma-thon/
├── csao/
│   ├── config/
│   │   └── taxonomies.py       # Cuisine menus, cities, and archetype metadata
│   ├── generators/
│   │   ├── user_generator.py   # Level 1 logic
│   │   ├── session_generator.py # Level 2 logic
│   │   ├── restaurant_generator.py # Level 3 logic
│   │   └── cart_assembler.py   # Level 4 sequential assembly
│   ├── models/
│   │   └── schema.py           # Pydantic-like Dataclasses for Items/Trajectories
│   ├── modifiers/
│   │   └── realism.py          # Peak-hour, price-anchoring, and geographic boosts
│   ├── validation/
│   │   └── validator.py        # Statistical χ² and KL tests
│   └── pipeline.py             # Orchestrates the generation flow
├── output/
│   └── cart_trajectories.csv   # Final synthetic training corpus
├── main.py                     # Entry point
└── requirements.txt            # Dependencies (numpy, pandas, scipy)
```

---

## ⚖️ Realism Modifiers (Contextual Dependencies)

- **Peak-Hour Penalty**: Add-on probabilities are reduced by 15-20% during lunch/dinner rushes to simulate urgency.
- **Price Anchoring**: Sharp drops in high-price item selection if the cart current total exceeds the user's AOV tier by 20%.
- **Geographic Taste Clusters**: Explicit boosts for city-specific pairings (e.g., Hyderabadi Biryani ↔ Salan).
- **Sequence Merging**: Intelligent cart logic that aggregates quantities for identical items added at different steps in the trajectory.
