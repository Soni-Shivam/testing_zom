# Walkthrough: Stage 2 — Advanced Feature Engineering

## What Was Built

A four-domain feature engineering pipeline with a three-tier feature store architecture producing a **171-dimensional** neural-network-ready feature vector in **~318 µs** per request.

### Modules

| File | Domain | Dims | Key Features |
|------|--------|------|-------------|
| [cart_features.py](file:///home/shivam/code-masala/zoma-thon/csao/features/cart_features.py) | Cart Aggregates | 12 | Meal gap vector, histogram, Herfindahl diversity, PAR |
| [item_features.py](file:///home/shivam/code-masala/zoma-thon/csao/features/item_features.py) | Candidate Item | 131 | 128-d SLM embedding, gap fill score, zone velocity, meal-time affinity |
| [context_features.py](file:///home/shivam/code-masala/zoma-thon/csao/features/context_features.py) | Context | 11 | Cyclical hour/day, day-type one-hot, weather proxy |
| [user_features.py](file:///home/shivam/code-masala/zoma-thon/csao/features/user_features.py) | User History | 17 | RFM triplet, cuisine preference (9-d), category acceptance (5-d) |
| [feature_store.py](file:///home/shivam/code-masala/zoma-thon/csao/features/feature_store.py) | Store | — | 3-tier: NightlyOffline, NearRealTime, OnlinePerRequest |

### Feature Store Statistics

| Tier | Entries | Latency |
|------|---------|---------|
| NightlyOfflineJob | 5,955 user vectors + 116 SLM embeddings | 4.62s batch |
| NearRealTimeJob | 558 zone-velocity entries | 0.02s |
| OnlinePerRequest | 171-dim vector | **318 µs** |

## Verification — **6/6 PASS** ✅

| Check | Result |
|-------|--------|
| Herfindahl D ∈ [0,1] | D = 0.5986 ✓ |
| Cyclical hour sin²+cos²=1 | 1.000000 ✓ |
| Cyclical day sin²+cos²=1 | 1.000000 ✓ |
| SLM embedding ‖v‖₂=1 | 1.000000 ✓ |
| PAR ≥ 0 | 2.8584 ✓ |
| Cuisine pref Σ=1 | 1.000000 ✓ |

## How to Run

```bash
cd /home/shivam/code-masala/zoma-thon && python stage2_main.py
```
