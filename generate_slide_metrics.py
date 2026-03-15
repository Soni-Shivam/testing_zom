import pandas as pd
import numpy as np
import ast
import json

def load_and_prep_data(filepath="artifacts/corpus_df.parquet"):
    import ast
    
    # Load the data
    try:
        df = pd.read_parquet(filepath)
    except FileNotFoundError:
        df = pd.read_csv("output/cart_trajectories.csv")
        
    print(f"[*] Available columns in dataset: {df.columns.tolist()}")
    
    # Scenario 1: The column is just named something else
    possible_list_cols = ['item_names', 'items', 'cart', 'cart_items', 'order_items']
    for col in possible_list_cols:
        if col in df.columns:
            print(f"[*] Found cart list in column: '{col}'")
            # Parse strings to lists if loaded from CSV
            if isinstance(df[col].iloc[0], str):
                df['item_names'] = df[col].apply(ast.literal_eval)
            elif col != 'item_names':
                df['item_names'] = df[col]
                
            # Ensure 'num_items' exists
            if 'num_items' not in df.columns:
                df['num_items'] = df['item_names'].apply(len)
            return df
            
    # Scenario 2: The dataframe is flattened (one row per item)
    if 'item_name' in df.columns and 'trajectory_id' in df.columns:
        print("[*] Dataset appears to be flattened. Reconstructing carts...")
        
        # Group items into lists per cart
        grouped_items = df.groupby('trajectory_id')['item_name'].apply(list).reset_index()
        grouped_items.rename(columns={'item_name': 'item_names'}, inplace=True)
        
        # Deduplicate session-level data (keep 1 row per cart)
        session_df = df.drop(columns=['item_name', 'price', 'quantity', 'category', 'step_index'], errors='ignore')
        session_df = session_df.drop_duplicates(subset=['trajectory_id'])
        
        # Merge them back together
        df = pd.merge(session_df, grouped_items, on='trajectory_id')
        df['num_items'] = df['item_names'].apply(len)
        return df
        
    raise KeyError(
        "Could not automatically find the items column. "
        "Please look at the 'Available columns' printed above and replace "
        "'item_names' in the script with the exact column name that holds your items."
    )

def generate_slide_1_geographic_lift(df):
    print("\n--- 1. Geographic Taste Clusters  ---")
    
    def get_conditional_prob(city_df, trigger_item, target_item):
        trigger_carts = city_df[city_df['item_names'].apply(lambda x: trigger_item in x)]
        if len(trigger_carts) == 0:
            return 0.0, 0
        target_carts = trigger_carts[trigger_carts['item_names'].apply(lambda x: target_item in x)]
        return len(target_carts) / len(trigger_carts), len(trigger_carts)

    hyd_df = df[df['city'] == 'Hyderabad']
    delhi_df = df[df['city'] == 'Delhi-NCR']

    trigger = 'Hyderabadi Chicken Biryani'
    target = 'Mirchi Ka Salan'

    hyd_prob, hyd_n = get_conditional_prob(hyd_df, trigger, target)
    delhi_prob, delhi_n = get_conditional_prob(delhi_df, trigger, target)

    print(f"[Hyderabad] P({target} | {trigger}): {hyd_prob:.1%} (n={hyd_n})")
    print(f"[Delhi-NCR] P({target} | {trigger}): {delhi_prob:.1%} (n={delhi_n})")
    if delhi_prob > 0:
        print(f"LIFT: Hyderabadi users are {hyd_prob/delhi_prob:.1f}x more likely to add {target}.")

def generate_slide_2_peak_hour_drop(df):
    print("\n--- 2. Peak-Hour Urgency  ---")
    # Add-on acceptance proxy: carts with > 1 item
    df['has_addon'] = df['num_items'] > 1
    
    peak_df = df[df['is_peak_hour'] == True]
    off_peak_df = df[df['is_peak_hour'] == False]
    
    peak_rate = peak_df['has_addon'].mean()
    off_peak_rate = off_peak_df['has_addon'].mean()
    
    print(f"Off-Peak Add-on Rate: {off_peak_rate:.1%}")
    print(f"Peak Lunch/Dinner Add-on Rate: {peak_rate:.1%}")
    print(f"Absolute Drop: {(off_peak_rate - peak_rate) * 100:.1f}%")

def generate_slide_3_archetypes(df):
    print("\n--- 3. Archetype Validation  ---")
    family = df[df['archetype'] == 'FamilyOrder']
    solo = df[df['archetype'] == 'Budget']
    
    print(f"[Family] Avg Cart Size: {family['num_items'].mean():.2f} items")
    print(f"[Family] Avg Cart Total: ₹{family['total_price'].mean():.2f}")
    print(f"[Budget] Avg Cart Size: {solo['num_items'].mean():.2f} items")
    print(f"[Budget] Avg Cart Total: ₹{solo['total_price'].mean():.2f}")

def generate_slide_4_template_fill(df):
    print("\n--- 4. Meal Template Fill Rate  ---")
    fill_rate = df['template_filled'].mean()
    print(f"Overall Template Fill Rate: {fill_rate:.1%}")
    
    # Breakdown by intent to show logic works
    print("\nFill Rate by Session Intent:")
    print(df.groupby('intent')['template_filled'].mean().apply(lambda x: f"{x:.1%}"))

if __name__ == "__main__":
    print("Loading synthetic dataset...")
    df = load_and_prep_data()
    
    generate_slide_1_geographic_lift(df)
    generate_slide_2_peak_hour_drop(df)
    generate_slide_3_archetypes(df)
    generate_slide_4_template_fill(df)
    
    print("\n--- 5. Statistical Validator Summary (From csao.validation.validator) ---")
    print("To get the exact KL-Divergence and Chi-Square outputs for the presentation, ")
    print("refer to the output generated by your stage 1 pipeline:")
    print("> python old_files/stage1_main.py")