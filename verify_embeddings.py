import os
import sys

# Ensure user's local site-packages are accessible avoiding ModuleNotFoundError
sys.path.append(os.path.expanduser("~/.local/lib/python3.10/site-packages"))

import torch
import numpy as np
import pandas as pd
try:
    import plotly.express as px
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "pandas", "scikit-learn"])
    import plotly.express as px
from sklearn.decomposition import PCA

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import CSAOEngine

def load_system():
    print("\n[1/4] Loading Trained Artifacts...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = CSAOEngine(device=device)
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    engine.load_pretrained_artifacts(artifacts_dir)
    return engine

def get_base_embedding(engine, item_name, quantity=1.0):
    idx = engine.item_to_idx.get(item_name)
    if not idx:
        return None
    with torch.no_grad():
        emb = engine.neural_model.item_embed(
            torch.tensor([[idx]], device=engine.device, dtype=torch.long),
            torch.tensor([[quantity]], device=engine.device, dtype=torch.float32)
        )
    return emb[0, 0]

def verify_cosine_similarity(engine, item_a, item_b):
    """Mathematical verification of item proximity."""
    emb_a = get_base_embedding(engine, item_a)
    emb_b = get_base_embedding(engine, item_b)
    
    if emb_a is None or emb_b is None:
        return "Item not found in vocabulary."

    sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
    return sim

def verify_quantity_shift(engine, item_name):
    """Proves that your f_q(log q) * v_q logic is working."""
    engine.neural_model.eval()
    
    shifts = []
    with torch.no_grad():
        for qty in [1.0, 2.0, 5.0, 10.0]:
            emb = get_base_embedding(engine, item_name, quantity=qty)
            shifts.append((qty, emb.cpu().numpy()))
            
    # Calculate distance shift from 1x to 10x
    dist = np.linalg.norm(shifts[3][1] - shifts[0][1])
    return dist

def generate_interactive_3d_plot(engine):
    """Generates a 3D HTML plot for the judges."""
    print("\n[4/4] Extracting embeddings for 3D projection...")
    engine.neural_model.eval()
    
    items = []
    vectors = []
    categories = []
    cuisines = []
    
    # Map metadata for colors/hover info
    meta_dict = {m['item_name']: m for m in engine.item_meta}
    
    with torch.no_grad():
        for item_name, idx in engine.item_to_idx.items():
            if idx == 0: continue # Skip padding token
            emb = get_base_embedding(engine, item_name)
            vectors.append(emb.cpu().numpy())
            items.append(item_name)
            
            meta = meta_dict.get(item_name, {})
            categories.append(meta.get("category", "Unknown"))
            cuisines.append(meta.get("cuisine", "Unknown"))
            
    vectors = np.array(vectors)
    
    # Use PCA for 3D projection
    print("      Running PCA Dimensionality Reduction...")
    pca = PCA(n_components=3)
    vecs_3d = pca.fit_transform(vectors)
    
    df = pd.DataFrame({
        'Item': items,
        'Category': categories,
        'Cuisine': cuisines,
        'PCA1': vecs_3d[:, 0],
        'PCA2': vecs_3d[:, 1],
        'PCA3': vecs_3d[:, 2]
    })
    
    print("      Generating Plotly interactive graph...")
    fig = px.scatter_3d(
        df, x='PCA1', y='PCA2', z='PCA3',
        color='Category',          # Color dots by category (Mains, Sides, Desserts)
        symbol='Cuisine',          # Shape dots by Cuisine
        hover_name='Item',         # Show item name on hover
        title="CSAO Learned Latent Space (Behavioral Embeddings)",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(scene=dict(xaxis_title='Latent Dim 1', yaxis_title='Latent Dim 2', zaxis_title='Latent Dim 3'))
    
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_viz.html")
    fig.write_html(html_path)
    return html_path


if __name__ == "__main__":
    engine = load_system()
    
    print("\n[2/4] Mathematical Sanity Checks (Affinity)")
    # Test High-Affinity Pair
    sim_high = verify_cosine_similarity(engine, "Butter Chicken", "Garlic Naan")
    print(f"      Similarity (Butter Chicken <-> Garlic Naan): {sim_high:.4f} (Expected: Higher)")
    
    # Test Low-Affinity Pair
    sim_low = verify_cosine_similarity(engine, "Butter Chicken", "Filter Coffee")
    if isinstance(sim_low, str):
        print(f"      Similarity (Butter Chicken <-> Filter Coffee): {sim_low}")
    else:
        print(f"      Similarity (Butter Chicken <-> Filter Coffee): {sim_low:.4f} (Expected: Lower)")
    
    print("\n[3/4] Mathematical Sanity Checks (Quantity-Aware Shift)")
    shift_dist = verify_quantity_shift(engine, "Garlic Naan")
    print(f"      L2 Distance shift from 1x to 10x quantity: {shift_dist:.4f}")
    if shift_dist > 0.01:
        print("      -> SUCCESS: The vector mathematically mutates as quantity increases.")
    else:
        print("      -> WARNING: Quantity vector is not shifting. Check training gradients.")
        
    html_out = generate_interactive_3d_plot(engine)
    print(f"\n========================================================")
    print(f"SUCCESS! Interactive 3D visualization saved to:")
    print(f"-> {html_out}")
    print(f"Double-click this file or open it in a browser to present it to the judges!")
    print(f"========================================================")
