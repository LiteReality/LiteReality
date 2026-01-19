# Disable xformers early to avoid compatibility issues with DINOv2
import os
os.environ["XFORMERS_DISABLED"] = "1"

# Standard library imports
import glob
import re

# Third-party imports
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import transforms
from tqdm import tqdm

# Constants
EXTS = [".png", ".jpg", ".jpeg"]
TARGET_SIZE = 392

# ───────────────── Color Naming Descriptor Setup ───────────────────
css4 = mcolors.CSS4_COLORS
names_all = list(css4.keys())
step = max(1, len(names_all)//36)
picked = names_all[::step][:36]
palette_hsv = []
for nm in picked:
    r,g,b = mcolors.to_rgb(css4[nm])
    bgr = np.uint8([[[int(b*255),int(g*255),int(r*255)]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0,0]
    palette_hsv.append(hsv)
palette_hsv = np.stack(palette_hsv, axis=0)

def extract_all_pixels(paths):
    arr = []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        arr.append(hsv.reshape(-1,3))
    return np.vstack(arr) if arr else None

def naming_descriptor(pix):
    dists = np.linalg.norm(pix[:,None,:] - palette_hsv[None,:,:], axis=2)
    idxs = dists.argmin(axis=1)
    hist = np.bincount(idxs, minlength=palette_hsv.shape[0]).astype(float)
    return hist / (hist.sum()+1e-6)

# -------------------------------
# 2. DINOv2 Feature Extractor
# -------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load DINOv2 model (xformers should be disabled via environment variable)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        # Move model to device
        self.model = self.model.to(self.device)
    def forward(self, x):
        with torch.no_grad():
            feats = self.model(x)
        return feats.view(feats.size(0), -1)

# -------------------------------
# 3. Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_feature(path, model):
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    # Move tensor to the same device as the model
    # FeatureExtractor has a device attribute, check that first
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'parameters'):
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)
    feat = model(tensor)
    return feat.squeeze().cpu().numpy()

# -------------------------------
# 4. Clustering Utility
# -------------------------------
def find_optimal_k(features, k_min=2, k_max=10):
    n = features.shape[0]
    upper = min(k_max, max(2, n-1))
    if n < 2:
        return 1
    best_k, best_score = k_min, -1
    for k in range(k_min, upper+1):
        if k >= n:
            continue
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(features)
        try:
            score = silhouette_score(features, labels, metric="cosine")
        except ValueError:
            continue
        if score > best_score:
            best_score, best_k = score, k
    return best_k

# -------------------------------
# 5. Visualization Utility
# -------------------------------

def create_collage(names, reps, labels, out_file, thumb=(200,200), margin=10):
    font = ImageFont.load_default()
    clusters = {}
    for name, rep, lbl in zip(names, reps, labels):
        clusters.setdefault(lbl, []).append((name, rep))
    rows, maxw = [], 0
    for lbl in sorted(clusters):
        thumbs = []
        for name, path in clusters[lbl]:
            img = Image.open(path).convert('RGB').resize(thumb)
            draw = ImageDraw.Draw(img)
            text = f"{name}-{lbl}"
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([0,0,tw,th], fill='black')
            draw.text((0,0), text, fill='white', font=font)
            thumbs.append(img)
        if not thumbs:
            continue
        row_w = sum(im.width for im in thumbs) + margin*(len(thumbs)-1)
        row_h = max(im.height for im in thumbs)
        row = Image.new('RGB', (row_w, row_h), 'white')
        x = 0
        for im in thumbs:
            row.paste(im, (x,0)); x += im.width + margin
        rows.append(row)
        maxw = max(maxw, row_w)
    total_h = sum(r.height for r in rows) + margin*(len(rows)-1)
    collage = Image.new('RGB', (maxw, total_h), 'white')
    y = 0
    for r in rows:
        collage.paste(r, (0,y)); y += r.height + margin
    collage.save(out_file)
    print(f"Saved collage: {out_file}")

# -------------------------------
# 6. Main Function for Chair Clustering
# -------------------------------
def cluster_chairs(scene_name, output_path=None, use_color_clustering=True):
    """
    Cluster chair images from a specified scene.

    Args:
        scene_name (str): Name of the scene (e.g., "Girton")
        output_path (str, optional): Path to save the output image.
                                     If None, will save to cache/clustering_cache/final_clustering.png
        use_color_clustering (bool): If True, use geometry+color clustering (fine-grained).
                                    If False, use geometry-only clustering (coarser).

    Returns:
        list: A list of lists, where each inner list contains chair names belonging to the same cluster.
              For example: [['Chair0', 'Chair1'], ['Chair2', 'Chair3']] represents two clusters.
              Returns None if no valid chair data is found.
    """
    # Setup paths
    scene_name = scene_name.split('/')[-1]
    base_dir = Path(f"output/object_stage/{scene_name}")
    
    # Create default output directory in cache/clustering_cache with scene_name folder
    output_dir = Path("cache/clustering_cache") / scene_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if output_path is None:
        # Use default path: final_clustering.png in cache/clustering_cache/{scene_name}
        output_path = output_dir / "final_clustering.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        # Also update output_dir to match the provided path's parent
        output_dir = output_path.parent
    
    # Define paths for the additional output images
    geom_output_path = output_dir / f"{scene_name.lower()}_chairs_geom_clusters.png"
    color_output_path = output_dir / f"{scene_name.lower()}_chairs_geom_color_clusters.png"
    
    print(f"Processing chairs from: {base_dir}")
    
    # Initialize feature extractor (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(device=device)
    if device == "cuda":
        print(f"Using GPU for feature extraction")
        # Verify model is actually on GPU
        try:
            test_param = next(extractor.model.parameters())
            if test_param.device.type != "cuda":
                print(f"Warning: Model parameters are on {test_param.device.type}, not CUDA!")
        except:
            pass
    else:
        print(f"Using CPU for feature extraction (GPU not available)")
    
    # Process all chair images from the specified directory
    feats, reps, names = [], [], []
    chair_paths = {}  # Dictionary to store all paths for each chair
    
    # Find all Chair directories
    chair_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.lower().startswith('chair'):
            chair_dirs.append(item)
    
    if not chair_dirs:
        print("No chair directories found in the specified location.")
        return None
    
    print(f"Found {len(chair_dirs)} chair directories.")
    
    # Process each chair directory
    for chair_dir in tqdm(chair_dirs, desc="Processing chair directories"):
        chair_name = chair_dir.name
        cropped_dir = chair_dir / "cropped_images"
        
        if not cropped_dir.exists() or not cropped_dir.is_dir():
            print(f"Warning: No cropped_images folder found in {chair_dir}")
            continue
        
        # Get all images for this chair
        chair_images = []
        for ext in EXTS:
            chair_images.extend(list(cropped_dir.glob(f"*{ext}")))
        
        if not chair_images:
            print(f"No images found for {chair_name}")
            continue
        
        # Store paths for later color analysis
        chair_paths[chair_name] = [str(img_path) for img_path in chair_images]
        
        # Extract features from all images for this chair
        arr = []
        successful_extractions = 0
        for img_path in chair_images:
            try:
                feat = extract_feature(str(img_path), extractor)
                if feat is not None and len(feat) > 0:
                    arr.append(feat)
                    successful_extractions += 1
            except Exception as e:
                error_msg = str(e).lower()
                # Log errors but continue - xformers should be disabled, so these shouldn't happen
                # But if they do, we'll skip this image
                if "xformers" not in error_msg and "memory_efficient_attention" not in error_msg:
                    # Only log non-xformers errors
                    print(f"Error processing {img_path}: {e}")
        
        if successful_extractions == 0 and len(chair_images) > 0:
            print(f"Warning: Failed to extract features from any images for {chair_name} (tried {len(chair_images)} images)")
        
        if arr:
            # Use mean of all image features to represent this chair
            feats.append(np.mean(arr, axis=0))
            reps.append(str(chair_images[0]))  # Use first image as representative
            names.append(chair_name)
    
    if not names:
        print("No valid chair data found.")
        return None
    
    print(f"Processed features for {len(names)} chairs.")
    
    # 1. Geometry-based clustering
    if len(names) == 1:
        geom_labels = [0]
        print("Single chair type, skipping geometry clustering.")
    else:
        feats_np = np.stack(feats)
        k_geom = find_optimal_k(feats_np)
        if k_geom < 2:
            geom_labels = [0] * len(names)
            print(f"Optimal number of geometry clusters is {k_geom}, using single cluster.")
        else:
            geom_labels = KMeans(n_clusters=k_geom, random_state=42).fit_predict(feats_np)
            print(f"Clustering chairs into {k_geom} geometry-based groups.")
    
    # Save geometry-only collage
    create_collage(names, reps, geom_labels, str(geom_output_path))
    print(f"Saved geometry-based clustering to: {geom_output_path}")

    if use_color_clustering:
        # 2. Color refinement within geometry clusters (fine-grained clustering)
        final_labels = [-1] * len(names)
        next_label = 0

        for gl in sorted(set(geom_labels)):
            # Get indices of chairs in this geometry cluster
            idxs = [i for i, l in enumerate(geom_labels) if l == gl]
            sub_names = [names[i] for i in idxs]
            sub_reps = [reps[i] for i in idxs]

            # Extract color features
            sub_feats = []
            for obj in sub_names:
                pix = extract_all_pixels(chair_paths[obj])
                if pix is not None:
                    sub_feats.append(naming_descriptor(pix))
                else:
                    # If color extraction fails, use a placeholder
                    sub_feats.append(np.zeros(len(palette_hsv)))

            # Cluster by color
            if len(sub_names) < 2:
                sub_labels = [0] * len(sub_names)
            else:
                sub_feats_np = np.stack(sub_feats)
                k_color = find_optimal_k(sub_feats_np)
                if k_color < 2:
                    sub_labels = [0] * len(sub_names)
                    print(f"  Color refinement for geometry cluster {gl}: using single color cluster")
                else:
                    sub_labels = KMeans(n_clusters=k_color, random_state=42).fit_predict(sub_feats_np)
                    print(f"  Color refinement for geometry cluster {gl}: found {k_color} color clusters")

            # Assign final labels (geometry + color)
            for idx, sl in zip(idxs, sub_labels):
                final_labels[idx] = next_label + sl
            next_label += max(sub_labels) + 1

        # Save combined geometry+color collage
        create_collage(names, reps, final_labels, str(color_output_path))
        print(f"Saved geometry+color clustering to: {color_output_path}")

        # Also save to the originally specified output path
        create_collage(names, reps, final_labels, str(output_path))

        # Organize chairs into clusters (using geometry + color for fine-grained clustering)
        clustering_labels = final_labels
        clustering_method = "geometry_color_hybrid"
    else:
        # Use geometry-only clustering
        print("Using geometry-only clustering (no color refinement)")
        create_collage(names, reps, geom_labels, str(output_path))
        clustering_labels = geom_labels
        clustering_method = "geometry_only"

    # Organize chairs into clusters
    chair_clusters = []
    unique_labels = sorted(set(clustering_labels))
    for label in unique_labels:
        cluster = [names[i] for i in range(len(names)) if clustering_labels[i] == label]
        chair_clusters.append(cluster)

    # Save clustering results
    results = {
        "scene_name": scene_name,
        "clusters": chair_clusters,
        "clustering_method": clustering_method,
        "total_objects": len(names),
        "num_clusters": len(chair_clusters),
        "cluster_sizes": [len(cluster) for cluster in chair_clusters],
        "use_color_clustering": use_color_clustering
    }

    # Save to appropriate cache file based on clustering method
    import json
    if use_color_clustering:
        # Save geometry+color clustering (for material painting optimization)
        cache_file = output_dir / "clustering_results_geom_color.json"
    else:
        # Save geometry-only clustering (for retrieval batch processing)
        cache_file = output_dir / "clustering_results.json"

    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved clustering results to: {cache_file}")

    return chair_clusters


def ensure_geometry_color_clustering(scene_name):
    """
    Ensure geometry+color clustering results exist for material painting optimization.
    This is called at the START of material painting, not during retrieval.

    Args:
        scene_name: Name of the scene

    Returns:
        dict: Clustering results with cluster assignments, or None if no valid data
    """
    import json

    scene_name = scene_name.split('/')[-1]
    cache_dir = Path("cache/clustering_cache") / scene_name
    cache_file = cache_dir / "clustering_results_geom_color.json"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded existing geometry+color clustering: {results['num_clusters']} clusters")
        return results

    # Generate geometry+color clustering if not exists
    print(f"🔄 Generating geometry+color clustering for material painting...")
    clusters = cluster_chairs(scene_name, use_color_clustering=True)

    if clusters:
        # Load the saved results
        with open(cache_file, 'r') as f:
            results = json.load(f)
        return results

    return None


def load_clustering_for_material_painting(scene_name):
    """
    Load or generate clustering results optimized for material painting.
    Prefers geometry+color clustering for fine-grained material selection.

    Args:
        scene_name: Name of the scene

    Returns:
        dict: Clustering results, or None if not available
    """
    import json

    scene_name = scene_name.split('/')[-1]
    cache_dir = Path("cache/clustering_cache") / scene_name

    # First, try geometry+color clustering (preferred for material painting)
    geom_color_file = cache_dir / "clustering_results_geom_color.json"
    if geom_color_file.exists():
        with open(geom_color_file, 'r') as f:
            return json.load(f)

    # Fallback: try geometry-only clustering
    geom_only_file = cache_dir / "clustering_results.json"
    if geom_only_file.exists():
        with open(geom_only_file, 'r') as f:
            results = json.load(f)
        print(f"⚠️  Using geometry-only clustering (geometry+color not available)")
        return results

    # Generate if nothing exists
    return ensure_geometry_color_clustering(scene_name)

# -------------------------------
# 7. Main Execution
# -------------------------------
if __name__ == '__main__':
    import sys
    
    # Default scene name
    scene_name = "BoardRoom"
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        scene_name = sys.argv[1]
    
    # Run the clustering
    chair_clusters = cluster_chairs(scene_name)
    
    # Print the clustering results
    if chair_clusters:
        print("\nChair clustering results:")
        for i, cluster in enumerate(chair_clusters):
            print(f"Cluster {i}: {cluster}")
    else:
        print("No clustering results available.")