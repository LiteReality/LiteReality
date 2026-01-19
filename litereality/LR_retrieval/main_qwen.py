#!/usr/bin/env python3
"""
3D Asset Retrieval Pipeline

This script processes images with bounding boxes, retrieves matching 3D assets from a database,
and selects the best match based on image similarity and LLM-based selection.

Usage:
    python main.py --name <path_to_camera_data>

Environment variables:
    OPENAI_API_KEY: Your OpenAI API key
"""

# Standard library imports
import argparse
import glob
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoProcessor, AutoModel

# Local imports
from identical_clustering import cluster_chairs
from qwan import load_model as qwen_load_model, unload_model as qwen_unload_model
from utils import (
    find_subfolder,
    multi_try_ensure_response_qwen,
    multi_try_ensure_response_qwen_multi,
    multi_try_ensure_response_qwen_multi_combined,
    prompt_for_categories,
    Prompt_final_select,
    Prompt_final_select_multi,
    two_stage_category_retrieval,
)

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# =====================================================================
# Configuration and Utility Functions
# =====================================================================

def setup_environment():
    """Setup environment variables and configuration"""
    # Load Qwen3-VL model at startup
    model_start = time.time()
    print(f"  {Colors.BLUE}🤖{Colors.RESET} Loading Qwen3-VL model...")
    qwen_load_model()  # Model path configurable via QWEN_MODEL_PATH env var
    model_elapsed = time.time() - model_start
    print(f"  {Colors.GREEN}✓{Colors.RESET} Model loaded successfully ({Colors.CYAN}{model_elapsed:.2f}s{Colors.RESET})")
    
    # Configuration (api_key kept for compatibility but not used with Qwen)
    config = {
        "api_key": None,  # Not needed for Qwen3-VL
        "data_base_path": "litereality_database/3D_assets/",
        "json_file": "litereality_database/3D_assets/tree.json",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return config


def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

# =====================================================================
# Image Processing Functions
# =====================================================================

def process_images(input_folder):
    """
    Process images by drawing bounding boxes and cropping objects
    
    Args:
        input_folder: Path to the input folder containing images and bbox_info.json
        
    Returns:
        tuple: (stitched_image_path, semantic_category)
    """
    start_time = time.time()
    images_folder = os.path.join(input_folder, "images")
    bbox_file = os.path.join(input_folder, "bbox_info_updated.json")
    
    if not os.path.exists(bbox_file):
        print(f"{Colors.RED}❌{Colors.RESET} Bounding box file not found: {bbox_file}")
        return None, None
    
    # Load bounding box information
    with open(bbox_file, "r") as f:
        bbox_info = json.load(f)
    
    # Create output folder structure
    output_folder = input_folder.replace("input", "output")
    bbox_output_folder = create_directory(os.path.join(output_folder, "bbox_images"))
    cropped_output_folder = create_directory(os.path.join(output_folder, "cropped_images"))
    
    stitched_images = []
    
    semantic = bbox_info.get("semantic", "Unknown")
    
    # Process images
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    count = 0
    
    for image_path in image_files:
        image_name = os.path.basename(image_path).split('.')[0]
        if image_name not in bbox_info:
            continue

        bbox = bbox_info[image_name]
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Draw bounding box (red, thicker)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
        
        # Add padding (10px white border)
        image_padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Crop the bounding box region
        x1, y1, x2, y2 = bbox
        cropped_image = image[y1:y2, x1:x2]
        
        # Rotate 90 deg clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        image_padded = cv2.rotate(image_padded, cv2.ROTATE_90_CLOCKWISE)
        
        # Save processed images
        bbox_image_path = os.path.join(bbox_output_folder, os.path.basename(image_path))
        cropped_image_path = os.path.join(cropped_output_folder, os.path.basename(image_path))
        
        # Reduce size to half height and width for visualization images
        h, w = image.shape[:2]
        image_vis = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(bbox_image_path, image_vis)
        
        cv2.imwrite(cropped_image_path, cropped_image)
        
        # Collect for stitching
        stitched_images.append(image_padded)
        count += 1
        
        if count >= 4:  # Process only the first four images
            break

   # Ensure we have 4 images by repeating the last one if needed
        while len(stitched_images) < 4:
            stitched_images.append(stitched_images[-1])

    # Now stitch and save
    h1 = np.hstack((stitched_images[0], stitched_images[1]))  # Top row
    h2 = np.hstack((stitched_images[2], stitched_images[3]))  # Bottom row
    stitched_image = np.vstack((h1, h2))  # Final stitched image

    stitched_image_path = os.path.join(output_folder, "stitched_image.jpg")
    cv2.imwrite(stitched_image_path, stitched_image)
    
    elapsed = time.time() - start_time
    obj_name = os.path.basename(input_folder)
    print(f"  {Colors.GREEN}✓{Colors.RESET} {obj_name} → stitched_image.jpg ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})")
    
    return stitched_image_path, semantic

# =====================================================================
# DINOv2 Image Feature Extraction and Similarity
# =====================================================================

def preprocess_image(image_path, device):
    """Preprocess image for DINOv2 model"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def get_image_features(image_path, model, device):
    """Extract DINOv2 features from image"""
    image = preprocess_image(image_path, device)
    with torch.no_grad():
        features = model(image).last_hidden_state.mean(dim=1)  # Global average pooling
        # Ensure features have the right dimensions
        features = features.squeeze().unsqueeze(0)  # Make sure it's [1, feature_dim]
        # Normalize the features
        features = features / features.norm(dim=1, keepdim=True)
    return features

def compute_average_similarity(query_folder, image_list_paths, model, device, top_k=10):
    """
    Compute average similarity scores between query images and database images
    
    Args:
        query_folder: Folder containing query images
        image_list_paths: List of database image paths
        model: Pre-trained DINOv2 model
        device: Computation device (cuda or cpu)
        top_k: Number of top results to return
        
    Returns:
        list: Top k results with (image_path, similarity_score)
    """
    frame_images = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith(('.jpg', '.png'))]
    
    if not frame_images:
        return []
    
    # Extract dataset folder from image paths
    if image_list_paths:
        first_image = image_list_paths[0]
        # Get the parent directory of the parent directory (UUID folder)
        dataset_folder = os.path.dirname(os.path.dirname(first_image))
        
        # Try to load pre-computed features
        precomputed_features = load_precomputed_features(dataset_folder)
        using_precomputed = len(precomputed_features) > 0
    else:
        precomputed_features = {}
        using_precomputed = False
    
    all_scores = []
    for i, frame_image in enumerate(frame_images):
        # Get query image features - always computed on the fly
        query_features = get_image_features(frame_image, model, device)
        
        scores = []
        for j, img_path in enumerate(image_list_paths):
            # Use pre-computed features if available, otherwise compute on-the-fly
            if using_precomputed and img_path in precomputed_features:
                image_features = precomputed_features[img_path].to(device)
                # Ensure correct dimension for precomputed features
                if len(image_features.shape) == 1:
                    image_features = image_features.unsqueeze(0)
            else:
                image_features = get_image_features(img_path, model, device)
            
            # Ensure both tensors have the same shape before computing similarity
            if query_features.shape != image_features.shape:
                # Adjust dimensions if needed
                if len(query_features.shape) == 1:
                    query_features = query_features.unsqueeze(0)
                if len(image_features.shape) == 1:
                    image_features = image_features.unsqueeze(0)
            
            # Compute cosine similarity between normalized vectors
            similarity = torch.nn.functional.cosine_similarity(query_features, image_features, dim=1)
            scores.append(similarity.item())
        
        all_scores.append(scores)
    
    avg_scores = np.mean(np.array(all_scores), axis=0)  # Compute mean across all frame images
    ranked_indices = np.argsort(avg_scores)[::-1]  # Sort indices based on similarity scores in descending order
    
    top_results = [(image_list_paths[i], avg_scores[i]) for i in ranked_indices[:top_k]]
    
    return top_results

def get_image_paths(dataset_folder):
    """Get all image paths in a dataset folder"""
    image_paths = []
    
    for uuid_folder in os.listdir(dataset_folder):
        uuid_path = os.path.join(dataset_folder, uuid_folder)
        
        if os.path.isdir(uuid_path):
            image_path = os.path.join(uuid_path, "image.jpg")
            if os.path.exists(image_path):
                image_paths.append(image_path)
    
    return image_paths

def load_precomputed_features(dataset_folder):
    """
    Load pre-computed DINOv2 features for images in the dataset folder
    
    Args:
        dataset_folder: Path to the dataset folder containing images
        
    Returns:
        dict: Dictionary mapping image paths to their precomputed features
    """
    
    # Get the category structure from the path 
    # e.g., "litereality_database/3D_assets/Chair/Dining_Chair" -> "Chair/Dining_Chair"
    relative_path = os.path.relpath(dataset_folder, "litereality_database/3D_assets")
    feature_folder = os.path.join("litereality_database/3D_assets_features", relative_path)
    
    if not os.path.exists(feature_folder):
        return {}
    
    image_features = {}
    feature_count = 0
    
    # Get all the images to create mapping
    image_paths = get_image_paths(dataset_folder)
    
    # Process each image and find its corresponding feature file
    for image_path in image_paths:
        # Extract UUID from image path
        uuid = os.path.basename(os.path.dirname(image_path))
        feature_path = os.path.join(feature_folder, f"{uuid}.npy")
        
        if os.path.exists(feature_path):
            try:
                # Load the pre-computed feature
                feature = np.load(feature_path)
                image_features[image_path] = torch.from_numpy(feature)
                feature_count += 1
            except Exception as e:
                pass
    
    return image_features


# =====================================================================
# Visualization Functions
# =====================================================================

def merge_selection_and_stitching(stitched_image_path, obj_paths, obj_name):
    """
    Merge stitched image with top selection candidates
    
    Args:
        stitched_image_path: Path to the stitched image
        obj_paths: List of paths to top candidate images
        obj_name: Name of the object
        
    Returns:
        str: Path to the final combined image
    """
    
    # Load stitched image
    stitched_image = Image.open(stitched_image_path)
    top_four_images = [Image.open(path.strip()) for path in obj_paths]  # Strip whitespace
    
    # Get stitched image dimensions
    stitched_width, stitched_height = stitched_image.size

    # Compute size for each image in the 2x2 grid (half of stitched image)
    grid_width = stitched_width // 2
    grid_height = stitched_height // 2

    # Resize images
    top_four_images = [img.resize((grid_width, grid_height)) for img in top_four_images]

    # Create a blank image for the 2x2 grid
    grid_image = Image.new("RGB", (stitched_width, stitched_height))

    # Try loading a large, legible font; fall back if unavailable
    # Make the index unmissable: big number + black background, placed top-left.
    label_font_size = max(120, min(grid_width, grid_height) // 3)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", label_font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", label_font_size)
        except Exception:
            font = ImageFont.load_default()

    # Arrange images in a 2x2 grid
    grid_positions = [(0, 0), (grid_width, 0), (0, grid_height), (grid_width, grid_height)]
    draw = ImageDraw.Draw(grid_image)
    for idx, (img, pos) in enumerate(zip(top_four_images, grid_positions), start=1):
        grid_image.paste(img, pos)

        # Draw index label at top-left with high-contrast background
        label = str(idx)
        pad = 18
        x0 = pos[0] + 18
        y0 = pos[1] + 18
        # textbbox returns (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2], fill="black")
        draw.text((x0 + pad - bbox[0], y0 + pad - bbox[1]), label, font=font, fill="white")

    # Create final combined image (stitched + 2x2 grid side by side)
    final_width = stitched_width * 2
    final_height = stitched_height

    final_image = Image.new("RGB", (final_width, final_height))
    final_image.paste(stitched_image, (0, 0))
    final_image.paste(grid_image, (stitched_width, 0))
    
    # Save the final image
    output_path = f"{obj_name}/final_combined_image.jpg".replace("input", "output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_image.save(output_path)
    
    return output_path

def create_retrieval_cache_image(obj_path, scene_name, obj_name):
    """
    Create a combined image for retrieval cache showing:
    - Left: query image (stitched_image.jpg or final_combined_image.jpg)
    - Right: selected_obj/image.jpg (selected result)

    Args:
        obj_path: Path to the object folder (input path)
        scene_name: Name of the scene (e.g., "Girton")
        obj_name: Name of the object (e.g., "Chair0")

    Returns:
        str: Path to the saved cache image, or None if failed
    """
    try:
        # Extract scene_name from obj_path if not provided
        if not scene_name:
            # Try to extract from obj_path: input/object_stage/{scene_name}/...
            path_parts = obj_path.split(os.sep)
            if "object_stage" in path_parts:
                idx = path_parts.index("object_stage")
                if idx + 1 < len(path_parts):
                    scene_name = path_parts[idx + 1]

        if not scene_name:
            print(f"Warning: Could not determine scene_name for {obj_name}")
            return None

        # Construct paths
        output_obj_path = obj_path.replace("input", "output")
        selected_obj_image_path = os.path.join(output_obj_path, "selected_obj", "image.jpg")

        # Check if selected object image exists
        if not os.path.exists(selected_obj_image_path):
            return None

        # Try to find query image - prefer final_combined_image.jpg (single-image mode),
        # fall back to stitched_image.jpg (multi-image mode)
        final_combined_path = os.path.join(output_obj_path, "final_combined_image.jpg")
        stitched_image_path = os.path.join(output_obj_path, "stitched_image.jpg")

        query_image_path = None
        if os.path.exists(final_combined_path):
            query_image_path = final_combined_path  # Single-image mode
        elif os.path.exists(stitched_image_path):
            query_image_path = stitched_image_path  # Multi-image mode
        else:
            print(f"Warning: No query image found for {obj_name}")
            return None

        # Load images
        img1 = Image.open(query_image_path).convert("RGB")
        img2 = Image.open(selected_obj_image_path).convert("RGB")
        
        # Resize to 1/4 of original size
        new_size1 = (img1.size[0] // 4, img1.size[1] // 4)
        new_size2 = (img2.size[0] // 4, img2.size[1] // 4)
        
        img1_resized = img1.resize(new_size1, Image.Resampling.LANCZOS)
        img2_resized = img2.resize(new_size2, Image.Resampling.LANCZOS)
        
        # Make both images the same height (use the smaller height)
        min_height = min(new_size1[1], new_size2[1])
        
        # Resize both to same height, maintaining aspect ratio
        aspect1 = new_size1[0] / new_size1[1]
        aspect2 = new_size2[0] / new_size2[1]
        
        new_width1 = int(min_height * aspect1)
        new_width2 = int(min_height * aspect2)
        
        img1_final = img1_resized.resize((new_width1, min_height), Image.Resampling.LANCZOS)
        img2_final = img2_resized.resize((new_width2, min_height), Image.Resampling.LANCZOS)
        
        # Combine side by side
        total_width = new_width1 + new_width2
        combined_image = Image.new("RGB", (total_width, min_height))
        combined_image.paste(img1_final, (0, 0))
        combined_image.paste(img2_final, (new_width1, 0))
        
        # Save to cache with scene_name folder
        cache_dir = os.path.join("cache", "retrieval_cache", scene_name)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{obj_name}.jpg")
        combined_image.save(cache_path, quality=85)
        
        return cache_path
        
    except Exception as e:
        print(f"Warning: Failed to create retrieval cache for {obj_name}: {e}")
        return None

# =====================================================================
# Main Processing Pipeline
# =====================================================================

def process_object(obj_path, config, stitched_image_path, semantic, use_multi_image_selection=True):
    """
    Process a single object folder
    
    Args:
        obj_path: Path to the object folder
        config: Configuration dictionary
        stitched_image_path: Path to the stitched image
        semantic: Semantic category of the object
        use_multi_image_selection: If True, use multi-image selection approach (default: True)
        
    Returns:
        dict: Log information for reporting
    """
    obj_name = os.path.basename(obj_path)
    obj_start_time = time.time()
    
    # Check if this object has already been processed
    output_obj_folder = f"{obj_path}/selected_obj".replace("input", "output")
    if os.path.exists(output_obj_folder) and os.listdir(output_obj_folder):
        return {
            "status": "already_processed",
            "output_path": output_obj_folder
        }
    
    log_for_report = {}
    
    try:
        # Clean semantic category
        semantic = ''.join([i for i in semantic if not i.isdigit()])
        log_for_report["stitched_image_path"] = stitched_image_path
        
        # Special case for floors
        if semantic == "Floor":
            print(f"Processing {obj_name}")
            print(f"  {Colors.YELLOW}⏩{Colors.RESET} Using predefined floor assets")
            floor_output = f"{obj_path}/selected_obj".replace("input", "output")
            create_directory(floor_output)
            shutil.copytree("litereality_database/3D_assets/floor", floor_output, dirs_exist_ok=True)
            elapsed = time.time() - obj_start_time
            print(f"  {Colors.GREEN}✓{Colors.RESET} Copied to selected_obj/ ({Colors.CYAN}{elapsed:.1f}s{Colors.RESET})")
            log_for_report["select_obj_image"] = "litereality_database/3D_assets/floor"
            return (log_for_report, floor_output)
        
        # Step 2: Category retrieval using LLM (Qwen3-VL) - Two-stage approach
        save_folder = f"{obj_path}/bbox_images".replace("input", "output")
        create_directory(save_folder)
        
        # Load JSON data for category tree
        with open(config["json_file"], "r") as f:
            json_data = json.load(f)
        
        # Use two-stage category retrieval
        cat_start = time.time()
        final_category, stage1_response, stage2_description, stage2_response, retrieval_info = two_stage_category_retrieval(
            stitched_image_path, semantic, json_data, save_folder, max_tries=5, Force=False
        )
        
        if not final_category:
            print(f"{Colors.RED}❌{Colors.RESET} Failed to get response from LLM for category retrieval. Skipping {obj_name}.")
            return None
        
        dataset_folder = find_subfolder(config["data_base_path"], final_category)
        cat_elapsed = time.time() - cat_start
        
        # Log retrieval information
        log_for_report["LLM_retrieval_method"] = "two_stage"
        log_for_report["LLM_retrieval_stage1_prompt"] = retrieval_info.get("stage1_prompt")
        log_for_report["LLM_retrieval_stage1_response"] = retrieval_info.get("stage1_response")
        log_for_report["LLM_retrieval_stage2_description_prompt"] = retrieval_info.get("stage2_description_prompt")
        log_for_report["LLM_retrieval_stage2_description"] = retrieval_info.get("stage2_description")
        log_for_report["LLM_retrieval_stage2_match_prompt"] = retrieval_info.get("stage2_match_prompt")
        log_for_report["LLM_retrieval_stage2_response"] = retrieval_info.get("stage2_response")
        log_for_report["LLM_retrieval_final_category"] = final_category
        log_for_report["LLM_retrieval_time_seconds"] = cat_elapsed
        
        if not dataset_folder:
            print(f"Processing {obj_name}")
            print(f"  {Colors.YELLOW}⚠️{Colors.RESET} Category '{final_category}' not found in database. Skipping {obj_name}.")
            return None
        
        category_name = os.path.basename(dataset_folder)
        print(f"Processing {obj_name}")
        print(f"  {Colors.BLUE}🤖{Colors.RESET} Category retrieval → {category_name} ({Colors.CYAN}{cat_elapsed:.1f}s{Colors.RESET})")
        
        # Step 3: Object selection using DINOv2
        dino_start = time.time()
        processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(config["device"])
        
        template_asset_image_paths = get_image_paths(dataset_folder)
        query_folder = f"{obj_path}/cropped_images".replace("input", "output")
        create_directory(query_folder)
        
        # Get top 10 results for visualization
        top_10_results = compute_average_similarity(query_folder, template_asset_image_paths, model, config["device"], 10)
        dino_elapsed = time.time() - dino_start
        
        # Extract top 4 for LLM selection (keep this unchanged)
        top_4_results = top_10_results[:4]
        obj_paths = [img_path for img_path, _ in top_4_results]
        
        # Save both top 4 (for backward compatibility) and top 10 (for visualization)
        log_for_report["DINOv2_top_4_results"] = top_4_results
        log_for_report["DINOv2_top_10_results"] = top_10_results
        log_for_report["DINOv2_time_seconds"] = dino_elapsed
        
        top_score = top_10_results[0][1] if top_10_results else 0.0
        print(f"  {Colors.BLUE}🔍{Colors.RESET} Object selection → Top match: {Colors.CYAN}{top_score:.4f}{Colors.RESET} ({Colors.CYAN}{dino_elapsed:.1f}s{Colors.RESET})")
        
        # Check if we have any candidate objects
        if not obj_paths:
            print(f"  {Colors.YELLOW}⚠️{Colors.RESET} No candidate objects found from DINOv2. Skipping {obj_name}.")
            return None
        
        # Step 4: LLM for final selection (Qwen3-VL)
        save_folder = f"{obj_path}".replace("input", "output")
        
        llm_start = time.time()
        
        if use_multi_image_selection:
            # Multi-image approach: pass stitched image + top 4 candidates separately
            # Need at least 1 candidate (plus stitched image = 2 total minimum)
            if len(obj_paths) < 1:
                print(f"  {Colors.YELLOW}⚠️{Colors.RESET} Not enough candidates for multi-image selection. Falling back to single-image approach.")
                use_multi_image_selection = False
            
        if use_multi_image_selection:
            image_paths = [stitched_image_path] + obj_paths[:4]
            result = multi_try_ensure_response_qwen_multi_combined(image_paths, None, Prompt_final_select_multi, save_folder, num_votes=3, Force=True)
            if not result:
                print(f"{Colors.RED}❌{Colors.RESET} Failed to get response from LLM for final selection. Skipping {obj_name}.")
                return None
            
            # Handle tuple return (message, stage_info)
            if isinstance(result, tuple) and len(result) == 2:
                message, stage_info = result
            else:
                # Backward compatibility: if it's just a string, wrap it
                message = result
                stage_info = {}
            
            if not message:
                print(f"{Colors.RED}❌{Colors.RESET} Failed to get response from LLM for final selection. Skipping {obj_name}.")
                return None
            llm_elapsed = time.time() - llm_start
            
            log_for_report["LLM_selection_method"] = "multi_image_combined"
            log_for_report["LLM_selection_prompt"] = "Reference Description + Candidate Descriptions + Voting"
            log_for_report["LLM_selection_images"] = image_paths
            log_for_report["LLM_selection_message"] = message
            log_for_report["LLM_selection_time_seconds"] = llm_elapsed
            # Add detailed stage information
            if stage_info:
                log_for_report["LLM_selection_stage1_description"] = stage_info.get("stage1_description")
                log_for_report["LLM_selection_stage1_prompt"] = stage_info.get("stage1_prompt")
                log_for_report["LLM_selection_stage2a_candidate_descriptions"] = stage_info.get("stage2a_candidate_descriptions", {})
                log_for_report["LLM_selection_stage2a_prompts"] = stage_info.get("stage2a_prompts", {})
                log_for_report["LLM_selection_stage2b_votes"] = stage_info.get("stage2b_votes", [])
                log_for_report["LLM_selection_stage2b_responses"] = stage_info.get("stage2b_responses", [])
                log_for_report["LLM_selection_stage2b_prompt"] = stage_info.get("stage2b_prompt")
                log_for_report["LLM_selection_final_candidate"] = stage_info.get("final_candidate")
                # Backward compatibility: keep old field names for stage2 (pointing to stage2b)
                log_for_report["LLM_selection_stage2_votes"] = stage_info.get("stage2b_votes", [])
                log_for_report["LLM_selection_stage2_responses"] = stage_info.get("stage2b_responses", [])
                log_for_report["LLM_selection_stage2_prompt"] = stage_info.get("stage2b_prompt")
        else:
            # Single-image approach: create combined image (original method)
            output_path = merge_selection_and_stitching(stitched_image_path, obj_paths, obj_path)
            message = multi_try_ensure_response_qwen(output_path, None, Prompt_final_select, save_folder, Force=True)
            if not message:
                print(f"{Colors.RED}❌{Colors.RESET} Failed to get response from LLM for final selection. Skipping {obj_name}.")
                return None
            llm_elapsed = time.time() - llm_start
            
            log_for_report["LLM_selection_method"] = "single_image"
            log_for_report["LLM_selection_prompt"] = Prompt_final_select
            log_for_report["LLM_selection_image"] = output_path
            log_for_report["LLM_selection_message"] = message
            log_for_report["LLM_selection_time_seconds"] = llm_elapsed
        
        print(f"  {Colors.BLUE}🎯{Colors.RESET} Final selection → Template #{message} ({Colors.CYAN}{llm_elapsed:.1f}s{Colors.RESET})")
        
        # Step 5: Save results
        try:
            selected_index = int(message) - 1
            if selected_index < 0 or selected_index >= len(obj_paths):
                print(f"  {Colors.YELLOW}⚠️{Colors.RESET} Invalid selection ({selected_index + 1} > {len(obj_paths)}), using top match instead")
                selected_index = 0
                
            select_obj_image = obj_paths[selected_index]
            log_for_report["select_obj_image"] = select_obj_image
            
            select_obj_image_folder = os.path.dirname(select_obj_image)
            output_obj_folder = f"{obj_path}/selected_obj".replace("input", "output")
            
            save_start = time.time()
            shutil.copytree(select_obj_image_folder, output_obj_folder, dirs_exist_ok=True)
            save_elapsed = time.time() - save_start

            print(f"  {Colors.BLUE}💾{Colors.RESET} Saved to selected_obj/ ({Colors.CYAN}{save_elapsed:.1f}s{Colors.RESET})")
            
            total_elapsed = time.time() - obj_start_time
            log_for_report["total_processing_time_seconds"] = total_elapsed
            print(f"  {Colors.GREEN}✓{Colors.RESET} {obj_name} complete ({Colors.CYAN}{total_elapsed:.1f}s{Colors.RESET})")
            
            return (log_for_report, output_obj_folder)
            
        except ValueError:
            print(f"{Colors.RED}❌{Colors.RESET} Invalid selection value from LLM: {message}. Expected a number.")
            return None
            
    except Exception as e:
        print(f"{Colors.RED}❌{Colors.RESET} Error processing object {obj_name}: {type(e).__name__}: {e}")
        return None


def process_chair_cluster(cluster, config, stitched_image_path_list, semantic_list, all_objs, scene_name, use_multi_image_selection=True):
    """
    Process a cluster of chair objects together
    
    Args:
        cluster: List of chair names in the cluster (e.g., ["Chair0", "Chair2"])
        config: Configuration dictionary
        stitched_image_path_list: List of stitched image paths for all objects
        semantic_list: List of semantic categories for all objects
        all_objs: List of all object paths
        scene_name: Name of the scene (e.g., "scene_3")
        use_multi_image_selection: If True, use multi-image selection approach (default: True)
        
    Returns:
        list: List of object indices that have been processed
    """
    if not cluster:
        return []
    
    # Map chair names to their indices in all_objs
    chair_indices = []
    chair_paths = {}
    
    for i, obj_path in enumerate(all_objs):
        obj_name = os.path.basename(obj_path)
        if obj_name in cluster:
            chair_indices.append(i)
            chair_paths[obj_name] = i
    
    if not chair_indices:
        return []
    
    # Check if all chairs in this cluster have already been processed
    processed_count = 0
    for chair_name in cluster:
        if chair_name not in chair_paths:
            continue
            
        idx = chair_paths[chair_name]
        obj_path = all_objs[idx]
        output_obj_folder = f"{obj_path}/selected_obj".replace("input", "output")
        if os.path.exists(output_obj_folder) and os.listdir(output_obj_folder):
            processed_count += 1
    
    if processed_count == len(cluster):
        return chair_indices
    
    # Load JSON data for category tree
    with open(config["json_file"], "r") as f:
        json_data = json.load(f)
    
    # Step 1: Category retrieval using LLM (Qwen3-VL) - sample max 6 chairs for voting
    print(f"  {Colors.BLUE}🤖{Colors.RESET} Category retrieval (LLM)")

    # Select subset of chairs for category voting (same logic as candidate selection)
    cluster_size = len(cluster)
    num_voters = max(3, cluster_size // 3)  # At least 3, or 1/3 of cluster
    num_voters = min(num_voters, 6)         # Max 6 voters

    # Select evenly distributed voters for diversity
    if cluster_size <= num_voters:
        voting_chairs = cluster  # Use all if small cluster
    else:
        # Select evenly spaced chairs for diversity
        indices = np.linspace(0, cluster_size - 1, num_voters, dtype=int)
        voting_chairs = [cluster[i] for i in indices]

    print(f"    Selected {len(voting_chairs)}/{cluster_size} chairs for category voting: {voting_chairs}")

    categories = {}  # Dictionary to store category votes
    llm_retrieval_data = []  # Store prompts and responses for each chair
    total_retrieval_time = 0.0

    for chair_name in voting_chairs:
        if chair_name not in chair_paths:
            continue
            
        idx = chair_paths[chair_name]
        obj_path = all_objs[idx]
        stitched_image_path = stitched_image_path_list[idx]
        semantic = semantic_list[idx]
        
        # Clean semantic category
        semantic = ''.join([i for i in semantic if not i.isdigit()])
        
        # Skip if no stitched image
        if not stitched_image_path:
            continue
        
        save_folder = f"{obj_path}/bbox_images".replace("input", "output")
        create_directory(save_folder)
        
        # Use two-stage category retrieval
        cat_start = time.time()
        final_category, stage1_response, stage2_description, stage2_response, retrieval_info = two_stage_category_retrieval(
            stitched_image_path, semantic, json_data, save_folder, max_tries=5, Force=False
        )
        
        if not final_category:
            continue
        
        cat_elapsed = time.time() - cat_start
        total_retrieval_time += cat_elapsed
        
        # Store retrieval data for this chair
        llm_retrieval_data.append({
            "chair_name": chair_name,
            "method": "two_stage",
            "stage1_prompt": retrieval_info.get("stage1_prompt"),
            "stage1_response": retrieval_info.get("stage1_response"),
            "stage2_description_prompt": retrieval_info.get("stage2_description_prompt"),
            "stage2_description": retrieval_info.get("stage2_description"),
            "stage2_match_prompt": retrieval_info.get("stage2_match_prompt"),
            "stage2_response": retrieval_info.get("stage2_response"),
            "final_category": final_category,
            "time_seconds": cat_elapsed
        })
        
        # Add to category votes
        category = final_category
        categories[category] = categories.get(category, 0) + 1
        
        dataset_folder = find_subfolder(config["data_base_path"], category)
        category_name = os.path.basename(dataset_folder) if dataset_folder else category
        print(f"    {Colors.GREEN}✓{Colors.RESET} {chair_name} → {category_name} ({Colors.CYAN}{cat_elapsed:.1f}s{Colors.RESET})")
    
    if not categories:
        return []
    
    # Find the most voted categories
    max_votes = max(categories.values())
    top_categories = [cat for cat, votes in categories.items() if votes == max_votes]
    
    # Find dataset folders for all top categories
    dataset_folders = []
    for category in top_categories:
        dataset_folder = find_subfolder(config["data_base_path"], category)
        if dataset_folder:
            dataset_folders.append(dataset_folder)
    
    if not dataset_folders:
        print(f"    {Colors.YELLOW}⚠️{Colors.RESET} No valid dataset folders found for categories: {top_categories}")
        return []
    
    category_name = os.path.basename(dataset_folders[0])
    print(f"    → Selected category: {category_name} ({max_votes} votes)")
    
    # Step 2: Object selection using DINOv2
    print(f"  {Colors.BLUE}🔍{Colors.RESET} Object selection (DINOv2)")
    
    # Load pre-trained DINOv2 model
    dino_start = time.time()
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(config["device"])
    print(f"    {Colors.GREEN}✓{Colors.RESET} Loading model... ({Colors.CYAN}{time.time() - dino_start:.1f}s{Colors.RESET})")
    
    # Select candidate chairs for this cluster using the specified logic:
    # At least 3 candidates, or 1/3 of cluster size (whichever is larger), max 6
    cluster_size = len(cluster)
    num_candidates = max(3, cluster_size // 3)  # At least 3, or 1/3 of cluster
    num_candidates = min(num_candidates, 6)     # Max 6 candidates

    # Select candidates evenly distributed across the cluster
    if cluster_size <= num_candidates:
        selected_chairs = cluster  # Use all if small cluster
    else:
        # Select evenly spaced candidates for diversity
        indices = np.linspace(0, cluster_size - 1, num_candidates, dtype=int)
        selected_chairs = [cluster[i] for i in indices]

    print(f"    Selected {len(selected_chairs)}/{cluster_size} candidates: {selected_chairs}")

    # Collect cropped images from selected candidate chairs only
    all_query_images = []
    for chair_name in selected_chairs:
        if chair_name not in chair_paths:
            continue

        idx = chair_paths[chair_name]
        obj_path = all_objs[idx]
        query_folder = f"{obj_path}/cropped_images".replace("input", "output")
        if os.path.exists(query_folder):
            chair_images = [os.path.join(query_folder, f) for f in os.listdir(query_folder)
                           if f.endswith(('.jpg', '.png'))]
            all_query_images.extend(chair_images)
    
    if not all_query_images:
        return []
    
    # Create a temporary directory to store all query images
    temp_query_dir = os.path.join(os.path.dirname(all_objs[0]).replace("input", "output"), "temp_cluster_query")
    create_directory(temp_query_dir)
    
    # Copy all query images to the temporary directory
    for img_path in all_query_images:
        shutil.copy(img_path, os.path.join(temp_query_dir, os.path.basename(img_path)))
    
    # Get all template asset image paths from all top category folders
    all_template_asset_image_paths = []
    for dataset_folder in dataset_folders:
        template_paths = get_image_paths(dataset_folder)
        all_template_asset_image_paths.extend(template_paths)
    
    # Compute similarity using all query images against all template images
    sim_start = time.time()
    top_10_results = compute_average_similarity(temp_query_dir, all_template_asset_image_paths, model, config["device"], 10)
    sim_elapsed = time.time() - sim_start
    
    # Clean up temporary directory
    shutil.rmtree(temp_query_dir, ignore_errors=True)
    
    if not top_10_results:
        return []
    
    # Extract top 4 for LLM selection
    top_4_results = top_10_results[:4]
    obj_paths = [img_path for img_path, _ in top_4_results]
    
    # Check if we have any candidate objects
    if not obj_paths:
        print(f"    {Colors.YELLOW}⚠️{Colors.RESET} No candidate objects found from DINOv2. Skipping cluster.")
        return []
    
    print(f"    {Colors.GREEN}✓{Colors.RESET} Computing similarity: {len(all_query_images)} queries × {len(all_template_asset_image_paths)} templates ({Colors.CYAN}{sim_elapsed:.1f}s{Colors.RESET})")
    print(f"    → Top 10 candidates identified, using top 4 for selection")
    
    # Step 3: LLM for final selection on candidate chairs only
    print(f"  {Colors.BLUE}🎯{Colors.RESET} Final selection (LLM) on {len(voting_chairs)} candidate chairs")
    
    selection_votes = {}  # Dictionary to store selection votes
    llm_selection_data = []  # Store prompts and responses for each chair
    total_selection_time = 0.0
    
    # Check if we have enough candidates for multi-image selection
    use_multi_for_cluster = use_multi_image_selection and len(obj_paths) >= 1
    
    # Only process the selected candidate chairs for LLM final selection
    for chair_name in voting_chairs:
        if chair_name not in chair_paths:
            continue

        idx = chair_paths[chair_name]
        obj_path = all_objs[idx]
        stitched_image_path = stitched_image_path_list[idx]

        # Skip if no stitched image
        if not stitched_image_path:
            continue
        
        save_folder = f"{obj_path}".replace("input", "output")
        
        llm_start = time.time()
        
        if use_multi_for_cluster:
            # Multi-image approach: pass stitched image + top 4 candidates separately
            image_paths = [stitched_image_path] + obj_paths[:4]
            result = multi_try_ensure_response_qwen_multi_combined(image_paths, None, Prompt_final_select_multi, save_folder, num_votes=3, Force=True)
            if not result:
                continue
            
            # Handle tuple return (message, stage_info)
            if isinstance(result, tuple) and len(result) == 2:
                message, stage_info = result
            else:
                # Backward compatibility: if it's just a string, wrap it
                message = result
                stage_info = {}
            
            if not message:
                continue
            llm_elapsed = time.time() - llm_start
            total_selection_time += llm_elapsed
            
            # Store selection data for this chair
            chair_selection_data = {
                "chair_name": chair_name,
                "selection_method": "multi_image_combined",
                "selection_images": image_paths,
                "prompt": "Reference Description + Candidate Descriptions + Voting",
                "response": message,
                "time_seconds": llm_elapsed
            }
            # Add detailed stage information if available
            if stage_info:
                chair_selection_data["stage1_description"] = stage_info.get("stage1_description")
                chair_selection_data["stage1_prompt"] = stage_info.get("stage1_prompt")
                chair_selection_data["stage2a_candidate_descriptions"] = stage_info.get("stage2a_candidate_descriptions", {})
                chair_selection_data["stage2a_prompts"] = stage_info.get("stage2a_prompts", {})
                chair_selection_data["stage2b_votes"] = stage_info.get("stage2b_votes", [])
                chair_selection_data["stage2b_responses"] = stage_info.get("stage2b_responses", [])
                chair_selection_data["stage2b_prompt"] = stage_info.get("stage2b_prompt")
                chair_selection_data["final_candidate"] = stage_info.get("final_candidate")
                # Backward compatibility
                chair_selection_data["stage2_votes"] = stage_info.get("stage2b_votes", [])
                chair_selection_data["stage2_responses"] = stage_info.get("stage2b_responses", [])
                chair_selection_data["stage2_prompt"] = stage_info.get("stage2b_prompt")
            
            llm_selection_data.append(chair_selection_data)
        else:
            # Single-image approach: create combined image (original method)
            output_path = merge_selection_and_stitching(stitched_image_path, obj_paths, obj_path)
            message = multi_try_ensure_response_qwen(output_path, None, Prompt_final_select, save_folder, Force=True)
            if not message:
                continue
            llm_elapsed = time.time() - llm_start
            total_selection_time += llm_elapsed
            
            # Store selection data for this chair
            llm_selection_data.append({
                "chair_name": chair_name,
                "selection_method": "single_image",
                "selection_image": output_path,
                "prompt": Prompt_final_select,
                "response": message,
                "time_seconds": llm_elapsed
            })
        
        try:
            # Parse the selection (expect a number from 1 to 4)
            selection = int(message)
            if selection < 1 or selection > 4:
                continue
                
            # Add to selection votes (convert to 0-based index)
            selection_idx = selection - 1
            selection_votes[selection_idx] = selection_votes.get(selection_idx, 0) + 1
            
            print(f"    {Colors.GREEN}✓{Colors.RESET} {chair_name} → Template #{selection} ({Colors.CYAN}{llm_elapsed:.1f}s{Colors.RESET})")
            
        except ValueError:
            continue
    
    if not selection_votes:
        return []
    
    # Find the most voted selections
    max_selection_votes = max(selection_votes.values())
    top_selections = [sel for sel, votes in selection_votes.items() if votes == max_selection_votes]
    
    if len(top_selections) == 1:
        print(f"    → Consensus: Template #{top_selections[0] + 1} ({max_selection_votes} votes)")
    else:
        print(f"    → Multiple selections, copying all top {len(top_selections)}")
    
    # Step 4: Save results for all chairs in the cluster
    print(f"  {Colors.BLUE}💾{Colors.RESET} Saving results")
    
    processed_chairs = []
    
    for chair_name in cluster:
        if chair_name not in chair_paths:
            continue
            
        idx = chair_paths[chair_name]
        obj_path = all_objs[idx]
        output_obj_folder = f"{obj_path}/selected_obj".replace("input", "output")
        
        # Skip if already processed
        if os.path.exists(output_obj_folder) and os.listdir(output_obj_folder):
            processed_chairs.append(idx)
            continue
            
        # Create log for report
        # Use the first chair's retrieval data as representative (they all share the same process)
        representative_retrieval = llm_retrieval_data[0] if llm_retrieval_data else None
        representative_selection = llm_selection_data[0] if llm_selection_data else None
        
        log_for_report = {
            "status": "processed_in_cluster",
            "cluster": cluster,
            "category_votes": categories,
            "selected_categories": top_categories,
            "dataset_folders": dataset_folders,
            "DINOv2_top_4_results": top_4_results,
            "DINOv2_top_10_results": top_10_results,
            "DINOv2_time_seconds": sim_elapsed,
            "selection_votes": selection_votes,
            "top_selections": top_selections,
            # LLM Category Retrieval data (using first chair as representative)
            "LLM_retrieval_method": representative_retrieval.get("method", "two_stage") if representative_retrieval else None,
            "LLM_retrieval_stage1_prompt": representative_retrieval.get("stage1_prompt") if representative_retrieval else None,
            "LLM_retrieval_stage1_response": representative_retrieval.get("stage1_response") if representative_retrieval else None,
            "LLM_retrieval_stage2_description_prompt": representative_retrieval.get("stage2_description_prompt") if representative_retrieval else None,
            "LLM_retrieval_stage2_description": representative_retrieval.get("stage2_description") if representative_retrieval else None,
            "LLM_retrieval_stage2_match_prompt": representative_retrieval.get("stage2_match_prompt") if representative_retrieval else None,
            "LLM_retrieval_stage2_response": representative_retrieval.get("stage2_response") if representative_retrieval else None,
            "LLM_retrieval_final_category": representative_retrieval.get("final_category") if representative_retrieval else None,
            "LLM_retrieval_time_seconds": total_retrieval_time,
            # Backward compatibility fields (for old code that might expect these)
            "LLM_retrieval_prompt": representative_retrieval.get("stage1_prompt") if representative_retrieval else None,
            "LLM_retrieval_message": representative_retrieval.get("stage2_response") or representative_retrieval.get("stage1_response") if representative_retrieval else None,
            "LLM_retrieval_retry_happened": False,  # Two-stage doesn't use retry, it's built-in
            "LLM_retrieval_retry_prompt": None,
            "LLM_retrieval_retry_message": None,
            # LLM Final Selection data (using first chair as representative)
            "LLM_selection_prompt": representative_selection["prompt"] if representative_selection else None,
            "LLM_selection_message": representative_selection["response"] if representative_selection else None,
            "LLM_selection_image": representative_selection.get("selection_image") if representative_selection else None,
            "LLM_selection_images": representative_selection.get("selection_images") if representative_selection else None,
            "LLM_selection_time_seconds": total_selection_time,
            # All retrieval and selection data for reference
            "LLM_retrieval_data_all_chairs": llm_retrieval_data,
            "LLM_selection_data_all_chairs": llm_selection_data
        }
        
        try:
            # Copy all top selected models to this chair's output folder
            create_directory(output_obj_folder)
            
            save_start = time.time()
            for selection in top_selections:
                if selection < 0 or selection >= len(obj_paths):
                    continue
                    
                select_obj_image = obj_paths[selection]
                log_for_report["select_obj_image"] = select_obj_image
                
                select_obj_image_folder = os.path.dirname(select_obj_image)
                shutil.copytree(select_obj_image_folder, output_obj_folder, dirs_exist_ok=True)
            
            save_elapsed = time.time() - save_start
            
            # Save the log
            local_log_save = os.path.join(output_obj_folder, "local_processing_log.json")
            with open(local_log_save, "w") as f:
                json.dump(log_for_report, f, indent=4)
                
            processed_chairs.append(idx)
            print(f"    {Colors.GREEN}✓{Colors.RESET} {chair_name} → selected_obj/ ({Colors.CYAN}{save_elapsed:.1f}s{Colors.RESET})")
            
            # Create retrieval cache image
            cache_path = create_retrieval_cache_image(obj_path, scene_name, chair_name)
            if cache_path:
                print(f"    {Colors.CYAN}📸{Colors.RESET} Cache saved: {cache_path}")
            
        except Exception as e:
            print(f"{Colors.RED}❌{Colors.RESET} Error processing {chair_name}: {type(e).__name__}: {e}")
    
    return processed_chairs


def main():
    """Main entry point for the script"""
    total_start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="3D Asset Retrieval Pipeline")
    parser.add_argument("--name", type=str, required=True, help="Path to camera data")
    parser.add_argument("--no-use-multi-image-selection", dest="use_multi_image_selection", action="store_false", default=True,
                       help="Disable multi-image selection and use single combined image approach (default: multi-image enabled)")
    args = parser.parse_args()
    path_camera = args.name
    use_multi_image_selection = args.use_multi_image_selection  # Defaults to True
    
    # Extract scene name from path
    scene_name = os.path.basename(path_camera.rstrip('/'))
    
    # Create retrieval cache directory with scene_name folder
    cache_dir = os.path.join("cache", "retrieval_cache", scene_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Print selection method being used
    selection_method_str = "multi-image" if use_multi_image_selection else "single-image (combined)"
    
    # Setup environment
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Initialization]{Colors.RESET} Setting up retrieval pipeline...")
    config = setup_environment()
    
    # Print script header
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET} {Colors.BOLD}🎯 3D Asset Retrieval Pipeline{Colors.RESET}                           {Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Started at:{Colors.RESET} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.CYAN}Scene:{Colors.RESET} {Colors.BOLD}{scene_name}{Colors.RESET}")
    print(f"{Colors.CYAN}Selection method:{Colors.RESET} {Colors.BOLD}{selection_method_str}{Colors.RESET}\n")
    
    # Get all object folders
    all_objs_input = f"{path_camera}/*"
    all_objs = glob.glob(all_objs_input)
    
    if not all_objs:
        print(f"{Colors.RED}❌{Colors.RESET} No objects found in {path_camera}")
        return
        
    # Step 1: Process images
    step1_start = time.time()
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 1/3]{Colors.RESET} Processing images and preparing data...")
    print(f"  Found {Colors.BOLD}{len(all_objs)}{Colors.RESET} potential objects to process")
    
    all_logs = {}
    successful_objects = 0
    
    stitched_image_path_list = []
    semantic_list = []
    valid_objs = []
    skipped_count = 0
    failed_count = 0
    
    for i, obj in enumerate(all_objs):
        # Skip folders that don't have the required bbox_info_updated.json file
        bbox_file = os.path.join(obj, "bbox_info_updated.json")
        if not os.path.exists(bbox_file):
            obj_name = os.path.basename(obj)
            skipped_count += 1
            continue
        
        stitched_image_path, semantic = process_images(obj)
        if stitched_image_path and semantic:
            stitched_image_path_list.append(stitched_image_path)
            semantic_list.append(semantic)
            valid_objs.append(obj)
        else:
            obj_name = os.path.basename(obj)
            print(f"  {Colors.RED}❌{Colors.RESET} Failed to process images for {obj_name}")
            failed_count += 1
    
    if not valid_objs:
        print(f"  {Colors.RED}❌{Colors.RESET} No valid objects found to process.")
        return
    
    # Update all_objs to only include valid objects
    all_objs = valid_objs
    
    elapsed = time.time() - step1_start
    print(f"  {Colors.GREEN}✓{Colors.RESET} Processed {Colors.BOLD}{len(all_objs)}{Colors.RESET} valid objects", end="")
    if skipped_count > 0:
        print(f" (skipped {skipped_count} invalid)", end="")
    if failed_count > 0:
        print(f" ({failed_count} failed)", end="")
    print(f" ({Colors.CYAN}{elapsed:.2f}s{Colors.RESET})\n")

    # Step 2: Chair clustering (generate both geometry-only and geometry+color)
    step2_start = time.time()
    print(f"{Colors.BOLD}{Colors.BLUE}[Step 2/3]{Colors.RESET} Chair clustering and batch processing...")
    scene_path = path_camera.replace("input", "output")

    # Generate geometry-only clustering for retrieval
    chair_clustering = cluster_chairs(scene_path, use_color_clustering=False)  # Geometry-only for retrieval

    # Generate geometry+color clustering for material painting
    print(f"  Generating geometry+color clustering for material painting...")
    chair_clustering_geom_color = cluster_chairs(scene_path, use_color_clustering=True)  # For material painting

    # Save geometry+color clustering for material painting stage
    if chair_clustering_geom_color:
        import json
        scene_name_clean = scene_name.split('/')[-1]
        cache_dir = f"cache/clustering_cache/{scene_name_clean}"
        os.makedirs(cache_dir, exist_ok=True)
        geom_color_results = {
            "scene_name": scene_name_clean,
            "clusters": chair_clustering_geom_color,
            "clustering_method": "geometry_color_hybrid",
            "total_objects": sum(len(c) for c in chair_clustering_geom_color) if chair_clustering_geom_color else 0,
            "num_clusters": len(chair_clustering_geom_color) if chair_clustering_geom_color else 0,
            "cluster_sizes": [len(c) for c in chair_clustering_geom_color] if chair_clustering_geom_color else [],
            "use_color_clustering": True
        }
        with open(f"{cache_dir}/clustering_results_geom_color.json", 'w') as f:
            json.dump(geom_color_results, f, indent=4)
        print(f"  ✅ Saved geometry+color clustering results for material painting")
    
    # Handle case where cluster_chairs returns None (no valid chair data)
    if chair_clustering is None:
        print(f"  {Colors.YELLOW}⚠️{Colors.RESET} No valid chair data found for clustering")
        chair_clustering = []
    
    chair_count = sum(len(c) for c in chair_clustering) if chair_clustering else 0
    if chair_count > 0:
        print(f"  Found {Colors.BOLD}{chair_count}{Colors.RESET} chair objects")
        print(f"  Clustering chairs by geometry and color...")
        for i, cluster in enumerate(chair_clustering):
            print(f"    Cluster {i+1}: {Colors.BOLD}{', '.join(cluster)}{Colors.RESET} ({len(cluster)} chairs)")
    
    processed_chair_indices = []
    for i, cluster in enumerate(chair_clustering):
        cluster_item_start = time.time()
        print(f"\n  {Colors.CYAN}📦{Colors.RESET} Processing Cluster {i+1}: {', '.join(cluster)}")
        processed_indices = process_chair_cluster(cluster, config, stitched_image_path_list, semantic_list, all_objs, scene_name, use_multi_image_selection)
        processed_chair_indices.extend(processed_indices)
        cluster_elapsed = time.time() - cluster_item_start
        print(f"    {Colors.GREEN}✓{Colors.RESET} Cluster {i+1} complete ({Colors.CYAN}{cluster_elapsed:.1f}s{Colors.RESET})")
    
    cluster_elapsed = time.time() - step2_start
    if chair_count > 0:
        print(f"  {Colors.GREEN}✓{Colors.RESET} Chair clustering complete ({Colors.CYAN}{cluster_elapsed:.1f}s{Colors.RESET})\n")
    else:
        print(f"  {Colors.GREEN}✓{Colors.RESET} No chairs to cluster ({Colors.CYAN}{cluster_elapsed:.1f}s{Colors.RESET})\n")

    # Step 3: Process individual objects
    step3_start = time.time()
    print(f"{Colors.BOLD}{Colors.BLUE}[Step 3/3]{Colors.RESET} Processing individual objects...")
    individual_objects = [i for i in range(len(all_objs)) if i not in processed_chair_indices]
    print(f"  Processing {Colors.BOLD}{len(individual_objects)}{Colors.RESET} individual objects...\n")

    for i in individual_objects:
        obj = all_objs[i]
        stitched_image_path = stitched_image_path_list[i]
        semantic = semantic_list[i]
        result = process_object(obj, config, stitched_image_path, semantic, use_multi_image_selection)
        
        # Handle the result
        if result is not None:
            # Check if it's a tuple (successful processing case)
            if isinstance(result, tuple) and len(result) == 2:
                log, output_obj_folder = result
                # Create output directory if it doesn't exist
                os.makedirs(output_obj_folder, exist_ok=True)
                # Save the log
                local_log_save = os.path.join(output_obj_folder, "local_processing_log.json")
                with open(local_log_save, "w") as f:
                    json.dump(log, f, indent=4)
                all_logs[os.path.basename(obj)] = log
                successful_objects += 1
                
                # Create retrieval cache image for successfully processed objects
                obj_name = os.path.basename(obj)
                cache_path = create_retrieval_cache_image(obj, scene_name, obj_name)
                if cache_path:
                    print(f"  {Colors.CYAN}📸{Colors.RESET} Cache saved: {cache_path}")
            # Handle dictionary case (already processed or special cases)
            elif isinstance(result, dict):
                all_logs[os.path.basename(obj)] = result
                if result.get("status") != "already_processed":
                    successful_objects += 1
                    
                    # Try to create cache for already processed objects too
                    obj_name = os.path.basename(obj)
                    cache_path = create_retrieval_cache_image(obj, scene_name, obj_name)
                    if cache_path:
                        print(f"  {Colors.CYAN}📸{Colors.RESET} Cache saved: {cache_path}")
    
    step3_elapsed = time.time() - step3_start
    if processed_chair_indices:
        print(f"\n  {Colors.YELLOW}⏩{Colors.RESET} Skipped {len(processed_chair_indices)} objects (already processed in chair clusters)")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Individual object processing complete ({Colors.CYAN}{step3_elapsed:.1f}s{Colors.RESET})\n")
    
    # Save overall log for reporting
    log_save_start = time.time()
    log_output_path = os.path.join(path_camera.replace("input", "output"), "processing_log.json")
    os.makedirs(os.path.dirname(log_output_path), exist_ok=True)
    with open(log_output_path, "w") as f:
        json.dump(all_logs, f, indent=4)
    log_save_elapsed = time.time() - log_save_start
    
    total_time = time.time() - total_start_time
    print(f"{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}║{Colors.RESET} {Colors.BOLD}✓ 3D Asset Retrieval completed successfully!{Colors.RESET}             {Colors.BOLD}{Colors.GREEN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Successfully processed:{Colors.RESET} {Colors.BOLD}{successful_objects}{Colors.RESET} objects")
    print(f"{Colors.CYAN}Total time:{Colors.RESET} {Colors.BOLD}{total_time:.2f}s{Colors.RESET}\n")
    
    # Unload Qwen3-VL model
    print(f"{Colors.BLUE}🤖{Colors.RESET} Unloading Qwen3-VL model...")
    qwen_unload_model()
    print(f"{Colors.GREEN}✓{Colors.RESET} Model unloaded successfully\n")


if __name__ == "__main__":
    main()