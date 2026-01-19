import sys
sys.path.append("litereality/LR_mat_painting")


# from utils.Decompose_mesh import center_obj, separate_obj_by_group
from utils.Render_Original_cycle import onboarding_rendering_with_pbr
# from utils.Render_Segmented_single import onboarding_rendering_Segment
import os
from glob import glob
from PIL import Image

import argparse

def parse_arguments():
    # Get the arguments passed after `--` to avoid Blender's internal arguments
    blender_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Render images for a specified scene with customizable image size.")
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Path to the folder containing the raw scan data for the scene."
    )
    parser.add_argument(
        "--image_size", type=int, default=500,
        help="Resolution of the output render (default: 500)."
    )
    return parser.parse_args(blender_args)

def create_pbr_cache_image(scene_path, obj_name):
    """
    Create a combined image for PBR cache showing:
    - Left: render_image_0.png (rendered result)
    - Right: stitched_image.jpg (original captured image)
    
    Args:
        scene_path: Full path to the scene folder (e.g., "output/mat_painting_stage/scene_1/Chair2")
        obj_name: Name of the object (e.g., "Chair2")
    
    Returns:
        str: Path to the saved cache image, or None if failed
    """
    try:
        # Extract scene_name from path (e.g., "scene_1" from "output/mat_painting_stage/scene_1/Chair2")
        path_parts = scene_path.split(os.sep)
        if "mat_painting_stage" in path_parts:
            idx = path_parts.index("mat_painting_stage")
            if idx + 1 < len(path_parts):
                scene_name = path_parts[idx + 1]
            else:
                print(f"Warning: Could not determine scene_name from {scene_path}")
                return None
        else:
            # Fallback: try to get second-to-last part
            if len(path_parts) >= 2:
                scene_name = path_parts[-2]
            else:
                print(f"Warning: Could not determine scene_name from {scene_path}")
                return None
        
        # Construct paths
        rendered_image_path = os.path.join(scene_path, "A-ReTextured", "OT_refined_with_adaptation", "render_image_0.png")
        stitched_image_path = os.path.join(scene_path, "captured_images", "stitched_image.jpg")
        
        # Check if both images exist
        if not os.path.exists(rendered_image_path):
            print(f"Warning: Rendered image not found: {rendered_image_path}")
            return None
        if not os.path.exists(stitched_image_path):
            print(f"Warning: Stitched image not found: {stitched_image_path}")
            return None
        
        # Load images
        img1 = Image.open(rendered_image_path).convert("RGB")
        img2 = Image.open(stitched_image_path).convert("RGB")
        
        # Resize to 1/4 of original size for cache
        new_size1 = (img1.size[0] // 2, img1.size[1] // 2)
        new_size2 = (img2.size[0] // 2, img2.size[1] // 2)
        
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
        cache_dir = os.path.join("cache", "PBR_cache", scene_name)
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{obj_name}.jpg")
        combined_image.save(cache_path, quality=85)
        
        return cache_path
        
    except Exception as e:
        # Suppress verbose error output
        return None

# Parse the arguments
args = parse_arguments()
# Retrieve arguments
scene_name = args.scene
image_size = args.image_size


sub_folders = os.listdir(scene_name)
# Remove 'captured_images' folder if present
if "captured_images" in sub_folders:
    sub_folders.remove("captured_images")
# Use the first remaining subdirectory as the type name
if not sub_folders:
    print("No valid subdirectories found other than 'captured_images'.")
    sys.exit(1)
type_name = sub_folders[0]


obj_name = scene_name.split("/")[-1]
# Remove digits
semantic_name = ''.join(filter(lambda x: not x.isdigit(), obj_name))

# Apply same naming logic as apply_mat_rotate.py for windows/doors
if "Window" not in semantic_name and "Door" not in semantic_name:
    semantic_name = semantic_name.replace("_", "").replace("Wall", "")
    object_name = semantic_name + "_gpt"
else:
    # Windows and Doors use special naming: "Wall_" + semantic_name + "__gpt"
    semantic_name = semantic_name.replace("_", "")
    semantic_name = semantic_name.replace("Wall", "")
    if "Window" in semantic_name:
        object_name = "Wall_" + semantic_name + "__gpt"
    elif "Door" in semantic_name:
        object_name = "Wall_" + semantic_name + "__gpt"

# obj_name = type_name # Replace with the object name
object_folder = f"{scene_name}/Onboarded/decomposed"
rendering_OT_with_adaptation = f"{scene_name}/A-ReTextured/OT_refined_with_adaptation"    # Replace with the desired output folder path
os.makedirs(rendering_OT_with_adaptation, exist_ok=True)

# Use resized materials (1000px) since originals are cleaned up after resize
material_file_OT_with_adaptation_1000 = f"{scene_name}/selected_material_OT_with_adaptation_1000/{object_name}"
material_file_OT_with_adaptation = f"{scene_name}/selected_material_OT_with_adaptation/{object_name}"

# Prefer resized version, fallback to original if resize didn't happen
if os.path.exists(material_file_OT_with_adaptation_1000):
    material_file_OT_with_adaptation = material_file_OT_with_adaptation_1000
    print(f"Using resized materials: {material_file_OT_with_adaptation_1000}")
elif not os.path.exists(material_file_OT_with_adaptation):
    print(f"Error: Neither resized nor original materials found for {object_name}")
    print(f"  Expected: {material_file_OT_with_adaptation_1000}")
    print(f"  Or: {material_file_OT_with_adaptation}")
    sys.exit(1)

onboarding_rendering_with_pbr(object_name, (image_size, image_size), rendering_OT_with_adaptation, object_folder, material_file_OT_with_adaptation)

rendering_org_path = f"{scene_name}/A-ReTextured/object_stage"    # Replace with the desired output folder path
os.makedirs(rendering_org_path, exist_ok=True)
material_file = f"{scene_name}/selected_material/{object_name}"

# Skip original material render if it was cleaned up (it's just for comparison anyway)
if os.path.exists(material_file):
    onboarding_rendering_with_pbr(obj_name, (image_size, image_size), rendering_org_path, object_folder, material_file)
else:
    print(f"Note: Skipping original material render - {material_file} was cleaned up after resize")

# Create PBR cache image after rendering completes
from litereality.LR_mat_painting.utils.output_formatter import formatter
cache_path = create_pbr_cache_image(scene_name, obj_name)
if cache_path:
    formatter.print_success("PBR cache created")
else:
    formatter.print_warning("Failed to create PBR cache image")


