import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Create an argument parser
parser = argparse.ArgumentParser(description="Resize material textures and optionally clean up originals.")

# Add arguments
parser.add_argument("--folder", type=str, required=True, help="Path to folder containing objects (or single object folder)")
parser.add_argument("--mat_size", type=int, required=True, help="Target texture size (e.g., 1000 for 1000x1000)")
parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
parser.add_argument("--keep-originals", action="store_true", help="Keep original material folders after resize (for debugging)")
parser.add_argument("--single-object", action="store_true", help="Process single object folder instead of scene folder")

# Parse the arguments
args = parser.parse_args()

# Assign the argument to a variable
folder_to_process = args.folder
resize_to = args.mat_size
num_workers = args.num_workers if args.num_workers else mp.cpu_count()
keep_originals = args.keep_originals
single_object_mode = args.single_object

# Print a nice header
print("\n" + "="*80)
print(f"🔍 TEXTURE RESIZING TOOL")
print(f"📂 Processing folder: {folder_to_process}")
print(f"🖼️ Target size: {resize_to}x{resize_to} pixels")
if keep_originals:
    print(f"⚠️  Keep originals mode: Original folders will be preserved")
else:
    print(f"🗑️  Cleanup mode: Original folders will be deleted after successful resize")
print("="*80 + "\n")

# Get all objects in the folder
if single_object_mode:
    # Single object mode: process the folder itself
    all_obj_in_folder = [folder_to_process]
else:
    # Scene mode: process all object folders
    all_obj_in_folder = glob.glob(os.path.join(folder_to_process, "*"))
    # Filter out non-directories
    all_obj_in_folder = [f for f in all_obj_in_folder if os.path.isdir(f)]

total_objects = len(all_obj_in_folder)

print(f"📋 Found {total_objects} objects to process")
print(f"⚡ Using {num_workers} parallel workers\n")

# Track statistics
total_textures_processed = 0
total_bytes_before = 0
total_bytes_after = 0

def resize_single_texture(args_tuple):
    """
    Worker function to resize a single texture file.
    
    Args:
        args_tuple: Tuple of (src_file, dest_folder, dst_file, resize_to)
    
    Returns:
        tuple: (success, original_size, new_size, error_message)
    """
    src_file, dest_folder, dst_file, resize_to = args_tuple
    
    try:
        # Create the destination folder if it doesn't exist
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder, exist_ok=True)
        
        # Get original file size
        original_size = os.path.getsize(src_file)
        
        # Open, resize and save the image
        im = Image.open(src_file)
        im_resized = im.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        im_resized.save(dst_file, optimize=True)
        
        # Get new file size
        new_size = os.path.getsize(dst_file)
        
        return (True, original_size, new_size, None)
    except Exception as e:
        return (False, 0, 0, str(e))

# Process each object
for obj_index, obj_mat_folder in enumerate(tqdm(all_obj_in_folder, desc="Processing objects", unit="obj")):
    obj_name = os.path.basename(obj_mat_folder)
    
    # Print current object being processed
    print(f"\n[{obj_index+1}/{total_objects}] 🔹 Processing object: {obj_name}")
    
    # Remove all digits from the object name to form the semantic name.
    semantic_name = ''.join([ch for ch in obj_name if not ch.isdigit()])
    semantic_name = semantic_name.replace("_", "")
    semantic_name = semantic_name.replace("Wall", "")

    if "Window" in semantic_name:
        semantic_name = "Wall_" + semantic_name + "__gpt"
    elif "Door" in semantic_name:
        semantic_name = "Wall_" + semantic_name + "__gpt"
    else: 
        semantic_name = semantic_name + "_gpt"

    # Original folder containing materials for the object.
    mat_folder = os.path.join(obj_mat_folder, "selected_material_OT_with_adaptation", semantic_name)
    
    # New destination folder for the resized textures.
    mat_folder_resized = os.path.join(obj_mat_folder, f"selected_material_OT_with_adaptation_{resize_to}", semantic_name)
    
    # Check if source folder exists
    if not os.path.exists(mat_folder):
        print(f"⚠️  Material folder not found: {mat_folder}")
        print(f"    Skipping this object...")
        continue
    
    print(f"📁 Source material folder: {mat_folder}")
    print(f"📁 Destination folder: {mat_folder_resized}")
    
    # Collect all texture files first for progress tracking
    texture_files = []
    for root, dirs, files in os.walk(mat_folder):
        for file in files:
            if file.lower().endswith(".png"):
                src_file = os.path.join(root, file)
                # Get the relative path from the material folder
                rel_path = os.path.relpath(root, mat_folder)
                dest_folder = os.path.join(mat_folder_resized, rel_path)
                dst_file = os.path.join(dest_folder, file)
                texture_files.append((src_file, dest_folder, dst_file))
    
    # Show how many textures will be processed for this object
    texture_count = len(texture_files)
    print(f"🖼️ Found {texture_count} textures to resize")
    
    # Prepare arguments for parallel processing
    texture_args = [(src_file, dest_folder, dst_file, resize_to) 
                    for src_file, dest_folder, dst_file in texture_files]
    
    # Process textures in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_texture = {executor.submit(resize_single_texture, args): args 
                            for args in texture_args}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_texture), total=texture_count, 
                         desc="Resizing textures", unit="tex"):
            success, orig_size, new_size, error = future.result()
            
            if success:
                total_bytes_before += orig_size
                total_bytes_after += new_size
                total_textures_processed += 1
            else:
                args = future_to_texture[future]
                print(f"  ❌ Error processing {args[0]}: {error}")
    
    # Verify resize succeeded before cleanup
    resize_succeeded = False
    if texture_count > 0:
        # Check if resized folder exists and has expected number of files
        resized_png_count = sum(1 for root, dirs, files in os.walk(mat_folder_resized) 
                               for f in files if f.lower().endswith('.png'))
        original_png_count = sum(1 for root, dirs, files in os.walk(mat_folder) 
                                for f in files if f.lower().endswith('.png'))
        
        if resized_png_count >= original_png_count * 0.9:  # Allow 10% tolerance
            resize_succeeded = True
            print(f"✅ Resize verification passed: {resized_png_count}/{original_png_count} textures")
        else:
            print(f"⚠️  Resize verification failed: {resized_png_count}/{original_png_count} textures")
    
    # Copy and resize "before" materials (selected_material) before deletion
    # Check both possible structures: selected_material/{semantic_name} and selected_material/{object_name}_gpt
    before_material_folder = os.path.join(obj_mat_folder, "selected_material", semantic_name)
    obj_name_clean = ''.join([ch for ch in obj_name if not ch.isdigit()])
    before_material_folder_alt = os.path.join(obj_mat_folder, "selected_material", obj_name_clean + "_gpt")
    
    # Determine which structure exists
    actual_before_folder = None
    backup_semantic_name = None
    if os.path.exists(before_material_folder):
        actual_before_folder = before_material_folder
        backup_semantic_name = semantic_name
    elif os.path.exists(before_material_folder_alt):
        actual_before_folder = before_material_folder_alt
        backup_semantic_name = obj_name_clean + "_gpt"
    
    if actual_before_folder:
        before_material_backup_folder = os.path.join(obj_mat_folder, f"selected_material_backup_{resize_to}", backup_semantic_name)
        print(f"📋 Copying and resizing 'before' materials from: {actual_before_folder}")
        print(f"📁 Backup destination: {before_material_backup_folder}")
        
        # Process each part folder
        for part_name in os.listdir(actual_before_folder):
            part_source = os.path.join(actual_before_folder, part_name)
            if os.path.isdir(part_source):
                basecolor_source = os.path.join(part_source, "basecolor.png")
                if os.path.exists(basecolor_source):
                    # Create destination folder
                    part_dest = os.path.join(before_material_backup_folder, part_name)
                    os.makedirs(part_dest, exist_ok=True)
                    
                    # Resize and copy basecolor only
                    basecolor_dest = os.path.join(part_dest, "basecolor.png")
                    try:
                        im = Image.open(basecolor_source)
                        im_resized = im.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
                        im_resized.save(basecolor_dest, optimize=True)
                        print(f"  ✅ Copied and resized {part_name}/basecolor.png")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not resize {part_name}/basecolor.png: {e}")
        
        print(f"✅ 'Before' materials backed up to: {before_material_backup_folder}")
    
    # Clean up original folders if resize succeeded and not keeping originals
    if resize_succeeded and not keep_originals:
        folders_to_delete = [
            os.path.join(obj_mat_folder, "selected_material", semantic_name),
            os.path.join(obj_mat_folder, "selected_material_OT_with_adaptation", semantic_name)
        ]
        
        # Also check alternative structure
        if backup_semantic_name and backup_semantic_name != semantic_name:
            folders_to_delete.append(os.path.join(obj_mat_folder, "selected_material", backup_semantic_name))
        
        for folder_to_delete in folders_to_delete:
            if os.path.exists(folder_to_delete):
                try:
                    shutil.rmtree(folder_to_delete)
                    print(f"🗑️  Deleted original folder: {os.path.basename(folder_to_delete)}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete {folder_to_delete}: {e}")
    elif not resize_succeeded:
        print(f"⚠️  Keeping original folders due to resize verification failure")
    elif keep_originals:
        print(f"ℹ️  Keeping original folders (--keep-originals flag set)")

# Print a summary at the end
print("\n" + "="*80)
print("✅ TEXTURE RESIZING COMPLETE")
print("="*80)
print(f"📊 Summary:")
print(f"  • Total objects processed: {total_objects}")
print(f"  • Total textures resized: {total_textures_processed}")

# Calculate size reduction
if total_bytes_before > 0:
    reduction_pct = (1 - (total_bytes_after / total_bytes_before)) * 100
    total_mb_before = total_bytes_before / (1024 * 1024)
    total_mb_after = total_bytes_after / (1024 * 1024)
    print(f"  • Total size before: {total_mb_before:.2f} MB")
    print(f"  • Total size after: {total_mb_after:.2f} MB")
    print(f"  • Size reduction: {reduction_pct:.2f}%")

if single_object_mode:
    print(f"  • Output location: {folder_to_process}/selected_material_OT_with_adaptation_{resize_to}")
else:
    print(f"  • Output location: {folder_to_process}/*/selected_material_OT_with_adaptation_{resize_to}")
print("="*80)