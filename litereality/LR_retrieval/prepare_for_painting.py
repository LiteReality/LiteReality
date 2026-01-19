import os
import argparse
import shutil
import json
import time
from datetime import datetime, timedelta


def process_single_object(obj_path, verbose=False):
    """
    Process a single object folder to prepare it for material painting.
    
    Args:
        obj_path: Path to the object folder (e.g., "output/object_stage/scene_1/Chair2")
        verbose: Whether to show detailed progress information
    
    Returns:
        str: Path to the prepared object folder in mat_painting_stage, or None if failed
    """
    try:
        obj_name = os.path.basename(obj_path)
        scene_path = os.path.dirname(obj_path)
        scene_name = os.path.basename(scene_path)
        
        # Determine semantic name
        semantic_name = ''.join([i for i in obj_name if not i.isdigit()])
        semantic_name = semantic_name.replace("_", "")
        semantic_name = semantic_name.replace("Wall", "")
        
        # Debug: Print semantic name calculation
        if verbose:
            print(f"   Debug: obj_name='{obj_name}', semantic_name='{semantic_name}'")
        
        # Copy object to mat_painting_stage
        mat_painting_scene_path = scene_path.replace("object_stage", "mat_painting_stage")
        os.makedirs(mat_painting_scene_path, exist_ok=True)
        
        obj_mat_painting_path = os.path.join(mat_painting_scene_path, obj_name)
        
        # Copy object folder into mat_painting_stage.
        # If the target already exists but is empty (can happen after partial runs),
        # treat it as not prepared and copy again.
        if (not os.path.exists(obj_mat_painting_path)) or (
            os.path.isdir(obj_mat_painting_path) and len(os.listdir(obj_mat_painting_path)) == 0
        ):
            if verbose:
                print(f"   📂 Copying object to {obj_mat_painting_path}...")
            try:
                shutil.copytree(obj_path, obj_mat_painting_path, dirs_exist_ok=True)
                if verbose:
                    print(f"   ✅ Copied successfully")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error copying object: {e}")
                return None
        else:
            if verbose:
                print(f"   ℹ️  Object already exists in mat_painting_stage and is non-empty, skipping copy")
        
        # Validate semantic_name is not empty before using it
        if not semantic_name:
            if verbose:
                print(f"   ⚠️  Warning: semantic_name is empty for {obj_name}, using object name as fallback")
            # Try a different approach - keep some structure
            semantic_name = obj_name.replace("_", "").replace("Wall", "")
            if not semantic_name:
                semantic_name = "Object"  # Last resort fallback
        
        # Rename selected_obj to semantic_name
        # Ensure obj_mat_painting_path exists before proceeding
        if not os.path.exists(obj_mat_painting_path):
            if verbose:
                print(f"   ❌ Error: Target folder does not exist: {obj_mat_painting_path}")
            return None
        
        selected_obj_path = os.path.join(obj_mat_painting_path, "selected_obj")
        new_selected_obj_path = os.path.join(obj_mat_painting_path, semantic_name)
        
        # Ensure new_selected_obj_path is actually a subdirectory, not the same as obj_mat_painting_path
        if new_selected_obj_path == obj_mat_painting_path or not semantic_name:
            if verbose:
                print(f"   ⚠️  Warning: Invalid semantic_name '{semantic_name}', cannot rename")
                print(f"   Debug: obj_mat_painting_path={obj_mat_painting_path}")
                print(f"   Debug: new_selected_obj_path={new_selected_obj_path}")
            return obj_mat_painting_path  # Return path even if rename fails
        
        if os.path.exists(selected_obj_path):
            if os.path.exists(new_selected_obj_path):
                if new_selected_obj_path != selected_obj_path:
                    shutil.rmtree(new_selected_obj_path)
                else:
                    if verbose:
                        print(f"   ℹ️  selected_obj already has correct name, skipping rename")
                    return obj_mat_painting_path
            try:
                os.rename(selected_obj_path, new_selected_obj_path)
                if verbose:
                    print(f"   ✅ Renamed selected_obj to {semantic_name}")
            except OSError as e:
                if verbose:
                    print(f"   ❌ Error renaming selected_obj: {e}")
                # Try to continue anyway - maybe the folder already has the right name
                if not os.path.exists(new_selected_obj_path):
                    if verbose:
                        print(f"   ⚠️  Rename failed and target doesn't exist, but continuing...")
        elif not os.path.exists(new_selected_obj_path):
            if verbose:
                print(f"   ⚠️  Neither selected_obj nor {semantic_name} found - object may not have been retrieved")
            # Still return the path so processing can continue
            return obj_mat_painting_path
        
        # Process onboarding files if local_processing_log.json exists
        # Use new_selected_obj_path if rename succeeded, otherwise try selected_obj_path
        log_file_path = None
        if os.path.exists(new_selected_obj_path):
            log_file_path = os.path.join(new_selected_obj_path, "local_processing_log.json")
        elif os.path.exists(selected_obj_path):
            log_file_path = os.path.join(selected_obj_path, "local_processing_log.json")
        
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'r') as file:
                    log_data = json.load(file)
                
                if "select_obj_image" in log_data:
                    selected_obj_path_from_log = log_data["select_obj_image"]
                    onboarding_path = selected_obj_path_from_log.replace("3D_assets", "3D_assets_pre_rendering")
                    
                    if "/image.jpg" in onboarding_path:
                        onboarding_path = onboarding_path.replace("/image.jpg", "")
                    
                    onboarded_dir = os.path.join(obj_mat_painting_path, "Onboarded")
                    os.makedirs(onboarded_dir, exist_ok=True)
                    
                    if os.path.exists(onboarding_path):
                        if verbose:
                            print(f"   📂 Copying onboarding files...")
                        shutil.copytree(onboarding_path, onboarded_dir, dirs_exist_ok=True)
                        if verbose:
                            print(f"   ✅ Onboarding files copied")
                    else:
                        if verbose:
                            print(f"   ⚠️  Onboarding path not found: {onboarding_path}")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error processing onboarding files: {str(e)}")
        
        # Remove bbox_images directory
        bbox_images_path = os.path.join(obj_mat_painting_path, "bbox_images")
        if os.path.exists(bbox_images_path):
            shutil.rmtree(bbox_images_path)
            if verbose:
                print(f"   ✅ Removed bbox_images")
        
        # Rename cropped_images to captured_images
        cropped_images_path = os.path.join(obj_mat_painting_path, "cropped_images")
        captured_images_path = os.path.join(obj_mat_painting_path, "captured_images")
        if os.path.exists(cropped_images_path):
            if os.path.exists(captured_images_path):
                shutil.rmtree(captured_images_path)
            os.rename(cropped_images_path, captured_images_path)
            if verbose:
                print(f"   ✅ Renamed cropped_images to captured_images")
        
        # Remove all files (not directories) from the object path
        files_removed = 0
        for file in os.listdir(obj_mat_painting_path):
            file_path = os.path.join(obj_mat_painting_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
        
        if verbose and files_removed > 0:
            print(f"   ✅ Removed {files_removed} files")
        
        return obj_mat_painting_path
        
    except Exception as e:
        print(f"❌ Error processing object {obj_path}: {str(e)}")
        return None


def process_folder(folder, verbose=False):
    """
    Process a retrieval folder to prepare it for rendering.
    
    Args:
        folder: Path to the retrieval folder
        verbose: Whether to show detailed progress information
    """
    print(f"🟢 Starting folder processing...")
    
    try:
        all_obj_in_folder = os.listdir(folder)
    except FileNotFoundError:
        print(f"❌ Folder '{folder}' not found.")
        return

    # Get the total number of objects for progress tracking
    total_objects = len([obj for obj in all_obj_in_folder if os.path.isdir(os.path.join(folder, obj))])
    print(f"📋 Found {total_objects} objects to process")
    
    # Copy folder to new folder with mat_painting_stage prefix
    new_folder = folder.replace("object_stage", "mat_painting_stage")
    print(f"🟢 Copying folder to {new_folder}...")
    shutil.copytree(folder, new_folder, dirs_exist_ok=True)
    print(f"✅ Folder copied successfully")

    folder = new_folder
    
    # Process each object
    start_time = time.time()
    processed_count = 0
    
    for obj_idx, obj in enumerate(all_obj_in_folder, 1):
        obj_path = os.path.join(folder, obj)
        if os.path.isdir(obj_path):
            # Calculate progress and estimated time remaining
            processed_count += 1
            elapsed_time = time.time() - start_time
            objects_per_second = processed_count / max(elapsed_time, 0.1)
            remaining_objects = total_objects - processed_count
            estimated_remaining_seconds = remaining_objects / max(objects_per_second, 0.001)
            remaining_time = str(timedelta(seconds=int(estimated_remaining_seconds)))
            
            # Update progress display
            progress_percent = (processed_count / total_objects) * 100
            print(f"\n🔹 [{obj_idx}/{total_objects}] Processing object: {obj} ({progress_percent:.1f}%)")
            print(f"   ⏱️ Estimated time remaining: {remaining_time}")
            
            # Remove all digits from the name
            semantic_name = ''.join([i for i in obj if not i.isdigit()])
            # remove "_" and "Wall"
            semantic_name = semantic_name.replace("_", "")
            semantic_name = semantic_name.replace("Wall", "")
            
            # Set up paths
            selected_obj_path = os.path.join(obj_path, "selected_obj")
            new_selected_obj_path = os.path.join(obj_path, semantic_name)

            # Track start time for this operation
            op_start_time = time.time()
            
            try:
                # Rename selected_obj to semantic_name
                if verbose:
                    print(f"   🟢 Renaming {selected_obj_path} to {semantic_name}...")
                    
                # Remove target folder if it exists
                if os.path.exists(new_selected_obj_path):
                    shutil.rmtree(new_selected_obj_path)
                    
                os.rename(selected_obj_path, new_selected_obj_path)
                
                if verbose:
                    print(f"   ✅ Renamed successfully")
                
                # Process onboarding files if local_processing_log.json exists
                log_file_path = os.path.join(new_selected_obj_path, "local_processing_log.json")
                if os.path.exists(log_file_path):
                    if verbose:
                        print(f"   🟢 Found processing log, setting up onboarding files...")
                        
                    try:
                        with open(log_file_path, 'r') as file:
                            log_data = json.load(file)
                        
                        if "select_obj_image" in log_data:
                            # Extract the path from selected object image
                            selected_obj_path = log_data["select_obj_image"]
                            
                            # Convert the path from 3D_assets to 3D_assets_pre_rendering
                            onboarding_path = selected_obj_path.replace("3D_assets", "3D_assets_pre_rendering")
                            
                            # Extract folder path without image.jpg at the end
                            if "/image.jpg" in onboarding_path:
                                onboarding_path = onboarding_path.replace("/image.jpg", "")
                            
                            print("onboarding_path", onboarding_path)
                            # Create Onboarded directory
                            onboarded_dir = os.path.join(obj_path, "Onboarded")
                            os.makedirs(onboarded_dir, exist_ok=True)
                            
                            # Copy the pre-rendering folder to Onboarded directory
                            if os.path.exists(onboarding_path):
                                print(f"   📂 Copying onboarding files to {onboarded_dir}")
                                shutil.copytree(onboarding_path, onboarded_dir, dirs_exist_ok=True)
                                print(f"   ✅ Onboarding files copied")
                            else:
                                print(f"   ⚠️ Onboarding path not found: {onboarding_path}")
                    except Exception as e:
                        print(f"   ❌ Error processing onboarding files: {str(e)}")
            except FileNotFoundError:
                print(f"   ⚠️ '{selected_obj_path}' not found, skipping rename operation")
            
            # Remove bbox_images directory
            bbox_images_path = os.path.join(obj_path, "bbox_images")
            if os.path.exists(bbox_images_path):
                if verbose:
                    print(f"   🟢 Removing bbox_images directory...")
                    
                shutil.rmtree(bbox_images_path)
                
                if verbose:
                    print(f"   ✅ Removed successfully")

            # Rename cropped_images to captured_images
            cropped_images_path = os.path.join(obj_path, "cropped_images")
            captured_images_path = os.path.join(obj_path, "captured_images")
            if os.path.exists(cropped_images_path):
                if verbose:
                    print(f"   🟢 Renaming cropped_images to captured_images...")
                    
                if os.path.exists(captured_images_path):
                    shutil.rmtree(captured_images_path)
                    
                os.rename(cropped_images_path, captured_images_path)
                
                if verbose:
                    print(f"   ✅ Renamed successfully")

            # Remove all files (not directories) from the object path
            if verbose:
                print(f"   🟢 Cleaning up non-directory files...")
            
            files_removed = 0
            for file in os.listdir(obj_path):
                file_path = os.path.join(obj_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_removed += 1
            
            if verbose:
                print(f"   ✅ Removed {files_removed} files")
            
            # Calculate operation time
            op_elapsed = time.time() - op_start_time
            if verbose:
                print(f"   ⏱️ Object processing took {op_elapsed:.2f} seconds")

    # Final stats
    total_elapsed = time.time() - start_time
    avg_time_per_object = total_elapsed / max(processed_count, 1)
    print(f"\n✅ Processed {processed_count} objects in {total_elapsed:.2f} seconds")
    print(f"📊 Average time per object: {avg_time_per_object:.2f} seconds")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process retrieval folders for material painting.")

    # Add arguments
    parser.add_argument("--name", type=str, default=None,
                      help="Path to the retrieval folder to process (for batch mode)")
    parser.add_argument("--single-object", type=str, default=None,
                      help="Path to a single object folder to process (for per-object mode)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output with detailed progress")

    # Parse the arguments
    args = parser.parse_args()

    # Check which mode to use
    if args.single_object:
        # Single object mode
        if args.verbose:
            print(f"🔧 Preparing single object: {args.single_object}")
        
        result = process_single_object(args.single_object, verbose=args.verbose)
        
        if result:
            if args.verbose:
                print(f"✅ Object prepared: {result}")
            exit(0)
        else:
            print(f"❌ Failed to prepare object: {args.single_object}")
            exit(1)
    
    elif args.name:
        # Batch mode (process entire folder)
        print("🔶 STARTING RENDER PREPARATION 🔶")
        print(f"📂 Processing folder: {args.name}")
        
        start_time = time.time()
        
        try:
            process_folder(args.name, verbose=args.verbose)
            
            # Calculate and display elapsed time
            elapsed_time = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            
            print(f"\n✅ Processing completed successfully!")
            print(f"⏱️ Total processing time: {elapsed_str}")
            print(f"📁 Output folder: {args.name.replace('object_stage', 'mat_painting_stage')}")
            
            # Show completion timestamp
            completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕒 Completed at: {completion_time}")
            
        except Exception as e:
            # Display error information
            elapsed_time = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            
            print(f"\n❌ Error during processing: {str(e)}")
            print(f"⏱️ Processing time before error: {elapsed_str}")
            print(f"Please check the input folder path and permissions.")
            
            # Show error timestamp
            error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕒 Error occurred at: {error_time}")
            
            # Re-raise to maintain original exit code
            raise
    else:
        parser.error("Either --name or --single-object must be provided")


