import bpy
import os
import math
import numpy as np
import mathutils

def apply_pbr_material(cube, folder_path):
    # Define texture filenames
    textures = {
        "base_color": "basecolor.png",
        "diffuse": "diffuse.png",
        "displacement": "displacement.png",
        "height": "height.png",
        "metallic": "metallic.png",
        "normal": "normal.png",
        "roughness": "roughness.png",
        "specular": "specular.png"
    }

    # Create a new material
    material = bpy.data.materials.new(name="PBR_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create the Principled BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Create output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Function to load and link texture
    def load_texture(input_name, texture_name, colorspace='sRGB'):
        texture_path = os.path.join(folder_path, textures[texture_name])
        print(f"Loading texture '{texture_name}' from {texture_path}")
        if os.path.exists(texture_path):
            texture_node = nodes.new(type="ShaderNodeTexImage")
            texture_node.image = bpy.data.images.load(texture_path)
            texture_node.location = (-400, 200 if input_name == "Base Color" else 0)
            if colorspace == 'Non-Color':
                texture_node.image.colorspace_settings.is_data = True
            if input_name in bsdf.inputs:
                links.new(texture_node.outputs["Color"], bsdf.inputs[input_name])
            return texture_node
        else:
            print(f"Texture '{texture_name}' not found at {texture_path}")

    # Load and connect each texture map
    load_texture("Base Color", "base_color")
    load_texture("Base Color", "diffuse")  # Optional: alternative for base color
    load_texture("Roughness", "roughness", colorspace='Non-Color')
    load_texture("Metallic", "metallic", colorspace='Non-Color')
    load_texture("Specular", "specular", colorspace='Non-Color')

    # Normal Map Handling
    normal_map = load_texture("Normal", "normal", colorspace='Non-Color')
    if normal_map:
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        normal_map_node.location = (-200, -200)
        links.new(normal_map.outputs["Color"], normal_map_node.inputs["Color"])
        links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

    # Displacement Map Handling
    displacement_path = os.path.join(folder_path, textures["displacement"])
    if os.path.exists(displacement_path):
        displacement_map = nodes.new(type="ShaderNodeTexImage")
        displacement_map.image = bpy.data.images.load(displacement_path)
        displacement_map.location = (-400, -300)
        displacement_map.image.colorspace_settings.is_data = True

        displacement_node = nodes.new(type="ShaderNodeDisplacement")
        displacement_node.location = (200, -200)
        links.new(displacement_map.outputs["Color"], displacement_node.inputs["Height"])
        links.new(displacement_node.outputs["Displacement"], output_node.inputs["Displacement"])

    # Assign the material to the cube
    if cube.data.materials:
        cube.data.materials[0] = material
    else:
        cube.data.materials.append(material)
    
    return cube

def calculate_center(obj):
    bounds = [child.matrix_world @ mathutils.Vector(corner) for child in obj.children for corner in child.bound_box]
    min_bound = mathutils.Vector(map(min, zip(*bounds)))
    max_bound = mathutils.Vector(map(max, zip(*bounds)))
    # bbox size 
    bbox_size = (max_bound - min_bound)
    return (min_bound + max_bound) / 2, bbox_size

def load_obj_with_pbr(object_folder, material_file, name_obj, glass_part_name="window_glass"):
    # Get parts from decomposed folder (OBJ files), not material folder
    # This ensures all parts are processed even if materials are missing
    obj_files = glob.glob(os.path.join(object_folder, "*.obj"))
    parts_in_folder = []
    for obj_file in obj_files:
        part_name = os.path.basename(obj_file).replace(".obj", "")
        # Skip .mtl files and other non-part files
        if part_name and not part_name.endswith(".mtl"):
            parts_in_folder.append(part_name)
    parts_in_folder = sorted(parts_in_folder)

    # Create an empty object to act as a parent
    parent_obj = bpy.data.objects.new(name=name_obj, object_data=None)
    bpy.context.collection.objects.link(parent_obj)

    for part in parts_in_folder:
        print("part!!!!!:", part)
        file_path = os.path.join(object_folder, f"{part}.obj")
        
        # Check if OBJ file exists
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: OBJ file not found for part '{part}': {file_path}, skipping...")
            continue
            
        # Import the .obj file
        bpy.ops.import_scene.obj(filepath=file_path)
        imported_obj = bpy.context.selected_objects[0]

        # --- SCALE UVS BY 10× ---
        if imported_obj.type == 'MESH' and imported_obj.data.uv_layers:
            for uv_layer in imported_obj.data.uv_layers:
                for uv_data in uv_layer.data:
                    uv = uv_data.uv
                    uv.x *= 3
                    uv.y *= 3
            print(f"Scaled UVs of '{imported_obj.name}' by 10×")

        # Special case: if this is the glass part, assign a transparent glass material
        if part == glass_part_name:
            # --- make / fetch material ---------------------------------------------------
            mat_name = f"{part}_Glass"
            mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Clear default nodes
            nodes.clear()

            # ---------------------------------------------------------------------------
            # Use a Principled BSDF set up for glass:
            #   • Transmission = 1  → full transparency
            #   • Roughness  = 0    → crystal-clear
            #   • IOR        = 1.45 → typical glass
            #   • Alpha      = 1    → leave at 1 (Cycles handles transparency via Transmission)
            # ---------------------------------------------------------------------------
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
            bsdf.inputs["Transmission"].default_value = 1.0
            bsdf.inputs["Roughness"].default_value = 0.0
            bsdf.inputs["IOR"].default_value = 1.45

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (200, 0)
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

            # ---------------------------------------------------------------------------
            # Make sure the material renders as transparent in Eevee / viewport
            # ---------------------------------------------------------------------------
            mat.blend_method = "BLEND"        # 'HASHED' also works; avoids solid opaque look
            mat.shadow_method = "NONE"        # prevents dark opaque shadows
            mat.use_screen_refraction = True  # only matters if you enable refraction in Eevee

            # ---------------------------------------------------------------------------
            # Assign to object
            # ---------------------------------------------------------------------------
            if imported_obj.data.materials:
                imported_obj.data.materials[0] = mat
            else:
                imported_obj.data.materials.append(mat)

            print(f"Assigned transparent glass material to '{imported_obj.name}'")

            # Parent and continue to next part
            imported_obj.parent = parent_obj

            continue

        # Otherwise apply your PBR material
        # Handle naming mismatch between decomposed OBJ parts and material folders
        material_part_name = part
        if part == "Door.001":
            material_part_name = "Door"
        elif part == "default_material":
            # default_material might not have a specific material folder, use fallback
            material_part_name = None

        if material_part_name:
            folder_path = os.path.join(material_file, material_part_name)
        else:
            folder_path = None
        
        # Check if material folder exists (skip if folder_path is None for parts without materials)
        if folder_path is None or not os.path.exists(folder_path):
            if folder_path is None:
                print(f"⚠️  Info: Part '{part}' does not require a specific material folder")
            else:
                print(f"⚠️  Warning: Material folder not found for part '{part}': {folder_path}")

            # FALLBACK: Check if LLM-retrieved materials exist in select_mat
            # This happens when visual retrieval failed and optional LLM queries ran
            select_mat_path = os.path.join(object_folder, "select_mat", part)
            if os.path.exists(select_mat_path):
                print(f"   ✓ Found LLM-retrieved materials in select_mat, using those...")
                folder_path = select_mat_path
            else:
                print(f"   No materials found (neither visual nor LLM fallback), assigning default material...")
                # Create a default material so the part still renders
                default_mat = bpy.data.materials.new(name=f"{part}_Default")
                default_mat.use_nodes = True
                bsdf = default_mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    # Set a neutral gray color
                    bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
                    bsdf.inputs["Roughness"].default_value = 0.5
                if imported_obj.data.materials:
                    imported_obj.data.materials[0] = default_mat
                else:
                    imported_obj.data.materials.append(default_mat)

                # Parent and continue to next part
                imported_obj.parent = parent_obj
                continue
        
        # Material folder exists (either from visual retrieval or LLM fallback), apply PBR material
        apply_pbr_material(imported_obj, folder_path)

        # Parent the part under our empty parent
        imported_obj.parent = parent_obj

    # Center the whole assembly
    center, bbox_size = calculate_center(parent_obj)
    parent_obj.location -= center

    # Add another empty at world origin and parent the group to it
    empty = bpy.data.objects.new(f"{name_obj}_ROOT", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0, 0, 0)
    parent_obj.parent = empty

    return empty, bbox_size

def load_obj_and_group_with_pbr(object_file, material_file, obj_name):

    parent_object, bbox_size = load_obj_with_pbr(object_file, material_file, obj_name)

    # Switch to object mode to ensure we're in the correct context
    bpy.context.view_layer.objects.active = parent_object
    bpy.ops.object.select_all(action='DESELECT')
    parent_object.select_set(True)

    # Apply a 90-degree rotation around the X-axis to convert Z-up to Y-up
    parent_object.rotation_euler = (math.radians(-90), 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

    bbox_size = (bbox_size[0], bbox_size[2], bbox_size[1])

    return parent_object, bbox_size

# Function to scale the UV coordinates in a given UV layer.
def scale_uvs(uv_layer, scale_x=10, scale_y=10):
    # Loop through all UVs in the layer.
    for uv_data in uv_layer.data:
        uv = uv_data.uv
        uv[0] *= scale_x  # Scale the U coordinate (X-axis).
        uv[1] *= scale_y  # Scale the V coordinate (Y-axis).


import sys
import argparse
import glob
import json

# ==================== CLUSTER-AWARE MATERIAL LOOKUP ====================

def load_cluster_processing_results(scene_dir):
    """
    Load cluster-aware processing results to identify winning candidates.

    Args:
        scene_dir: Path to the mat_painting_stage scene directory

    Returns:
        dict: Cluster processing results, or None if not available
    """
    results_file = os.path.join(scene_dir, "cluster_aware_processing_results.json")

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cluster results: {e}")

    return None


def find_material_source_for_object(obj_name, scene_dir, cluster_results):
    """
    Find the material source for an object, considering cluster propagation.

    For objects that were non-candidates in cluster-aware processing,
    this returns the winning candidate's object name whose materials should be used.

    Args:
        obj_name: Name of the current object (e.g., "Chair5")
        scene_dir: Path to the mat_painting_stage scene directory
        cluster_results: Loaded cluster processing results (or None)

    Returns:
        str: Object name to use for materials (either self or winning candidate)
    """
    if not cluster_results or "cluster_results" not in cluster_results:
        return obj_name

    # Search through cluster results to find this object
    for cluster_result in cluster_results["cluster_results"]:
        candidates = cluster_result.get("candidates", [])
        non_candidates = cluster_result.get("non_candidates", [])
        winning_candidate = cluster_result.get("winning_candidate")

        # Check if this object is a non-candidate in this cluster
        if obj_name in non_candidates and winning_candidate:
            print(f"   ✓ {obj_name} is a cluster non-candidate, using materials from winner: {winning_candidate}")
            return winning_candidate

        # Check if this object is a candidate but not the winner
        # In this case, it has its own materials, so use them
        if obj_name in candidates:
            return obj_name

    return obj_name


def get_material_folder_path(obj_mat_folder, semantic_name, image_size, scene_dir, cluster_results):
    """
    Get the material folder path for an object, with cluster-aware fallback.

    This function handles the case where an object doesn't have its own materials
    because it received propagated materials from a cluster winning candidate.

    Args:
        obj_mat_folder: Path to the current object's mat_painting folder
        semantic_name: Semantic name for material lookup (e.g., "Chair_gpt")
        image_size: Image size for resized materials
        scene_dir: Path to the mat_painting_stage scene directory
        cluster_results: Loaded cluster processing results (or None)

    Returns:
        str: Path to the material folder to use, or None if not found
    """
    obj_name = os.path.basename(obj_mat_folder)

    # First, try the object's own material folder
    mat_folder = os.path.join(obj_mat_folder, f"selected_material_OT_with_adaptation_{image_size}", semantic_name)

    if os.path.exists(mat_folder):
        return mat_folder

    # Try non-resized folder
    mat_folder_non_resized = os.path.join(obj_mat_folder, "selected_material_OT_with_adaptation", semantic_name)
    if os.path.exists(mat_folder_non_resized):
        print(f"   Using non-resized material folder for {obj_name}")
        return mat_folder_non_resized

    # If cluster results available, try to find material from winning candidate
    if cluster_results:
        source_obj = find_material_source_for_object(obj_name, scene_dir, cluster_results)

        if source_obj != obj_name:
            # Use winning candidate's materials
            source_folder = os.path.join(scene_dir, source_obj)

            # Try resized folder first
            source_mat_folder = os.path.join(source_folder, f"selected_material_OT_with_adaptation_{image_size}", semantic_name)
            if os.path.exists(source_mat_folder):
                print(f"   Using cluster winner's materials: {source_mat_folder}")
                return source_mat_folder

            # Try non-resized folder
            source_mat_folder_non_resized = os.path.join(source_folder, "selected_material_OT_with_adaptation", semantic_name)
            if os.path.exists(source_mat_folder_non_resized):
                print(f"   Using cluster winner's non-resized materials: {source_mat_folder_non_resized}")
                return source_mat_folder_non_resized

            # Try winner's semantic name (it might differ slightly)
            source_semantic = ''.join([i for i in source_obj if not i.isdigit()])
            source_semantic = source_semantic.replace("_", "").replace("Wall", "") + "_gpt"

            source_mat_folder_alt = os.path.join(source_folder, f"selected_material_OT_with_adaptation_{image_size}", source_semantic)
            if os.path.exists(source_mat_folder_alt):
                print(f"   Using cluster winner's materials (alt semantic): {source_mat_folder_alt}")
                return source_mat_folder_alt

            source_mat_folder_alt_non_resized = os.path.join(source_folder, "selected_material_OT_with_adaptation", source_semantic)
            if os.path.exists(source_mat_folder_alt_non_resized):
                print(f"   Using cluster winner's non-resized materials (alt semantic): {source_mat_folder_alt_non_resized}")
                return source_mat_folder_alt_non_resized

    return None


# ==================== END CLUSTER-AWARE MATERIAL LOOKUP ====================

def parse_arguments():
    # Get the arguments passed after `--` to avoid Blender's internal arguments
    blender_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Render images for a specified scene with customizable image size.")
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the folder containing the raw scan data for the scene."
    )
    parser.add_argument(
        "--image_size", type=int, default=500,
        help="Resolution of the output render (default: 500)."
    )
    return parser.parse_args(blender_args)

# Parse the arguments
args = parse_arguments()
# Retrieve arguments
folder_to_process = args.path
image_size = args.image_size
all_items_in_folder = glob.glob(f"{folder_to_process}/*")

# Filter out log files and other non-object files that don't need processing
all_obj_in_folder = []
for item in all_items_in_folder:
    item_name = os.path.basename(item)

    # Skip log files and other non-object files
    if (item_name.endswith('.log') or
        item_name.endswith('.json') or
        item_name.endswith('.txt') or
        item_name.endswith('.png') or
        item_name.endswith('.jpg') or
        item_name.endswith('.mp4') or
        item_name.endswith('.glb') or
        item_name.endswith('.gltf') or
        'cache' in item_name.lower() or
        'video' in item_name.lower()):
        print(f"Skipping non-object file: {item_name}")
        continue

    # Only include directories (object folders)
    if os.path.isdir(item):
        all_obj_in_folder.append(item)

print(f"Found {len(all_obj_in_folder)} object folders to process")

# Load cluster processing results (if available) for cluster-aware material lookup
cluster_results = load_cluster_processing_results(folder_to_process)
if cluster_results:
    print(f"Loaded cluster-aware processing results: {cluster_results.get('total_clusters', 0)} clusters")
else:
    print("No cluster-aware processing results found (processing all objects individually)")

for obj_mat_folder in all_obj_in_folder:
    try:
        # remove all in the scene 
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        obj_name = obj_mat_folder.split("/")[-1]

        # remove all digits
        semantic_name = ''.join([i for i in obj_name if not i.isdigit()])

        if "Window" not in semantic_name and "Door" not in semantic_name:
            semantic_name = semantic_name.replace("_", "")
            semantic_name = semantic_name.replace("Wall","")
            semantic_name = semantic_name + "_gpt"
        else:
            # Windows and Doors use special naming: "Wall_" + semantic_name + "__gpt"
            # First remove "Wall" and "_" from semantic_name (like resize_texture.py does)
            semantic_name = semantic_name.replace("_", "")
            semantic_name = semantic_name.replace("Wall", "")
            # Then add "Wall_" prefix and "__gpt" suffix
            if "Window" in semantic_name:
                semantic_name = "Wall_" + semantic_name + "__gpt"
            elif "Door" in semantic_name:
                semantic_name = "Wall_" + semantic_name + "__gpt"

        # Use cluster-aware material folder lookup
        # This handles both normal objects and cluster-propagated objects
        mat_folder = get_material_folder_path(
            obj_mat_folder, semantic_name, image_size, folder_to_process, cluster_results
        )

        # If cluster-aware lookup failed, try direct path as fallback
        if mat_folder is None:
            direct_path = f"{obj_mat_folder}/selected_material_OT_with_adaptation_{str(image_size)}/{semantic_name}"
            if os.path.exists(direct_path):
                mat_folder = direct_path
            else:
                # Try non-resized folder
                non_resized_path = f"{obj_mat_folder}/selected_material_OT_with_adaptation/{semantic_name}"
                if os.path.exists(non_resized_path):
                    mat_folder = non_resized_path
                    print(f"⚠️  Resized folder not found, using non-resized: {mat_folder}")
                else:
                    print(f"⚠️  No material folder found for {obj_name}, will use fallback materials if available")

        print(f"Processing {semantic_name} (material folder: {mat_folder if mat_folder else 'NONE - will use fallback'})")
        obj_folder =  f"{obj_mat_folder}/Onboarded/decomposed"
        output_folder = f"{folder_to_process}_output_gltf"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        parent_obj, _= load_obj_and_group_with_pbr(obj_folder, mat_folder, obj_name)

        # Iterate over all objects in the scene.
        for obj in bpy.data.objects:
            # Process only mesh objects.
            if obj.type == 'MESH':
                mesh = obj.data

                # Check if the mesh has an active UV map.
                if mesh.uv_layers.active is not None:
                    print(f"Scaling UVs for object: {obj.name}")
                    scale_uvs(mesh.uv_layers.active, scale_x=10, scale_y=10)
                else:
                    print(f"Object '{obj.name}' has no active UV map.")
        bpy.context.view_layer.update()
        parent_obj.rotation_euler = (math.radians(90), 0, 0)
        bpy.ops.object.select_all(action='DESELECT')
        parent_obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
        bpy.ops.export_scene.gltf(
        filepath=f"{output_folder}/{obj_name}.glb",
        export_format='GLB')
    except Exception as e:
        print(f"Error processing {obj_name}: {e}")
        continue
