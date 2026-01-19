import numpy as np
import os
import bpy
import mathutils
import glob
import bpy

def setup_render_engine(engine='BLENDER_EEVEE'):
    """
    Set up render engine. Default is Eevee, but you can pass 'CYCLES' for Cycles.
    """
    scene = bpy.context.scene
    scene.render.engine = engine

    if engine == 'CYCLES':
        # --- Cycles settings ---
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 512
        scene.cycles.max_bounces = 8
        scene.cycles.min_bounces = 3
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        # scene.render.film_transparent = True  # uncomment if you need alpha

    else:
        # --- Eevee settings ---
        eevee = scene.eevee
        # viewport TAA samples
        eevee.taa_samples = 64
        # render TAA samples (was render_antialiasing_samples)
        eevee.taa_render_samples = 16  
        # common effects
        eevee.use_bloom = True
        eevee.use_ssr = True
        eevee.use_ssr_refraction = True
        eevee.use_gtao = True
        eevee.gtao_distance = 0.2
        eevee.use_soft_shadows = True



    return scene


def load_and_center_object(filepath):
    # Import .obj file and center the object in the scene
    bpy.ops.import_scene.obj(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # translate the object to the center, i.e. the max x, y, z and min x, y, z has same absolute value
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    center = obj.location.copy()

    # print("obj.location", obj.location)
    obj.location = (0, 0, 0)
    
    # Apply a simple shader with some emission for a material preview effect
    mat = bpy.data.materials.new(name="PreviewMaterial")
    mat.use_nodes = True
    bsdf_node = mat.node_tree.nodes.get("Principled BSDF")
    bsdf_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1)  # Base color, adjust as needed
    
    obj.data.materials.append(mat)

    # output the object_size information 
    obj_size = obj.dimensions
    return obj_size, center

def render_image(output_path, resolution_x=256, resolution_y=192):
    # Render the current scene and save the image
    bpy.context.scene.camera = bpy.data.objects["CustomCamera"]
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.ops.render.render(write_still=True)

def setup_camera(intrinsic_matrix, extrinsic_matrix, render_width, render_height):
    # Add and configure a new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "CustomCamera"
    cam_data = camera.data
    cam_data.type = 'PERSP'
    
    # Set intrinsic parameters
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    sensor_width = 32  # mm, adjustable if necessary
    cam_data.lens = fx * (sensor_width / render_width)
    cam_data.sensor_width = sensor_width
    cam_data.shift_x = (cx - render_width / 2) / render_width
    cam_data.shift_y = (cy - render_height / 2) / render_height

    # Convert and apply extrinsic matrix for Blender's coordinate system
    extrinsic_matrix_blender = np.array([
        [extrinsic_matrix[0, 0], -extrinsic_matrix[0, 1], -extrinsic_matrix[0, 2], extrinsic_matrix[0, 3]],
        [extrinsic_matrix[1, 0], -extrinsic_matrix[1, 1], -extrinsic_matrix[1, 2], extrinsic_matrix[1, 3]],
        [extrinsic_matrix[2, 0], -extrinsic_matrix[2, 1], -extrinsic_matrix[2, 2], extrinsic_matrix[2, 3]],
        [0, 0, 0, 1]
    ])
    camera.matrix_world = mathutils.Matrix(extrinsic_matrix_blender.tolist())
    return camera

def resize_intrinsic(intrinsic, original_size, new_size):
    """
    Adjust the intrinsic matrix for a resized image.
    
    Parameters:
    intrinsic (np.array): Original 3x3 intrinsic matrix.
    original_size (tuple): Original image size as (height, width).
    new_size (tuple): New image size as (height, width).
    
    Returns:
    np.array: Rescaled 3x3 intrinsic matrix.
    """
    # Calculate scaling factors
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    
    # Scale the focal lengths and principal point
    resized_intrinsic = intrinsic.copy()
    resized_intrinsic[0, 0] *= scale_x  # f_x
    resized_intrinsic[1, 1] *= scale_y  # f_y
    resized_intrinsic[0, 2] *= scale_x  # c_x
    resized_intrinsic[1, 2] *= scale_y  # c_y
    
    return resized_intrinsic

def onboarding_rendering(obj_files, img_size, output_path):
    # Clean scene by removing all objects, cameras, and lights
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Set render engine to material preview settings
    setup_render_engine()

    # Define paths and parameters
    predefined_poses_path = "litereality/LR_mat_painting/utils/cam_poses_level0.npy"
    output_folder = output_path

    # Load and center the object
    size, center = load_and_center_object(obj_files)
    size = max(size)

    radius = 0.001 * size * 1.5 # Scale radius by object size

    # Load and scale predefined poses
    poses = np.load(predefined_poses_path, allow_pickle=True)
    poses[:, :3, 3] *= radius  # Scale translation by radius

    # Filter poses based on z-axis position
    poses = poses[poses[:, 2, 3] >= 0]

    # Camera intrinsic matrix
    intrinsic = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ])

    width, height = img_size
    intrinsic = resize_intrinsic(intrinsic, [480, 640], img_size)
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Render images from each predefined camera pose
    for obj_id, extrinsic_matrix in enumerate(poses):
        camera = setup_camera(intrinsic, extrinsic_matrix, render_width=width, render_height=height)
        render_image(f"{output_folder}/render_image_{obj_id}.png", resolution_x=width, resolution_y=height)
        bpy.data.objects.remove(camera)

    return size, center








########## onboarding_rendering_with_pbr ##########


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
        file_path = os.path.join(object_folder, f"{part}.obj")
        
        # Check if OBJ file exists
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: OBJ file not found for part '{part}': {file_path}, skipping...")
            continue
            
        # Import the .obj file
        bpy.ops.import_scene.obj(filepath=file_path)
        imported_obj = bpy.context.selected_objects[0]

        # --- SCALE UVS BY 3× ---
        if imported_obj.type == 'MESH' and imported_obj.data.uv_layers:
            for uv_layer in imported_obj.data.uv_layers:
                for uv_data in uv_layer.data:
                    uv = uv_data.uv
                    uv.x *= 3
                    uv.y *= 3
            print(f"Scaled UVs of '{imported_obj.name}' by 3×")

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
        folder_path = os.path.join(material_file, part)
        
        # Check if material folder exists
        if not os.path.exists(folder_path):
            # Try to match without numeric suffix (OBJ files sometimes have .001, .002, etc. suffixes but material folders don't)
            part_name_without_suffix = part
            import re
            # Remove any . followed by digits at the end (e.g., .001, .002, .123)
            suffix_match = re.search(r'\.\d+$', part)
            if suffix_match:
                part_name_without_suffix = part[:suffix_match.start()]
                folder_path_alt = os.path.join(material_file, part_name_without_suffix)
                if os.path.exists(folder_path_alt):
                    print(f"⚠️  Material folder not found for '{part}', but found '{part_name_without_suffix}', using that...")
                    folder_path = folder_path_alt
                else:
                    folder_path = os.path.join(material_file, part)  # Reset to original
            
            # If still not found, try fallback
            if not os.path.exists(folder_path):
                print(f"⚠️  Warning: Material folder not found for part '{part}': {folder_path}")
                
                # FALLBACK: Check if LLM-retrieved materials exist in select_mat
                # Try both with and without numeric suffix
                select_mat_path = os.path.join(object_folder, "select_mat", part)
                import re
                suffix_match_fallback = re.search(r'\.\d+$', part)
                if not os.path.exists(select_mat_path) and suffix_match_fallback:
                    select_mat_path = os.path.join(object_folder, "select_mat", part_name_without_suffix)
                
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

    return bbox_size




def setup_world_lighting(hdri_path=None):
    """
    Set up world/environment lighting for the scene.
    Uses HDRI for lighting but renders with transparent background (HDRI won't be visible).
    If hdri_path is provided, uses HDRI. Otherwise, uses a bright background.
    """
    # Enable transparent background - HDRI provides lighting but won't show in render
    bpy.context.scene.render.film_transparent = True
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    if hdri_path and os.path.exists(hdri_path):
        # Use HDRI for realistic lighting (background will be transparent)
        env_tex = nodes.new(type='ShaderNodeTexEnvironment')
        env_tex.image = bpy.data.images.load(hdri_path)
        env_tex.location = (-300, 0)
        
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.location = (0, 0)
        bg_node.inputs["Strength"].default_value = 1.5  # Increase brightness for better visibility
        
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (300, 0)
        
        links.new(env_tex.outputs['Color'], bg_node.inputs['Color'])
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        # Suppress verbose output
    else:
        # Use bright background lighting (fallback)
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.location = (0, 0)
        bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # White
        bg_node.inputs["Strength"].default_value = 3.0  # Very bright lighting to avoid dark renders
        
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        output_node.location = (300, 0)
        
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
        # Suppress verbose output


def onboarding_rendering_with_pbr(obj_name, img_size, output_path, object_folder, material_file):
    # Clean scene by removing all objects, cameras, and lights
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Set render engine to material preview settings
    setup_render_engine()
    
    # Set up world lighting - try HDRI first, fallback to bright background
    hdri_path = "litereality_database/symmetrical_garden_02_4k.exr"
    setup_world_lighting(hdri_path)

    # Define paths and parameters
    predefined_poses_path = "litereality/LR_mat_painting/utils/cam_poses_level0.npy"
    output_folder = output_path


    # Load and center the object

    # object_folder, material_file, name_obj
    size = load_obj_with_pbr(object_folder=object_folder, material_file=material_file, name_obj=obj_name)

    size = max(size)

    radius = 0.001 * size * 1.5 # Scale radius by object size

    # Load and scale predefined poses
    poses = np.load(predefined_poses_path, allow_pickle=True)
    poses[:, :3, 3] *= radius  # Scale translation by radius

    # Filter poses based on z-axis position
    poses = poses[poses[:, 2, 3] >= 0]

    # Camera intrinsic matrix
    intrinsic = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ])

    width, height = img_size
    intrinsic = resize_intrinsic(intrinsic, [480, 640], img_size)
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Render images from each predefined camera pose
    for obj_id, extrinsic_matrix in enumerate(poses):
        if obj_id >=1:
            continue
        camera = setup_camera(intrinsic, extrinsic_matrix, render_width=width, render_height=height)
        render_image(f"{output_folder}/render_image_{obj_id}.png", resolution_x=width, resolution_y=height)
        bpy.data.objects.remove(camera)

    return size


import os
import numpy as np
import bpy

def setup_render_engine_and_world(hdri_path):
    scene = bpy.context.scene

    # 1) Cycles engine
    scene.render.engine = 'CYCLES'
    scene.cycles.feature_set = 'SUPPORTED'
    scene.cycles.device = 'GPU'             # or 'CPU' if you don't have GPU
    scene.cycles.samples = 128              # adjust as needed

    # 2) Transparent background
    scene.render.film_transparent = True

    # 3) Set up world nodes for HDRI lighting
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    for n in nodes:
        nodes.remove(n)

    # Create nodes
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    mapping   = nodes.new(type='ShaderNodeMapping')
    env_tex   = nodes.new(type='ShaderNodeTexEnvironment')
    bg_node   = nodes.new(type='ShaderNodeBackground')
    out_node  = nodes.new(type='ShaderNodeOutputWorld')

    # Load HDRI image
    env_tex.image = bpy.data.images.load(hdri_path)

    # Link them up
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'],   env_tex.inputs['Vector'])
    links.new(env_tex.outputs['Color'],    bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])


def onboarding_rendering_with_pbr_highquality(obj_name, img_size, output_path, object_folder, material_file, hdri_path):
    # --- Clean scene ---
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # --- Engine & World Setup ---
    setup_render_engine_and_world(hdri_path)

    # --- Load your object with PBR materials (your existing function) ---
    size = load_obj_with_pbr(
        object_folder=object_folder,
        material_file=material_file,
        name_obj=obj_name
    )
    size = max(size)
    radius = 0.001 * size * 1.5

    # --- Camera poses from numpy array ---
    predefined_poses_path = "litereality/LR_mat_painting/utils/cam_poses_level0.npy"
    poses = np.load(predefined_poses_path, allow_pickle=True)
    poses[:, :3, 3] *= radius
    poses = poses[poses[:, 2, 3] >= 0]

    # --- Intrinsic matrix resizing (your existing utility) ---
    width, height = img_size
    intrinsic = np.array([
        [572.4114,    0.0,      325.2611],
        [   0.0,    573.57043,  242.04899],
        [   0.0,       0.0,       1.0   ]
    ])
    intrinsic = resize_intrinsic(intrinsic, [480, 640], img_size)

    # --- Ensure output folder exists ---
    os.makedirs(output_path, exist_ok=True)

    # --- Render loop ---
    for idx, extrinsic in enumerate(poses):

        if idx >= 1:
            continue
        cam = setup_camera(
            intrinsic,
            extrinsic,
            render_width=width,
            render_height=height
        )
        filename = os.path.join(output_path, f"render_image_{idx:03d}.png")
        render_image(
            output_path=filename,
            resolution_x=width,
            resolution_y=height
        )
        # Remove camera to keep scene clean
        bpy.data.objects.remove(cam, do_unlink=True)

    return size
