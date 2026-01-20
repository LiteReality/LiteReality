import pickle
import sys
import numpy as np
import json
import os
import sys
import bpy
import mathutils
import math
from mathutils import Vector
import trimesh
import random
import glob
import argparse

from shapely.geometry import Polygon
import pickle


def add_skybox(exr_file_path, rotation_x_deg=90):

        # Set the world to use nodes
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_environment = nodes.new(type='ShaderNodeTexEnvironment')
    node_background = nodes.new(type='ShaderNodeBackground')
    node_output = nodes.new(type='ShaderNodeOutputWorld')

    # Load the HDRI file
    lightbox_path = exr_file_path
    node_environment.image = bpy.data.images.load(lightbox_path)

    # Set up mapping: rotate 90 degrees along X
    node_mapping.inputs['Rotation'].default_value[0] = math.radians(rotation_x_deg)

    # Arrange nodes (optional, for UI clarity)
    node_tex_coord.location = (-600, 0)
    node_mapping.location = (-400, 0)
    node_environment.location = (-200, 0)
    node_background.location = (0, 0)
    node_output.location = (300, 0)

    # Link nodes
    links.new(node_tex_coord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_environment.inputs['Vector'])
    links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

    # Set environment strength
    node_background.inputs['Strength'].default_value = 1.0  # Adjust as needed

    print("World environment set with rotated HDRI (90° on X axis)")


def create_pbr_material_from_folder(folder_path, material_name="floor_PBR_Material"):
    # Supported texture types and their corresponding BSDF input
    texture_map = {
        "basecolor": ("Base Color", "sRGB"),
        "roughness": ("Roughness", "Non-Color"),
        "metallic": ("Metallic", "Non-Color"),
        "specular": ("Specular", "Non-Color"),  # This will be skipped if the input doesn't exist.
        "normal": ("Normal", "Non-Color"),
        "height": ("Displacement", "Non-Color"),
        "displacement": ("Displacement", "Non-Color"),
    }

    # Create a new material and enable nodes
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear existing nodes
    for node in list(nodes):
        nodes.remove(node)

    # Create the Principled BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Create the Material Output node and connect the BSDF to it
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # Helper function to load and connect a texture
    def load_texture(texture_type, input_name, colorspace):
        texture_file = next((f for f in os.listdir(folder_path) if texture_type in f.lower()), None)
        if texture_file:
            texture_path = os.path.join(folder_path, texture_file)
            texture_node = nodes.new(type="ShaderNodeTexImage")
            texture_node.image = bpy.data.images.load(texture_path)
            texture_node.location = (-400, len(nodes) * -200)
            if colorspace == "Non-Color":
                texture_node.image.colorspace_settings.is_data = True
            # Check if the BSDF node has the expected input socket
            if input_name in bsdf.inputs:
                links.new(texture_node.outputs["Color"], bsdf.inputs[input_name])
            else:
                print(f"Input '{input_name}' not found in BSDF node. Skipping texture '{texture_type}'.")
            return texture_node
        else:
            print(f"Texture for '{texture_type}' not found in {folder_path}")
            return None

    # Load and connect each texture based on the texture_map
    for texture_type, (input_name, colorspace) in texture_map.items():
        if input_name == "Normal":
            normal_map = load_texture(texture_type, input_name, colorspace)
            if normal_map:
                normal_map_node = nodes.new(type="ShaderNodeNormalMap")
                normal_map_node.location = (-200, -200)
                links.new(normal_map.outputs["Color"], normal_map_node.inputs["Color"])
                links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])
        elif input_name == "Displacement":
            # Handle displacement separately to add a Displacement node
            displacement_file = next((f for f in os.listdir(folder_path) if texture_type in f.lower()), None)
            if displacement_file:
                displacement_path = os.path.join(folder_path, displacement_file)
                displacement_node = nodes.new(type="ShaderNodeDisplacement")
                displacement_node.location = (200, -200)

                displacement_map = nodes.new(type="ShaderNodeTexImage")
                displacement_map.image = bpy.data.images.load(displacement_path)
                displacement_map.location = (-400, -300)
                displacement_map.image.colorspace_settings.is_data = True

                links.new(displacement_map.outputs["Color"], displacement_node.inputs["Height"])
                links.new(displacement_node.outputs["Displacement"], output_node.inputs["Displacement"])
            else:
                print(f"Displacement texture for '{texture_type}' not found in {folder_path}")
        else:
            load_texture(texture_type, input_name, colorspace)

    print(f"Material '{material_name}' created with textures from {folder_path}")
    return material

def setup_render_engine(render_engine='CYCLES'):
    # Set the render engine based on the parameter
    bpy.context.scene.render.engine = render_engine
    
    if render_engine == 'CYCLES':
        # Configure Cycles specific settings - optimized for speed with good quality
        try:
            # Enable GPU in Blender preferences if available
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX', 'OPENCL'
            bpy.context.preferences.addons['cycles'].preferences.get_devices()
            for device in bpy.context.preferences.addons['cycles'].preferences.devices:
                device.use = True  # Enable all available GPUs
            
            # Set Cycles settings - balanced for speed and quality
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.scene.cycles.samples = 64  # Lower samples for speed (was 128)
            bpy.context.scene.cycles.preview_samples = 32  # Preview samples
            
            # Light bounces - reduced for speed while maintaining realism
            bpy.context.scene.cycles.max_bounces = 6  # Total bounces (was 8)
            bpy.context.scene.cycles.diffuse_bounces = 3  # Diffuse bounces
            bpy.context.scene.cycles.glossy_bounces = 3  # Glossy/specular bounces
            bpy.context.scene.cycles.transmission_bounces = 4  # Glass/transparency
            bpy.context.scene.cycles.volume_bounces = 0  # Disable volume for speed
            bpy.context.scene.cycles.transparent_max_bounces = 4
            
            # Performance optimizations
            bpy.context.scene.cycles.use_adaptive_sampling = True  # Adaptive sampling for speed
            bpy.context.scene.cycles.adaptive_threshold = 0.05  # Stop sampling when pixels converge
            
            # Tile size optimization for GPU
            bpy.context.scene.cycles.tile_size = 256  # Larger tiles for GPU rendering
            
            # Enable denoising - crucial for lower sample counts
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.denoiser = 'OPTIX'  # Fast GPU denoiser (use 'OPENIMAGEDENOISE' for CPU)
            bpy.context.scene.cycles.use_preview_denoising = True
            
            # Additional quality settings
            bpy.context.scene.cycles.caustics_reflective = False  # Disable caustics for speed
            bpy.context.scene.cycles.caustics_refractive = False
            bpy.context.scene.cycles.blur_glossy = 0.5  # Slight glossy blur for noise reduction
            
            print("✓ Cycles renderer configured with GPU acceleration")
            print(f"  Samples: 64 (adaptive)")
            print(f"  Bounces: 6 max (diffuse: 3, glossy: 3)")
            print(f"  Denoiser: OPTIX (GPU)")
        except Exception as e:
            print(f"⚠️  Error configuring Cycles: {e}")
            print("   Using Cycles with default settings.")
    elif render_engine == 'BLENDER_EEVEE':
        # Configure EEVEE specific settings
        bpy.context.scene.eevee.taa_render_samples = 64  # Adjust anti-aliasing samples
        bpy.context.scene.eevee.use_soft_shadows = True
        bpy.context.scene.eevee.use_ssr = True  # Enable screen space reflections
        bpy.context.scene.eevee.use_ssr_refraction = True
        bpy.context.scene.eevee.use_gtao = True  # Enable ambient occlusion
        bpy.context.scene.eevee.gtao_distance = 0.2
        print("✓ EEVEE renderer configured")

def create_random_material(name):
    """Creates a random material with a unique name."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")

    if principled_bsdf:
        # Random color
        principled_bsdf.inputs["Base Color"].default_value = (
            0.7, 0.7, 0.5, 1)
        # Random roughness
        principled_bsdf.inputs["Roughness"].default_value = random.uniform(0.1, 0.8)
        # Random metallic
        principled_bsdf.inputs["Metallic"].default_value = random.uniform(0, 1)

    return mat

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

def load_obj_with_pbr_glb_corrected(filepath, name_obj):
    # Check if file exists
    if not filepath or not os.path.exists(filepath):
        # Try with absolute path from project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        abs_filepath = os.path.join(project_root, filepath)
        if os.path.exists(abs_filepath):
            filepath = abs_filepath
        else:
            raise FileNotFoundError(f"GLB file not found: {filepath} (also tried: {abs_filepath})")
    
    # Import the GLB file
    bpy.ops.import_scene.gltf(filepath=filepath)
    
    # Capture imported objects. Depending on your scene,
    # you might need a more selective filter than just selected_objects.
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    # Optionally, apply transforms to ensure correct bounding box calculations.
    for obj in imported_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Create an empty parent object.
    parent_obj = bpy.data.objects.new(name=name_obj, object_data=None)
    bpy.context.collection.objects.link(parent_obj)
    
    # Parent the imported objects to this new empty.
    for obj in imported_objects:
        obj.parent = parent_obj

    # Calculate the center and bounding box size.
    center, bbox_size = calculate_center_and_bbox_size_new(imported_objects)
    print("Calculated center:", center)
    
    # Move the parent so that the bounding box is centered at the origin.
    parent_obj.location = -center
    
    # Optionally, update the scene so that transforms are recalculated.
    bpy.context.view_layer.update()
    
    return parent_obj, bbox_size

def calculate_center_and_bbox_size_new(objects):
    """Compute the overall center and size of the bounding box for given objects."""
    all_verts = []
    for obj in objects:
        if obj.type == 'MESH' and obj.data is not None:
            for v in obj.data.vertices:
                # Transform the vertex coordinate to world space.
                all_verts.append(obj.matrix_world @ v.co)
    
    if not all_verts:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((0, 0, 0))
    
    # Compute min and max for each axis.
    min_x = min(v.x for v in all_verts)
    min_y = min(v.y for v in all_verts)
    min_z = min(v.z for v in all_verts)
    max_x = max(v.x for v in all_verts)
    max_y = max(v.y for v in all_verts)
    max_z = max(v.z for v in all_verts)
    
    center = mathutils.Vector(((min_x + max_x) / 2,
                               (min_y + max_y) / 2,
                               (min_z + max_z) / 2))
    
    bbox_size = mathutils.Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    
    return center, bbox_size

def calculate_center(objects):
        """
        Calculate the center and bounding box size of a collection of objects.
        """
        vertices_coords = [
            obj.matrix_world @ mathutils.Vector(vertex.co)
            for obj in objects.children
            for vertex in obj.data.vertices
        ]

        # Calculate the minimum and maximum bounds based on all vertices
        min_bound = mathutils.Vector(map(min, zip(*vertices_coords)))
        max_bound = mathutils.Vector(map(max, zip(*vertices_coords)))

        center = (min_bound + max_bound) / 2
        bbox_size = max_bound - min_bound

        return center, bbox_size

def load_obj_and_group_with_pbr_glb(object_file, obj_name):
    
    parent_object, bbox_size = load_obj_with_pbr_glb_corrected(object_file, obj_name)

    # Switch to object mode to ensure we're in the correct context
    bpy.context.view_layer.objects.active = parent_object
    bpy.ops.object.select_all(action='DESELECT')
    parent_object.select_set(True)

    # Apply a 90-degree rotation around the X-axis to convert Z-up to Y-up
    parent_object.rotation_euler = (math.radians(-90), 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

    bbox_size = (bbox_size[0], bbox_size[2], bbox_size[1])

    return parent_object, bbox_size

def load_obj_and_group(filepath):
    """Load OBJ file and group imported objects under a parent."""
    bpy.ops.import_scene.obj(filepath=filepath)

    # Get the imported objects
    imported_objects = [obj for obj in bpy.context.selected_objects]

    # Create an empty object to act as a parent
    parent_object = bpy.data.objects.new("Parent", None)
    bpy.context.collection.objects.link(parent_object)
    for obj in imported_objects:
        obj.parent = parent_object

    scene = trimesh.load(filepath)
    bbox = scene.bounds
    xyz_range = (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2])
    
    # Switch to object mode to ensure we're in the correct context
    bpy.context.view_layer.objects.active = parent_object
    bpy.ops.object.select_all(action='DESELECT')
    parent_object.select_set(True)

    # Apply a 90-degree rotation around the X-axis to convert Z-up to Y-up
    parent_object.rotation_euler = (math.radians(-90), 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

    return parent_object, xyz_range

def load_obj_with_pbr(object_folder, material_file, name_obj):
    
    parts_in_folder = os.listdir(object_folder)

    # Create an empty object to act as a parent
    parent_obj = bpy.data.objects.new(name=name_obj, object_data=None)
    bpy.context.collection.objects.link(parent_obj)
    for part in parts_in_folder:
        if not part.endswith("obj"):
            continue
        # Define the path to your .obj file
        file_path = f"{object_folder}/{part}"
        # Import the .obj file
        bpy.ops.import_scene.obj(filepath=file_path)
        folder_path = f"{material_file}/{part}"  # Replace with the path to your texture folder
        imported_obj = bpy.context.selected_objects[0]
        # apply_pbr_material(imported_obj, folder_path)
        # Set the imported object as a child of the empty parent object
        imported_obj.parent = parent_obj

        # Assuming 'parent_obj' is the parent of all imported objects
    center, bbox_size = calculate_center(parent_obj)

    bpy.context.view_layer.objects.active = parent_obj
    # Move the parent back to its original position if needed
    parent_obj.location -= center

    empty = bpy.data.objects.new(f"{name_obj}", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0, 0, 0)
    # Set the empty object as the parent of the parent_obj
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

def load_rgbd_scans(file_path):

    # Import the .obj file
    bpy.ops.import_scene.obj(filepath=file_path)
    
    # Get the imported object (assuming it's the first selected object)
    obj = bpy.context.selected_objects[0]
    
    # Apply transformations to normalize the object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Rotate the object to correct for the coordinate system difference
    obj.rotation_euler = (math.radians(-90), 0, 0)  # Rotate 90 degrees on the X-axis
    return obj

def add_light(location):
    # point light shotting down from the ceiling
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=location)
    light = bpy.context.object
    light.data.energy = 30
    light.data.shadow_soft_size = 0.1

def create_holes_on_walls(room, hole_info, wall_obj, bbox_of_wall, index, wall_raw_name):
    
    if hole_info != []:
        num_hole = len(hole_info["type"])

        hole_model_list = []
        
        for i in range(num_hole):
            type = hole_info["type"][i]

            obj_name = f"{wall_raw_name}_{type}_{i}"

            hole_dim = hole_info["dimension"][i]
            hole_x_1, hole_y_1, hole_x_2, hole_y_2 = hole_dim[0][0], hole_dim[0][1], hole_dim[1][0], hole_dim[1][1]
            size_of_hole = (hole_x_2 - hole_x_1, hole_y_2 - hole_y_1,  bbox_of_wall[2])
            position_of_hole = ((hole_x_1 + hole_x_2) / 2, (hole_y_1 + hole_y_2) / 2, 0)

            hole_type = type
            object_file = f"output/mat_painting_stage/{room}_output_gltf/{obj_name}.glb"

            print("object_file", object_file)

            # Check if GLB file exists (try relative path first, then absolute)
            glb_exists = os.path.exists(object_file)
            if not glb_exists:
                # Try absolute path from script location
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(script_dir))
                    abs_object_file = os.path.join(project_root, object_file)
                    glb_exists = os.path.exists(abs_object_file)
                    if glb_exists:
                        object_file = abs_object_file
                except:
                    pass

            # Use GLB if it exists, otherwise fall back to OBJ template
            if glb_exists:
                hole_model, size = load_obj_and_group_with_pbr_glb(object_file, "hole_type")
            else:
                # Fall back to OBJ template files
                hole_type_lower = hole_type.lower()
                if hole_type_lower == "door":
                    template_file = "litereality_database/ob_template/Door.obj"
                elif hole_type_lower == "window":
                    template_file = "litereality_database/ob_template/Window.obj"
                else:
                    print(f"Warning: Unknown hole type '{hole_type}', skipping...")
                    continue

                # Resolve template file path (try relative first, then absolute)
                if not os.path.exists(template_file):
                    try:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(os.path.dirname(script_dir))
                        abs_template_file = os.path.join(project_root, template_file)
                        if os.path.exists(abs_template_file):
                            template_file = abs_template_file
                    except:
                        pass

                print(f"Warning: GLB file not found: {object_file}")
                print(f"   Using template file instead: {template_file}")
                hole_model, size = load_obj_and_group(template_file)
            hole_model.name = f"hole_type_{wall_raw_name}_{i}"

            # Scale the object
            hole_model.scale = (size_of_hole[0] / size[0], size_of_hole[1] / size[1], size_of_hole[2] / size[2])
            hole_model.location = position_of_hole

            # Set the hole_model as the active object
            bpy.context.view_layer.objects.active = hole_model
            hole_model.select_set(True)

            # Apply transformations to the active object
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
            hole_model_list.append(hole_model)
            # create cube for hole
            bpy.ops.mesh.primitive_cube_add(size=1)

            hole_obj = bpy.context.object
            hole_obj.name = f"Hole_{index}"
            hole_obj.scale = size_of_hole
            hole_obj.location = position_of_hole

            # Perform boolean difference to cut the hole in the wall
            boolean_modifier = wall_obj.modifiers.new(name="Hole_Cutter", type='BOOLEAN')
            boolean_modifier.operation = 'DIFFERENCE'
            boolean_modifier.object = hole_obj
            # Apply the boolean modifier
            bpy.context.view_layer.objects.active = wall_obj
            bpy.ops.object.modifier_apply(modifier=boolean_modifier.name)
            # Remove the hole object after applying the boolean
            bpy.data.objects.remove(hole_obj, do_unlink=True)

    return hole_model_list


def create_floor(walls, thickness, room):
    """
    Create 3D floor meshes in Blender from wall polygons,
    apply the imported GLB floor material, then rebuild & scale UVs.
    
    Args:
        walls (dict): Dictionary of walls loaded from the pickle file.
        thickness (float): Thickness of the floor extrusion.
        room (str): Room name, used to locate the GLB file.
    """
    def create_mesh(polygon, name, base_height, thickness):
        exterior_coords = list(polygon.exterior.coords)
        vertices = [(x, base_height, z) for x, z in exterior_coords]
        vertices += [(x, base_height + thickness, z) for x, z in exterior_coords]

        num_points = len(exterior_coords)
        faces = [[i, i + 1, i + 1 + num_points, i + num_points] 
                 for i in range(num_points - 1)]
        faces.append(list(range(num_points)))                # Bottom face
        faces.append([i + num_points for i in range(num_points)])  # Top face

        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)

        return obj

    # --- build floors from wall polygons ---
    floor_objects = []
    for i, wall in walls.items():
        pts = [(x, z) for x, _, z in wall]
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        poly = Polygon(pts)
        if not poly.is_valid:
            print(f"Skipping invalid polygon for wall {i}")
            continue

        avg_height = sum(y for _, y, _ in wall) / len(wall)
        obj_floor = create_mesh(poly, f"floor_{i}", avg_height, thickness)
        floor_objects.append(obj_floor)

    # --- import GLB and grab material ---
    glb_path = f"output/mat_painting_stage/{room}_output_gltf/Floor.glb"
    
    # Try absolute path if relative doesn't exist
    if not os.path.exists(glb_path):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            abs_glb_path = os.path.join(project_root, glb_path)
            if os.path.exists(abs_glb_path):
                glb_path = abs_glb_path
        except:
            pass
    
    floor_material = None
    imported = []
    
    if not os.path.exists(glb_path):
        print(f"⚠️  Warning: Floor GLB not found at {glb_path}, floor will be created without material")
    else:
        bpy.ops.import_scene.gltf(filepath=glb_path)
        imported = bpy.context.selected_objects

        for obj in imported:
            if obj.type == 'MESH' and obj.material_slots:
                floor_material = obj.material_slots[0].material
                if floor_material:
                    floor_material.name = "Floor"
                    floor_material.use_fake_user = True
                    break

        # delete the imported helper objects
        if imported:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in imported:
                obj.select_set(True)
            bpy.ops.object.delete()

    # --- apply material, rebuild & scale UVs ---
    if floor_material:
        for obj_floor in floor_objects:
            # assign material
            obj_floor.data.materials.clear()
            obj_floor.data.materials.append(floor_material)

            # rebuild UV map
            mesh = obj_floor.data
            while mesh.uv_layers:
                mesh.uv_layers.remove(mesh.uv_layers[0])
            mesh.uv_layers.new(name="UVMap")

            # unwrap in Edit Mode
            bpy.context.view_layer.objects.active = obj_floor
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')

            # scale UVs by 5× in X and Y
            uv_data = mesh.uv_layers.active.data
            for luv in uv_data:
                luv.uv.x *= 5.0
                luv.uv.y *= 5.0
    else:
        print("Warning: No floor material found in the imported GLB file")

    print("Floor meshes created, UV-mapped & materials applied.")


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


def matrix_to_rotation_translation(matrix):
    """Extract rotation matrix and translation vector from a 4x4 transformation matrix."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return rotation, translation


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the dot product
    dot = np.dot(q1, q2)

    # If the dot product is negative, negate one quaternion to take the shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Compute the angle between quaternions
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    # Compute the interpolated quaternion
    q2_perp = q2 - dot * q1
    q2_perp = q2_perp / np.linalg.norm(q2_perp)

    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


def interpolate_extrinsics(extrinsic1, extrinsic2, num_interpolations=3):
    """
    Interpolate between two extrinsic matrices to create smooth camera transitions.

    Parameters:
    extrinsic1 (np.array): First 4x4 extrinsic matrix.
    extrinsic2 (np.array): Second 4x4 extrinsic matrix.
    num_interpolations (int): Number of intermediate frames to create between the two key frames.

    Returns:
    list: List of interpolated 4x4 extrinsic matrices (excluding the endpoints).
    """
    # Extract rotation and translation from both matrices
    R1, t1 = matrix_to_rotation_translation(extrinsic1)
    R2, t2 = matrix_to_rotation_translation(extrinsic2)

    # Convert rotation matrices to quaternions
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)

    interpolated_matrices = []

    for i in range(1, num_interpolations + 1):
        # Interpolation parameter (0 to 1, excluding endpoints)
        t = i / (num_interpolations + 1)

        # Linear interpolation for translation
        t_interp = t1 * (1 - t) + t2 * t

        # SLERP for rotation
        q_interp = slerp(q1, q2, t)
        R_interp = quaternion_to_rotation_matrix(q_interp)

        # Construct the interpolated 4x4 matrix
        matrix_interp = np.eye(4)
        matrix_interp[:3, :3] = R_interp
        matrix_interp[:3, 3] = t_interp

        interpolated_matrices.append(matrix_interp)

    return interpolated_matrices


def create_smooth_camera_trajectory(extrinsic_list, num_interpolations=3):
    """
    Create a smooth camera trajectory by interpolating between key camera poses.

    Parameters:
    extrinsic_list (list): List of (id, extrinsic_matrix) tuples for key camera poses.
    num_interpolations (int): Number of intermediate frames between each pair of key frames.

    Returns:
    list: List of (id_str, extrinsic_matrix) tuples for all camera poses (key + interpolated).
    """
    if len(extrinsic_list) < 2:
        return [(str(id), ext) for id, ext in extrinsic_list]

    smooth_trajectory = []

    for i in range(len(extrinsic_list)):
        id_curr, ext_curr = extrinsic_list[i]

        # Add the current key frame
        smooth_trajectory.append((f"{id_curr}", ext_curr))

        # If there's a next key frame, interpolate
        if i < len(extrinsic_list) - 1:
            id_next, ext_next = extrinsic_list[i + 1]

            # Create interpolated frames
            interpolated = interpolate_extrinsics(ext_curr, ext_next, num_interpolations)

            # Add interpolated frames with IDs like "5_interp_1", "5_interp_2", etc.
            for j, ext_interp in enumerate(interpolated):
                interp_id = f"{id_curr}_interp_{j+1}"
                smooth_trajectory.append((interp_id, ext_interp))

    return smooth_trajectory

def load_processed_data(room):
    """Load preprocessed scene data with validation."""
    scene_data_dir = f"input/scene_data/{room}"
    
    # Check if scene data directory exists
    if not os.path.exists(scene_data_dir):
        # List available scenes
        scene_base = "input/scene_data"
        available_scenes = []
        if os.path.exists(scene_base):
            available_scenes = [d for d in os.listdir(scene_base) 
                              if os.path.isdir(os.path.join(scene_base, d))]
        
        error_msg = (
            f"Scene data directory not found: {scene_data_dir}\n"
            f"Please run preprocessing steps first:\n"
            f"  1. preprocessing.py --raw <scan_path> --name {room}\n"
            f"  2. scene_parsing.py --name {room}\n"
            f"  3. bbox_polish.py --scene {room}\n"
        )
        if available_scenes:
            error_msg += f"\nAvailable scenes: {', '.join(available_scenes)}"
        raise FileNotFoundError(error_msg)
    
    # Required files
    required_files = {
        "objects_organized.pkl": "objects_organized.pkl",
        "walls_organized.pkl": "walls_organized.pkl",
        "walls_hole_organized.pkl": "walls_hole_organized.pkl",
        "floor_new_final.pkl": "floor_new_final.pkl",
        "wall_hole_reorder.txt": "wall_hole_reorder.txt"
    }
    
    # Check each required file
    missing_files = []
    for file_key, file_name in required_files.items():
        file_path = os.path.join(scene_data_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files for scene '{room}':\n" +
            "\n".join(f"  - {f}" for f in missing_files) +
            f"\n\nPlease ensure preprocessing steps have been completed for scene '{room}'."
        )
    
    # Load the files
    with open(f"input/scene_data/{room}/objects_organized.pkl", 'rb') as f:
        object_lists = pickle.load(f)
    with open(f"input/scene_data/{room}/walls_organized.pkl", 'rb') as f:
        walls = pickle.load(f)
    with open(f"input/scene_data/{room}/walls_hole_organized.pkl", 'rb') as f:
        wall_holes = pickle.load(f)
    with open(f"input/scene_data/{room}/floor_new_final.pkl", 'rb') as f:
        floor_pose = pickle.load(f)
    
    print(f"✓ Successfully loaded scene data for '{room}'")
    return object_lists, walls, wall_holes, floor_pose

def main(room, load_textured_scan=False, take_image=False, render_engine='EEVEE', num_interpolations=3):
    # load the preprocessed data
    
    # Get current script directory to use for relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to project root

    # Use litereality_database at project root
    database_dir = os.path.join(project_root, "litereality_database")
    skybox_path = os.path.join(database_dir, "symmetrical_garden_02_4k.exr")
    add_skybox(skybox_path)
    object_lists, walls, wall_holes, floor_pose = load_processed_data(room)

    # load the material for the wall
    wall_pbr_folder = os.path.join(database_dir, "wall_pbr")
    wall_mat = create_pbr_material_from_folder(wall_pbr_folder)

    # Clear all existing objects in Blender (be cautious, this deletes everything in the scene)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # load the input/scene_data/Girton/wall_hole_reorder.txt as a list
    with open(f"input/scene_data/{room}/wall_hole_reorder.txt", 'r') as f:
        wall_hole_reorder = f.readlines()
    wall_hole_reorder = [x.strip() for x in wall_hole_reorder]


    # create the walls with holes
    for index, (key, hole_info) in enumerate(wall_holes.items()):

        print("hole_info", hole_info)
        hole_info = hole_info["holes"]

        wall = walls[index]
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]

        # Create a cube for the wall
        bpy.ops.mesh.primitive_cube_add(size=1)


        # --- assume you just appended the material ---
        wall_obj = bpy.context.object
        wall_obj.data.materials.append(wall_mat)

        # -------------------------------------------------
        # 1) Re-unwrap the wall’s UVs
        # -------------------------------------------------
        bpy.context.view_layer.objects.active = wall_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.03)

        # -------------------------------------------------
        # 2) Scale the UVs 10× in both U and V
        # -------------------------------------------------
        bpy.ops.object.mode_set(mode="OBJECT")  # switch out of Edit mode
        uv_layer = wall_obj.data.uv_layers.active
        for loop in uv_layer.data:
            loop.uv *= 10.0

        # -------------------------------------------------
        # 3) Ensure we're back in Object mode for later operations
        # -------------------------------------------------
        bpy.ops.object.mode_set(mode="OBJECT")


        wall_obj.name = f"Wall_{index}"
        wall_obj.scale = (bbox[0] , bbox[1], bbox[2])

        if hole_info != []:
            wall_raw_name = wall_hole_reorder[index]
            hole_objs = create_holes_on_walls(room, hole_info, wall_obj, bbox, index, wall_raw_name)
            parent_object = bpy.data.objects.new("Parent", None)
            bpy.context.collection.objects.link(parent_object)
            wall_obj.parent = parent_object
            for hole_obj in hole_objs:
                hole_obj.parent = parent_object
            parent_object.location = position
            parent_object.rotation_euler = (
                math.radians(rotation[0]),
                math.radians(rotation[1]),
                math.radians(rotation[2]),
            )
            parent_object.name = f"Wall_{index}_with_hole"
        else:
            wall_obj.location = position
            # Apply rotation (converting degrees to radians using math.radians)
            wall_obj.rotation_euler = (
                math.radians(rotation[0]),
                math.radians(rotation[1]),
                math.radians(rotation[2]),
            )
        
        # get the max y value of the wall
        max_y = position[1] + bbox[1]/2

    # create the objects
    for index, object_info in enumerate(object_lists):
        try:
            object_name = object_info["object_type"]
            object_name_raw = object_name
            object_bbox = object_info["bbox"]
            object_position = object_info["position"]
            object_rotation = object_info["rotation"]
            # object_name without the number, remove all number at the end
            object_name = ''.join([i for i in object_name if not i.isdigit()])
            # Updated path: GLB files are exported to mat_painting_stage/{room}_output_gltf/
            save_folder = f"output/mat_painting_stage/{room}_output_gltf"
            object_file = f"{save_folder}/{object_name_raw}.glb"
            print("object_file", object_file)
            
            # Try absolute path if relative doesn't exist
            if not os.path.exists(object_file):
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(os.path.dirname(script_dir))
                    abs_object_file = os.path.join(project_root, object_file)
                    if os.path.exists(abs_object_file):
                        object_file = abs_object_file
                        print(f"   Using absolute path: {object_file}")
                except Exception as e:
                    print(f"   Error resolving absolute path: {e}")
            
            if not os.path.exists(object_file):
                print(f"⚠️  Warning: GLB file not found for {object_name_raw} at {object_file}, skipping...")
                continue
                
            parent_object, bbox_size = load_obj_and_group_with_pbr_glb(object_file, object_name_raw)
            parent_object.name = f"{object_name}_real_{index}"
            parent_object.scale = (object_bbox[0]/bbox_size[0], object_bbox[1]/bbox_size[1], object_bbox[2]/bbox_size[2])
            parent_object.rotation_euler = (0, math.radians(object_rotation), 0)
            parent_object.location = object_position
            # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception as e:
            print("Error loading object:", e)
    # # # create the floor
    create_floor(floor_pose, -0.16, room)

    # # load rgbd 
    if load_textured_scan:
        load_rgbd_scans(f"input/textured_scans/{room}/textured_output.obj")

    bpy.ops.file.make_paths_relative()

    # -------------------------------------------------
    # 2) Pack every external resource into the .blend
    #    – images, HDRIs, PBR maps, linked libraries, etc.
    # -------------------------------------------------
    bpy.ops.file.pack_all()                # equivalent to File ▸ External Data ▸ Pack Resources

    os.makedirs("output/whole_scene_model/blender", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(
            filepath=f"output/whole_scene_model/blender/{room}.blend",
            compress=True)                 # compress=True keeps file size down

    # save everything into a glb file

    if not os.path.exists("output/whole_scene_model/glb/"):
        os.makedirs(f"output/whole_scene_model/glb")

    bpy.ops.export_scene.gltf(filepath=f"output/whole_scene_model/glb/{room}.glb", export_format='GLB', use_selection=False)


    setup_render_engine(render_engine)
    # # topview()

    if take_image:

            # Set render engine to material preview settings

        folder_to_save = f"output/whole_scene_render/rendered_rgbd/{room}"
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)

        camera_data_folder = f"input/rgbd/{room}"
        intrinsic_matrix_list = f"{camera_data_folder}/intrinsic/*"

        intrinsic_files = glob.glob(intrinsic_matrix_list)

        width = 960
        height = 720

        # Load all camera poses first (key frames)
        key_camera_poses = []
        intrinsic_data = None  # We'll use the same intrinsic for all frames

        for intrinsic_file in sorted(intrinsic_files, key=lambda x: int(x.split("_")[-1].split(".")[0])):
            id = int(intrinsic_file.split("_")[-1].split(".")[0])
            extrinsic_file = f"{camera_data_folder}/extrinsic/extrinsic_{id}.npy"

            if intrinsic_data is None:
                intrinsic_data = np.load(intrinsic_file)
                intrinsic_data = resize_intrinsic(intrinsic_data, (192, 256), (height, width))

            extrinsic = np.load(extrinsic_file)
            key_camera_poses.append((id, extrinsic))

        print(f"Loaded {len(key_camera_poses)} key camera poses")

        # Create smooth trajectory by interpolating between key frames
        if num_interpolations > 0:
            smooth_trajectory = create_smooth_camera_trajectory(key_camera_poses, num_interpolations)
            print(f"Created smooth trajectory with {len(smooth_trajectory)} total frames ({num_interpolations} interpolations between each key frame)")
        else:
            smooth_trajectory = [(str(id), ext) for id, ext in key_camera_poses]
            print(f"No interpolation, using {len(smooth_trajectory)} key frames only")

        # Render all frames (key frames + interpolated frames)
        for frame_idx, (frame_id, extrinsic) in enumerate(smooth_trajectory):
            print(f"Rendering frame {frame_idx + 1}/{len(smooth_trajectory)}: {frame_id}")

            # Set up the camera
            camera = setup_camera(intrinsic_data, extrinsic, render_width=width, render_height=height)
            # Render the image
            render_image(f"{folder_to_save}/render_image_{frame_id}.png", resolution_x=width, resolution_y=height)

            bpy.data.objects.remove(camera)

        print(f"Finished rendering {len(smooth_trajectory)} frames to {folder_to_save}")

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

def render_image(output_path, resolution_x=1920/2, resolution_y=1440/2):

    bpy.context.scene.camera = bpy.data.objects["CustomCamera"]
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Blender integration script with smooth camera trajectory")
    parser.add_argument('--scene', type=str, default="Girton", help="Scene name")
    parser.add_argument('--load_textured_scan', action='store_true', help="Load textured scan")
    parser.add_argument('--take_image', action='store_true', default=True, help="Take image")
    parser.add_argument('--render_engine', type=str, choices=['BLENDER_EEVEE', 'CYCLES'], default='CYCLES',
                        help="Rendering engine: CYCLES (better quality, optimized) or BLENDER_EEVEE (faster but lower quality)")
    parser.add_argument('--num_interpolations', type=int, default=3,
                        help="Number of interpolated frames between each key camera pose (default: 3). Set to 0 to disable smoothing.")

    # Get all args after "--" which are intended for this script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)

    # Run main function with provided arguments
    main(args.scene, args.load_textured_scan, args.take_image, args.render_engine, args.num_interpolations)