from scene_graph import (
    plot_objs_walls,
    line_obj_process_with_vis,
    offset_wall_to_ensure_include)
from layout_parsing import Wall_Parsing
from layout_utils import *
from shapely.geometry import JOIN_STYLE
from collisionsolver import CollisionSolver2d

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


def create_closed_area_and_offset_lines_with_visualization(wall_line_list, offset_distance, enable_visualization=True):
    """
    Create a closed area from wall lines and offset inside for each wall line.
    Saves three visualization images if enable_visualization is True:
    1) Original lines
    2) Merged closed polygon
    3) Inward-offset polygon
    """
    
    # -----------------------------
    # Step 1: Create original lines
    # -----------------------------
    lines = [LineString(line) for line in wall_line_list]

    # Visualization: Original wall lines
    if enable_visualization:
        plt.figure()
        for ls in lines:
            x, y = ls.xy
            plt.plot(x, y, 'b-', alpha=0.8)
        plt.title("Step 1: Original Wall Lines")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("output/visualization_offset_step1_original_lines.png")
        plt.close()

    # -----------------------------
    # Step 2: Merge to form polygon
    # -----------------------------
    merged_lines = unary_union(lines)

    if isinstance(merged_lines, LineString):
        closed_polygon = Polygon(merged_lines.coords)
    elif merged_lines.geom_type == "MultiLineString":
        # Flatten the list of coordinates
        coords = []
        for geom in merged_lines.geoms:
            coords.extend(geom.coords)
        closed_polygon = Polygon(coords)
    else:
        raise ValueError("Invalid wall lines: Unable to form a closed area.")

    # Visualization: Closed polygon from merged lines
    if enable_visualization:
        plt.figure()
        # Plot the original lines in the background (optional)
        for ls in lines:
            x, y = ls.xy
            plt.plot(x, y, color='gray', alpha=0.5)
        
        if not closed_polygon.is_empty:
            exterior_x, exterior_y = closed_polygon.exterior.xy
            plt.plot(exterior_x, exterior_y, 'r-', label="Closed Polygon")
        plt.title("Step 2: Merged Closed Polygon")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("output/visualization_offset_step2_closed_polygon.png")
        plt.close()

    # -----------------------------
    # Step 3: Inward offset polygon
    # -----------------------------
    offset_polygon = closed_polygon.buffer(-offset_distance, join_style=JOIN_STYLE.mitre)

    if offset_polygon.is_empty or not offset_polygon.is_valid:
        raise ValueError("Offset resulted in an invalid or empty polygon.")

    # Visualization: Offset polygon
    if enable_visualization:
        plt.figure()
        # Plot the original closed polygon in gray for reference
        if not closed_polygon.is_empty:
            exterior_x, exterior_y = closed_polygon.exterior.xy
            plt.plot(exterior_x, exterior_y, color='gray', label="Original Polygon", alpha=0.5)
        
        # Plot the offset polygon boundary in blue
        if offset_polygon.boundary.geom_type == "LineString":
            ox, oy = offset_polygon.boundary.xy
            plt.plot(ox, oy, 'b-', label="Offset Polygon")
        elif offset_polygon.boundary.geom_type == "MultiLineString":
            for geom in offset_polygon.boundary.geoms:
                ox, oy = geom.xy
                plt.plot(ox, oy, 'b-', label="Offset Polygon")
        plt.title("Step 3: Inward Offset Polygon")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("output/visualization_offset_step3_offset_polygon.png")
        plt.close()

   # --------------------------------
    # Step 4: Extract offset wall lines
    # --------------------------------
    boundary = offset_polygon.boundary

    # Only merge if the boundary is a MultiLineString
    if boundary.geom_type == "MultiLineString":
        merged_boundary = linemerge(boundary)
    else:
        merged_boundary = boundary

    offset_lines = []

    if merged_boundary.geom_type == "LineString":
        coords = list(merged_boundary.coords)
        offset_lines = [[coords[i], coords[i+1]] for i in range(len(coords)-1)]
    elif merged_boundary.geom_type == "MultiLineString":
        for line in merged_boundary.geoms:
            coords = list(line.coords)
            offset_lines.extend([[coords[i], coords[i+1]] for i in range(len(coords)-1)])

    return offset_lines

def get_obj_attributes(obj_bboxs, average_wall_min_y, floor_offset=0.3):
    """
    Process object bounding boxes to extract attributes and classify objects as on-floor or off-floor.
    
    Parameters:
        obj_bboxs (list): List of objects' bounding box data.
        average_wall_min_y (float): Average minimum Y coordinate of the walls.
        floor_offset (float): Offset to determine the floor threshold.

    Returns:
        dict: Dictionary containing object attributes classified by on-floor and off-floor.
    """
    obj_data = {"on_floor": [], "off_floor": []}

    for obj_bbox in obj_bboxs:
        # Extract object attributes
        position = obj_bbox["position"]
        rotation = obj_bbox["rotation"]
        bbox = obj_bbox["bbox"]
        name = obj_bbox["object_type"]
        corners_2d = calculate_obj_corners(position, rotation, bbox)
        obj_min_y = position[1] - bbox[1] / 2

        # Determine if object is on the floor or off the floor
        category = "on_floor" if obj_min_y < average_wall_min_y + floor_offset else "off_floor"

        # Store object attributes
        obj_data[category].append({
            "type": "".join([char for char in name if not char.isdigit()]),  # Remove digits from type
            "raw_type": name,
            "corners_2d": corners_2d.tolist(),
            "bbox_raw": bbox,
            "rotation": rotation,
            "position": position,
            "min_y": obj_min_y,
            "category": category  # Add category field (on_floor or off_floor)
        })

    return obj_data

if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process scene data for 3D reconstruction.')
    parser.add_argument('--name', type=str, required=True, help='Name of the scene to process (e.g., "Girton")')
    args = parser.parse_args()
    
    scene_name = args.name
    
    # Remove any path components if passed in like "input/object_stage/Girton"
    if '/' in scene_name:
        scene_name = scene_name.split('/')[-1]

    # Create cache directory for scene parsing visualizations with scene_name folder
    scene_parsing_cache_dir = os.path.join("cache", "scene_parsing_cache", scene_name)
    os.makedirs(scene_parsing_cache_dir, exist_ok=True)

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}🔍 Scene Graph Processing Pipeline{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"{Colors.CYAN}Processing scene:{Colors.RESET} {Colors.BOLD}{scene_name}{Colors.RESET}\n")
    
    # -----------------------------
    # STEP 1: Load scene data and plot initial state
    # -----------------------------
    print(f"{Colors.BOLD}{Colors.BLUE}[Step 1/9]{Colors.RESET} Loading scene data: objects, walls, and wall holes...")
    object_lists, walls, wall_holes, _ = load_processed_data(scene_name)
    before_path = f"{scene_parsing_cache_dir}/{scene_name}_before.png"
    plot_objs_walls(object_lists, walls, before_path)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Found {Colors.BOLD}{len(object_lists)}{Colors.RESET} objects and {Colors.BOLD}{len(walls)}{Colors.RESET} walls")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Initial scene state saved to {Colors.CYAN}{before_path}{Colors.RESET}")
    raw_obj_list = object_lists.copy()
    
    # -----------------------------
    # STEP 2: Process wall lines and ensure enclosed area
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 2/9]{Colors.RESET} Processing wall lines to ensure enclosed area...")
    floor_level_scan = False
    wall_parser = Wall_Parsing(walls, wall_holes, floor_level_scan, tolerence=0.4)
    wall_parser.output_wall_hole_floor(0.16, scene_name)
    wall_lines = wall_parser.wall_lines
    print(f"  {Colors.GREEN}✓{Colors.RESET} Processed {Colors.BOLD}{len(wall_lines)}{Colors.RESET} wall lines")
    
    # -----------------------------
    # STEP 3: Categorize objects as floor or off-floor objects
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 3/9]{Colors.RESET} Categorizing objects as on-floor or off-floor...")
    wall_min_y_s = [wall["pose"]["position"][1] - wall["pose"]["bbox"][1] / 2 for wall in walls]
    average_wall_min_y = np.mean(wall_min_y_s)
    obj_data = get_obj_attributes(object_lists, average_wall_min_y)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Found {Colors.BOLD}{len(obj_data['on_floor'])}{Colors.RESET} on-floor objects")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Found {Colors.BOLD}{len(obj_data['off_floor'])}{Colors.RESET} off-floor objects")
    
    # Create original object list with all objects
    orginal_object_list = []
    for obj in obj_data["on_floor"]:
        orginal_object_list.append(obj)
    for obj in obj_data["off_floor"]:
        orginal_object_list.append(obj)
    
    # -----------------------------
    # STEP 4: Offset wall lines inward for object placement
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 4/9]{Colors.RESET} Creating inward wall offsets for object placement...")
    wall_line_inward = create_closed_area_and_offset_lines_with_visualization(wall_lines, 0.08, enable_visualization=False)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Created {Colors.BOLD}{len(wall_line_inward)}{Colors.RESET} inward-offset wall lines")
    
    # Extract object corners and types
    object_list = []
    obj_type = []
    for obj in orginal_object_list:
        object_list.append(obj["corners_2d"])
        obj_type.append(obj["raw_type"])
    
    # -----------------------------
    # STEP 5: Adjust walls to ensure objects are included
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 5/9]{Colors.RESET} Adjusting wall lines to ensure objects are included...")
    wall_line_inward = offset_wall_to_ensure_include(object_list, wall_line_inward, scene_name)
    new_wall_lines = offset_wall_to_ensure_include(object_list, wall_line_inward, scene_name)
    new_wall_lines = offset_wall_to_ensure_include(object_list, new_wall_lines, scene_name)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Wall adjustment completed")
    
    # -----------------------------
    # STEP 6: Process object positions relative to wall lines
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 6/9]{Colors.RESET} Processing object positions relative to walls...")
    object_lists_update, _ = line_obj_process_with_vis(
        object_list, obj_type, new_wall_lines, scene_name, 
        distance_threshold=0.2, angle_threshold=10
    )
    print(f"  {Colors.GREEN}✓{Colors.RESET} Updated {Colors.BOLD}{len(object_lists_update)}{Colors.RESET} object positions")
    
    # -----------------------------
    # STEP 7: Create final wall representation
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 7/9]{Colors.RESET} Creating final wall representation...")
    new_wall_lines = create_closed_area_and_offset_lines_with_visualization(new_wall_lines, -0.08, enable_visualization=False)
    update_lines_parsing = Wall_Parsing(
        walls, wall_holes, floor_level_scan, 
        tolerence=0.4, output_only=True, wall_lines_process=new_wall_lines
    )
    update_lines_parsing.output_wall_hole_floor(0.16, scene_name)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Final wall representation created")
    
    # Load updated walls
    with open(f"input/scene_data/{scene_name}/walls_organized.pkl", "rb") as f:
        walls = pickle.load(f)
    
    # -----------------------------
    # STEP 8: Resolve object collisions
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 8/9]{Colors.RESET} Resolving object collisions...")
    all_obj_updated = []
    
    # Process on-floor objects
    collision_solver = CollisionSolver2d(orginal_object_list, new_wall_lines, 0, on_floor=True)
    collision_solver.update_object_list()
    for update_obj_dict in collision_solver.updated_obj_bbox_info.values():
        all_obj_updated.append(update_obj_dict)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Updated {Colors.BOLD}{len(all_obj_updated)}{Colors.RESET} on-floor objects after collision resolution")
    
    # Process off-floor objects (e.g., TVs, wall-mounted items)
    collision_solver_off_floor = CollisionSolver2d(orginal_object_list, new_wall_lines, 0, on_floor=False)
    collision_solver_off_floor.update_object_list()
    for update_obj_dict in collision_solver_off_floor.updated_obj_bbox_info.values():
        all_obj_updated.append(update_obj_dict)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Updated {Colors.BOLD}{len(collision_solver_off_floor.updated_obj_bbox_info)}{Colors.RESET} off-floor objects after collision resolution")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Total updated objects: {Colors.BOLD}{len(all_obj_updated)}{Colors.RESET}")
    
    # Final wall adjustment
    new_wall_lines = offset_wall_to_ensure_include(object_list, new_wall_lines, scene_name)
    
    # -----------------------------
    # STEP 9: Save results and visualization
    # -----------------------------
    print(f"\n{Colors.BOLD}{Colors.BLUE}[Step 9/9]{Colors.RESET} Saving processed results...")
    with open(f"input/scene_data/{scene_name}/objects_organized.pkl", "wb") as f:
        pickle.dump(all_obj_updated, f)
    
    # Load final data for visualization
    with open(f"input/scene_data/{scene_name}/walls_organized.pkl", "rb") as f:
        walls = pickle.load(f)
    with open(f"input/scene_data/{scene_name}/objects_organized.pkl", "rb") as f:
        objects = pickle.load(f)
        
    # Create final visualization
    after_path = f"{scene_parsing_cache_dir}/{scene_name}_after.png"
    plot_objs_walls(objects, walls, after_path)
    print(f"  {Colors.GREEN}✓{Colors.RESET} Final scene state saved to {Colors.CYAN}{after_path}{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}╔═══════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}║{Colors.RESET} {Colors.BOLD}✓ Scene processing complete for: {scene_name}{Colors.RESET} {Colors.BOLD}{Colors.GREEN}║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
