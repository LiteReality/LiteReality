import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from layout_utils import *

# Minimum line length tolerance to avoid division by zero
MIN_LINE_LENGTH = 1e-6

def plot_objs_walls(obj_bboxs, walls, save_path):
    """Plot 2D layout of walls and objects with enhanced visualization."""

    plt.figure(figsize=(10, 10))
    wall_color = (0.3, 0.3, 0.3, 0.5)
    color_palette = [
        (0.90, 0.49, 0.13, 0.5), (0.17, 0.63, 0.17, 0.5), 
        (0.12, 0.47, 0.71, 0.5), (0.84, 0.15, 0.16, 0.5), 
        (0.58, 0.40, 0.74, 0.5), (0.55, 0.34, 0.29, 0.5), 
        (0.95, 0.77, 0.06, 0.5), (0.50, 0.50, 0.50, 0.5)
    ]

    def get_corners_and_plot(elements, calculate_corners, color_fn, label_fn):
        corners_list = []
        for idx, elem in enumerate(elements):
            pos, rot, bbox = elem["position"], elem["rotation"], elem["bbox"]
            corners = calculate_corners(pos, rot, bbox)
            corners_list.append(corners.tolist())
            plt.fill(*np.append(corners, [corners[0]], axis=0).T, color=color_fn(idx, elem))
            center = np.mean(corners, axis=0)
            plt.text(*center, label_fn(idx, elem), ha="center", va="center", fontsize=10)
        return corners_list

    walls = walls.values() if isinstance(walls, dict) else walls
    obj_bboxs = obj_bboxs.values() if isinstance(obj_bboxs, dict) else obj_bboxs

    wall_lines = get_corners_and_plot(
        walls, 
        calculate_wall_corners, 
        lambda idx, _: wall_color, 
        lambda idx, _: f"Wall {idx}"
    )

    obj_colors = {}
    obj_corners = get_corners_and_plot(
        obj_bboxs, 
        calculate_obj_corners, 
        lambda idx, obj: obj_colors.setdefault(
            "".join(c for c in obj["object_type"] if not c.isdigit()), 
            color_palette[len(obj_colors) % len(color_palette)]
        ), 
        lambda _, obj: f'{obj["position"][:2]}'
    )

    # plt.xlabel("X Position")
    # plt.ylabel("Z Position")
    # plt.title("2D Layout of Walls and Objects")
    # plt.axis("equal")
    # plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, transparent=True)
    plt.show()

    return obj_corners, wall_lines

def create_line_segments(points):
    """
    Creates line segments from a list of ordered points.
    Each pair of consecutive points forms a line segment.
    """
    segments = []
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]  # Wrap around to form a closed loop
        segments.append([start, end])
    return segments

def get_wall_lines(walls):
    """Extract 2D wall lines and minimum Y-coordinates from walls."""
    return [
        calculate_wall_line(wall["pose"]["position"], wall["pose"]["rotation"], wall["pose"]["bbox"]).tolist()
        for wall in walls
    ]

def calculate_wall_corners(position, rotation, bbox):
    """Calculate 2D wall corners after applying rotation and translation."""
    # Convert rotations from degrees to radians
    rotation_rad = np.radians(rotation)

    # Define rotation matrices
    def rotation_matrix_x(angle):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rotation_matrix_y(angle):
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rotation_matrix_z(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    # Combine rotation matrices
    rotation_matrix = (
        rotation_matrix_x(rotation_rad[0])
        @ rotation_matrix_y(rotation_rad[1])
        @ rotation_matrix_z(rotation_rad[2])
    )

    # Wall dimensions (width, depth) from bounding box
    width, _, depth = bbox

    corners = np.array(
        [
            [-width / 2, 0, -depth / 2],
            [width / 2, 0, -depth / 2],
            [width / 2, 0, depth / 2],
            [-width / 2, 0, depth / 2],
        ]
    )

    rotated_corners = (corners @ rotation_matrix.T) + position
    return rotated_corners[:, [0, 2]]  # Return X-Z coordinates for 2D plotting

def calculate_wall_line(position, rotation, bbox):
    """Calculate 2D wall corners after applying rotation and translation."""
    # Convert rotations from degrees to radians
    rotation_rad = np.radians(rotation)

    # Define rotation matrices
    def rotation_matrix_x(angle):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rotation_matrix_y(angle):
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rotation_matrix_z(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    # Combine rotation matrices
    rotation_matrix = (
        rotation_matrix_x(rotation_rad[0])
        @ rotation_matrix_y(rotation_rad[1])
        @ rotation_matrix_z(rotation_rad[2])
    )

    # Wall dimensions (width, depth) from bounding box
    width, _, depth = bbox

    corners = np.array(
        [
            [-width / 2, 0, 0],
            [width / 2, 0, 0],
        ]
    )

    rotated_corners = (corners @ rotation_matrix.T) + position
    return rotated_corners[:, [0, 2]]  # Return X-Z coordinates for 2D plotting

def calculate_obj_corners(position, rotation, bbox):
    """Calculate 2D wall corners after applying rotation and translation."""
    # Convert rotations from degrees to radians
    rotation_rad = np.radians(rotation)

    # Define rotation matrices
    def rotation_matrix_x(angle):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rotation_matrix_y(angle):
        return np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rotation_matrix_z(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    # Combine rotation matrices
    rotation_matrix = rotation_matrix_y(rotation_rad)

    # Wall dimensions (width, depth) from bounding box
    width, _, depth = bbox
    corners = np.array(
        [
            [-width / 2, 0, -depth / 2],
            [width / 2, 0, -depth / 2],
            [width / 2, 0, depth / 2],
            [-width / 2, 0, depth / 2],
        ]
    )

    # Rotate and translate corners
    rotated_corners = (corners @ rotation_matrix.T) + position
    return rotated_corners[:, [0, 2]]  # Return X-Z coordinates for 2D plotting

def get_bbox_coordinates(obj_bboxs, walls):

    obj_corners = {}
    for index, obj_bbox in enumerate(obj_bboxs):
        position = obj_bbox["position"]
        rotation = obj_bbox["rotation"]
        bbox = obj_bbox["bbox"]
        name = obj_bbox["object_type"]
        # Calculate 2D corners of the object
        corners_2d = calculate_obj_corners(position, rotation, bbox)
        obj_corners[name] = corners_2d

    wall_corners = {}
    for index, wall in enumerate(walls):
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]
        # Calculate 2D corners of the wall
        corners_2d = calculate_wall_corners(position, rotation, bbox)
        wall_corners[f"wall_{index}"] = corners_2d

    return obj_corners, wall_corners

def wall_line_to_close(walls):
    # Step 1: Collect all endpoints
    endpoints = []
    for wall in walls:
        endpoints.extend(wall)
    endpoints = np.array(endpoints)

    # Step 2: Snap points together using KDTree
    from scipy.spatial import KDTree

    epsilon = 0.16  # Tolerance threshold
    tree = KDTree(endpoints)
    groups = tree.query_ball_tree(tree, r=epsilon)

    # Create a mapping from original point indices to snapped points
    snapped_points = {}
    for idx, group in enumerate(groups):
        if idx not in snapped_points:
            # Compute the centroid of the group
            group_points = endpoints[group]
            centroid = np.mean(group_points, axis=0)
            for g_idx in group:
                snapped_points[g_idx] = centroid

    # Step 3: Update line segments with snapped points
    snapped_walls = {}
    for key, wall in enumerate(walls):
        p1_idx = np.where((endpoints == wall[0]).all(axis=1))[0][0]
        p2_idx = np.where((endpoints == wall[1]).all(axis=1))[0][0]
        p1_snapped = snapped_points[p1_idx]
        p2_snapped = snapped_points[p2_idx]
        snapped_walls[key] = np.array([p1_snapped, p2_snapped])
    # Step 4: Build connectivity graph
    from collections import defaultdict

    graph = defaultdict(list)
    point_indices = {}
    index = 0
    for wall in snapped_walls.values():
        p1_tuple = tuple(wall[0])
        p2_tuple = tuple(wall[1])
        for p in [p1_tuple, p2_tuple]:
            if p not in point_indices:
                point_indices[p] = index
                index += 1
        i1 = point_indices[p1_tuple]
        i2 = point_indices[p2_tuple]
        graph[i1].append(i2)
        graph[i2].append(i1)

    # Step 5: Traverse the graph to find the closed loop
    def find_closed_loop(graph, start_node):
        path = []
        visited = set()

        def dfs(node, parent):
            visited.add(node)
            path.append(node)
            for neighbor in graph[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a loop
                    return True
                if dfs(neighbor, node):
                    return True
            path.pop()
            return False

        dfs(start_node, None)
        return path

    start_node = next(iter(graph))
    loop = find_closed_loop(graph, start_node)

    # Step 6: Reconstruct the polygon
    ordered_points = [list(point_indices.keys())[node] for node in loop]
    polygon = np.array(ordered_points)
    return polygon

def load_processed_data(room):
    """Load preprocessed data for the given room."""
    base_path = f"scene_data/{room}"
    with open(os.path.join(base_path, "objects.pkl"), "rb") as f:
        object_lists = pickle.load(f)
    with open(os.path.join(base_path, "walls.pkl"), "rb") as f:
        walls = pickle.load(f)
    with open(os.path.join(base_path, "wall_holes.pkl"), "rb") as f:
        wall_holes = pickle.load(f)
    with open(os.path.join(base_path, "floor.pkl"), "rb") as f:
        floor_pose = pickle.load(f)
    return object_lists, walls, wall_holes, floor_pose

def extend_line_if_needed(start, end, thickness, line_set):
    """
    Extend line ends if not connected to other lines.
    
    Parameters:
        start (tuple): Starting point (x1, y1).
        end (tuple): Ending point (x2, y2).
        thickness (float): Thickness of the wall.
        line_set (set): Set of all endpoints to check connectivity.

    Returns:
        tuple: Extended start and end points.
    """
    start, end = np.array(start, dtype=float), np.array(end, dtype=float)
    line_vector = end - start
    line_length = np.linalg.norm(line_vector)
    
    # Handle zero-length lines
    if line_length < MIN_LINE_LENGTH:
        return tuple(start), tuple(end)
    
    line_unit_vector = line_vector / line_length
    extension = 0.5 * thickness * line_unit_vector

    # Extend start if not connected
    if tuple(start) not in line_set:
        start = start - extension
    # Extend end if not connected
    if tuple(end) not in line_set:
        end = end + extension

    return tuple(start), tuple(end)

def line_to_thick_rectangle(start, end, thickness):
    """
    Generate a rectangle around a line segment with given thickness.
    
    Parameters:
        start (tuple): Starting point (x1, y1).
        end (tuple): Ending point (x2, y2).
        thickness (float): Thickness of the wall.

    Returns:
        Polygon: Shapely Polygon representing the rectangle.
    """
    start, end = np.array(start, dtype=float), np.array(end, dtype=float)
    line_vector = end - start
    line_length = np.linalg.norm(line_vector)
    
    # Handle zero-length lines - create a small square instead
    if line_length < MIN_LINE_LENGTH:
        half_thickness = thickness / 2
        corner1 = start + np.array([half_thickness, half_thickness])
        corner2 = start + np.array([-half_thickness, half_thickness])
        corner3 = start + np.array([-half_thickness, -half_thickness])
        corner4 = start + np.array([half_thickness, -half_thickness])
        return Polygon([corner1, corner2, corner3, corner4])
    
    line_unit_vector = line_vector / line_length

    # Compute perpendicular vector for thickness
    perp_vector = np.array([-line_unit_vector[1], line_unit_vector[0]]) * (thickness / 2)

    # Compute rectangle corners
    corner1 = start + perp_vector
    corner2 = start - perp_vector
    corner3 = end - perp_vector
    corner4 = end + perp_vector

    # Return as a Polygon with 4 corners (ensure it's closed)
    corners = [corner1, corner2, corner3, corner4]
    # Ensure polygon is closed by checking if first and last are different
    if not np.allclose(corners[0], corners[-1]):
        corners.append(corners[0])
    return Polygon(corners)

def process_lines_to_rectangles(lines, thickness):
    """
    Process a list of lines into thick rectangles, extending non-connected ends.

    Parameters:
        lines (list): List of lines where each line is [(x1, y1), (x2, y2)].
        thickness (float): Thickness of the walls.

    Returns:
        list: List of Shapely Polygon objects representing the thick lines.
    """
    rectangles = []
    # Collect all endpoints to check connectivity
    endpoints = [point for line in lines for point in line]
    line_set = set(tuple(point) for point in endpoints)

    for line in lines:
        start, end = line
        # Check for zero-length using tolerance
        start_arr = np.array(start)
        end_arr = np.array(end)
        line_length = np.linalg.norm(end_arr - start_arr)
        
        if line_length < MIN_LINE_LENGTH:
            continue  # Skip zero-length lines

        # Extend line ends if needed
        start, end = extend_line_if_needed(start, end, thickness, line_set)
        
        # Double-check after extension
        start_arr = np.array(start)
        end_arr = np.array(end)
        line_length = np.linalg.norm(end_arr - start_arr)
        if line_length < MIN_LINE_LENGTH:
            continue  # Skip if still zero-length after extension

        # Convert to thick rectangle
        try:
            rectangle = line_to_thick_rectangle(start, end, thickness)
            rectangles.append(rectangle)
        except Exception as e:
            # Skip lines that fail to create rectangles
            print(f"Warning: Failed to create rectangle for line {start}->{end}: {e}")
            continue

    return rectangles

def plot_rectangles(rectangles):
    """
    Plot the rectangles using Matplotlib.

    Parameters:
        rectangles (list): List of Shapely Polygon objects.
    """
    fig, ax = plt.subplots()
    for rect in rectangles:
        x, y = rect.exterior.xy
        ax.fill(x, y, alpha=0.6, edgecolor='black')

    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

def map_to_parsed_wall_lines(old_line, parsed_wall_lines):
    """
    Finds the closest matching wall line from the parsed wall lines to a given old wall line,
    considering endpoint distances and wall lengths, and also determines if the directions are the same.
    
    Args:
        old_line (list): A wall line in the format [[x1, y1], [x2, y2]].
        parsed_wall_lines (list): A list of wall lines, each in the format [[x1, y1], [x2, y2]].

    Returns:
        tuple: A tuple containing:
            - list: The closest matching wall line from parsed_wall_lines.
            - int: The index of the closest matching wall line in parsed_wall_lines.
            - bool: Whether the directions of the old line and the matched line are the same.
    """
    def line_length(line):
        """Compute the length of a line."""
        return np.linalg.norm(np.array(line[1]) - np.array(line[0]))

    def endpoint_distance(line1, line2):
        """Compute the minimum distance between endpoints of two lines, regardless of direction."""
        d1 = np.linalg.norm(np.array(line1[0]) - np.array(line2[0]))
        d2 = np.linalg.norm(np.array(line1[0]) - np.array(line2[1]))
        d3 = np.linalg.norm(np.array(line1[1]) - np.array(line2[0]))
        d4 = np.linalg.norm(np.array(line1[1]) - np.array(line2[1]))
        return min(d1 + d4, d2 + d3)  # Minimum total distance, accounting for direction ambiguity

    def direction_same(line1, line2):
        """Determine if the directions of two lines are the same."""
        vector1 = np.array(line1[1]) - np.array(line1[0])
        vector2 = np.array(line2[1]) - np.array(line2[0])
        dot_product = np.dot(vector1, vector2)
        return dot_product > 0  # Same direction if dot product is positive

    old_line_length = line_length(old_line)
    best_match = None
    best_score = float("inf")  # Lower score is better
    best_match_idx = -1
    best_direction_same = False

    for i, parsed_line in enumerate(parsed_wall_lines):
        parsed_line_length = line_length(parsed_line)
        length_diff = abs(old_line_length - parsed_line_length)  # Penalize length differences
        endpoint_dist = endpoint_distance(old_line, parsed_line)
        score = endpoint_dist + length_diff  # Combine metrics into a single score

        if score < best_score:
            best_score = score
            best_match = parsed_line
            best_match_idx = i
            best_direction_same = direction_same(old_line, parsed_line)

    return best_match, best_match_idx, best_direction_same

def visualize_lines(old_line, parsed_wall_lines, best_match):
    """
    Visualizes the old line, parsed wall lines, and the best match.
    """
    plt.figure(figsize=(8, 8))
    plt.axis("equal")
    plt.title("Wall Line Matching")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Plot the old line
    old_line = np.array(old_line)
    plt.plot(old_line[:, 0], old_line[:, 1], color="blue", label="Old Line", linewidth=2)

    # Plot the parsed wall lines
    for i, parsed_line in enumerate(parsed_wall_lines):
        parsed_line = np.array(parsed_line)
        plt.plot(
            parsed_line[:, 0],
            parsed_line[:, 1],
            color="gray",
            linestyle="--",
            label="Parsed Wall Line" if i == 0 else None,
        )

    # Highlight the best match
    if best_match is not None:
        best_match = np.array(best_match)
        plt.plot(
            best_match[:, 0],
            best_match[:, 1],
            color="red",
            label="Best Match",
            linewidth=3,
        )

    plt.legend()
    plt.grid(True)
    plt.show()

def wall_line_closed_rect(overall_wall_lines, thickness):
    # Filter out zero-length lines first
    valid_lines = []
    for i in range(len(overall_wall_lines)):
        start, end = overall_wall_lines[i]
        p1 = np.array(start)
        p2 = np.array(end)
        line_length = np.linalg.norm(p2 - p1)
        
        # Skip zero-length lines
        if line_length < MIN_LINE_LENGTH:
            continue
        
        # ensure p1 is always on the left of p2
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        line_vector = p2 - p1
        line_length = np.linalg.norm(line_vector)
        
        line_unit_vector = line_vector / line_length
        extension = 0.5 * thickness * line_unit_vector
        p1 = p1 - extension
        p2 = p2 + extension
        valid_lines.append((tuple(p1), tuple(p2)))
    
    rectangles_list = []
    rectangles = process_lines_to_rectangles(valid_lines, thickness)
    for rect in rectangles:
        rectangles_list.append(list(rect.exterior.coords))
    return rectangles_list

if __name__ == "__main__":

    scene_name = "1_nov"
    room_type = 1  # 0: Single-room, 1: Multi-room

    # Load preprocessed scene data
    object_lists, walls, wall_holes, _ = load_processed_data(scene_name)
    # Step 1: Extract object and wall details
    (
        obj_corners_on_floor,
        obj_corners_not_on_floor,
        obj_type_on_floor,
        obj_type_not_on_floor,
        wall_lines,
    ) = get_obj_4p_line_2p(object_lists, walls)

    wall_parser = Wall_Parsing(wall_lines, wall_holes, True, tolerence = 0.3)
    areas = wall_parser.closed_area()

    visualize_lines_and_objects(wall_parser.wall_lines, [], "name_of_plot", line_labels=None, object_labels=None)

    parsed_wall_lines = wall_parser.wall_lines # wall line of every room

    thickness = 0.16  # Thickness of the "wall" in meters

    overall_wall_lines = [line for line in parsed_wall_lines if line[0] != line[1]]
    rectangles_list = wall_line_closed_rect(overall_wall_lines, thickness)

    # plot this rectangles list 

    visualize_lines_and_objects([], rectangles_list, "plot", line_labels=None, object_labels=None)

    Hole_file = {}
    Wall_flie = {}

    new_idx = 0
    used_wall_idx = []
    for index, (key, hole_info) in enumerate(wall_holes.items()):
        wall = walls[index]
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]
        position_y = position[1]
        bbox_y = bbox[1]

        if hole_info != []:
            wall_line = wall_lines[index]
            best_match, best_match_idx = map_to_parsed_wall_lines(wall_line, overall_wall_lines)
            # visualize_lines(wall_line, overall_wall_lines, best_match)
            center_wall = np.mean(rectangles_list[best_match_idx], axis=0)
            print("point1", np.array(overall_wall_lines[best_match_idx][0]))
            print("point2", np.array(overall_wall_lines[best_match_idx][1]))
            length_of_wall = np.linalg.norm(np.array(overall_wall_lines[best_match_idx][0]) - np.array(overall_wall_lines[best_match_idx][1]))
            thickness_of_wall = thickness
            line_for_wall = overall_wall_lines[best_match_idx]
            center_wall = np.mean(line_for_wall, axis=0)
            used_wall_idx.append(best_match_idx)
            orientation = math.atan2(
                line_for_wall[1][1] - line_for_wall[0][1], line_for_wall[1][0] - line_for_wall[0][0]
            )
            orientation = math.degrees(orientation)
            # start to update the wall
            position_new = [center_wall[0], position_y, center_wall[1]]
            bbox_new = [length_of_wall, bbox_y, thickness_of_wall]
            rotation_new = [180, orientation, 180]
            Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall}}
            Hole_file[new_idx] = {"file": new_idx, "holes": hole_info}
            new_idx += 1
    for i in range(len(rectangles_list)):
        if i in used_wall_idx:
            continue
        # center_wall = np.mean(rectangles_list[i], axis=0)
        length_of_wall = np.linalg.norm(np.array(overall_wall_lines[i][0]) - np.array(overall_wall_lines[i][1]))
        thickness_of_wall = thickness
        line_for_wall = overall_wall_lines[i]
        center_wall = np.mean(line_for_wall, axis=0)
        orientation = math.atan2(
            line_for_wall[1][1] - line_for_wall[0][1], line_for_wall[1][0] - line_for_wall[0][0]
        )
        orientation = math.degrees(orientation)
        position_new = [center_wall[0], position_y, center_wall[1]]
        bbox_new = [length_of_wall, bbox_y, thickness_of_wall]
        rotation_new = [180, orientation, 180]
        Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall}}
        Hole_file[new_idx] = {"file": new_idx, "holes": []}
        new_idx += 1
    
    
    # dump the wall and hole file
    with open(f"scene_data/{scene_name}/walls_organized.pkl", "wb") as f:
        pickle.dump(Wall_flie, f)
    with open(f"scene_data/{scene_name}/walls_hole_organized.pkl", "wb") as f:
        pickle.dump(Hole_file, f)

    with open(f"scene_data/{scene_name}/walls_organized.pkl", "rb") as f:
        walls = pickle.load(f)

    plot_objs_walls([], walls, "./")


