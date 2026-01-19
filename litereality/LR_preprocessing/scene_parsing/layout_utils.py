

import os
import math
import pickle
import numpy as np
import matplotlib.colors as mcolors
import random
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

import numpy as np
import matplotlib.pyplot as plt

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
        if start == end:
            continue

        # Extend line ends if needed
        start, end = extend_line_if_needed(start, end, thickness, line_set)

        # Convert to thick rectangle
        rectangle = line_to_thick_rectangle(start, end, thickness)
        rectangles.append(rectangle)

    return rectangles


def wall_line_closed_rect(overall_wall_lines, thickness):
    for i in range(len(overall_wall_lines)):
        start, end = overall_wall_lines[i]
        p1 = np.array(start)
        p2 = np.array(end)
        # ensure p1 is always on the left of p2
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        line_vector = p2 - p1
        line_length = np.linalg.norm(line_vector)
        line_unit_vector = line_vector / line_length
        extension = 0.5 * thickness * line_unit_vector
        p1 = p1 - extension
        p2 = p2 + extension
        overall_wall_lines[i] = (tuple(p1), tuple(p2))
    
    rectangles_list = []
    rectangles = process_lines_to_rectangles(overall_wall_lines, thickness)
    for rect in rectangles:
    
        rectangles_list.append(list(rect.exterior.coords))
    return rectangles_list


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



def plot_objs_walls(obj_bboxs, walls, save_path):
    """Plot 2D layout of walls and objects with enhanced visualization."""
    plt.figure(figsize=(10, 10))

    wall_color = (0.3, 0.3, 0.3, 0.5)  # Set wall color with transparency

    # Define a predefined color palette for object categories
    color_palette = [
        (0.90, 0.49, 0.13, 0.5),  # Orange
        (0.17, 0.63, 0.17, 0.5),  # Green
        (0.12, 0.47, 0.71, 0.5),  # Blue
        (0.84, 0.15, 0.16, 0.5),  # Red
        (0.58, 0.40, 0.74, 0.5),  # Purple
        (0.55, 0.34, 0.29, 0.5),  # Brown
        (0.95, 0.77, 0.06, 0.5),  # Yellow
        (0.50, 0.50, 0.50, 0.5),  # Gray
    ]

    wall_corners = []
    wall_min_y_s = []
    wall_lines = []
    # Plot walls
    # if walls are a dict 
    if isinstance(walls, dict):
        walls = walls.values()
    for index, wall in enumerate(walls):
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]

        wall_min_y = position[1] - bbox[1] / 2
        wall_min_y_s.append(wall_min_y)

        # Calculate 2D corners of the wall
        corners_2d = calculate_wall_corners(position, rotation, bbox)
        wall_corners.append(corners_2d.tolist())
        wall_line = calculate_wall_line(position, rotation, bbox)
        wall_lines.append(wall_line.tolist())

        # Plot the wall as a polygon with specified wall color and transparency
        plt.fill(
            *np.append(corners_2d, [corners_2d[0]], axis=0).T,
            color=wall_color,
            label=f"Wall {index}" if index == 0 else "",
        )

        # Calculate the center position of the wall to place the label
        center_x = np.mean(corners_2d[:, 0])
        center_y = np.mean(corners_2d[:, 1])
        plt.text(
            center_x,
            center_y,
            f"Wall {index}",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )

    # Average to get the ground level of the wall
    wall_ground = np.mean(wall_min_y_s)

    obj_colors = {}  # Dictionary to store colors for each object type
    obj_corners = []


    if isinstance(obj_bboxs, dict):
        obj_bboxs = obj_bboxs.values()

    for index, obj_bbox in enumerate(obj_bboxs):
        position = obj_bbox["position"]
        rotation = obj_bbox["rotation"]
        bbox = obj_bbox["bbox"]
        name = obj_bbox["object_type"]
        raw_name = name
        # take out digits from the name
        name = "".join([i for i in raw_name if not i.isdigit()])

        # Calculate 2D corners of the object


        corners_2d = calculate_obj_corners(position, rotation, bbox)



        obj_corners.append(corners_2d.tolist())

        # Assign color to each object type, cycling through the predefined palette
        if name not in obj_colors:
            obj_colors[name] = color_palette[len(obj_colors) % len(color_palette)]
        color = obj_colors[name]

        # Plot the object as a polygon with category-based color and transparency
        plt.fill(*np.append(corners_2d, [corners_2d[0]], axis=0).T, color=color)

        # Calculate the center position of the bounding box to place the label
        center_x = np.mean(corners_2d[:, 0])
        center_y = np.mean(corners_2d[:, 1])
        plt.text(
            center_x,
            center_y,
            name,
            ha="center",
            va="center",
            fontsize=10,
            color="blue",
        )

    # Formatting the plot
    # plt.xlabel("X Position")
    # plt.ylabel("Z Position")
    # plt.title("2D Layout of Walls and Objects")
    plt.axis("equal")
    # plt.axis("off")
    # plt.grid(True)
    # plt.grid(False)
    # do not show the axis
    plt.axis('off')
    # set transparent background
    plt.gca().set_facecolor((0, 0, 0, 0))

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, transparent=True)
    plt.show()
    # plt.close()

    return obj_corners, wall_lines


def load_processed_data(room):
    """Load preprocessed data for the given room."""
    base_path = f"input/scene_data/{room}"
    with open(os.path.join(base_path, "objects.pkl"), "rb") as f:
        object_lists = pickle.load(f)
    with open(os.path.join(base_path, "walls.pkl"), "rb") as f:
        walls = pickle.load(f)
    with open(os.path.join(base_path, "wall_holes.pkl"), "rb") as f:
        wall_holes = pickle.load(f)
    with open(os.path.join(base_path, "floor.pkl"), "rb") as f:
        floor_pose = pickle.load(f)
    return object_lists, walls, wall_holes, floor_pose

def calculate_average_angle(wall_lines):

    angles_in_degrees = []
    for start, end in wall_lines:
        # Calculate angle in radians
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        angle = (angle + math.pi) % math.pi  # Normalize angle to [0, pi]
        
        # Adjust angle to range [0, pi/2]
        if angle > math.pi / 2:
            angle -= math.pi / 2

        # Convert angle to degrees and round to 2 decimal places
        angle_in_degrees = round(math.degrees(angle), 2)
        angles_in_degrees.append(angle_in_degrees)

    # Step 1: Sort the angles
    angle_list = sorted(angles_in_degrees)

    # Step 2: Use the IQR to filter out outliers
    q1 = np.percentile(angle_list, 25)  # First quartile (25th percentile)
    q3 = np.percentile(angle_list, 75)  # Third quartile (75th percentile)
    iqr = q3 - q1  # Interquartile range

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Step 3: Filter out angles that are outside the bounds
    filtered_angles = [angle for angle in angle_list if lower_bound <= angle <= upper_bound]

    # Step 4: Calculate the mean of the remaining angles
    if filtered_angles:  # Ensure there are angles left after filtering
        average_angle = round(np.mean(filtered_angles), 2)
    else:
        average_angle = None  # No angles left after filtering

    return average_angle, filtered_angles

def center_rotate_and_translate_wall_lines(wall_lines, angle_deg):
    """
    Center the wall lines around the origin, rotate them by a given angle,
    and then translate them back to their original position.
    """
    # Flatten all points to calculate the centroid
    all_points = [point for line in wall_lines for point in line]
    centroid_x = np.mean([p[0] for p in all_points])
    centroid_y = np.mean([p[1] for p in all_points])

    # Translate all points to center the layout
    centered_lines = [
        [[p[0] - centroid_x, p[1] - centroid_y] for p in line]
        for line in wall_lines
    ]

    # Rotate the centered wall lines
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    rotated_lines = [
        [
            [
                p[0] * cos_theta - p[1] * sin_theta,
                p[0] * sin_theta + p[1] * cos_theta,
            ]
            for p in line
        ]
        for line in centered_lines
    ]

    # Translate back to the original position
    translated_lines = [
        [[p[0] + centroid_x, p[1] + centroid_y] for p in line]
        for line in rotated_lines
    ]

    return translated_lines

def visualize_lines(before_lines, after_lines, name_of_plot, txt=None):
    """
    Visualizes lines before and after adjustment on the same figure with proper snapping
    and intersections, assigning a unique color to each line.

    Parameters:
        before_lines (list): List of lines before adjustment.
        after_lines (list): List of lines after adjustment.
        name_of_plot (str): Name of the plot for saving output images.
        txt (list): Optional. List of labels for each line.
    """
    distinct_colors = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
        "lime", "teal", "pink", "gold", "brown", "black"
    ]
    random.shuffle(distinct_colors)  # Shuffle to add randomness

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot "Before" visualization
    axs[0].set_title(f"{name_of_plot} - Before")
    for idx, line in enumerate(before_lines):
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        axs[0].plot(
            x_values,
            y_values,
            color=distinct_colors[idx % len(distinct_colors)],
            linewidth=2,
        )
        if txt:
            axs[0].text(np.mean(x_values), np.mean(y_values), f"{txt[idx]}", fontsize=8, color="black")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    axs[0].grid(True)

    # Plot "After" visualization
    axs[1].set_title(f"{name_of_plot} - After")
    for idx, line in enumerate(after_lines):
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        axs[1].plot(
            x_values,
            y_values,
            color=distinct_colors[idx % len(distinct_colors)],
            linewidth=2,
        )
        if txt:
            axs[1].text(np.mean(x_values), np.mean(y_values), f"{txt[idx]}", fontsize=8, color="black")
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    axs[1].grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{name_of_plot}_before_after.png")
    plt.close()

def correct_and_filter_lines(rotated_wall_lines, angle_threshold=10):
    corrected_lines = []

    for line in rotated_wall_lines:
        # Calculate the angle of the line
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        angle = math.degrees(math.atan2(dy, dx))  # Angle in degrees

        # Normalize the angle to 0-90 degrees
        if angle < 0:
            angle += 180
        if angle > 90:
            angle = 180 - angle

        # Check if the angle is close to 0 or 90 degrees
        if abs(angle - 0) <= angle_threshold:
            # Snap to horizontal axis (align y-coordinates)
            corrected_lines.append([
                [line[0][0], line[0][1]],
                [line[1][0], line[0][1]]  # Same y-coordinate as start point
            ])
        elif abs(angle - 90) <= angle_threshold:
            # Snap to vertical axis (align x-coordinates)
            corrected_lines.append([
                [line[0][0], line[0][1]],
                [line[0][0], line[1][1]]  # Same x-coordinate as start point
            ])
        else:
            if angle < 45:  # Snap to horizontal axis
                corrected_lines.append([
                    [line[0][0], line[0][1]],
                    [line[1][0], line[0][1]]  # Same y-coordinate as start point
                ])
            else:  # Snap to vertical axis
                corrected_lines.append([
                    [line[0][0], line[0][1]],
                    [line[0][0], line[1][1]]  # Same x-coordinate as start point
                ])


    visualize_lines(rotated_wall_lines, corrected_lines, "output/corrected_lines.png")
    return corrected_lines

def group_lines_by_grid(lines, tolerance=0.1):

    # TODO: pay more attention to area that has a bend
    horizontal_groups = {}
    vertical_groups = {}
    for line in lines:
        if np.isclose(line[0][1], line[1][1]):  # Horizontal line
            y_coord = round((line[0][1] + line[1][1]) / 2, 1)  # Average y-coordinate
            for key in horizontal_groups:
                if abs(key - y_coord) < tolerance:
                    horizontal_groups[key].append(line)
                    break
            else:
                horizontal_groups[y_coord] = [line]

        elif np.isclose(line[0][0], line[1][0]):  # Vertical line
            x_coord = round((line[0][0] + line[1][0]) / 2, 1)  # Average x-coordinate
            for key in vertical_groups:
                if abs(key - x_coord) < tolerance:
                    vertical_groups[key].append(line)
                    break
            else:
                vertical_groups[x_coord] = [line]

    # Combine horizontal and vertical groups into a single list of clusters
    clusters = list(horizontal_groups.values()) + list(vertical_groups.values())
    return clusters

def visualize_line_clusters(clusters):
    """
    Visualizes clustered lines with different colors.

    Parameters:
        clusters (list): List of clusters, where each cluster is a list of wall lines.
    """
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    num_clusters = len(clusters)

    for idx, cluster in enumerate(clusters):
        color = cmap(idx % 10)  # Cycle through the colormap if more than 10 clusters
        for line in cluster:
            x_values = [line[0][0], line[1][0]]
            y_values = [line[0][1], line[1][1]]
            plt.plot(x_values, y_values, color=color, linewidth=2)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Clustered Wall Lines with Bends Highlighted")
    plt.grid(True)
    plt.savefig("output/clustered_lines.png")

def is_line_on_line(line1, line2):
    """
    Check if line1 is completely on line2 (i.e., it is collinear and overlaps).
    
    Args:
        line1 (list): First line segment as [[x1, y1], [x2, y2]].
        line2 (list): Second line segment as [[x1, y1], [x2, y2]].
    
    Returns:
        bool: True if line1 is completely on line2; False otherwise.
    """
    def is_point_on_segment(point, segment):
        """Check if a point lies on a given line segment."""
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        px, py = point
        
        # Check collinearity
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_product) > 1e-6:  # Not collinear
            return False
        
        # Check if point is within segment bounds
        dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        if dot_product < 0:
            return False
        segment_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dot_product > segment_length_squared:
            return False
        
        return True
    
    # Check if both endpoints of line1 are on line2
    return is_point_on_segment(line1[0], line2) and is_point_on_segment(line1[1], line2)

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


def wall_line_to_close_with_visualization(walls):
    """
    This function processes a list of 2D wall segments (each defined by two endpoints)
    to automatically form a closed polygon. In addition to the core logic, it saves
    visualization images at each step for debugging and inspection.

    The following images are generated:
      - visualization_step1_endpoints.png: Original walls and endpoints.
      - visualization_step2_snapping.png: Original endpoints with their snapped centroids.
      - visualization_step3_snapped_walls.png: Wall segments after snapping endpoints.
      - visualization_step4_connectivity_graph.png: The connectivity graph (nodes and edges).
      - visualization_step5_closed_loop.png: The connectivity graph with the detected loop highlighted.
      - visualization_step6_polygon.png: The final reconstructed polygon.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import KDTree
    from collections import defaultdict

    # ------------------------------
    # Step 1: Collect all endpoints
    # ------------------------------
    endpoints = []
    for wall in walls:
        endpoints.extend(wall)
    endpoints = np.array(endpoints)

    # Visualization Step 1: Plot original walls and endpoints
    plt.figure()
    for wall in walls:
        wall_arr = np.array(wall)
        plt.plot(wall_arr[:, 0], wall_arr[:, 1], 'b-', alpha=0.5)  # draw original wall segments in blue
    plt.scatter(endpoints[:, 0], endpoints[:, 1], c='r', label='Endpoints')
    plt.title("Step 1: Original Walls and Endpoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step1_endpoints.png")
    plt.close()

    # -------------------------------------------
    # Step 2: Snap points together using KDTree
    # -------------------------------------------
    epsilon = 0.2  # Tolerance threshold for snapping
    tree = KDTree(endpoints)
    groups = tree.query_ball_tree(tree, r=epsilon)

    snapped_points = {}
    for idx, group in enumerate(groups):
        if idx not in snapped_points:
            group_points = endpoints[group]
            centroid = np.mean(group_points, axis=0)
            for g_idx in group:
                snapped_points[g_idx] = centroid

    # Visualization Step 2: Plot original endpoints and unique snapped centroids
    plt.figure()
    plt.scatter(endpoints[:, 0], endpoints[:, 1], c='r', label='Original Endpoints')
    # Get unique centroids from snapped_points mapping
    unique_centroids = np.array(list({tuple(val) for val in snapped_points.values()}))
    plt.scatter(unique_centroids[:, 0], unique_centroids[:, 1], c='g', marker='x', label='Snapped Centroids')
    plt.title("Step 2: KDTree Snapping of Endpoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step2_snapping.png")
    plt.close()

    # ------------------------------------------------------
    # Step 3: Update line segments with snapped endpoints
    # ------------------------------------------------------
    snapped_walls = {}
    for key, wall in enumerate(walls):
        wall_arr = np.array(wall)
        p1_idx = np.where((endpoints == wall_arr[0]).all(axis=1))[0][0]
        p2_idx = np.where((endpoints == wall_arr[1]).all(axis=1))[0][0]
        p1_snapped = snapped_points[p1_idx]
        p2_snapped = snapped_points[p2_idx]
        snapped_walls[key] = np.array([p1_snapped, p2_snapped])

    # Visualization Step 3: Plot wall segments after snapping endpoints
    plt.figure()
    for wall in snapped_walls.values():
        plt.plot(wall[:, 0], wall[:, 1], 'c-', alpha=0.7)
    plt.scatter(unique_centroids[:, 0], unique_centroids[:, 1], c='g', marker='x', label='Snapped Centroids')
    plt.title("Step 3: Walls after Snapping Endpoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step3_snapped_walls.png")
    plt.close()

    # --------------------------------------------
    # Step 4: Build the connectivity graph
    # --------------------------------------------
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

    # Create a reverse mapping for easier plotting (index -> point)
    index_to_point = {v: k for k, v in point_indices.items()}
    points_graph = np.array(list(index_to_point.values()))

    # Visualization Step 4: Plot the connectivity graph (nodes and edges)
    plt.figure()
    plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
    # Annotate each node with its index
    for idx, point in index_to_point.items():
        plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
    # Draw the edges between connected nodes
    for node, neighbors in graph.items():
        p1 = np.array(index_to_point[node])
        for neighbor in neighbors:
            p2 = np.array(index_to_point[neighbor])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.5)
    plt.title("Step 4: Connectivity Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step4_connectivity_graph.png")
    plt.close()

    # -----------------------------------------------------
    # Step 5: Traverse the graph to find the closed loop
    # -----------------------------------------------------
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
                    # Found a loop; add neighbor to complete the loop
                    path.append(neighbor)
                    return True
                if dfs(neighbor, node):
                    return True
            path.pop()
            return False

        dfs(start_node, None)
        return path

    start_node = next(iter(graph))
    loop = find_closed_loop(graph, start_node)

    # Visualization Step 5: Highlight the detected closed loop on the graph
    plt.figure()
    plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
    for idx, point in index_to_point.items():
        plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
    # Draw all edges in light gray
    for node, neighbors in graph.items():
        p1 = np.array(index_to_point[node])
        for neighbor in neighbors:
            p2 = np.array(index_to_point[neighbor])
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)
    # Extract loop points and highlight them in red
    loop_points = [index_to_point[node] for node in loop if node in index_to_point]
    loop_points = np.array(loop_points)
    plt.plot(loop_points[:, 0], loop_points[:, 1], 'r-', linewidth=2, label='Closed Loop')
    plt.title("Step 5: Closed Loop in Connectivity Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step5_closed_loop.png")
    plt.close()

    # -------------------------------------------
    # Step 6: Reconstruct the polygon from loop
    # -------------------------------------------
    ordered_points = [index_to_point[node] for node in loop if node in index_to_point]
    polygon = np.array(ordered_points)

    # Visualization Step 6: Plot the final polygon
    plt.figure()
    plt.plot(polygon[:, 0], polygon[:, 1], 'b-', marker='o', label='Polygon Boundary')
    plt.fill(polygon[:, 0], polygon[:, 1], 'b', alpha=0.1)  # optionally fill the polygon area
    plt.title("Step 6: Reconstructed Polygon")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_step6_polygon.png")
    plt.close()

    return polygon


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict

def wall_line_to_close_with_visualization_simple_underflow(walls, visualization=True):
    """
    A simplified version of the wall_line_to_close_with_visualization function
    that only handles the 'no closed loop' (underflow) scenario by connecting
    any two 'endpoint' nodes in the graph. If that creates a loop, we use it.
    Otherwise, the function proceeds without forming a loop.

    Visualization files (saved, not shown) will be generated only if 
    visualization is True.
    """

    # -------------------------------------------
    # Step 1: Collect all endpoints
    # -------------------------------------------
    endpoints = []
    for wall in walls:
        endpoints.extend(wall)
    endpoints = np.array(endpoints)

    if visualization:
        plt.figure()
        for wall in walls:
            w_arr = np.array(wall)
            plt.plot(w_arr[:, 0], w_arr[:, 1], 'b-', alpha=0.5)
        plt.scatter(endpoints[:, 0], endpoints[:, 1], c='r', label='Endpoints')
        plt.title("Step 1: Original Walls and Endpoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("visualization_step1_endpoints.png")
        plt.close()

    # -------------------------------------------
    # Step 2: Snap points together using KDTree
    # -------------------------------------------
    epsilon = 0.16
    tree = KDTree(endpoints)
    groups = tree.query_ball_tree(tree, r=epsilon)

    snapped_points = {}
    for idx, group in enumerate(groups):
        if idx not in snapped_points:
            group_points = endpoints[group]
            centroid = np.mean(group_points, axis=0)
            for g_idx in group:
                snapped_points[g_idx] = centroid

    if visualization:
        unique_centroids = np.array(list({tuple(v) for v in snapped_points.values()}))
        plt.figure()
        plt.scatter(endpoints[:, 0], endpoints[:, 1], c='r', label='Original Endpoints')
        plt.scatter(unique_centroids[:, 0], unique_centroids[:, 1], c='g', marker='x', label='Snapped Centroids')
        plt.title("Step 2: KDTree Snapping of Endpoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("visualization_step2_snapping.png")
        plt.close()

    # -------------------------------------------
    # Step 3: Update line segments with snapped points
    # -------------------------------------------
    snapped_walls = {}
    for key, wall in enumerate(walls):
        wall_arr = np.array(wall)
        p1_idx = np.where((endpoints == wall_arr[0]).all(axis=1))[0][0]
        p2_idx = np.where((endpoints == wall_arr[1]).all(axis=1))[0][0]
        p1_snapped = snapped_points[p1_idx]
        p2_snapped = snapped_points[p2_idx]
        
        # Filter out zero-length walls after snapping
        wall_length = np.linalg.norm(p2_snapped - p1_snapped)
        if wall_length > 1e-6:  # Only keep walls with meaningful length
            snapped_walls[key] = np.array([p1_snapped, p2_snapped])

    if visualization:
        unique_centroids = np.array(list({tuple(v) for v in snapped_points.values()}))
        plt.figure()
        for wall_arr in snapped_walls.values():
            plt.plot(wall_arr[:, 0], wall_arr[:, 1], 'c-', alpha=0.7)
        plt.scatter(unique_centroids[:, 0], unique_centroids[:, 1], c='g', marker='x', label='Snapped Centroids')
        plt.title("Step 3: Walls after Snapping Endpoints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("visualization_step3_snapped_walls.png")
        plt.close()

    # -------------------------------------------
    # Step 4: Build the connectivity graph
    # -------------------------------------------
    graph = defaultdict(list)
    point_indices = {}
    index = 0
    for wall_arr in snapped_walls.values():
        p1_tuple = tuple(wall_arr[0])
        p2_tuple = tuple(wall_arr[1])
        for p in [p1_tuple, p2_tuple]:
            if p not in point_indices:
                point_indices[p] = index
                index += 1
        i1 = point_indices[p1_tuple]
        i2 = point_indices[p2_tuple]
        graph[i1].append(i2)
        graph[i2].append(i1)

    index_to_point = {v: k for k, v in point_indices.items()}
    points_graph = np.array(list(index_to_point.values()))

    if visualization:
        plt.figure()
        plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
        for idx, point in index_to_point.items():
            plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
        for node, neighbors in graph.items():
            p1 = np.array(index_to_point[node])
            for neighbor in neighbors:
                p2 = np.array(index_to_point[neighbor])
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.5)
        plt.title("Step 4: Connectivity Graph")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("visualization_step4_connectivity_graph.png")
        plt.close()

    # -----------------------------------------------------
    # Step 5: Find a closed loop; if none, connect endpoints
    # -----------------------------------------------------
    def find_closed_loop(graph, start_node):
        """
        Returns a list of nodes representing the first loop found via DFS,
        or an empty list if no loop is found.
        """
        visited = set()
        path = []

        def dfs(node, parent):
            visited.add(node)
            path.append(node)
            for neighbor in graph[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a loop
                    path.append(neighbor)  # Add neighbor to close
                    return True
                if dfs(neighbor, node):
                    return True
            path.pop()
            return False

        if dfs(start_node, -1):
            return path
        return []

    start_node = next(iter(graph)) if len(graph) > 0 else None
    loop_path = []
    if start_node is not None:
        loop_path = find_closed_loop(graph, start_node)

    if not loop_path:
        if visualization:
            plt.figure()
            plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
            for idx, point in index_to_point.items():
                plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
            for node, neighbors in graph.items():
                p1 = np.array(index_to_point[node])
                for neighbor in neighbors:
                    p2 = np.array(index_to_point[neighbor])
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)
            plt.title("Step 5: No Loop Found (Underflow)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.savefig("visualization_step5_no_loop.png")
            plt.close()

        # --- Naive Fix: connect the two endpoints (nodes with degree 1) ---
        endpoints_in_graph = [n for n, nbrs in graph.items() if len(nbrs) == 1]
        if len(endpoints_in_graph) == 2:
            n1, n2 = endpoints_in_graph
            # Connect them if not already connected
            if n2 not in graph[n1]:
                graph[n1].append(n2)
            if n1 not in graph[n2]:
                graph[n2].append(n1)

            # Try finding the loop again
            loop_path = find_closed_loop(graph, n1)

            if loop_path and visualization:
                plt.figure()
                plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
                for idx, point in index_to_point.items():
                    plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
                for node, neighbors in graph.items():
                    p1 = np.array(index_to_point[node])
                    for neighbor in neighbors:
                        p2 = np.array(index_to_point[neighbor])
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)

                # Highlight the loop
                loop_coords = [index_to_point[n] for n in loop_path]
                loop_coords = np.array(loop_coords)
                # Typically, loop_path has the last node repeated. We'll close it ourselves:
                if len(loop_coords) > 1 and np.all(loop_coords[0] == loop_coords[-1]):
                    loop_coords = loop_coords[:-1]
                loop_coords_closed = np.vstack([loop_coords, loop_coords[0]])
                plt.plot(loop_coords_closed[:, 0], loop_coords_closed[:, 1],
                         'r-', linewidth=2, label='Newly Formed Loop')
                plt.title("Step 5: Fixed Loop by Connecting Endpoints")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.legend()
                plt.savefig("visualization_step5_fixed_loop.png")
                plt.close()

    # ------------------------------------------------
    # Step 6: Reconstruct the polygon from the loop
    # ------------------------------------------------
    polygon = np.array([])
    if loop_path:
        # Remove repeated last node if present
        if len(loop_path) > 1 and loop_path[0] == loop_path[-1]:
            loop_path = loop_path[:-1]
        polygon = np.array([index_to_point[n] for n in loop_path])

    if visualization:
        plt.figure()
        if len(polygon) >= 2:
            plt.plot(polygon[:, 0], polygon[:, 1], 'b-', marker='o', label='Polygon Boundary')
            plt.fill(polygon[:, 0], polygon[:, 1], 'b', alpha=0.1)
            plt.title("Step 6: Reconstructed Polygon")
        else:
            # If no valid loop, just show the final graph
            plt.scatter(points_graph[:, 0], points_graph[:, 1], c='m', label='Graph Nodes')
            for idx, point in index_to_point.items():
                plt.text(point[0], point[1], str(idx), color='black', fontsize=8)
            for node, neighbors in graph.items():
                p1 = np.array(index_to_point[node])
                for neighbor in neighbors:
                    p2 = np.array(index_to_point[neighbor])
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.5)
            plt.title("Step 6: Final Graph (No Polygon Formed)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.savefig("visualization_step6_polygon.png")
        plt.close()

    return polygon

def adjust_lines_with_intersections(clusters):
    """
    Adjust lines so that they align properly, snapping endpoints to intersection points.

    Parameters:
        clusters (list): List of clusters, where each cluster is a list of wall lines.
        bends (list): List of bend lines connecting different groups.
        tolerance (float): Tolerance for snapping endpoints to align with clusters.

    Returns:
        adjusted_lines (list): List of adjusted wall lines, including bends.
    """
    adjusted_lines = []
    intersection_points = set()

    def add_intersection(point):
        """Add a point to the set of intersection points, ensuring uniqueness."""
        point = (round(point[0], 2), round(point[1], 2))
        intersection_points.add(point)

    # Process clusters and gather potential intersection points
    for cluster in clusters:
        if np.isclose(cluster[0][0][1], cluster[0][1][1]):  # Horizontal cluster
            y_value = round(np.mean([line[0][1] for line in cluster]), 1)
            grid_points = sorted(set([p[0] for line in cluster for p in line]))  # Unique x-coordinates
            for line in cluster:
                x1, x2 = sorted([line[0][0], line[1][0]])
                adjusted_lines.append([[x1, y_value], [x2, y_value]])
                add_intersection([x1, y_value])
                add_intersection([x2, y_value])
        elif np.isclose(cluster[0][0][0], cluster[0][1][0]):  # Vertical cluster
            x_value = round(np.mean([line[0][0] for line in cluster]), 1)
            grid_points = sorted(set([p[1] for line in cluster for p in line]))  # Unique y-coordinates
            for line in cluster:
                y1, y2 = sorted([line[0][1], line[1][1]])
                adjusted_lines.append([[x_value, y1], [x_value, y2]])
                add_intersection([x_value, y1])
                add_intersection([x_value, y2])

    # Snap all endpoints of lines to the nearest intersection points
    def snap_to_intersections(point):
        """Snap a point to the nearest intersection point."""
        return min(intersection_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(point)))

    snapped_lines = []
    for line in adjusted_lines:
        snapped_start = snap_to_intersections(line[0])
        snapped_end = snap_to_intersections(line[1])
        snapped_lines.append([list(snapped_start), list(snapped_end)])

    return snapped_lines

def filter_and_adjust_wall_lines(wall_lines, extension_length=0.2):
    """
    Filters and adjusts wall lines:
    1. Removes lines that are completely overlapped (subsections of others).
    2. Extends lines with endpoints that have no intersections.
    """

    
    from shapely.geometry import LineString

    def do_lines_intersect(line1, line2):
        """
        Check if two line segments intersect using Shapely.
        
        Args:
            line1 (list): Line segment as [[x1, y1], [x2, y2]].
            line2 (list): Line segment as [[x1, y1], [x2, y2]].
        
        Returns:
            bool: True if the lines intersect; False otherwise.
        """
        # Create LineString objects for the two lines
        line1_geom = LineString(line1)
        line2_geom = LineString(line2)

        # Check for intersection
        return line1_geom.intersects(line2_geom)

    def count_endpoint_intersections(line, all_lines):
        """
        Count intersections for each endpoint by splitting the line into halves.
        """

        import math
        def calculate_distance(point1, point2):
            return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

        def interpolate(point1, point2, distance):
            total_distance = calculate_distance(point1, point2)
            ratio = distance / total_distance
            return [
                point1[0] + ratio * (point2[0] - point1[0]),
                point1[1] + ratio * (point2[1] - point1[1])
            ]

        # Input line endpoints
        p1, p2 = line[0], line[1]

        # Calculate points for the first 0.2 m and last 0.2 m
        first_0_2 = interpolate(p1, p2, 0.2)
        last_0_2 = interpolate(p2, p1, 0.2)

        # Define the first 0.2 m and last 0.2 m segments
        first_half = [p1, first_0_2]
        second_half = [last_0_2, p2]


        intersect_p1 = any(
            do_lines_intersect(first_half, other_line) for other_line in all_lines if other_line != line
        )
        intersect_p2 = any(
            do_lines_intersect(second_half, other_line) for other_line in all_lines if other_line != line
        )

        return intersect_p1, intersect_p2

    
    adjusted_lines = []
    intersect_record = []

    for line1 in wall_lines:
        print("line1 before", line1)
        # Validate input line
        if not isinstance(line1, list) or len(line1) != 2:
            raise ValueError(f"Unexpected line format: {line1}. Expected [[x1, y1], [x2, y2]]")

        # Count intersections at endpoints
        intersect_start, intersect_end = count_endpoint_intersections(line1, wall_lines)
        intersect_record.append((intersect_start, intersect_end))

        # Extend line if needed
        if not intersect_start and intersect_end:
            line1 = extend_until_intersect(line1, wall_lines, max_extension=extension_length, step_size=0.02, end="start")

        elif not intersect_end and intersect_start:
            line1 = extend_until_intersect(line1, wall_lines, max_extension=extension_length, step_size=0.02, end="end")
        elif not intersect_start and not intersect_end:
            line1 = extend_until_intersect(line1, wall_lines, max_extension=extension_length, step_size=0.02, end="both")
        else:
            pass
        adjusted_lines.append(line1)
    return adjusted_lines, intersect_record

def trim_line_to_intersections(line, all_lines):
    """
    Trim a line to only keep the segment within two intersections. 
    If there are more than two intersections, keep the longest segment.

    Args:
        line (list): Line segment as [[x1, y1], [x2, y2]].
        all_lines (list): List of all other line segments.

    Returns:
        list: The trimmed line as [[x1, y1], [x2, y2]].
    """
    # Convert the line to a Shapely LineString
    line_geom = LineString(line)
    
    # Find all intersection points with other lines
    intersection_points = []
    for other_line in all_lines:
        if other_line != line:
            other_geom = LineString(other_line)
            if line_geom.intersects(other_geom):
                intersection = line_geom.intersection(other_geom)
                if isinstance(intersection, Point):
                    intersection_points.append(intersection)
                elif intersection.geom_type == "MultiPoint":
                    intersection_points.extend(intersection.geoms)

    # Sort intersection points along the line
    intersection_points = sorted(
        intersection_points,
        key=lambda point: line_geom.project(point)
    )
    
    # If fewer than two intersections, return the original line
    if len(intersection_points) < 2:
        return line
    
    # Find the longest segment between intersection points
    longest_segment = None
    max_length = 0
    for i in range(len(intersection_points) - 1):
        segment = LineString([intersection_points[i], intersection_points[i + 1]])
        if segment.length > max_length:
            longest_segment = segment
            max_length = segment.length

    # Return the coordinates of the longest segment
    return list(longest_segment.coords)

def trim_all_lines_to_intersections(filtered_lines):
    """
    Trim all lines in filtered_lines to keep only the segment within two intersections.
    If there are more than two intersections, keep the longest segment.
    
    Args:
        filtered_lines (list): List of wall lines, where each line is [[x1, y1], [x2, y2]].
    
    Returns:
        list: List of trimmed lines.
    """

    def trim_line_to_intersections(line, all_lines):
        """
        Trim a single line to keep the segment within two intersections or the longest segment
        if there are multiple intersections.

        Args:
            line (list): Line segment as [[x1, y1], [x2, y2]].
            all_lines (list): List of all other line segments.

        Returns:
            list: Trimmed line as [[x1, y1], [x2, y2]].
        """
        # Convert the line to a Shapely LineString
        line_geom = LineString(line)

        # Find all intersection points with other lines
        intersection_points = []
        for other_line in all_lines:
            if other_line != line:
                other_geom = LineString(other_line)
                if line_geom.intersects(other_geom):
                    intersection = line_geom.intersection(other_geom)
                    if isinstance(intersection, Point):
                        intersection_points.append(intersection)
                    elif intersection.geom_type == "MultiPoint":
                        intersection_points.extend(intersection.geoms)

        # Sort intersection points along the line
        intersection_points = sorted(
            intersection_points,
            key=lambda point: line_geom.project(point)
        )

        # If fewer than two intersections, return the original line as a list
        if len(intersection_points) < 2:
            return list(line_geom.coords)

        # Find the longest segment between intersection points
        longest_segment = None
        max_length = 0
        for i in range(len(intersection_points) - 1):
            segment = LineString([intersection_points[i], intersection_points[i + 1]])
            if segment.length > max_length:
                longest_segment = segment
                max_length = segment.length

        # Ensure the return value is a list of coordinates
        if longest_segment is not None:
            return [list(coord) for coord in longest_segment.coords]
        else:
            return list(line_geom.coords)

    # Process all lines
    trimmed_lines = []
    for line in filtered_lines:
        trimmed_line = trim_line_to_intersections(line, filtered_lines)
        trimmed_lines.append(trimmed_line)

    return trimmed_lines

import networkx as nx
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

def merge_lines_by_axis(lines):
    """
    Merge lines that are collinear and connected on the same axis.
    Args:
        lines (list): List of line segments in the format [[[x1, y1], [x2, y2]], ...]
    Returns:
        merged_lines (list): List of merged lines in the same format as input.
    """
    from collections import defaultdict

    # Separate horizontal and vertical lines
    horizontal_lines = defaultdict(list)
    vertical_lines = defaultdict(list)

    for line in lines:
        (x1, y1), (x2, y2) = line
        if y1 == y2:  # Horizontal line
            horizontal_lines[y1].append(sorted([x1, x2]))
        elif x1 == x2:  # Vertical line
            vertical_lines[x1].append(sorted([y1, y2]))

    def merge_sorted_intervals(intervals):
        """Merge overlapping or adjacent intervals."""
        intervals.sort()
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlapping or touching
                merged[-1] = [last[0], max(last[1], current[1])]
            else:
                merged.append(current)
        return merged

    # Merge horizontal lines
    merged_horizontal = []
    for y, x_intervals in horizontal_lines.items():
        merged_intervals = merge_sorted_intervals(x_intervals)
        for x1, x2 in merged_intervals:
            merged_horizontal.append([[x1, y], [x2, y]])

    # Merge vertical lines
    merged_vertical = []
    for x, y_intervals in vertical_lines.items():
        merged_intervals = merge_sorted_intervals(y_intervals)
        for y1, y2 in merged_intervals:
            merged_vertical.append([[x, y1], [x, y2]])

    # Combine results
    return merged_horizontal + merged_vertical

from shapely.geometry import LineString
import numpy as np

from shapely.geometry import LineString
import numpy as np

from shapely.geometry import LineString
import numpy as np


from shapely.geometry import LineString
import numpy as np


import numpy as np

def refine_lines(lines, threshold=0.2):
    """
    Post-process lines by snapping horizontal and vertical line endpoints to nearby axes.
    
    Args:
        lines (list): List of lines in the format [[[x1, y1], [x2, y2]], ...]
        threshold (float): Maximum distance to snap to an axis.
        
    Returns:
        refined_lines (list): List of refined lines with snapped endpoints.
    """
    horizontal_lines = []
    vertical_lines = []
    
    # Step 1: Separate horizontal and vertical lines
    for line in lines:
        (x1, y1), (x2, y2) = line
        if np.isclose(y1, y2):  # Horizontal line (y1 == y2)
            horizontal_lines.append(line)
        elif np.isclose(x1, x2):  # Vertical line (x1 == x2)
            vertical_lines.append(line)
    
    # Step 2: Record horizontal and vertical axis locations
    horizontal_axes = sorted(set(y1 for line in horizontal_lines for _, y1 in line))
    vertical_axes = sorted(set(x1 for line in vertical_lines for x1, _ in line))
    
    # Step 3: Snap horizontal line endpoints to nearby vertical axes
    def snap_to_axes(coord, axes):
        """Snap a coordinate to the nearest axis if within threshold."""
        for axis in axes:
            if abs(coord - axis) <= threshold:
                return axis
        return coord

    refined_horizontal_lines = []
    for line in horizontal_lines:
        (x1, y1), (x2, y2) = line
        snapped_x1 = snap_to_axes(x1, vertical_axes)
        snapped_x2 = snap_to_axes(x2, vertical_axes)
        refined_horizontal_lines.append([[snapped_x1, y1], [snapped_x2, y2]])
    
    # Step 4: Snap vertical line endpoints to nearby horizontal axes
    refined_vertical_lines = []
    for line in vertical_lines:
        (x1, y1), (x2, y2) = line
        snapped_y1 = snap_to_axes(y1, horizontal_axes)
        snapped_y2 = snap_to_axes(y2, horizontal_axes)
        refined_vertical_lines.append([[x1, snapped_y1], [x2, snapped_y2]])
    
    # Combine refined lines
    return refined_horizontal_lines + refined_vertical_lines

def normalize_line_direction(line):
    """
    Normalize the direction of a line (as a list).
    For vertical lines, ensure the start is at the top.
    For horizontal lines, ensure the start is on the left.
    For diagonal or other lines, no specific normalization is applied.

    Parameters:
    - line: A list [[x1, y1], [x2, y2]] representing the start and end points of the line.

    Returns:
    - A normalized line (list) with the start and end points reordered if needed.
    """
    [x1, y1], [x2, y2] = line

    # Check for vertical line
    if x1 == x2:  # Same x-coordinate, vertical line
        if y1 > y2:  # Ensure the start is at the top (smaller y-coordinate)
            return [[x2, y2], [x1, y1]]
        else:
            return line

    # Check for horizontal line
    elif y1 == y2:  # Same y-coordinate, horizontal line
        if x1 > x2:  # Ensure the start is on the left (smaller x-coordinate)
            return [[x2, y2], [x1, y1]]
        else:
            return line

    # For diagonal or other lines, no specific normalization
    return line

def extend_until_intersect(line, all_lines, max_extension=1.0, step_size=0.01, end="both"):
    """
    Extends a line until it intersects with any other line, ensuring it stops precisely at the intersection point.

    Args:
        line (list): Line segment as [[x1, y1], [x2, y2]].
        all_lines (list): List of all line segments to check for intersection.
        max_extension (float): Maximum extension length (in meters).
        step_size (float): Incremental extension length for each step (in meters).
        end (str): "start", "end", or "both" specifying which end(s) to extend.

    Returns:
        list: Adjusted line segment with extended endpoints.
    """

    def extend_line_segment(line, extension, end):
        """Helper function to extend a single endpoint of the line."""
        p1, p2 = np.array(line[0]), np.array(line[1])
        direction = p2 - p1

        length = np.linalg.norm(direction)
        if length == 0:
            return line  # Skip zero-length lines
        
        # Normalize direction with explicit handling of horizontal and vertical lines
        if p1[1] == p2[1]:  # Horizontal line
            direction = np.array([1, 0]) if p2[0] > p1[0] else np.array([-1, 0])
        elif p1[0] == p2[0]:  # Vertical line
            direction = np.array([0, 1]) if p2[1] > p1[1] else np.array([0, -1])

        print(p1, p2, direction)

        if end == "start":
            p1 = p1 - direction * extension
        elif end == "end":
            p2 = p2 + direction * extension

        return [list(p1), list(p2)]


    extended_line = line[:]
    current_extension = 0

    while current_extension <= max_extension:
        # Extend the specified endpoint(s)
        if end in ["start", "both"]:
            extended_line = extend_line_segment(extended_line, step_size, "start")
        if end in ["end", "both"]:
            extended_line = extend_line_segment(extended_line, step_size, "end")
        
        print("extended_line", extended_line)

        current_extension += step_size

        # Check for intersection with any other line
        line_geom = LineString([extended_line[0], extended_line[1]])
        for other_line in all_lines:
            if other_line != line:
                other_geom = LineString([other_line[0], other_line[1]])
                if line_geom.intersects(other_geom):
                    # print("intersect")
                    # # Get the exact intersection point
                    # intersection_point = line_geom.intersection(other_geom)
                    # if not intersection_point.is_empty:
                    #     # Adjust endpoint to the exact intersection point
                    #     if end == "start":
                    #         return [[list(intersection_point.coords[0]), extended_line[1]]]
                    #     elif end == "end":
                    #         return [[extended_line[0], list(intersection_point.coords[0])]]
                    return extended_line

    # Return the original line if no intersection occurs within max_extension
    return line

from shapely.geometry import LineString, Point
import numpy as np

def split_lines(lines):
    """
    Splits lines at all intersection points.
    
    Args:
        lines: List of lines, where each line is represented as a tuple of start and end points (x1, y1), (x2, y2).

    Returns:
        List of split lines.
    """
    # Convert lines to Shapely LineString objects
    shapely_lines = [LineString(line) for line in lines]
    
    # Find all intersection points
    intersection_points = set()
    for i in range(len(shapely_lines)):
        for j in range(i + 1, len(shapely_lines)):
            intersection = shapely_lines[i].intersection(shapely_lines[j])
            if isinstance(intersection, Point):
                intersection_points.add((intersection.x, intersection.y))
            elif isinstance(intersection, LineString):  # Handle overlapping lines
                for coord in intersection.coords:
                    intersection_points.add(coord)
    
    # Split each line at intersection points
    new_lines = []
    for line in shapely_lines:
        points = [Point(line.coords[0]), Point(line.coords[1])]
        for inter_point in intersection_points:
            if line.distance(Point(inter_point)) < 1e-8 and not Point(inter_point).equals(points[0]) and not Point(inter_point).equals(points[1]):
                points.append(Point(inter_point))
        points = sorted(points, key=lambda p: (p.x, p.y))
        for i in range(len(points) - 1):
            new_lines.append(((points[i].x, points[i].y), (points[i + 1].x, points[i + 1].y)))

    return new_lines

def visualize_line(adjusted_lines, name_of_plot, txt = None):
    """
    Visualizes adjusted lines with proper snapping and intersections, assigning a unique color to each line.

    Parameters:
        adjusted_lines (list): List of adjusted wall lines.
    """
    plt.figure(figsize=(8, 8))
    
    distinct_colors = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
        "lime", "teal", "pink", "gold", "brown", "black"
    ]
    random.shuffle(distinct_colors)  # Shuffle to add randomness

    # Plot the lines with distinct colors
    for idx, line in enumerate(adjusted_lines):
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        plt.plot(
            x_values, 
            y_values, 
            color=distinct_colors[idx % len(distinct_colors)], 
            linewidth=2
        )
        if txt:
            plt.text(np.mean(x_values), np.mean(y_values), f"{txt[idx]}", fontsize=8, color="black")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.title(f"{name_of_plot}")
    # plt.grid(True)
    plt.show()




import matplotlib.pyplot as plt
import random
import numpy as np

def visualize_lines_and_objects(adjusted_lines, objects, name_of_plot, line_labels=None, object_labels=None):

    """
    Visualizes adjusted lines (walls) and objects (rectangles/polygons) with distinct colors.

    Parameters:
        adjusted_lines (list): List of adjusted wall lines (each line represented as [(x1, y1), (x2, y2)]).
        objects (list): List of objects, where each object is represented by corner points [(x1, y1), (x2, y2), ...].
        name_of_plot (str): Title of the plot.
        line_labels (list, optional): Labels for the lines (walls).
        object_labels (list, optional): Labels for the objects.
    """
    plt.figure(figsize=(10, 10))
    
    distinct_colors = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
        "lime", "teal", "pink", "gold", "brown", "black"
    ]
    random.shuffle(distinct_colors)  # Shuffle to add randomness
    
    # Plot the lines with distinct colors
    for idx, line in enumerate(adjusted_lines):
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        plt.plot(
            x_values, 
            y_values, 
            color=distinct_colors[idx % len(distinct_colors)], 
            linewidth=2,
            label=f"Wall {idx+1}" if line_labels is None else line_labels[idx]
        )
        if line_labels:
            plt.text(np.mean(x_values), np.mean(y_values), f"{line_labels[idx]}", fontsize=10, color="black")
    
    # Plot the objects as filled polygons
    for idx, obj in enumerate(objects):
        print("obj!!!", obj)
        x_coords = [point[0] for point in obj]
        y_coords = [point[1] for point in obj]
        color = distinct_colors[(idx + len(adjusted_lines)) % len(distinct_colors)]
        plt.fill(x_coords, y_coords, color=color, alpha=0.4, label=f"Object {idx+1}" if object_labels is None else object_labels[idx])
        centroid = np.mean(obj, axis=0)
        # plt.text(centroid[0], centroid[1], f"{object_labels[idx]}", fontsize=8, color="blue")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"{name_of_plot}")
    # plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize
import matplotlib.pyplot as plt

from shapely.ops import polygonize, unary_union

def find_closed_areas_and_label(lines):
    """
    Fix the closed area and line-to-area association.
    Args:
        lines (list): List of line segments in the format [[[x1, y1], [x2, y2]], ...]
    Returns:
        areas (list): List of closed polygons (areas) with their coordinates.
        line_to_area (list): List of lists, where each sublist contains the indices of areas the corresponding line belongs to.
    """
    # Step 1: Convert lines to Shapely LineString objects
    shapely_lines = [LineString(line) for line in lines]
    
    # Step 2: Combine all lines into a single geometry and polygonize
    combined_lines = unary_union(shapely_lines)
    closed_areas = list(polygonize(combined_lines))
    
    # Step 3: Collect the coordinates of each closed area
    areas = [list(polygon.exterior.coords) for polygon in closed_areas]

    line_to_area = [
    [i for i, area in enumerate(areas) if tuple(line[0]) in area and tuple(line[1]) in area]
    for line in lines]
    
    visualize_lines_and_areas_with_labels(lines, areas, line_to_area)
    return areas, line_to_area
    # return areas


def visualize_lines_and_areas_with_labels(lines, areas, line_to_area):
    """
    Visualize lines and closed areas, labeling each line and area.
    
    Args:
        lines (list): List of line segments in the format [[[x1, y1], [x2, y2]], ...].
        areas (list): List of closed polygons (areas) with their coordinates.
        line_to_area (list): List of lists, where each sublist contains the indices of areas a line belongs to.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot lines with labels
    for i, line in enumerate(lines):
        x_coords, y_coords = zip(*line)
        ax.plot(x_coords, y_coords, color="blue", linewidth=1.5, label=f"Line {i}" if i == 0 else "")
        mid_x = (x_coords[0] + x_coords[1]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        area_indices = line_to_area[i]
        ax.text(mid_x, mid_y, f"L{i}\nA{area_indices}", color="blue", fontsize=10, ha="center", va="center")

    # Plot and label areas
    for i, area_coords in enumerate(areas):
        x_coords, y_coords = zip(*area_coords)
        polygon = Polygon(area_coords)
        centroid = polygon.centroid
        ax.fill(x_coords, y_coords, alpha=0.3, label=f"Area {i}" if i == 0 else "")
        ax.text(centroid.x, centroid.y, f"A{i}", color="green", fontsize=10, ha="center", va="center")

    # Customize plot
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Lines and Closed Areas with Labels")
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_closed_areas_with_labels(lines, areas):
    """
    Plot lines and closed areas with labels.
    Args:
        lines (list): List of line segments in the format [[[x1, y1], [x2, y2]], ...]
        areas (list): List of closed polygons (areas) with their coordinates.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot lines
    for line in lines:
        x, y = zip(*line)
        plt.plot(x, y, color='black', linewidth=1)
    
    # Plot closed areas with labels
    for i, area in enumerate(areas):
        x, y = zip(*area)
        plt.fill(x, y, alpha=0.3, label=f"Area {i}")
        
        # Label the area at its centroid
        centroid_x = sum(x[:-1]) / len(x[:-1])  # Exclude the duplicate last point
        centroid_y = sum(y[:-1]) / len(y[:-1])
        plt.text(centroid_x, centroid_y, str(i), fontsize=12, ha='center', va='center', color='red')
    
    plt.title("Closed Areas with Labels")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

def is_open_line(line, all_lines, tolerance=1e-6):
    """
    Check if a line is open (one or both of its endpoints are not connected to others).
    
    Args:
        line (LineString): The line to check.
        all_lines (list): List of all LineString objects.
        tolerance (float): Tolerance to consider two points as connected.
        
    Returns:
        bool: True if the line is open, False otherwise.
    """
    # Extract the endpoints of the line
    start_point = Point(line.coords[0])
    end_point = Point(line.coords[1])
    
    # Check if both endpoints are connected to other lines
    connected_start = False
    connected_end = False
    
    for other_line in all_lines:
        if line.equals(other_line):  # Skip the same line
            continue
        other_start = Point(other_line.coords[0])
        other_end = Point(other_line.coords[1])
        
        # Check if start_point is connected to other_line
        if start_point.distance(other_start) <= tolerance or start_point.distance(other_end) <= tolerance:
            connected_start = True
        
        # Check if end_point is connected to other_line
        if end_point.distance(other_start) <= tolerance or end_point.distance(other_end) <= tolerance:
            connected_end = True
        
        # If both endpoints are connected, no need to check further
        if connected_start and connected_end:
            break
    
    return not (connected_start and connected_end)

def filter_lines(lines, areas, tolerance=1e-6):
    """
    Remove lines that are open (endpoints not connected to others) and not within any closed area.
    
    Args:
        lines (list): List of line segments in the format [[[x1, y1], [x2, y2]], ...]
        areas (list): List of closed polygons (areas) with their coordinates.
        tolerance (float): Tolerance to consider two points as connected.
        
    Returns:
        filtered_lines (list): List of lines after filtering.
    """
    # Convert areas to Shapely Polygon objects
    polygons = [Polygon(area) for area in areas]
    
    # Convert lines to Shapely LineString objects
    shapely_lines = [LineString(line) for line in lines]
    
    # Filter lines
    filtered_lines = []
    for line in shapely_lines:
        # Check if the line is fully contained in any polygon
        contained_in_polygon = any(line.within(polygon) for polygon in polygons)
        
        # Check if the line is open
        if not contained_in_polygon and is_open_line(line, shapely_lines, tolerance):
            continue  # Skip this line (it's open and not within any polygon)
        
        # Keep the line if it passes the checks
        filtered_lines.append(list(line.coords))
    
    return filtered_lines


def assign_objects_to_areas(areas, objects):
    """
    Assign objects to the corresponding closed area based on overlap percentage.

    Args:
        areas (list): List of closed polygons (Shapely Polygon objects).
        objects (list): List of objects represented as corner points [[x1, y1], [x2, y2], ...]

    Returns:
        dict: A dictionary mapping area indices to their corresponding objects.
    """
    area_objects_map = {i: [] for i in range(len(areas))}
    
    for obj_idx, obj in enumerate(objects):
        obj_polygon = Polygon(obj)
        for area_idx, area in enumerate(areas):
            overlap_area = area.intersection(obj_polygon).area
            if overlap_area / obj_polygon.area > 0.5:  # More than 50% overlap
                area_objects_map[area_idx].append(obj)
                break  # Assign the object to the first matching area

    return area_objects_map


def visualize_areas_and_objects(areas, area_objects_map, lines, name_of_plot):
    """
    Visualize closed areas, wall lines, and objects in a single plot.

    Args:
        areas (list): List of closed polygons (Shapely Polygon objects).
        area_objects_map (dict): A dictionary mapping area indices to their objects.
        lines (list): List of wall lines to visualize.
        name_of_plot (str): Title of the plot.
    """
    plt.figure(figsize=(10, 10))
    
    # Define distinct colors for areas and objects
    distinct_colors = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
        "lime", "teal", "pink", "gold", "brown", "black"
    ]
    
    # Plot wall lines
    for line in lines:
        x_values = [line[0][0], line[1][0]]
        y_values = [line[0][1], line[1][1]]
        plt.plot(x_values, y_values, color="black", linewidth=1, linestyle="--", label="Wall Line")
    
    # Plot closed areas
    for idx, area in enumerate(areas):
        x_coords, y_coords = area.exterior.xy
        plt.fill(x_coords, y_coords, color=distinct_colors[idx % len(distinct_colors)], alpha=0.3, label=f"Area {idx}")
    
    # Plot objects inside areas
    for area_idx, objects in area_objects_map.items():
        for obj in objects:
            x_coords = [point[0] for point in obj]
            y_coords = [point[1] for point in obj]
            color = distinct_colors[area_idx % len(distinct_colors)]
            plt.fill(x_coords, y_coords, color=color, alpha=0.6, edgecolor="black", label=f"Object in Area {area_idx}")
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"{name_of_plot}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.axis("equal")
    plt.show()





if __name__ == "__main__":

    scene_name = "oct_31_2"

    object_lists, walls, wall_holes, floor_pose = load_processed_data(scene_name)
    obj_corners, wall_lines = plot_objs_walls(object_lists, walls, f"output/{scene_name}_layout.png")

    average_angle, filtered_angles = calculate_average_angle(wall_lines)
    rotated_wall_lines = center_rotate_and_translate_wall_lines(wall_lines, -average_angle)
    # visualize_line(rotated_wall_lines, "rotated_line")
    corrected_lines = correct_and_filter_lines(rotated_wall_lines, angle_threshold=10)
    # visualize_line(corrected_lines, "corrected_line")
    clusters = group_lines_by_grid(corrected_lines, tolerance=0.3)
    visualize_line_clusters(clusters)

    adjusted_lines = adjust_lines_with_intersections(clusters, tolerance=0.3)
    # visualize_line(adjusted_lines, "adjusted_line")

    normalized_adjusted_lines = [normalize_line_direction(line) for line in adjusted_lines]

    adjusted_lines = normalized_adjusted_lines

    connect_lines_ed = merge_lines_by_axis(adjusted_lines)



def find_dominant_orientations(polygon_points):
    """
    Find the two dominant wall orientations (perpendicular to each other) using weighted PCA.
    Walls are weighted by their length - longer walls have more influence.
    
    Parameters:
        polygon_points: List of (x, y) vertices forming a closed loop
    
    Returns:
        tuple: (main_angle, perpendicular_angle) in radians
    """
    n = len(polygon_points)
    
    # Collect all wall direction vectors with their lengths
    directions = []
    weights = []
    
    for i in range(n):
        p1 = np.array(polygon_points[i])
        p2 = np.array(polygon_points[(i+1) % n])
        v = p2 - p1
        length = np.linalg.norm(v)
        
        if length > 1e-6:
            v_normalized = v / length
            directions.append(v_normalized)
            weights.append(length)
            # Also add opposite direction
            directions.append(-v_normalized)
            weights.append(length)
    
    if len(directions) == 0:
        return 0.0, np.pi/2
    
    directions = np.array(directions)
    weights = np.array(weights)
    
    # Weighted mean of directions
    weighted_dirs = directions * weights[:, np.newaxis]
    
    # Compute weighted covariance matrix
    mean_dir = np.sum(weighted_dirs, axis=0) / np.sum(weights)
    centered = directions - mean_dir
    cov_matrix = (centered.T @ (centered * weights[:, np.newaxis])) / np.sum(weights)
    
    # Eigenvalue decomposition to find principal directions
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Main direction (eigenvector with largest eigenvalue)
    main_idx = np.argmax(eigenvalues)
    main_direction = eigenvectors[:, main_idx]
    main_angle = np.arctan2(main_direction[1], main_direction[0])
    
    # Perpendicular direction
    perp_angle = main_angle + np.pi/2
    
    return main_angle, perp_angle


def angular_difference(angle1, angle2):
    """
    Compute the smallest angular difference between two angles.
    Returns value in [-pi, pi]
    """
    diff = angle1 - angle2
    # Normalize to [-pi, pi]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def snap_angle_to_grid(angle, grid_angles, threshold_degrees=15.0):
    """
    Snap an angle to the nearest grid axis if within threshold.
    
    Parameters:
        angle: Angle in radians
        grid_angles: List of grid axis angles in radians
        threshold_degrees: Threshold in degrees
    
    Returns:
        tuple: (snapped_angle, was_snapped)
    """
    threshold_rad = np.radians(threshold_degrees)
    
    # Check all grid angles (including π rotations)
    all_grid_angles = []
    for ga in grid_angles:
        all_grid_angles.extend([ga, ga + np.pi, ga - np.pi])
    
    min_diff = float('inf')
    best_angle = angle
    
    for grid_angle in all_grid_angles:
        diff = abs(angular_difference(angle, grid_angle))
        if diff < min_diff:
            min_diff = diff
            best_angle = grid_angle
    
    was_snapped = min_diff < threshold_rad
    
    if was_snapped:
        return best_angle, True
    else:
        return angle, False


def line_intersection_2d(p1, dir1, p2, dir2):
    """
    Find intersection point of two lines defined by point + direction.
    
    Parameters:
        p1, p2: Points on each line (np.array)
        dir1, dir2: Direction vectors (np.array)
    
    Returns:
        intersection point or None if parallel
    """
    # Line 1: p1 + t * dir1
    # Line 2: p2 + s * dir2
    # Solve: p1 + t * dir1 = p2 + s * dir2
    
    # Convert to matrix form: A * [t, s]^T = b
    A = np.column_stack([dir1, -dir2])
    b = p2 - p1
    
    # Check if lines are parallel
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None
    
    params = np.linalg.solve(A, b)
    intersection = p1 + params[0] * dir1
    
    return intersection


def manhattan_world_regularization(polygon_points, angle_threshold=15.0, n_iterations=3, damping=0.6):
    """
    Apply Manhattan World regularization to snap walls to a perpendicular grid.
    Uses iterative refinement for stability.
    
    Parameters:
        polygon_points: List of (x, y) vertices forming a closed loop
        angle_threshold: Threshold in degrees for snapping (default 15°)
        n_iterations: Number of refinement iterations (default 3)
        damping: Damping factor for updates (0-1, lower = more conservative)
    
    Returns:
        List of adjusted vertices
    """
    points = [np.array(p) for p in polygon_points]
    n = len(points)
    
    if n < 3:
        return polygon_points
    
    # Find dominant grid orientations
    main_angle, perp_angle = find_dominant_orientations(polygon_points)
    grid_angles = [main_angle, perp_angle]
    
    print(f"  → Dominant wall orientations: {np.degrees(main_angle):.1f}° and {np.degrees(perp_angle):.1f}°")
    
    # Iterative refinement
    for iteration in range(n_iterations):
        new_points = []
        n_snapped = 0
        
        for i in range(n):
            # Get adjacent walls
            p_prev = points[i-1]
            p_curr = points[i]
            p_next = points[(i+1) % n]
            
            # Wall directions
            wall_before = p_curr - p_prev
            wall_after = p_next - p_curr
            
            len_before = np.linalg.norm(wall_before)
            len_after = np.linalg.norm(wall_after)
            
            if len_before < 1e-6 or len_after < 1e-6:
                new_points.append(p_curr)
                continue
            
            # Current angles
            angle_before = np.arctan2(wall_before[1], wall_before[0])
            angle_after = np.arctan2(wall_after[1], wall_after[0])
            
            # Snap to grid
            snapped_angle_before, was_snapped_before = snap_angle_to_grid(
                angle_before, grid_angles, angle_threshold
            )
            snapped_angle_after, was_snapped_after = snap_angle_to_grid(
                angle_after, grid_angles, angle_threshold
            )
            
            if was_snapped_before or was_snapped_after:
                n_snapped += 1
            
            # Create snapped direction vectors
            dir_before = np.array([np.cos(snapped_angle_before), np.sin(snapped_angle_before)])
            dir_after = np.array([np.cos(snapped_angle_after), np.sin(snapped_angle_after)])
            
            # Find intersection of snapped walls
            new_position = line_intersection_2d(p_prev, dir_before, p_next, -dir_after)
            
            if new_position is None:
                # Lines are parallel, keep original position
                new_points.append(p_curr)
            else:
                # Apply damped update (decreasing with iterations for stability)
                alpha = damping / (iteration + 1)
                updated_position = (1 - alpha) * p_curr + alpha * new_position
                new_points.append(updated_position)
        
        points = new_points
        print(f"  → Iteration {iteration+1}/{n_iterations}: {n_snapped}/{n} vertices adjusted")
    
    # Convert back to list of tuples/lists
    result = [p.tolist() for p in points]
    
    return result



