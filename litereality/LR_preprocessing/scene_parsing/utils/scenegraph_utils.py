import os
import pickle
import numpy as np
import matplotlib.pyplot as plt




def plot_objs_walls_2d(obj_corners, wall_lines, save_path):
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
    for corners_2d in enumerate(wall_lines):

        # Plot the wall as a polygon with specified wall color and transparency
        plt.fill(
            *np.append(corners_2d, [corners_2d[0]], axis=0).T,
            color=wall_color,
            label=f"Wall {index}" if index == 0 else "",
        )

        # Calculate the center position of the wall to place the label
        center_x = np.mean(corners_2d[:, 0])
        center_y = np.mean(corners_2d[:, 1])
        # plt.text(
        #     center_x,
        #     center_y,
        #     f"Wall {index}",
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     color="red",
        # )

   
    for index, corners_2d in enumerate(obj_corners):



        # Plot the object as a polygon with category-based color and transparency
        plt.fill(*np.append(corners_2d, [corners_2d[0]], axis=0).T, color="blue")

        # Calculate the center position of the bounding box to place the label
        center_x = np.mean(corners_2d[:, 0])
        center_y = np.mean(corners_2d[:, 1])
        # plt.text(
        #     center_x,
        #     center_y,
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     color="blue",
        # )

    # Formatting the plot
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    # plt.title("2D Layout of Walls and Objects")
    plt.axis("equal")
    plt.axis("off")
    plt.grid(False)

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, transparent=True)
    # plt.show()

    return obj_corners, wall_lines


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
            raw_name,
            ha="center",
            va="center",
            fontsize=10,
            color="blue",
        )

    # Formatting the plot
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    # plt.title("2D Layout of Walls and Objects")
    plt.axis("equal")
    plt.axis("off")
    plt.grid(False)

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, transparent=True)
    # plt.show()
    plt.close()

    return obj_corners, wall_lines

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
        # plt.text(
        #     center_x,
        #     center_y,
        #     f"Wall {index}",
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     color="red",
        # )

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

        print("position, rotation, bbox", position, rotation, bbox)

        corners_2d = calculate_obj_corners(position, rotation, bbox)



        if index == 0:
            print("corners_2d___", corners_2d)
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
        # plt.text(
        #     center_x,
        #     center_y,

        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     color="blue",
        # )

    # Formatting the plot
    # plt.xlabel("X Position")
    # plt.ylabel("Z Position")
    # plt.title("2D Layout of Walls and Objects")
    plt.axis("equal")
    # plt.axis("off")
    # plt.grid(True)
    # plt.grid(False)

    print("save_path!!!!", save_path)
    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, transparent=True)
    plt.show()
    # plt.close()

    return obj_corners, wall_lines


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


# get the wall bbox and object bbox
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


def get_bbox_from_poly(points):
    # Close the polygon by appending the first point at the end
    points = np.vstack([points, points[0]])

    # Width of the walls (0.16 meters outward)
    wall_width = 0.16

    # Function to compute the signed area to determine the orientation
    def signed_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Determine if the polygon is clockwise or counter-clockwise
    area = signed_area(points)
    if area > 0:
        orientation = "counter-clockwise"
        normal_sign = 1
    else:
        orientation = "clockwise"
        normal_sign = -1

    # List to store wall bounding boxes
    wall_bounding_boxes = []

    # Compute wall bounding boxes
    for i in range(len(points) - 1):
        A = points[i]
        B = points[i + 1]
        edge = B - A
        # Compute outward normal vector
        normal = normal_sign * np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Offset distance (wall extends outward)
        offset = wall_width * normal
        # Four corners of the wall bounding box
        p1 = A
        p2 = B
        p3 = B - offset
        p4 = A - offset
        wall = np.array([p1, p2, p3, p4])
        wall_bounding_boxes.append(wall)

    return wall_bounding_boxes
