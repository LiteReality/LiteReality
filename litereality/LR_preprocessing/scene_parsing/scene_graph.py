import os
import math
import pickle
import numpy as np
from utils.scenegraph_utils import (
    plot_objs_walls,
    wall_line_to_close,
    get_bbox_from_poly,
    calculate_wall_corners,
)
from shapely.geometry import Polygon, LineString

from utils.snap_to_wall import snap_and_align_bbox, overlapping_length
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


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


def load_processed_data_refined(room):
    """Load preprocessed data for the given room."""
    base_path = f"scene_data/{room}"
    with open(os.path.join(base_path, "objects_processed.pkl"), "rb") as f:
        object_lists = pickle.load(f)
    with open(os.path.join(base_path, "walls_processed.pkl"), "rb") as f:
        walls = pickle.load(f)
    with open(os.path.join(base_path, "wall_holes.pkl"), "rb") as f:
        wall_holes = pickle.load(f)
    with open(os.path.join(base_path, "floor.pkl"), "rb") as f:
        floor_pose = pickle.load(f)
    return object_lists, walls




import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint

# ---------- Basic Utilities ----------

def polygon_area(vertices):
    """Compute signed area using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

def outward_normal(v0, v1, ccw=True):
    """
    Given an edge from v0 to v1, return the outward normal (unit vector)
    assuming the polygon is in CCW order if ccw=True (interior is left),
    so outward is to the right (rotate by -90°). For CW, the reverse.
    """
    edge = v1 - v0
    length = np.linalg.norm(edge)
    if length < 1e-12:
        return np.array([0.0, 0.0])
    direction = edge / length
    if ccw:
        n = np.array([direction[1], -direction[0]])
    else:
        n = np.array([-direction[1], direction[0]])
    return n

def line_equation(v0, v1, offset=0.0, normal=None):
    """
    Returns a parametric line (point, direction).
    If offset>0 and normal is provided, shift both v0 and v1 outward by offset.
    """
    if offset != 0.0 and normal is not None:
        v0 = v0 + offset * normal
        v1 = v1 + offset * normal
    return (v0, v1 - v0)

def intersect_lines(p1, d1, p2, d2):
    """
    Solve p1 + t*d1 = p2 + s*d2 for t,s. Return the intersection point,
    or None if the lines are parallel.
    """
    mat = np.column_stack((d1, -d2))
    det = np.linalg.det(mat)
    if abs(det) < 1e-12:
        return None
    rhs = p2 - p1
    sol = np.linalg.solve(mat, rhs)
    t = sol[0]
    return p1 + t * d1

# ---------- Selective Offset for a General Polygon ----------

def selective_offset_polygon(vertices, offset_edges, offset_dist=0.16):
    """
    Given a closed polygon with vertices (ordered, not repeating the first),
    offset only the edges whose indices are in 'offset_edges' by offset_dist outward.
    
    Each edge i goes from vertices[i] to vertices[(i+1)%n].
    For each edge:
      - If it is in offset_edges, use the infinite line shifted outward by offset_dist.
      - Otherwise, use the original infinite line.
    
    New vertices are computed as the intersection of consecutive infinite lines.
    (If intersection fails, the original vertex is used.)
    
    Returns:
        new_vertices: an array of shape (n,2) for the final polygon.
    """
    n = len(vertices)
    # Compute orientation using closed vertices (append first)
    closed = np.vstack([vertices, vertices[0]])
    area = polygon_area(closed)
    ccw = (area > 0)
    
    # Build infinite line for each edge
    lines = []
    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i+1) % n]
        if i in offset_edges:
            n_out = outward_normal(v0, v1, ccw=ccw)
            p, d = line_equation(v0, v1, offset=offset_dist, normal=n_out)
        else:
            p, d = line_equation(v0, v1, offset=0.0, normal=None)
        lines.append((p, d))
    
    # Compute new vertices as intersections of line (i-1) and line i
    new_vertices = []
    for i in range(n):
        prev = (i - 1) % n
        p1, d1 = lines[prev]
        p2, d2 = lines[i]
        inter = intersect_lines(p1, d1, p2, d2)
        if inter is None:
            inter = vertices[i]
        new_vertices.append(inter)
    return np.array(new_vertices)

# ---------- Main Function ----------

def offset_wall_to_ensure_include(obj_corners, wall_lines, scenename, check_offset=0.3):
    """
    Given a closed wall boundary (wall_lines) and objects (each defined by corner points in obj_corners),
    this function selectively offsets wall edges only if one or more bbox points lie within the
    external rectangular area (of height check_offset) adjacent to the edge.
    
    For each edge:
      1. Construct a rectangle defined by the edge and its check_offset (e.g., 0.3m) offset in the outward direction.
      2. Subtract from that rectangle any area that is inside the original wall.
      3. If any bbox point (of an object) lies in the remaining external area,
         compute the required offset (the maximum distance along the outward normal among those points),
         and cap it to check_offset.
    
    Then, new wall vertices are computed as intersections of the offset (or original) infinite lines.
    
    Finally, the function plots the original and offset wall along with:
      - The object bounding boxes (in green)
      - The detection (external) areas for each edge (in orange dashed lines)
      
    Returns:
        polygon_line: a list of wall segments (each segment is a pair of endpoints) for the new polygon.
    """
    # Step 1: Convert wall_lines to vertices.
    vertices = [np.array(wall_lines[0][0])]
    for wall in wall_lines:
        vertices.append(np.array(wall[1]))
    # Remove duplicate of first vertex if present.
    vertices = np.array(vertices[:-1])
    n = len(vertices)
    
    # Build the original wall polygon for containment tests.
    poly = ShapelyPolygon(np.vstack([vertices, vertices[0]]))
    
    # Determine polygon orientation (ccw if area > 0).
    closed = np.vstack([vertices, vertices[0]])
    area = polygon_area(closed)
    ccw = (area > 0)
    
    offset_info = {}  # Dictionary to record the computed offset for each edge.
    detection_rects = []  # List to store (edge_index, external detection rectangle) for plotting.
    
    # Step 2: Check each edge.
    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i+1) % n]
        edge_vec = v1 - v0
        L = np.linalg.norm(edge_vec)
        if L < 1e-12:
            continue
        edge_dir = edge_vec / L
        n_out = outward_normal(v0, v1, ccw=ccw)
        
        # Define the rectangle along the edge that extends check_offset outward.
        v0_off = v0 + check_offset * n_out
        v1_off = v1 + check_offset * n_out
        rect = ShapelyPolygon([tuple(v0), tuple(v1), tuple(v1_off), tuple(v0_off)])
        # Remove any area that is inside the original wall.
        ext_rect = rect.difference(poly)
        
        # Save this detection area for plotting.
        detection_rects.append((i, ext_rect))
        
        max_offset = 0.0
        found = False
        # For each object, check its bounding box points.
        for obj in obj_corners:
            for pt in obj:
                sp = ShapelyPoint(pt)
                # Only count points that lie within the external rectangle.
                if not ext_rect.is_empty and ext_rect.contains(sp):
                    found = True
                    # Compute how far the point is along the outward normal.
                    d = np.dot(np.array(pt) - v0, n_out)
                    if d > max_offset:
                        max_offset = d
        # Only record an offset if at least one bbox point was found.
        if found and max_offset > 0:
            offset_info[i] = min(max_offset, check_offset)
    
    # Step 3: Build new infinite lines for each edge using the computed offsets.
    lines = []
    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i+1) % n]
        if i in offset_info:
            offset_val = offset_info[i]
            n_out = outward_normal(v0, v1, ccw=ccw)
            p, d = line_equation(v0, v1, offset=offset_val, normal=n_out)
        else:
            p, d = line_equation(v0, v1, offset=0.0, normal=None)
        lines.append((p, d))
    
    # Step 4: Compute new vertices as intersections of consecutive offset lines.
    new_vertices = []
    for i in range(n):
        prev = (i - 1) % n
        p1, d1 = lines[prev]
        p2, d2 = lines[i]
        inter = intersect_lines(p1, d1, p2, d2)
        if inter is None:
            inter = vertices[i]
        new_vertices.append(inter)
    new_vertices = np.array(new_vertices)
    
    # Step 5: Visualization.
    plt.figure(figsize=(8,8))
    
    # Plot original wall in blue.
    closed_orig = np.vstack([vertices, vertices[0]])
    plt.plot(closed_orig[:,0], closed_orig[:,1], 'b-', lw=2, label='Original Wall')
    for i, v in enumerate(vertices):
        plt.text(v[0], v[1], f"V{i}", color='blue', fontsize=10)
    for i in range(n):
        mid = (vertices[i] + vertices[(i+1)%n]) / 2
        plt.text(mid[0], mid[1], f"L{i}", color='blue', fontsize=10)
    
    # Plot the new (offset) wall in red dashed.
    closed_new = np.vstack([new_vertices, new_vertices[0]])
    plt.plot(closed_new[:,0], closed_new[:,1], 'r--', lw=2, label='Offset Wall')
    for i, v in enumerate(new_vertices):
        plt.text(v[0], v[1], f"V'{i}", color='red', fontsize=10)
    for i in range(len(new_vertices)):
        mid_new = (new_vertices[i] + new_vertices[(i+1)%len(new_vertices)]) / 2
        plt.text(mid_new[0], mid_new[1], f"L'{i}", color='red', fontsize=10)
    
    # Indicate which edges were offset.
    for i, offset_val in offset_info.items():
        mid = (vertices[i] + vertices[(i+1)%n]) / 2
        plt.text(mid[0], mid[1], f"offset: {offset_val:.2f}", color='magenta', fontsize=10)
    
    # Plot object bounding boxes in green.
    for j, obj in enumerate(obj_corners):
        obj = np.array(obj)
        closed_obj = np.vstack([obj, obj[0]])
        plt.plot(closed_obj[:,0], closed_obj[:,1], 'g-', lw=2, label='Object BBox' if j == 0 else None)
    
    # Plot detection (external) areas in orange dashed lines.
    detection_label_plotted = False
    for i, ext_rect in detection_rects:
        if ext_rect.is_empty:
            continue
        # ext_rect might be a Polygon or MultiPolygon.
        if ext_rect.geom_type == 'Polygon':
            x, y = ext_rect.exterior.xy
            plt.plot(x, y, 'orange', linestyle='--', lw=1, 
                     label='Detection Area' if not detection_label_plotted else None)
            detection_label_plotted = True
        elif ext_rect.geom_type == 'MultiPolygon':
            for part in ext_rect.geoms:
                x, y = part.exterior.xy
                plt.plot(x, y, 'orange', linestyle='--', lw=1, 
                         label='Detection Area' if not detection_label_plotted else None)
                detection_label_plotted = True
    
    # plt.title("Before & After: Selective Offset Wall with Object BBoxes and Detection Areas\n(Only edges with bbox points in external area are offset)")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # plt.axis('equal')
    # plt.savefig(f"output/{scenename}_offset_wall_before_after.png")
    plt.close()
    
    # Reconstruct polygon segments for output.
    polygon_line = []
    for i in range(len(new_vertices)):
        polygon_line.append([new_vertices[i], new_vertices[(i+1)%len(new_vertices)]])
        
    return polygon_line

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


def snap_and_scene_graph_intilized(scene_name, distance_threshold, angle_threshold):

    print("-----snap and scene graph initialized start-----")
    """
    1. give input object_list and walls, snap the object to the wall if they are close and in parallel
    2. Get walls to be connected into a closed loop
    3. Save the updated object_list and walls
    4. Save the scene graph
    """

    # Load processed data for the scene
    object_lists, walls, wall_holes, floor_pose = load_processed_data(scene_name)

    # Visualize the original layout and save the plot
    obj_corners, wall_lines = plot_objs_walls(
        object_lists, walls, f"output/{scene_name}_layout.png"
    )


    
    # Close the wall lines to form a complete polygon and compute bounding box
    polygon = wall_line_to_close(wall_lines)
    bbox_walls = get_bbox_from_poly(polygon)

    # the bbox_walls are the new walls

    corners_2d_list = []

    # Calculate the 2D corners for each wall and add to the list
    for index, wall in enumerate(walls):
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]
        # Calculate 2D corners of the wall
        corners_2d = calculate_wall_corners(position, rotation, bbox)
        corners_2d_list.append(corners_2d)

    # Compute center points for each wall's bounding box
    reconstructed_wall_centers = [
        np.mean(corners, axis=0) for corners in corners_2d_list
    ]
    original_wall_centers = [np.mean(bbox, axis=0) for bbox in bbox_walls]

    # Create a distance matrix for the Hungarian algorithm
    num_reconstructed = len(reconstructed_wall_centers)
    num_original = len(original_wall_centers)
    distance_matrix = np.zeros((num_reconstructed, num_original))

    for i, rec_center in enumerate(reconstructed_wall_centers):
        for j, orig_center in enumerate(original_wall_centers):
            distance_matrix[i, j] = np.linalg.norm(rec_center - orig_center)
    # Apply the Hungarian algorithm to minimize total correspondence distance
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # now save the wall out as pickle.
    walls_update = walls.copy()
    for index, wall in enumerate(walls_update):
        pose = wall["pose"]
        position = pose["position"]
        rotation = pose["rotation"]
        bbox = pose["bbox"]

        # their best match new wall bounding box
        new_bbox = bbox_walls[col_ind[index]]

        corners_2d = calculate_wall_corners(position, rotation, bbox)

        center_wall = np.mean(corners_2d, axis=0)
        center_bbox = np.mean(new_bbox, axis=0)

        # now update the position and rotation
        pose["position"] = [center_bbox[0], pose["position"][1], center_bbox[1]]
        length = np.linalg.norm(new_bbox[0] - new_bbox[1])
        width = np.linalg.norm(new_bbox[1] - new_bbox[2])
        pose["bbox"] = [length, pose["bbox"][1], width]

        # rotation is a bit tricky
        orientation = math.atan2(
            new_bbox[1][1] - new_bbox[0][1], new_bbox[1][0] - new_bbox[0][0]
        )
        orientation = math.degrees(orientation)
        sign_of_rotation = 1 if rotation[1] > 0 else -1
        distance_1 = abs(abs(rotation[1]) - abs(orientation))
        distance_2 = abs((180 - abs(rotation[1])) - abs(orientation))
        if distance_1 < distance_2:
            orientation = abs(orientation) * sign_of_rotation
        else:
            orientation = (180 - abs(orientation)) * sign_of_rotation
        pose["rotation"] = [pose["rotation"][0], orientation, pose["rotation"][2]]


    base_path = f"scene_data/{scene_name}"
    object_pickle_path = os.path.join(base_path, "objects_processed.pkl")
    wall_pickle_path = os.path.join(base_path, "walls_processed.pkl")

    with open(wall_pickle_path, "wb") as f:
        pickle.dump(walls_update, f)

    wall_segments = create_line_segments(polygon)

    new_object_list = []
    attached_wall_list = []
    for i, bbox_points in enumerate(obj_corners):
        bbox_points = np.array(bbox_points)
        correct_bbox = bbox_points
        length_aligned = 0
        for j, wall in enumerate(wall_segments):
            point_a, point_b = np.array(wall[0]), np.array(wall[1])
            aligned_bbox_points = snap_and_align_bbox(
                point_a,
                point_b,
                bbox_points,
                distance_threshold=distance_threshold,
                angle_threshold=angle_threshold,
            )
            overlap_length, overlap_start, overlap_end = overlapping_length(
                point_a, point_b, aligned_bbox_points
            )

            if overlap_length > length_aligned:
                length_aligned = overlap_length
                correct_bbox = aligned_bbox_points
                attached_wall_idx = j

        new_object_list.append(correct_bbox)

        if length_aligned < 0.1:
            attached_wall_idx = -1
        attached_wall_list.append(attached_wall_idx)

    wall_center = []
    for wall in wall_segments:
        wall_center.append(np.mean(wall, axis=0).tolist())
    object_center = []
    for obj in new_object_list:
        object_center.append(np.mean(obj, axis=0).tolist())
    

    import json

    print(wall_segments[0])

    # get the center of old_object_list and new_object_list
    old_object_list_center = []
    new_object_list_center = []

    for i in range(len(new_object_list)):
        old_object_list_center.append(np.mean(obj_corners[i], axis=0))
        new_object_list_center.append(np.mean(new_object_list[i], axis=0))


    # Create a distance matrix for the Hungarian algorithm
    num_reconstructed = len(new_object_list_center)
    num_original = len(old_object_list_center)
    distance_matrix = np.zeros((num_reconstructed, num_original))

    for i, rec_center in enumerate(new_object_list_center):
        for j, orig_center in enumerate(old_object_list_center):
            distance_matrix[i, j] = np.linalg.norm(rec_center - orig_center)

    # Apply the Hungarian algorithm to minimize total correspondence distance
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # resave the object_pkl:

    object_lists_update = object_lists.copy()

    for index, obj_bbox in enumerate(object_lists_update):
        position = obj_bbox["position"]
        rotation = obj_bbox["rotation"]
        bbox = obj_bbox["bbox"]
        name = obj_bbox["object_type"]

        new_position = new_object_list_center[col_ind[index]]

        obj_bbox["position"] = [new_position[0], position[1], new_position[1]]

        orientation = math.atan2(
            new_object_list[col_ind[index]][1][1]
            - new_object_list[col_ind[index]][0][1],
            new_object_list[col_ind[index]][1][0]
            - new_object_list[col_ind[index]][0][0],
        )
        orientation = math.degrees(orientation)

        sign_of_rotation = 1 if rotation > 0 else -1

        distance_1 = abs(abs(rotation) - abs(orientation))
        distance_2 = abs((180 - abs(rotation)) - abs(orientation))

        if distance_1 < distance_2:
            orientation = abs(orientation) * sign_of_rotation
        else:
            orientation = (180 - abs(orientation)) * sign_of_rotation

    # save the updated layout as pickle
    with open(object_pickle_path, "wb") as f:
        pickle.dump(object_lists_update, f)

    obj_corners, wall_lines = plot_objs_walls(
        object_lists_update, walls_update, f"output/{scene_name}_layout_updated.png"
    )


    scene_graph = {
            "walls": [
                {"id": i, "center": center, "name": f"Wall {i}", "line_segment": [arr.tolist() for arr in wall_segments[i]] }
                for i, center in enumerate(wall_center)
            ],
            "objects": [
                {
                    "id": i,
                    "center": center,
                    "attached_wall": attached_wall_list[i],
                    "type": object_lists[i]["object_type"],
                    "four_point": [arr.tolist() for arr in new_object_list[i]],
                    "bbox3d": [arr.tolist() for arr in object_lists_update[i]["bbox"]] ,
                    "position_3d": [arr.tolist() for arr in object_lists_update[i]["position"]],
                }
                for i, center in enumerate(object_center)
                
            ],
            "wall_connectivity": [
                {"wall_1": i, "wall_2": (i + 1) % len(wall_segments)}
                for i in range(len(wall_segments))
            ],
        }

    # Output the scene graph to a JSON file
    with open(f"scene_data/{scene_name}/scene_graph.json", "w") as json_file:
        json.dump(scene_graph, json_file, indent=4)

    print("updated object_list saved to {}".format(object_pickle_path))
    print("updated wall_list saved to {}".format(wall_pickle_path))
    print("scene graph saved to {}".format(f"scene_data/{scene_name}/scene_graph.json"))

    print("-----snap and scene graph initialized finish-----")

    return walls_update, object_lists_update

import random

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
            plt.text(np.mean(x_values), np.mean(y_values), f"{line_labels[idx]}", fontsize=8, color="black")
    
    # Plot the objects as filled polygons
    for idx, obj in enumerate(objects):
        x_coords = [point[0] for point in obj]
        y_coords = [point[1] for point in obj]
        color = distinct_colors[(idx + len(adjusted_lines)) % len(distinct_colors)]
        plt.fill(x_coords, y_coords, color=color, alpha=0.4, label=f"Object {idx+1}" if object_labels is None else object_labels[idx])
        if object_labels:
            centroid = np.mean(obj, axis=0)
            plt.text(centroid[0], centroid[1], f"{object_labels[idx]}", fontsize=8, color="black")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"{name_of_plot}")
    # plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def update_wall_line_with_max_shift(obj_corners, obj_type, wall_lines, max_shift=0.16, margin=0.0, tolerance=1e-5):
    """
    Moves each wall line at most `max_shift` along its outward normal to try to enclose
    all object corners. Each line is processed once, in the original order. After a line 
    moves, we rebuild the polygon and proceed to the next line.

    Args:
        obj_corners (list of list of [x, y]):
            Each object's bounding box corners (e.g., 4 corners).
        obj_type (list of str):
            The type of each object (not used here, but kept for signature consistency).
        wall_lines (list of [[x1, y1], [x2, y2]]):
            Ordered wall segments forming a closed loop.
        max_shift (float):
            The maximum distance any single line can move.
        margin (float):
            Extra offset so corners aren't just on the line but slightly inside.
        tolerance (float):
            Distance threshold for treating corners as "on" the boundary.

    Returns:
        updated_lines (list):
            The final updated wall lines after shifting each line up to max_shift.
    """
    import numpy as np
    from shapely.geometry import Polygon, Point
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------
    # Helper: build a polygon from the ordered wall lines
    # -----------------------------------------------------------
    def polygon_from_lines(lines):
        pts = []
        for i, seg in enumerate(lines):
            if i == 0:
                pts.append(seg[0])
                pts.append(seg[1])
            else:
                pts.append(seg[1])
        if len(lines) > 0:
            pts.append(lines[0][0])  # close the loop
        return Polygon(pts)

    # -----------------------------------------------------------
    # Helper: check if a point is inside or on boundary
    #         within a small tolerance
    # -----------------------------------------------------------
    def is_inside_or_on(pt, poly):
        shapely_pt = Point(pt)
        if poly.contains(shapely_pt):
            return True
        dist = poly.boundary.distance(shapely_pt)
        return (dist <= tolerance)

    # -----------------------------------------------------------
    # Find corners outside the polygon
    # -----------------------------------------------------------
    def corners_outside_polygon(corners_list, polygon):
        outs = []
        for c in corners_list:
            if not is_inside_or_on(c, polygon):
                outs.append(c)
        return outs

    # -----------------------------------------------------------
    # Distance from a point to a line segment
    # plus the param t in [0..1]
    # -----------------------------------------------------------
    def point_segment_dist_and_t(pt, seg_start, seg_end):
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-12:
            return np.linalg.norm(pt - seg_start), 0.0
        t = np.dot(pt - seg_start, seg_vec) / seg_len_sq
        if t < 0:
            closest = seg_start
        elif t > 1:
            closest = seg_end
        else:
            closest = seg_start + t * seg_vec
        dist = np.linalg.norm(pt - closest)
        return dist, t

    # -----------------------------------------------------------
    # Outward normal for a line in the polygon
    # We guess "outside" by checking a small offset from midpoint
    # -----------------------------------------------------------
    def outward_normal(p1, p2, polygon):
        midpoint = (p1 + p2) * 0.5
        direction = p2 - p1
        perp = np.array([direction[1], -direction[0]])
        test_pt = midpoint + 0.001 * perp
        if polygon.contains(Point(test_pt)):
            perp = -perp
        length = np.linalg.norm(perp)
        if length > 1e-12:
            perp /= length
        return perp

    # -----------------------------------------------------------
    # Single pass approach: 
    # - For each line in original order
    #   * find corners outside the polygon
    #   * among those, pick corners that are on the "outside" side of this line
    #   * compute how far to shift so each such corner is on/inside
    #   * shift by min(that distance + margin, max_shift)
    #   * rebuild polygon
    # -----------------------------------------------------------

    # 1) Convert input lines to np arrays
    lines_np = []
    for seg in wall_lines:
        p1 = np.array(seg[0], dtype=float)
        p2 = np.array(seg[1], dtype=float)
        lines_np.append((p1, p2))

    # 2) Build the "before" polygon for visualization
    original_polygon = polygon_from_lines(lines_np)

    # 3) Single pass: iterate lines in original order
    for i_line in range(len(lines_np)):
        # Rebuild polygon from the updated lines so far
        poly_current = polygon_from_lines(lines_np)

        p1, p2 = lines_np[i_line]
        normal_vec = outward_normal(p1, p2, poly_current)

        # Gather corners outside
        outside_pts = []
        for bbox in obj_corners:
            outs = corners_outside_polygon(bbox, poly_current)
            outside_pts.extend(outs)

        if not outside_pts:
            continue  # nothing is outside, move on

        # Among the outside corners, we only care about corners that are 
        # actually on the outward side of this line
        needed_shift = 0.0
        for corner in outside_pts:
            dist, _ = point_segment_dist_and_t(np.array(corner), p1, p2)
            # Check side
            side = np.dot(np.array(corner) - p1, normal_vec)
            if side > 0:
                # corner is on outward side
                # shift needed so corner is on/inside => dist + margin
                shift_required = dist + margin
                if shift_required > needed_shift:
                    needed_shift = shift_required

        # Shift the line by min(needed_shift, max_shift)
        if needed_shift > 0:
            shift_amount = min(needed_shift, max_shift)
            p1_new = p1 + shift_amount * normal_vec
            p2_new = p2 + shift_amount * normal_vec
            lines_np[i_line] = (p1_new, p2_new)

    # 4) Build final polygon
    final_polygon = polygon_from_lines(lines_np)

    # -----------------------------
    # Visualization
    # -----------------------------
    import matplotlib.pyplot as plt

    # (A) Before
    plt.figure(figsize=(8,8))
    ox, oy = original_polygon.exterior.xy
    plt.plot(ox, oy, 'b-', linewidth=2, label="Original Wall")

    # Plot bounding boxes (red if partially outside originally, else green)
    for bbox in obj_corners:
        corners_arr = np.array(bbox)
        if not np.array_equal(corners_arr[0], corners_arr[-1]):
            corners_arr = np.vstack([corners_arr, corners_arr[0]])
        color = 'g'
        for c in bbox:
            if not is_inside_or_on(c, original_polygon):
                color = 'r'
                break
        plt.plot(corners_arr[:,0], corners_arr[:,1], color=color, linewidth=2, marker='o')
    plt.title("Before: Single-Pass, Max Shift = %.2f\n(Red boxes partially outside)" % max_shift)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_max_shift_before.png")
    plt.close()

    # (B) After
    plt.figure(figsize=(8,8))
    fx, fy = final_polygon.exterior.xy
    plt.plot(fx, fy, 'k-', linewidth=2, label="Updated Wall")

    # Plot bounding boxes in green if inside final polygon, else red
    for bbox in obj_corners:
        corners_arr = np.array(bbox)
        if not np.array_equal(corners_arr[0], corners_arr[-1]):
            corners_arr = np.vstack([corners_arr, corners_arr[0]])
        # Check corners
        color = 'g'
        for c in bbox:
            if not is_inside_or_on(c, final_polygon):
                color = 'r'
                break
        plt.plot(corners_arr[:,0], corners_arr[:,1], color=color, linewidth=2, marker='o')
    plt.title("After: Single-Pass, Max Shift = %.2f" % max_shift)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_max_shift_after.png")
    plt.close()

    # Convert lines_np back to Python lists
    updated_lines = []
    for (p1, p2) in lines_np:
        updated_lines.append([p1.tolist(), p2.tolist()])

    return updated_lines
def update_wall_line_for_collision_minimal(obj_corners, obj_type, wall_lines, max_iterations=1000, tolerance=1e-5):
    """
    Iteratively move one wall line at a time so that any outside object corner
    is brought just onto or inside the polygon boundary. This aims for minimal
    movement: each corner push only moves the closest line enough to include
    that corner, then we re-check from the beginning.

    Args:
        obj_corners (list of list of [x, y]):
            Each entry is an object's bounding box corners (e.g., 4 corners).
        obj_type (list of str):
            The type of each object (not used directly here, but kept for signature consistency).
        wall_lines (list of [[x1, y1], [x2, y2]]):
            Ordered wall segments forming a closed loop.
        max_iterations (int):
            Maximum times we’ll iterate over all corners.
        tolerance (float):
            A small distance threshold to treat a corner as "inside" if it's
            extremely close to the boundary.

    Returns:
        final_lines (list):
            Updated wall lines after minimal shifting.
    """
    import numpy as np
    from shapely.geometry import Polygon, Point

    # ------------------------------------------------
    # Helper: build a polygon from ordered wall lines
    # ------------------------------------------------
    def polygon_from_lines(lines):
        pts = []
        for i, seg in enumerate(lines):
            if i == 0:
                pts.append(seg[0])
                pts.append(seg[1])
            else:
                pts.append(seg[1])
        # Close the loop
        if len(lines) > 0:
            pts.append(lines[0][0])
        return Polygon(pts)

    # ------------------------------------------------
    # Helper: check if a point is inside or on boundary
    #         of polygon (within a small tolerance).
    # ------------------------------------------------
    def point_inside_or_on(pt, polygon):
        # If inside or on boundary within tolerance, return True
        shapely_pt = Point(pt)
        dist = polygon.boundary.distance(shapely_pt)
        if polygon.contains(shapely_pt) or dist <= tolerance:
            return True
        return False

    # ------------------------------------------------
    # Helper: find corners outside the polygon
    # ------------------------------------------------
    def find_outside_corners(corner_lists, polygon):
        outside = []
        for i_obj, corners in enumerate(corner_lists):
            for corner in corners:
                if not point_inside_or_on(corner, polygon):
                    outside.append(corner)
        return outside

    # ------------------------------------------------
    # Distance from a point to a segment + the param t
    # ------------------------------------------------
    def point_segment_dist_and_t(pt, seg_start, seg_end):
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-12:
            # degenerate line
            return np.linalg.norm(pt - seg_start), 0.0
        t = np.dot(pt - seg_start, seg_vec) / seg_len_sq
        if t < 0:
            closest = seg_start
        elif t > 1:
            closest = seg_end
        else:
            closest = seg_start + t * seg_vec
        dist = np.linalg.norm(pt - closest)
        return dist, t

    # ------------------------------------------------
    # Find the closest line (by perpendicular distance)
    # ------------------------------------------------
    def find_closest_line(pt, lines):
        pt_arr = np.array(pt)
        best_idx = None
        best_dist = None
        for i, (p1, p2) in enumerate(lines):
            dist, _ = point_segment_dist_and_t(pt_arr, p1, p2)
            if best_idx is None or dist < best_dist:
                best_idx = i
                best_dist = dist
        return best_idx, best_dist

    # ------------------------------------------------
    # Determine outward normal for a line in the polygon
    # We'll test a midpoint and see which side is outside
    # ------------------------------------------------
    def outward_normal(e1, e2, polygon):
        e1, e2 = np.array(e1), np.array(e2)
        direction = e2 - e1
        perp = np.array([direction[1], -direction[0]])  # a perpendicular
        mid = (e1 + e2) * 0.5
        test_pt = mid + perp * 0.001  # small step in perp direction
        if polygon.contains(Point(test_pt)):
            # If that test point is inside, flip
            perp = -perp
        # Normalize
        length = np.linalg.norm(perp)
        if length > 1e-12:
            perp /= length
        return perp

    # ------------------------------------------------
    # Shift one wall line outward so that 'corner' is
    # just on or inside the polygon boundary
    # ------------------------------------------------
    def shift_line_for_corner(line_idx, corner):
        p1, p2 = new_lines[line_idx]
        dist, _ = point_segment_dist_and_t(np.array(corner), p1, p2)
        if dist < tolerance:
            return  # already close enough

        # Build current polygon from lines
        poly = polygon_from_lines(new_lines)
        # Outward normal
        n = outward_normal(p1, p2, poly)

        # Shift amount = exactly dist, so corner is on the line
        shift_amt = dist

        # Move the line
        new_p1 = p1 + shift_amt * n
        new_p2 = p2 + shift_amt * n
        new_lines[line_idx] = (new_p1, new_p2)

    # -----------------------------
    # Make a mutable copy of lines
    # -----------------------------
    new_lines = []
    for seg in wall_lines:
        p1, p2 = np.array(seg[0]), np.array(seg[1])
        new_lines.append((p1, p2))

    # -----------------------------
    # Iterative approach
    # -----------------------------
    iteration = 0
    while iteration < max_iterations:
        iteration += 1

        # Build polygon
        current_poly = polygon_from_lines(new_lines)
        # Find outside corners
        outside = find_outside_corners(obj_corners, current_poly)
        if not outside:
            # all corners inside or on boundary
            break

        # We'll fix the first outside corner
        corner_to_fix = outside[0]
        # Find closest line
        idx_line, dist_line = find_closest_line(corner_to_fix, new_lines)
        if dist_line < tolerance:
            # corner is effectively on boundary; skip
            continue

        # Shift that line outward
        shift_line_for_corner(idx_line, corner_to_fix)
        # Then re-check from scratch

    # Build final polygon
    final_poly = polygon_from_lines(new_lines)

    # -----------------------------
    # Visualization: Before & After
    # -----------------------------
    import matplotlib.pyplot as plt

    # (A) Before
    original_poly = polygon_from_lines(wall_lines)
    plt.figure(figsize=(8, 8))
    ox, oy = original_poly.exterior.xy
    plt.plot(ox, oy, 'b-', linewidth=2, label="Original Wall")

    # Draw bounding boxes in red if any corner is outside, else green
    for corners in obj_corners:
        corners_arr = np.array(corners)
        if not np.array_equal(corners_arr[0], corners_arr[-1]):
            corners_arr = np.vstack([corners_arr, corners_arr[0]])
        # Check corners
        color = 'g'
        for c in corners_arr[:-1]:
            if not point_inside_or_on(c, original_poly):
                color = 'r'
                break
        plt.plot(corners_arr[:, 0], corners_arr[:, 1], color=color, linewidth=2, marker='o')
    plt.title("Before Minimal Line-by-Line Update\n(Red boxes partially outside)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_minimal_wall_before.png")
    plt.close()

    # (B) After
    final_x, final_y = final_poly.exterior.xy
    plt.figure(figsize=(8, 8))
    plt.plot(final_x, final_y, 'k-', linewidth=2, label="Updated Wall")

    # Check corners again
    for corners in obj_corners:
        corners_arr = np.array(corners)
        if not np.array_equal(corners_arr[0], corners_arr[-1]):
            corners_arr = np.vstack([corners_arr, corners_arr[0]])
        color = 'g'
        for c in corners_arr[:-1]:
            if not point_inside_or_on(c, final_poly):
                color = 'r'
                break
        plt.plot(corners_arr[:, 0], corners_arr[:, 1], color=color, linewidth=2, marker='o')
    plt.title("After Minimal Line-by-Line Update\n(All boxes should be inside or on boundary)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("visualization_minimal_wall_after.png")
    plt.close()

    # Convert back to Python lists
    final_lines = []
    for (p1, p2) in new_lines:
        final_lines.append([p1.tolist(), p2.tolist()])

    return final_lines

def line_obj_process_with_vis(obj_corners, obj_type, wall_lines, scene_name, distance_threshold, angle_threshold):

    """
    1. Input the object corners and wall line segments.
    2. Snap the objects to the walls and resolve collisions.
    3. Return the updated object corners and a scene graph.
    
    Additionally, a visualization of the scene graph is saved to disk, where:
      - Walls are drawn in blue with their centers labeled (W0, W1, etc.).
      - Objects are drawn in green with their centers labeled (O0, O1, etc.).
      - If an object is attached to a wall, a red dashed line connects the object center to the wall center.
    """

    from shapely.geometry import Polygon
    import numpy as np
    import matplotlib.pyplot as plt

    # Build closed polygon from wall lines
    points_for_polygon = []
    for i, wall in enumerate(wall_lines):
        if i == 0:
            points_for_polygon.append(wall[0])
            points_for_polygon.append(wall[1])
        else:
            points_for_polygon.append(wall[1])
    points_for_polygon.append(wall_lines[0][0])
    polygon_closed = Polygon(points_for_polygon)

    wall_segments = wall_lines
    new_object_list = []
    attached_wall_list = []

    # Process each object bounding box
    for i, bbox_points in enumerate(obj_corners):
        bbox_type = obj_type[i]
        # if "Chair" in bbox_type:
        #     new_object_list.append(np.array(bbox_points))
        #     attached_wall_list.append(-1)
        #     continue
        bbox_points = np.array(bbox_points)
        correct_bbox = bbox_points
        length_aligned = 0
        for j, wall in enumerate(wall_segments):
            point_a, point_b = np.array(wall[0]), np.array(wall[1])
            aligned_bbox_points = snap_and_align_bbox(
                point_a,
                point_b,
                bbox_points,
                polygon_closed,
                distance_threshold=distance_threshold,
                angle_threshold=angle_threshold,
            )

            overlap_length, overlap_start, overlap_end = overlapping_length(
                point_a, point_b, aligned_bbox_points
            )
            if overlap_length > length_aligned:
                length_aligned = overlap_length
                correct_bbox = aligned_bbox_points
                attached_wall_idx = j

        new_object_list.append(correct_bbox)
        if length_aligned < 0.1:
            attached_wall_idx = -1
        attached_wall_list.append(attached_wall_idx)

    # Compute centers of walls and objects
    wall_center = []
    for wall in wall_segments:
        wall_center.append(np.mean(wall, axis=0).tolist())
    object_center = []
    for obj in new_object_list:
        object_center.append(np.mean(obj, axis=0).tolist())

    # Build scene graph
    scene_graph = {
        "walls": [
            {"id": i, "center": center, "name": f"Wall {i}", "line_segment": [list(arr) for arr in wall_segments[i]]}
            for i, center in enumerate(wall_center)
        ],
        "objects": [
            {
                "id": i,
                "center": center,
                "attached_wall": attached_wall_list[i],
                "four_point": [arr for arr in obj_corners[i]]
            }
            for i, center in enumerate(object_center)
        ],
        "wall_connectivity": [
            {"wall_1": i, "wall_2": (i + 1) % len(wall_segments)}
            for i in range(len(wall_segments))
        ],
    }

    # --- Visualization of the Scene Graph ---

    # create cache folder for scene graph visualizations with scene_name folder
    scene_graph_cache_dir = os.path.join("cache", "scene_graph_cache", scene_name)
    os.makedirs(scene_graph_cache_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot walls
    for wall in scene_graph["walls"]:
        segment = wall["line_segment"]
        x_vals = [segment[0][0], segment[1][0]]
        y_vals = [segment[0][1], segment[1][1]]
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label="Wall" if wall["id"] == 0 else "")
        center = wall["center"]
        ax.plot(center[0], center[1], 'bo')
        ax.text(center[0], center[1], f"W{wall['id']}", color='blue', fontsize=9)

    # Plot objects and their relationships
    for obj in scene_graph["objects"]:
        pts = obj["four_point"]
        pts_arr = np.array(pts)
        # Ensure polygon is closed
        if not np.array_equal(pts_arr[0], pts_arr[-1]):
            pts_arr = np.vstack([pts_arr, pts_arr[0]])
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], 'g-', linewidth=2, label="Object" if obj["id"] == 0 else "")
        center = obj["center"]
        ax.plot(center[0], center[1], 'go')
        ax.text(center[0], center[1], f"O{obj['id']}", color='green', fontsize=9)
        # If object is attached to a wall, draw a connection line
        if obj["attached_wall"] != -1:
            wall_id = obj["attached_wall"]
            wall_center_pt = scene_graph["walls"][wall_id]["center"]
            ax.plot([center[0], wall_center_pt[0]], [center[1], wall_center_pt[1]], 'r--', linewidth=1)
            mid_x = (center[0] + wall_center_pt[0]) / 2
            mid_y = (center[1] + wall_center_pt[1]) / 2
            ax.text(mid_x, mid_y, f"A{wall_id}", color='red', fontsize=8)

    ax.set_title("Scene Graph: Walls and Objects")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.savefig(os.path.join(scene_graph_cache_dir, f"a_{scene_name}_visualization_scene_graph.png"))
    plt.close()
    # --- End Visualization ---

    return new_object_list, scene_graph


def line_obj_process(obj_corners, obj_type, wall_lines, distance_threshold, angle_threshold):

    """
    1. input the obj and line segments, snap the obj to the line segments
    2. resolve collision
    3. return the updated obj_corners and wall_lines
    """
    

    points_for_polygon = []
    for i, wall in enumerate(wall_lines):
        if i == 0:
            points_for_polygon.append(wall[0])
            points_for_polygon.append(wall[1])
        else:
            points_for_polygon.append(wall[1])
    
    points_for_polygon.append(wall_lines[0][0])
    polygon_closed = Polygon(points_for_polygon)


    # Close the wall lines to form a complete polygon and compute bounding box
    wall_segments = wall_lines

    new_object_list = []
    attached_wall_list = []

    # visualize_lines_and_objects(wall_segments, obj_corners, "Original Layout")


    for i, bbox_points in enumerate(obj_corners):
        bbox_type = obj_type[i]
        if "Chair" in bbox_type:
            new_object_list.append(np.array(bbox_points))
            attached_wall_list.append(-1)
            continue
        bbox_points = np.array(bbox_points)

        correct_bbox = bbox_points
        length_aligned = 0
        for j, wall in enumerate(wall_segments):
            point_a, point_b = np.array(wall[0]), np.array(wall[1])
            aligned_bbox_points = snap_and_align_bbox(
                point_a,
                point_b,
                bbox_points,
                polygon_closed,
                distance_threshold=distance_threshold,
                angle_threshold=angle_threshold,
            )
            overlap_length, overlap_start, overlap_end = overlapping_length(
                point_a, point_b, aligned_bbox_points
            )

            if overlap_length > length_aligned:
                length_aligned = overlap_length
                correct_bbox = aligned_bbox_points
                attached_wall_idx = j

        new_object_list.append(correct_bbox)

        if length_aligned < 0.1:
            attached_wall_idx = -1
        attached_wall_list.append(attached_wall_idx)

    wall_center = []
    for wall in wall_segments:
        wall_center.append(np.mean(wall, axis=0).tolist())
    object_center = []
    for obj in new_object_list:
        object_center.append(np.mean(obj, axis=0).tolist())
    
    scene_graph = {
            "walls": [
                {"id": i, "center": center, "name": f"Wall {i}", "line_segment": [list(arr) for arr in wall_segments[i]] }
                for i, center in enumerate(wall_center)
            ],
            "objects": [
                {
                    "id": i,
                    "center": center,
                    "attached_wall": attached_wall_list[i],
                    "four_point": [arr.tolist() for arr in new_object_list[i]]
                }
                for i, center in enumerate(object_center)
            ],
            "wall_connectivity": [
                {"wall_1": i, "wall_2": (i + 1) % len(wall_segments)}
                for i in range(len(wall_segments))
            ],
        }

    return new_object_list, scene_graph


    
if __name__ == "__main__":
    scene_name = "girton"
    distance_threshold = 0.4
    angle_threshold = 5

    snap_and_scene_graph_intilized(scene_name, distance_threshold, angle_threshold)
