import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString

def line_intersects_or_near_bbox(
    p1, p2, bbox_points, tol=1e-9, distance_threshold=0.3, angle_threshold=5
):
    """Finds intersections between a line segment and a bounding box, including near-miss cases."""
    intersections = []

    def get_line_params(p_start, p_end):
        """Returns the coefficients (A, B, C) of the line equation Ax + By + C = 0."""
        x1, y1 = p_start
        x2, y2 = p_end
        A = y2 - y1
        B = x1 - x2
        C = -(A * x1 + B * y1)
        return A, B, C

    def is_parallel(A1, B1, A2, B2, angle_tol):
        """Checks if two lines are parallel within an angle tolerance."""
        dot_product = A1 * A2 + B1 * B2
        mag1 = np.hypot(A1, B1)
        mag2 = np.hypot(A2, B2)
        cos_theta = dot_product / (mag1 * mag2)
        angle_diff = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return min(angle_diff, 180 - angle_diff) < angle_tol

    def point_to_segment_distance(p, a, b):
        """Compute the minimal distance between point p and segment a-b."""
        ap = p - a
        ab = b - a
        ab_squared = np.dot(ab, ab)
        if ab_squared == 0:
            # a and b are the same point
            return np.linalg.norm(p - a)
        t = np.dot(ap, ab) / ab_squared
        if t < 0.0:
            closest_point = a
        elif t > 1.0:
            closest_point = b
        else:
            closest_point = a + t * ab
        return np.linalg.norm(p - closest_point)

    # Get line parameters for the line segment
    A1, B1, C1 = get_line_params(p1, p2)

    # Define edges of the bounding box
    edges = [
        (bbox_points[0], bbox_points[1]),  # Left edge
        (bbox_points[1], bbox_points[2]),  # Top edge
        (bbox_points[2], bbox_points[3]),  # Right edge
        (bbox_points[3], bbox_points[0]),  # Bottom edge
    ]

    # Initialize list to collect parallel edges and their distances
    parallel_edges = []

    for edge_start, edge_end in edges:
        # Get line parameters for the edge
        A2, B2, C2 = get_line_params(edge_start, edge_end)

        # Check if lines are parallel within angle threshold
        if is_parallel(A1, B1, A2, B2, angle_threshold):
            # Compute the minimal distance between the two segments
            # Compute distances between endpoints and opposite segments
            d1 = point_to_segment_distance(p1, edge_start, edge_end)
            d2 = point_to_segment_distance(p2, edge_start, edge_end)
            d3 = point_to_segment_distance(edge_start, p1, p2)
            d4 = point_to_segment_distance(edge_end, p1, p2)
            min_distance = min(d1, d2, d3, d4)
            parallel_edges.append((edge_start, edge_end, min_distance))
        else:
            # Lines are not parallel, check for actual intersection
            determinant = A1 * B2 - A2 * B1
            if abs(determinant) < tol:
                # Lines are parallel, skip
                continue
            # Compute intersection point
            x = (B2 * (-C1) - B1 * (-C2)) / determinant
            y = (A1 * (-C2) - A2 * (-C1)) / determinant

            intersection_point = np.array([x, y])

            # Check if the intersection point lies within both line segments
            if (
                min(p1[0], p2[0]) - tol <= x <= max(p1[0], p2[0]) + tol
                and min(p1[1], p2[1]) - tol <= y <= max(p1[1], p2[1]) + tol
                and min(edge_start[0], edge_end[0]) - tol
                <= x
                <= max(edge_start[0], edge_end[0]) + tol
                and min(edge_start[1], edge_end[1]) - tol
                <= y
                <= max(edge_start[1], edge_end[1]) + tol
            ):
                intersections.append(intersection_point)

    if parallel_edges:
        # Find the edge with minimal distance
        parallel_edges.sort(key=lambda x: x[2])  # Sort by distance
        closest_edge = parallel_edges[0]
        edge_start, edge_end, min_distance = closest_edge

        if min_distance < distance_threshold:
            # Project the edge's endpoints onto the line
            def project_point(p):
                # Project point p onto the line defined by A1 x + B1 y + C1 = 0
                D = A1 * p[0] + B1 * p[1] + C1
                denom = A1**2 + B1**2
                x_proj = p[0] - A1 * D / denom
                y_proj = p[1] - B1 * D / denom
                return np.array([x_proj, y_proj])

            proj_start = project_point(edge_start)
            proj_end = project_point(edge_end)

            # Check if projections lie within the line segment
            def is_between(a, b, c):
                return min(a, b) - tol <= c <= max(a, b) + tol

            t_start = ((proj_start - p1) @ (p2 - p1)) / np.dot(p2 - p1, p2 - p1)
            t_end = ((proj_end - p1) @ (p2 - p1)) / np.dot(p2 - p1, p2 - p1)

            if 0 - tol <= t_start <= 1 + tol:
                intersections.append(proj_start)
            if 0 - tol <= t_end <= 1 + tol:
                intersections.append(proj_end)
            # If neither projection lies within the line segment, find the closest point
            if not ((0 - tol <= t_start <= 1 + tol) or (0 - tol <= t_end <= 1 + tol)):
                intersections.append(proj_end)
            return intersections


def closest_bbox_point(point, bbox_points):
    """Finds the closest point on the bounding box to the specified point."""
    distances = [np.linalg.norm(point - bbox_point) for bbox_point in bbox_points]
    closest_index = np.argmin(distances)
    return bbox_points[closest_index], closest_index


import numpy as np
import matplotlib.pyplot as plt


# Helper function to calculate the angle between two points
def calculate_angle(p1, p2):
    line_vector = p2 - p1
    angle = np.arctan2(line_vector[1], line_vector[0])
    return angle


# Helper function to rotate a point around a pivot by a specified angle
def rotate_point(point, pivot, angle):
    translated_point = point - pivot
    rotated_point = np.array(
        [
            translated_point[0] * np.cos(angle) - translated_point[1] * np.sin(angle),
            translated_point[0] * np.sin(angle) + translated_point[1] * np.cos(angle),
        ]
    )
    return rotated_point + pivot


# Function to rotate the bounding box around a pivot point to a target angle
def rotate_bbox(bbox_points, pivot, target_angle):
    return [rotate_point(point, pivot, target_angle) for point in bbox_points]


import numpy as np
import numpy as np

def compute_edges_right_to_left(bbox_points):
    """
    Compute bounding box edges where each edge starts from the rightmost point 
    and ends at the leftmost point.
    
    Args:
        bbox_points: List or array of 2D points (x, y) of the bounding box.
    
    Returns:
        edges: List of edge vectors (right_point - left_point).
    """
    edges = []
    n = len(bbox_points)

    for i in range(n):
        # Current point and the next point (loop around)
        p1 = bbox_points[i]
        p2 = bbox_points[(i + 1) % n]

        # Determine rightmost (x is larger) and leftmost point
        if p1[0] > p2[0]:  # Compare x-coordinates
            right_point = p1
            left_point = p2
        else:
            right_point = p2
            left_point = p1

        # Compute edge vector: right_point -> left_point
        edge_vector = np.array(left_point) - np.array(right_point)
        edges.append(edge_vector)

    return edges


def calculate_perpendicular_angle_diff(bbox_points, line_start, line_end):
    """
    Calculate angle differences for two bounding box edges most perpendicular to the line segment.

    Parameters:
        bbox_points: np.array - (4, 2) array of bounding box corner points.
        line_start: np.array - (2,) start point of the line segment.
        line_end: np.array - (2,) end point of the line segment.

    Returns:
        angle_diffs: list - Two angle differences in degrees, guaranteed to be within 90 degrees.
    """
    # Step 1: Compute all four bounding box edge vectors
    edges = compute_edges_right_to_left(bbox_points)

    # Step 2: Normalize the line vector to ensure right-to-left direction
    if line_start[0] < line_end[0]:
        line_start, line_end = line_end, line_start  # Swap if needed
    line_vector = line_end - line_start
    line_vector /= np.linalg.norm(line_vector)  # Normalize

    # Step 3: Find two edges closest to parallel (dot product close to ±1)
    parallel_edges = sorted(edges, key=lambda v: -abs(np.dot(line_vector, v) / (np.linalg.norm(v) * np.linalg.norm(line_vector))))[:2]
    # Step 4: Compute angle differences
    angle_diffs = []
    for edge_vector in parallel_edges:
        edge_vector = -edge_vector if edge_vector[0] > 0 else edge_vector  # Ensure right-to-left direction
        edge_vector /= np.linalg.norm(edge_vector)  # Normalize

        # Calculate angle difference
        angle_diff = np.arctan2(line_vector[1], line_vector[0]) - np.arctan2(edge_vector[1], edge_vector[0])
        angle_diff_deg = np.degrees(angle_diff)
        angle_diffs.append(angle_diff_deg)

    
    return sum(angle_diffs) / len(angle_diffs)

    

# Function to align the bounding box to the line segment considering its original orientation
def snap_and_align_bbox(
    line_start, line_end, bbox_points, closed_polygon, distance_threshold = 0.4, angle_threshold = 5, 
):

    intersections = line_intersects_or_near_bbox(
        line_start,
        line_end,
        bbox_points,
        distance_threshold=distance_threshold,
        angle_threshold=angle_threshold,
    )

    if intersections:
        intersection_point = intersections[0]

        # Find the closest point on the bounding box to the intersection point
        closest_point, closest_index = closest_bbox_point(
            intersection_point, bbox_points
        )

        # Translate the bounding box so that the closest point aligns with the intersection point
        translation_vector = intersection_point - closest_point
        translated_bbox_points = [point + translation_vector for point in bbox_points]

      
        angle_difference = calculate_perpendicular_angle_diff(bbox_points, line_start, line_end)
       
        angle_difference = np.radians(angle_difference)
        aligned_bbox_points = [
            rotate_point(point, intersection_point, angle_difference)
            for point in translated_bbox_points
        ]


        aligned_bbox_points = within_check(closed_polygon, line_start, line_end, aligned_bbox_points)

        return aligned_bbox_points    
    else:
        # print("No intersection found.")
        bbox_points = within_check(closed_polygon, line_start, line_end, bbox_points)
        return bbox_points


# Function to calculate the overlapping length and points for visualization
def overlapping_length(line_start, line_end, bbox_points):

    # check intersection between two line segments

    intersections = line_intersects_or_near_bbox(line_start, line_end, bbox_points)
    if not intersections:
        return 0, None, None

    def line_segment_overlap(start1, end1, start2, end2):
        overlap_start = max(min(start1, end1), min(start2, end2))
        overlap_end = min(max(start1, end1), max(start2, end2))
        overlap_length = max(0, overlap_end - overlap_start)
        return overlap_length, overlap_start, overlap_end

    line_vector = line_end - line_start
    line_angle = np.arctan2(line_vector[1], line_vector[0])
    aligned_edge = None
    min_angle_diff = np.inf
    for i in range(len(bbox_points)):
        edge_start, edge_end = bbox_points[i], bbox_points[(i + 1) % len(bbox_points)]
        edge_vector = edge_end - edge_start
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])
        angle_diff = abs(line_angle - edge_angle)
        if angle_diff < min_angle_diff:
            min_angle_diff = angle_diff
            aligned_edge = (edge_start, edge_end)

    line_proj_start = np.dot(line_start, line_vector) / np.linalg.norm(line_vector)
    line_proj_end = np.dot(line_end, line_vector) / np.linalg.norm(line_vector)
    edge_proj_start = np.dot(aligned_edge[0], line_vector) / np.linalg.norm(line_vector)
    edge_proj_end = np.dot(aligned_edge[1], line_vector) / np.linalg.norm(line_vector)

    overlap_length, overlap_proj_start, overlap_proj_end = line_segment_overlap(
        line_proj_start, line_proj_end, edge_proj_start, edge_proj_end
    )

    overlap_start_point = line_start + (
        overlap_proj_start - line_proj_start
    ) * line_vector / np.linalg.norm(line_vector)
    overlap_end_point = line_start + (
        overlap_proj_end - line_proj_start
    ) * line_vector / np.linalg.norm(line_vector)

    return overlap_length, overlap_start_point, overlap_end_point


# Visualization function
def plot_bounding_boxes(
    line_start,
    line_end,
    bbox_points,
    aligned_bbox_points=None,
    overlap_start=None,
    overlap_end=None,
    polygon=None,
):
    plt.figure()
    plt.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        "r-",
        label="Line Segment",
    )

    # Plot original bounding box
    for i in range(len(bbox_points)):
        start, end = bbox_points[i], bbox_points[(i + 1) % len(bbox_points)]
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            "b--",
            label="Original Bounding Box" if i == 0 else "",
        )

    # Plot aligned bounding box
    if aligned_bbox_points is not None:
        for i in range(len(aligned_bbox_points)):
            start, end = (
                aligned_bbox_points[i],
                aligned_bbox_points[(i + 1) % len(aligned_bbox_points)],
            )
            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                "g-",
                label="Aligned Bounding Box" if i == 0 else "",
            )

    # Plot overlapping section
    if overlap_start is not None and overlap_end is not None:
        plt.plot(
            [overlap_start[0], overlap_end[0]],
            [overlap_start[1], overlap_end[1]],
            "m-",
            label="Overlap Section",
            linewidth=2,
        )

    # Plot the polygon
    if polygon is not None:
        x, y = polygon.exterior.xy
        plt.plot(x, y, "c-", label="Polygon", linewidth=1.5)

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Line Segment, Bounding Box, and Overlapping Section with Polygon")
    plt.axis("equal")
    plt.show()



import numpy as np
from shapely.geometry import Polygon, LineString
import numpy as np
from shapely.geometry import Polygon, LineString, Point

def within_check(room_polygon, point_a, point_b, obj_points):
    """
    Adjusts aligned_bbox_points to ensure they are within the polygon or just touch
    the line segment defined by point_a and point_b if they are outside.
    
    Args:
        bbox_polygon (shapely.geometry.Polygon): Polygon representation of the bounding box.
        point_a (tuple): Start point of the line segment as (x, y).
        point_b (tuple): End point of the line segment as (x, y).
        aligned_bbox_points (list): List of numpy arrays representing the four corners of the bounding box [(x1, y1), ..., (x4, y4)].

    Returns:
        list: Adjusted aligned_bbox_points.
    """
    # Check the overlapping area between the bbox and the polygon
    bbox_area = Polygon(obj_points).area
    overlap_area = room_polygon.intersection(Polygon(obj_points)).area
    # def plot_polygon(polygon, color, label):
    #     x, y = polygon.exterior.xy
    #     plt.fill(x, y, alpha=0.5, fc=color, ec="black", label=label)

    # # Plotting
    # plt.figure(figsize=(8, 6))
    # plot_polygon(Polygon(aligned_bbox_points), "blue", "Original Polygon")
    # plot_polygon(bbox_polygon, "green", "Buffered Area")

    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.legend()
    # plt.title("Polygon Visualization")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.grid(True)
    # plt.show()

    if overlap_area / bbox_area > 0.5:
        # If the overlap area is greater than 50%, return the original points
        return obj_points

    # Otherwise, calculate the normal to the line segment (point_a, point_b)
    line_segment = LineString([point_a, point_b])
    line_vector = np.array(point_b) - np.array(point_a)
    normal_vector = np.array([-line_vector[1], line_vector[0]])  # Perpendicular to the line segment
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector

    # Move the bounding box along the normal direction
    step_size = 0.01  # Small step size for movement
    moved_bbox_points = obj_points[:]

    # Move the bounding box until it is no longer intersecting or touching the line
    max_iterations = 50  # Prevent infinite loops
    iteration_count = 0
    
    while room_polygon.intersection(Polygon(obj_points)).area < 0 and iteration_count < max_iterations:
        moved_bbox_points = [point - step_size * normal_vector for point in moved_bbox_points]
        iteration_count += 1
    
    iteration_count = 0  # Reset for second loop
    while Polygon(moved_bbox_points).intersects(line_segment) and iteration_count < max_iterations:
        moved_bbox_points = [point - step_size * normal_vector for point in moved_bbox_points]
        iteration_count += 1
        
        # Safety check: ensure object stays within room polygon
        moved_polygon = Polygon(moved_bbox_points)
        if not room_polygon.contains(moved_polygon.centroid):
            # Object is being pushed outside, stop and return original points
            print(f"  ⚠ Warning: Object would be pushed outside room, keeping original position")
            return obj_points
        
    if iteration_count >= max_iterations:
        print(f"  ⚠ Warning: Max iterations reached for wall snapping, keeping current position")
        
    return moved_bbox_points


if __name__ == "__main__":

    # Updated bbox_points in the same format
    bbox_points = [
        np.array([3.86955184, 2.14824738]),
        np.array([2.17932401, 3.03519771]),
        np.array([1.84955865, 2.40677629]),
        np.array([3.53978648, 1.51982596]),
    ]

    # Updated wall points in the same format
    point_a = np.array([2.27654451, 3.02559139])
    point_b = np.array([3.81727056, 2.21703513])

    # create a polygon 


    indoor_point = [
    np.array([1.37505074, 1.30777039]),
    np.array([2.27654451, 3.02559139]),
    np.array([3.81727056, 2.21703513]),
    np.array([2.9157768, 0.49921413]),
    np.array([2.10114003, 0.92672663]),
]
    


    # indoor_point = [point_a, point_b, point_d, point_c, point_a]

    bbox_polygon_to_be_move = Polygon(bbox_points)

    closed_polygon = Polygon(indoor_point)

    
    # Align bounding box to the line segment
    aligned_bbox_points = snap_and_align_bbox(point_a, point_b, bbox_points, closed_polygon)

    # check if within the polygon




    
    # print("aligned_bbox_points", aligned_bbox_points)

    # print("bbox_polygon", bbox_polygon)
    # print("point_a, point_b", point_a, point_b)
    # # check if within the polygon

    
    # aligned_bbox_points = within_check(bbox_polygon, point_a, point_b, aligned_bbox_points)




    # Calculate overlap length and points
    overlap_length, overlap_start, overlap_end = overlapping_length(
        point_a, point_b, aligned_bbox_points
    )

    # Visualize
    plot_bounding_boxes(
        point_a, point_b, bbox_points, aligned_bbox_points, overlap_start, overlap_end
    )
