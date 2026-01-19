import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely.affinity import translate
from shapely.ops import unary_union

from scene_graph import (
    plot_objs_walls,
    line_obj_process,
    line_obj_process_with_vis
)

from layout_parsing import Wall_Parsing
from layout_utils import *


class CollisionSolver2d:
    def __init__(self, object_list_dict, wall_lines, group_index = 0, on_floor=True):
        
        wall_lines = self.create_closed_area_and_offset_lines(wall_lines, 0.08) # Offset the wall lines inwards by half width of the walls

        object_list = []
        obj_type = []
        self.group_index = group_index
        self.orginal_object_list = []
        
        # Filter objects based on on_floor flag
        for obj in object_list_dict:
            if on_floor:
                if obj["category"] == "on_floor":
                    self.orginal_object_list.append(obj)
            else:
                if obj["category"] == "off_floor":
                    self.orginal_object_list.append(obj)

        for obj in self.orginal_object_list:
                object_list.append(obj["corners_2d"])
                obj_type.append(obj["raw_type"])

        self.object_lists_update, scene_graph = line_obj_process(object_list, obj_type, wall_lines,distance_threshold = 0.5, angle_threshold = 10)
        self.walls, self.object_list = scene_graph["walls"], scene_graph["objects"]

        # Extract wall lines
        self.all_wall_line = []
        for wall in self.walls:
            self.all_wall_line.append(wall["line_segment"])

        # Extract object polygons and constraints
        self.polygons = []
        self.constraints = []  # Directions for constrained movement
        self.object_types = []  # Store object types for each polygon
        self.outside_objects = []  # Mark objects outside the closed area
        for i, obj in enumerate(self.object_list):
            obj_four_point = obj["four_point"]  # Four corners of the object
            self.polygons.append(Polygon(obj_four_point))
            self.object_types.append(obj_type[i])
            if obj["attached_wall"] != -1:
                # Object is attached to a wall
                wall_attached = self.walls[obj["attached_wall"]]
                wall_line = wall_attached["line_segment"]  # Line direction [[x1, y1], [x2, y2]]
                direction = np.array([wall_line[1][0] - wall_line[0][0], wall_line[1][1] - wall_line[0][1]])
                direction = direction / np.linalg.norm(direction)  # Normalize direction
                self.constraints.append(direction)
            else:
                # No constraint for this object
                self.constraints.append(None)

        # Create a closed polygon from wall lines
        self.closed_area = Polygon([line[0] for line in self.all_wall_line] + [self.all_wall_line[0][0]])

        buffered_area = self.closed_area.buffer(0.02)  # Expand the closed area by 0.01
        for poly in self.polygons:
            if not buffered_area.contains(poly):
                print(f"Object outside the closed area: {poly.centroid}")
                self.outside_objects.append(poly)

    def resolve_collisions(self, max_iterations=10):

        """
        Resolve collisions among objects while respecting movement constraints.
        """
        for _ in range(max_iterations):
            has_overlap = False
            for i, poly1 in enumerate(self.polygons):
                for j, poly2 in enumerate(self.polygons):
                    if i != j and poly1.intersects(poly2):
                        # Check if one is a Chair and the other is a Table

                        if ("Chair" in self.object_types[i] and "Table" in self.object_types[j]) or \
                           ("Table" in self.object_types[i] and "Chair" in self.object_types[j]) or \
                           ("Chair" in self.object_types[i] and "Chair" in self.object_types[j]):
                            continue  # Skip resolving collision for Chair and Table pairs

                        has_overlap = True
                        # Resolve collision by moving poly2
                        overlap = poly1.intersection(poly2).area
                        dx = poly2.centroid.x - poly1.centroid.x
                        dy = poly2.centroid.y - poly1.centroid.y
                        distance = np.sqrt(dx**2 + dy**2)

                        # Handle overlapping centroids
                        if distance == 0:
                            dx, dy = 1, 1
                            distance = np.sqrt(2)

                        # Calculate offset to move poly2 apart
                        offset = max(overlap, 0.1) / distance

                        # Apply movement constraint if any
                        constraint = self.constraints[j]
                        if constraint is not None:
                            # Project movement onto the constraint direction
                            movement = np.array([dx * offset, dy * offset])
                            constrained_movement = np.dot(movement, constraint) * constraint
                            dx, dy = constrained_movement
                        else:
                            # Free movement
                            dx *= offset
                            dy *= offset

                        # Translate the polygon
                        poly2 = translate(poly2, xoff=dx, yoff=dy)
                        self.polygons[j] = poly2

            # Stop if no overlaps are found
            if not has_overlap:
                break

    def visualize(self, title, color, ax):
        """
        Visualize polygons and wall lines on a Matplotlib axis.
        """

        # TODO: make sure the one on the left us the raw input
        
        # Plot the closed area
        x, y = self.closed_area.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2, label='Closed Area')

        # Plot the polygons
        for poly in self.polygons:
            if poly in self.outside_objects:
                # Highlight outside objects with purple
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='purple', edgecolor='black', label='Outside Object')
            else:
                # Plot normal objects
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc=color, edgecolor='black')

        # Plot the wall lines
        for line in self.all_wall_line:
            x_coords = [line[0][0], line[1][0]]
            y_coords = [line[0][1], line[1][1]]
            ax.plot(x_coords, y_coords, color='blue', linewidth=2)

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()

    def resolve_outside_objects(self):
        """
        Adjust the positions of objects outside the closed area by moving them along constrained directions.
        The movement direction is determined once and applied repeatedly until the overlap stops increasing.
        """
        buffered_area = self.closed_area.buffer(0.03)  # Slightly expand the closed area for tolerance

        for i, poly in enumerate(self.polygons):
            if poly in self.outside_objects:
                constraint = self.constraints[i]

                # Test both directions to find the one that increases overlap
                dx1, dy1 = constraint[0] * 0.01, constraint[1] * 0.01
                dx2, dy2 = -constraint[0] * 0.01, -constraint[1] * 0.01

                # Calculate initial overlaps for both test directions
                poly1 = translate(poly, xoff=dx1, yoff=dy1)
                overlap1 = buffered_area.intersection(poly1).area

                poly2 = translate(poly, xoff=dx2, yoff=dy2)
                overlap2 = buffered_area.intersection(poly2).area

                # Determine the final movement direction
                if overlap1 > overlap2:
                    print(f"Moving object {i} in direction 1")
                    dx, dy = dx1, dy1
                elif overlap2 > overlap1:
                    print(f"Moving object {i} in direction 2")
                    dx, dy = dx2, dy2
                else:
                    print(f"Skipping object {i} as both directions have same overlap")
                    # If neither direction improves, skip this polygon
                    continue

                last_overlap = max(overlap1, overlap2)
                if overlap1 > overlap2:
                    poly = poly1
                else:
                    poly = poly2

                # Move the polygon in the chosen direction until overlap stops increasing
                while True:
                    poly = translate(poly, xoff=dx, yoff=dy)
                    current_overlap = buffered_area.intersection(poly).area
                    if current_overlap - last_overlap <= 0.01:
                        break  # Stop if overlap is no longer increasing

                    # Update for the next iteration
                    last_overlap = current_overlap

                    # Update the polygon in the list
                    self.polygons[i] = poly



                    

    def run(self):

        # Resolve objects outside the closed area
        # self.resolve_outside_objects()

        # Resolve collisions
        self.resolve_collisions()

        # Update object list after collision resolution
        self.update_object_list()

    def run_and_visualize(self):
        """
        Run collision resolution and visualize before and after.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # # Plot before any resolution
        # self.visualize("Before Collision Resolution", 'red', axes[0])

        # Resolve objects outside the closed area
        self.resolve_outside_objects()

        # # Plot before any resolution
        # self.visualize("resolve_outside_objects", 'red', axes[0])

        # Resolve collisions
        # self.resolve_collisions()


        # # Plot after resolution
        self.visualize("After Collision Resolution", 'green', axes[1])
        plt.tight_layout()
        plt.show()
        # Update object list after collision resolution
        self.update_object_list()


        
    def update_object_list(self):
        rectangles_list_on_group = []
        for poly in self.polygons:
            rectangles_list_on_group.append(list(poly.exterior.coords))
        rectangles_list_on_group = self.object_lists_update
        self.updated_obj_bbox_info = {}

        for i in range(len(self.orginal_object_list)):

            self.updated_obj_bbox_info[i] = {}
            self.updated_obj_bbox_info[i]["bbox"] = self.orginal_object_list[i]["bbox_raw"]
            center = np.mean(rectangles_list_on_group[i][:4], axis=0)
            center_y = self.orginal_object_list[i]["position"][1]
            self.updated_obj_bbox_info[i]["position"] = [center[0], center_y, center[1]]
            points = rectangles_list_on_group[i][:4]
            orientation = - math.atan2(
                points[1][1]
                - points[0][1],
                points[1][0]
                - points[0][0],
            )
            orientation = math.degrees(orientation)
            self.updated_obj_bbox_info[i]["rotation"] = orientation
            self.updated_obj_bbox_info[i]["object_type"] = self.orginal_object_list[i]["raw_type"]
            self.updated_obj_bbox_info[i]["group_index"] = self.group_index
            self.updated_obj_bbox_info[i]["attached_wall"] = self.object_list[i]["attached_wall"]


                
    def create_closed_area_and_offset_lines(self, wall_line_list, offset_distance):
        """
        Create a closed area from wall lines and offset inside for each wall line.

        Args:
            wall_line_list (list): List of wall lines (each a tuple of two points).
            offset_distance (float): Distance to offset the walls inward.

        Returns:
            list: Offset wall lines as tuples of two points.
        """
        # Create LineString objects
        lines = [LineString(line) for line in wall_line_list]

        # Merge the lines to create a closed polygon
        merged_lines = unary_union(lines)

        if isinstance(merged_lines, LineString):
            closed_polygon = Polygon(merged_lines.coords)
        elif merged_lines.geom_type == "MultiLineString":
            # Flatten the list of coordinates to avoid nested tuples
            coords = []
            for line in merged_lines.geoms:
                coords.extend(line.coords)
            closed_polygon = Polygon(coords)
        else:
            raise ValueError("Invalid wall lines: Unable to form a closed area.")

        # Offset the polygon inward
        offset_polygon = closed_polygon.buffer(-offset_distance)

        if offset_polygon.is_empty or not offset_polygon.is_valid:
            raise ValueError("Offset resulted in an invalid or empty polygon.")

        # Extract lines from the offset polygon boundary
        offset_lines = []
        if offset_polygon.boundary.geom_type == "LineString":
            coords = list(offset_polygon.boundary.coords)
            offset_lines = [[coords[i], coords[i + 1]] for i in range(len(coords) - 1)]
        elif offset_polygon.boundary.geom_type == "MultiLineString":
            for line in offset_polygon.boundary.geoms:
                coords = list(line.coords)
                offset_lines.extend([[coords[i], coords[i + 1]] for i in range(len(coords) - 1)])
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
            "min_y": obj_min_y
        })

    return obj_data


def assign_objects_to_areas(obj_data, area_polygons, grouping):
    """
    Assign objects into groups based on intersection with area polygons.

    Parameters:
        obj_data (dict): Object attributes classified by on-floor and off-floor.
        area_polygons (list): List of area polygons.
        grouping (dict): Grouping structure to hold walls and objects for each area.

    Returns:
        dict: Updated grouping structure with objects assigned.
    """
    for category, objects in obj_data.items():  # category is "on_floor" or "off_floor"
        for obj in objects:
            obj_polygon = Polygon(obj["corners_2d"])
            for i, area_polygon in enumerate(area_polygons):
                # Check for significant intersection
                if (area_polygon.intersection(obj_polygon).area / obj_polygon.area) > 0.5:
                    grouping[i]["objects"].append({
                        "type": obj["type"],
                        "raw_type": obj["raw_type"],
                        "corners_2d": obj["corners_2d"],
                        "bbox_raw": obj["bbox_raw"],
                        "rotation": obj["rotation"],
                        "position": obj["position"],
                        "min_y": obj["min_y"],
                        "category": category,  # on_floor or off_floor
                    })
                    break
    return grouping

if __name__ == "__main__":
        
    scene_name = "girton"


    object_lists, walls, wall_holes, _ = load_processed_data(scene_name)
    floor_level_scan = False
    wall_parser = Wall_Parsing(walls, wall_holes, floor_level_scan, tolerence = 0.4)
    wall_parser.output_wall_hole_floor(0.16, scene_name)
    area_polygons = wall_parser.room_polygons
    wall_line_by_area = wall_parser.wall_line_by_area

    # get started:
 
    wall_min_y_s = [wall["pose"]["position"][1] - wall["pose"]["bbox"][1] / 2 for wall in walls]
    average_wall_min_y = np.mean(wall_min_y_s)

    # Get object attributes and classify
    obj_data = get_obj_attributes(object_lists, average_wall_min_y)

    grouping = {
        i: {
            "walls": wall_line_by_area[i],
            "objects": [],
        }
        for i in range(len(wall_line_by_area))
    }

        # Assign objects to areas
    grouping = assign_objects_to_areas(obj_data, area_polygons, grouping)


    all_obj_updated = []
    # # # Example Usage: Access grouped walls and objects
    for group_id, group in grouping.items():
        collision_solver = CollisionSolver2d(group["objects"], group["walls"], group_id, on_floor=True)
        collision_solver.run_and_visualize()
        for update_obj_dict in collision_solver.updated_obj_bbox_info.values():
            all_obj_updated.append(update_obj_dict)

    for group_id, group in grouping.items():
        collision_solver = CollisionSolver2d(group["objects"], group["walls"], group_id, on_floor=False)
        collision_solver.run_and_visualize()
        for update_obj_dict in collision_solver.updated_obj_bbox_info.values():
            all_obj_updated.append(update_obj_dict)


    with open(f"scene_data/{scene_name}/objects_organized.pkl", "wb") as f:
        pickle.dump(all_obj_updated, f)

    with open(f"scene_data/{scene_name}/walls_organized.pkl", "rb") as f:
        walls = pickle.load(f)

    with open(f"scene_data/{scene_name}/objects_organized.pkl", "rb") as f:
        objects = pickle.load(f)

    plot_objs_walls(objects, walls, "./")

