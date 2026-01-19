from layout_utils import *
from holes_utils import wall_line_closed_rect, map_to_parsed_wall_lines, get_wall_lines


class Wall_Parsing:

    def __init__(self, walls, wall_holes, floor_level = False, tolerence = 0.0, output_only = False, wall_lines_process = None):
        self.walls = walls
        self.floor_level = floor_level
        wall_lines = get_wall_lines(walls)
        self.tolerence = tolerence
        self.raw_lines = wall_lines.copy()
        self.wall_lines = wall_lines

        if not output_only:
            if floor_level:
                self.parsing_flat()            
            else:
                self.parsing_room()
            
        else:
            self.wall_lines = wall_lines_process
            segments = [list(line) for line in self.wall_lines]
            # start with first segment
            loop = [segments[0][0], segments[0][1]]
            segments.pop(0)
            # keep chaining until none left
            while segments:
                last = loop[-1]
                for i, (a, b) in enumerate(segments):
                    if a == last:
                        loop.append(b)
                        segments.pop(i)
                        break
                    elif b == last:
                        loop.append(a)
                        segments.pop(i)
                        break
                else:
                    # if we get here, pieces don't connect
                    raise ValueError(f"Broken wall at {last}")
            # close the loop if needed
            if loop[0] != loop[-1]:
                loop.append(loop[0])

            self.room_points = [loop]

        self.wall_holes = wall_holes

    def parsing_room(self):
        polygon_point = wall_line_to_close_with_visualization_simple_underflow(self.wall_lines, False)
        
        # Apply Manhattan World regularization to enforce 90-degree angles
        print("  → Applying Manhattan World regularization...")
        polygon_point = manhattan_world_regularization(
            polygon_point, 
            angle_threshold=15.0,  # Snap walls within 15° of grid axes
            n_iterations=2,         # 2 iterations to minimize wall movement
            damping=0.3            # Very conservative updates to keep objects inside
        )
        
        # self.visulize()
        self.polygon_point = [polygon_point]
        polygon_line = []
        for i in range(len(polygon_point)):
            polygon_line.append([polygon_point[i], polygon_point[(i+1)%len(polygon_point)]])
        self.wall_lines = polygon_line
        # self.visulize()
        self.room_points = [polygon_point]
        self.room_polygons = [Polygon(area) for area in self.room_points]
        self.wall_line_by_area = [self.wall_lines]


    def parsing_flat(self):
        self.axis_aligned_walls()
        # self.visulize()
        self.snap_walls_to_grid()
        # self.visulize()
        self.merge_line_on_same_axis()
        # self.visulize()
        self.snap_end_point_to_grid()
        # self.visulize()
        self.split_line_by_intersection()
        # self.visulize()
        self.rotate_back()
        self.room_points = self.closed_area()
        self.room_polygons = [Polygon(area) for area in self.room_points]


        self.wall_line_by_area = []
        for polygon_points in self.room_points:
            wall_line_numpy = [
                [np.array(polygon_points[j]), np.array(polygon_points[(j + 1) % len(polygon_points)])]
                for j in range(len(polygon_points))
            ]
            self.wall_line_by_area.append(wall_line_numpy)
        
    def axis_aligned_walls(self):
        # Rotate the wall lines to be axis aligned
        self.average_angle, _ = calculate_average_angle(self.wall_lines)
        self.wall_lines = center_rotate_and_translate_wall_lines(self.wall_lines, -self.average_angle)
        self.wall_lines = correct_and_filter_lines(self.wall_lines, 10)
        return self.wall_lines

    def snap_walls_to_grid(self):
        clusters = group_lines_by_grid(self.wall_lines, tolerance=self.tolerence)
        self.wall_lines = adjust_lines_with_intersections(clusters)
    
    def merge_line_on_same_axis(self):
        self.wall_lines = merge_lines_by_axis(self.wall_lines)
    
    def split_line_by_intersection(self):
        self.wall_lines = split_lines(self.wall_lines)
    
    def visulize(self):
        visualize_line(self.wall_lines, "wall_lines", txt = [f"wall_{i}" for i in range(len(self.wall_lines))])
    
    def snap_end_point_to_grid(self):
        self.wall_lines = refine_lines(self.wall_lines)

    def rotate_back(self):
        self.wall_lines = center_rotate_and_translate_wall_lines(self.wall_lines, self.average_angle)

    def closed_area(self):
        areas, self.line_to_area = find_closed_areas_and_label(self.wall_lines)
        self.wall_lines, self.line_to_area = zip(*[
            (line, area) for line, area in zip(self.wall_lines, self.line_to_area) if area
        ]) # this only keeps line that contributes to a closed area, works for most cases
        return areas

    def output_wall_hole_floor(self, thickness, scene_name):

        if self.floor_level:
            self.output_walls_and_holes_room(thickness, scene_name)
        else:
            self.output_walls_and_holes_single(thickness, scene_name)
        
        room_points_3d = {}
        for idx, area in enumerate(self.room_points):
            this_room_3d = []
            for point in area:
                this_room_3d.append([point[0], self.floor_height, point[1]])
            room_points_3d[idx] = this_room_3d
    
        with open(f"input/scene_data/{scene_name}/floor_new_final.pkl", "wb") as f:
            pickle.dump(room_points_3d, f)
        

    def output_walls_and_holes_room(self, thickness, scene_name):
        overall_wall_lines = [line for line in self.wall_lines if line[0] != line[1]]
        rectangles_list = wall_line_closed_rect(overall_wall_lines, thickness)
        Hole_file = {}
        Wall_flie = {}
        new_idx = 0
        used_wall_idx = []
        if isinstance(self.wall_holes, dict):
            self.wall_holes = self.wall_holes.items()
        for index, (key, hole_info) in enumerate(self.wall_holes):
            wall = self.walls[index]
            pose = wall["pose"]
            position = pose["position"]
            bbox = pose["bbox"]
            position_y = position[1]
            bbox_y = bbox[1]

            if hole_info != []:
                wall_line = self.raw_lines[index]
                _, best_match_idx, best_direction_same = map_to_parsed_wall_lines(wall_line, overall_wall_lines)
                if best_direction_same:
                    overall_wall_lines[best_match_idx] = [overall_wall_lines[best_match_idx][1], overall_wall_lines[best_match_idx][0]]

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
                Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall, "area_belongs": self.line_to_area[best_match_idx]}}
                Hole_file[new_idx] = {"file": new_idx, "holes": hole_info}
                new_idx += 1

        for i in range(len(rectangles_list)):
            if i in used_wall_idx:
                continue
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
            Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall, "area_belongs": self.line_to_area[i]}}
            Hole_file[new_idx] = {"file": new_idx, "holes": []}
            new_idx += 1

        self.floor_height = position_y - bbox_y / 2  # this assume all the floor heights are the same, but they are not, should be updated

        # dump the wall and hole file
        with open(f"scene_data/{scene_name}/walls_organized.pkl", "wb") as f:
            pickle.dump(Wall_flie, f)
        with open(f"scene_data/{scene_name}/walls_hole_organized.pkl", "wb") as f:
            pickle.dump(Hole_file, f)

    def output_walls_and_holes_single(self, thickness, scene_name):

        overall_wall_lines = list(self.wall_lines)
        rectangles_list = wall_line_closed_rect(overall_wall_lines, thickness)

        Hole_file = {}
        Wall_flie = {}
        new_idx = 0
        used_wall_idx = []
        wall_hole_reorder = []
        if isinstance(self.wall_holes, dict):
            self.wall_holes = self.wall_holes.items()
        for index, (key, hole_info) in enumerate(self.wall_holes):
            wall = self.walls[index]
            pose = wall["pose"]
            position = pose["position"]
            bbox = pose["bbox"]
            position_y = position[1]
            bbox_y = bbox[1]
            if hole_info != []:
                wall_hole_reorder.append(key.split("/")[-1].split(".")[0])
                wall_line = self.raw_lines[index]
                _, best_match_idx, best_direction_same = map_to_parsed_wall_lines(wall_line, overall_wall_lines)

                if best_direction_same:
                    overall_wall_lines[best_match_idx] = [overall_wall_lines[best_match_idx][1], overall_wall_lines[best_match_idx][0]]

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
                Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall, "area_belongs": [0]}}
                Hole_file[new_idx] = {"file": new_idx, "holes": hole_info}
                new_idx += 1

        for i in range(len(rectangles_list)): # rectangles_list is all the wall rectangles box after parsing this is hard to get, we need to output all of them
            if i in used_wall_idx:  # alreay output
                continue

            # i need to know the original height of wall. this is important
            # here we should find the original wall height, this information only contains 
            # technicall we should compare these wall rectangles with the original wall rectangles, in that case, we can find the height of them

            _, best_match_idx, best_direction_same = map_to_parsed_wall_lines(overall_wall_lines[i], self.raw_lines)
            # height 
            wall = self.walls[best_match_idx]
            # height of wall 
            pose = wall["pose"]
            position = pose["position"]
            bbox = pose["bbox"]
            position_y = position[1]
            bbox_y = bbox[1]

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
            Wall_flie[new_idx] = {"file": new_idx, "pose": {"position": position_new, "rotation": rotation_new, "bbox": bbox_new, "2d_line": line_for_wall, "area_belongs": [0]}}
            Hole_file[new_idx] = {"file": new_idx, "holes": []}
            new_idx += 1

        self.floor_height = position_y - bbox_y / 2
        # dump the wall and hole file
        with open(f"input/scene_data/{scene_name}/walls_organized.pkl", "wb") as f:
            pickle.dump(Wall_flie, f)
        with open(f"input/scene_data/{scene_name}/walls_hole_organized.pkl", "wb") as f:
            pickle.dump(Hole_file, f)
        
        # save wall_hole_reorder as a text file
        with open(f"input/scene_data/{scene_name}/wall_hole_reorder.txt", "w") as f:
            for item in wall_hole_reorder:
                f.write("%s\n" % item)

if __name__ == "__main__":

    # scene_name_list = ["Trinity_BA", "bedroom", "0312_test",  "d35", "demo_today", "boardroom", "ba_room","girton", "meetingroom", "sigproc"]

    scene_name_list = ["Girton"]
    floor_level_scan = False
    # Load preprocessed scene data
    for scene_name in scene_name_list:
        print("start to process", scene_name)
        object_lists, walls, wall_holes, _ = load_processed_data(scene_name)
        wall_parser = Wall_Parsing(walls, wall_holes, floor_level_scan, tolerence = 0.4)
        wall_parser.output_wall_hole_floor(0.16, scene_name)

        # check if objects are on the floor or off the floor
        wall_min_y_s = [wall["pose"]["position"][1] - wall["pose"]["bbox"][1] / 2 for wall in walls]
        average_wall_min_y = np.mean(wall_min_y_s)
        # obj_data = get_obj_attributes(object_lists, average_wall_min_y)
        print("finish processing", scene_name)