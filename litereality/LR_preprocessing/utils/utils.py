import os
from typing import Any, Dict, List
import numpy as np
import filecmp
import os
import shutil
from zipfile import ZipFile
from itertools import combinations
from typing import Any, Dict, List, Tuple
import numpy as np
from trimesh import Trimesh
import utils.scannerapp_utils as scannerapp_utils


def get_object_pose(
    object_file: str, flip_x: bool = True, flip_y: bool = False
) -> Dict[str, Any]:
    """
    object_file: path to the usda file of the object.
    """
    with open(object_file, "r") as f:
        lines = f.readlines()
        position, bbox, rotation, point_center, faces, normals = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for line in lines:
            if "normal3f[] normals" in line:
                assert normals is None
                normals = eval(line.split("=")[1].strip())
            if "int[] faceVertexIndices" in line:
                assert faces is None
                faces = eval(line.split("=")[1].strip())
            if "point3f[] points" in line:
                assert bbox is None
                points = np.array(eval(line.split("=")[1].strip()))
                bbox = points.max(axis=0) - points.min(axis=0)
                point_center = (
                    np.abs(points.max(axis=0)) - np.abs(points.min(axis=0))
                ) / 2
            if "matrix4d xformOp:transform" in line:
                assert rotation is None and position is None
                transform = np.array(eval(line.split("=")[1].strip()))

                position = transform[3, :3]
                position += point_center

                roll = np.arctan2(transform[2, 1], transform[2, 2]) * 180 / np.pi
                pitch = (
                    np.arctan2(
                        transform[2, 0],
                        np.sqrt(transform[2, 1] ** 2 + transform[2, 2] ** 2),
                    )
                    * 180
                    / np.pi
                )
                yaw = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
                rotation = np.array([roll, pitch, yaw])
    assert (
        position is not None
        and bbox is not None
        and rotation is not None
        and faces is not None
        and normals is not None
    )
    pose = {
        "position": position.copy(),
        "rotation": rotation.copy(),
        "bbox": bbox.copy(),
        "transform": transform.copy(),
        "points": points.copy(),
        "faces": faces,
        "normals": normals,
    }

    return pose


def get_wall_and_object_floor_files(room_usda: str) -> Dict[str, List[str]]:
    base_dir = os.path.dirname(room_usda)

    with open(room_usda, "r") as f:
        lines = f.readlines()

        walls = []
        objects = []
        floor = []
        append_to_walls = False
        append_to_objects = True
        for i, line in enumerate(lines):
            if "Walls" in line:
                append_to_walls = True
                append_to_objects = False
            if "Object_grp" in line:
                append_to_objects = True
                append_to_walls = False
            if "Floors" in line:
                line = line[line.find("./") + 2 : line.rfind(".usda") + 5]
                line = os.path.join(base_dir, line)
                floor.append(line)
                continue
            if ".usda" in line:
                if not append_to_walls and not append_to_objects:
                    raise Exception("No group found")
                line = line[line.find("./") + 2 : line.rfind(".usda") + 5]
                mesh_id = line.split("/")[-1].split(".")[0]
                line = os.path.join(base_dir, line)
                if append_to_walls:
                    walls.append(line)
                if append_to_objects:
                    objects.append(line)
        walls_final = []
        for i, wall_name in enumerate(walls):
            if "Window" in wall_name or "Door" in wall_name or "Floor" in wall_name:
                continue
            walls_final.append(wall_name)
        walls = walls_final
        print("Walls:", len(walls), "Objects:", len(objects), "Floor:", len(floor))
    return {"walls": walls, "objects": objects, "Floor": floor}

def get_object_pose(
    object_file: str, flip_x: bool = True, flip_y: bool = False
) -> Dict[str, Any]:
    """
    object_file: path to the usda file of the object.
    """
    with open(object_file, "r") as f:
        lines = f.readlines()
        position, bbox, rotation, point_center, faces, normals = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for line in lines:
            if "normal3f[] normals" in line:
                assert normals is None
                normals = eval(line.split("=")[1].strip())
            if "int[] faceVertexIndices" in line:
                assert faces is None
                faces = eval(line.split("=")[1].strip())
            if "point3f[] points" in line:
                assert bbox is None
                points = np.array(eval(line.split("=")[1].strip()))
                bbox = points.max(axis=0) - points.min(axis=0)
                point_center = (
                    np.abs(points.max(axis=0)) - np.abs(points.min(axis=0))
                ) / 2
            if "matrix4d xformOp:transform" in line:
                assert rotation is None and position is None
                transform = np.array(eval(line.split("=")[1].strip()))

                position = transform[3, :3]
                position += point_center

                roll = np.arctan2(transform[2, 1], transform[2, 2]) * 180 / np.pi
                pitch = (
                    np.arctan2(
                        transform[2, 0],
                        np.sqrt(transform[2, 1] ** 2 + transform[2, 2] ** 2),
                    )
                    * 180
                    / np.pi
                )
                yaw = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
                rotation = np.array([roll, pitch, yaw])
    assert (
        position is not None
        and bbox is not None
        and rotation is not None
        and faces is not None
        and normals is not None
    )
    pose = {
        "position": position.copy(),
        "rotation": rotation.copy(),
        "bbox": bbox.copy(),
        "transform": transform.copy(),
        "points": points.copy(),
        "faces": faces,
        "normals": normals,
    }

    return pose


def get_line_segments_and_walls(wall_files):
    walls = []
    line_segments = []
    for file in wall_files:
        pose = get_object_pose(file)
        walls.append({"file": file, "pose": pose})
        points = np.array(
            [[-pose["bbox"][0] / 2, 0, 0, 1], [pose["bbox"][0] / 2, 0, 0, 1]]
        )
        t_points = pose["transform"].T @ points.T
        x1 = t_points[0, 0]
        z1 = t_points[2, 0]
        x2 = t_points[0, 1]
        z2 = t_points[2, 1]
        line_segments.append([(x1, z1), (x2, z2)])
    return line_segments, walls


def extract_scene_usdz(scene_usdz: str, output_zip_path: str) -> str:
    # NOTE: The usdz is actually just a zip file. So we're copying it to be one.
    # create the directory
    base_path = os.path.dirname(output_zip_path)
    os.makedirs(base_path)

    # make a copy of file_id.usdz to the base_path
    shutil.copyfile(scene_usdz, output_zip_path)

    with ZipFile(output_zip_path, "r") as zip_ref:
        zip_ref.extractall(base_path)

    scene_usda = os.path.join(base_path, "room.usda")
    assert os.path.exists(scene_usda), f"Could not find Room.usda in {scene_usdz}"


def get_scene_usda(scene_usdz: str) -> str:
    # remove the extension and directory from file_id, if it has one
    scene_id = os.path.splitext(os.path.basename(scene_usdz))[0]
    base_path = os.path.join(os.path.expanduser("./"), ".processed_usdz", scene_id)
    usdz_zip_path = os.path.join(base_path, f"{scene_id}.zip")
    if not os.path.exists(base_path):
        # extract the usdz file
        extract_scene_usdz(scene_usdz, usdz_zip_path)
    elif not filecmp.cmp(scene_usdz, usdz_zip_path):
        raise Exception(f"Cannot overwrite existing scene {scene_id}.")

    room_usda = os.path.join(base_path, "room.usda")
    return room_usda


def get_object_yaw(rotation: np.ndarray) -> float:
    """Get the yaw of an object from its xyz rotation.

    Assumes x and z are either both 0 or both 180, which is true from the RoomPlan
    API.
    """
    yaw_deg = rotation[1]
    if rotation[0] == 180 and rotation[2] == 180:
        yaw_deg = yaw_deg % 360
        dist_to_270 = abs(yaw_deg - 270)
        dist_to_90 = abs(yaw_deg - 90)
        if dist_to_270 < dist_to_90:
            yaw_deg = yaw_deg + 2 * (270 - yaw_deg)
        elif dist_to_270 > dist_to_90:
            yaw_deg = yaw_deg + 2 * (90 - yaw_deg)
    return yaw_deg


def get_wall_holes(
    wall_pose: Dict[str, Any]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Returns a list of holes in the wall.

    Returns in the format [((x0, y0), (x1, y1)), ...]
    """
    max_point = wall_pose["points"].max(axis=0)
    min_point = wall_pose["points"].min(axis=0)

    points = set([(round(p[0], 3), round(p[1], 3)) for p in wall_pose["points"]])


    # get all the points 
    # print(wall_pose["points"])
    # add the corners of the wall incase they are not there
    points.add((round(min_point[0], 3), round(min_point[1], 3)))
    points.add((round(max_point[0], 3), round(max_point[1], 3)))
    points.add((round(min_point[0], 3), round(max_point[1], 3)))
    points.add((round(max_point[0], 3), round(min_point[1], 3)))

    unique_x_points = sorted(list(set([p[0] for p in points])))
    xs_to_ys = {x: set([p[1] for p in points if p[0] == x]) for x in unique_x_points}

    rectangles = []
    for x0, x1 in combinations(unique_x_points, 2):
        x0, x1 = sorted([x0, x1])
        ints = xs_to_ys[x0].intersection(xs_to_ys[x1])
        if len(ints) >= 2:
            for y0, y1 in combinations(ints, 2):
                y0, y1 = sorted([y0, y1])
                rectangles.append(((x0, y0), (x1, y1)))

    mesh = Trimesh(
        vertices=wall_pose["points"],
        faces=np.array(wall_pose["faces"]).reshape(-1, 3),
        process=False,
    )
    mesh.fill_holes()

    holes = []
    eps = 5e-3
    for (x0, y0), (x1, y1) in rectangles:
        nearby_points = np.array(
            [
                [x0 + eps, y0 + eps, -0.08],
                [x1 - eps, y0 + eps, -0.08],
                [x1 - eps, y1 - eps, -0.08],
                [x0 + eps, y1 - eps, -0.08],
            ]
        )
        inner_points = mesh.contains(nearby_points)
        if sum(inner_points) == 0:
            holes.append(((x0, y0), (x1, y1)))
    return holes


def get_objects(object_files):
    objects = []
    object_type_count = {}
    for i, file in enumerate(object_files):
        pose = get_object_pose(file)
        object_type = file.split("/")[-2]
        if object_type not in object_type_count:
            object_type_count[object_type] = 0
        object_type_name = object_type + str(object_type_count[object_type])
        object_type_count[object_type] += 1
        
        rect = [
            (pose["bbox"][0] / 2, pose["bbox"][2] / 2),
            (pose["bbox"][0] / 2, -pose["bbox"][2] / 2),
            (-pose["bbox"][0] / 2, -pose["bbox"][2] / 2),
            (-pose["bbox"][0] / 2, pose["bbox"][2] / 2),
        ]
        unity_rotation = get_object_yaw(pose["rotation"])

        room_plan_rotation = pose["rotation"][1]
        if pose["rotation"][0] != 180:
            room_plan_rotation = -room_plan_rotation
        room_plan_rad = room_plan_rotation * np.pi / 180
        for i, p in enumerate(rect):
            rect[i] = (
                p[0] * np.cos(room_plan_rad) - p[1] * np.sin(room_plan_rad),
                p[0] * np.sin(room_plan_rad) + p[1] * np.cos(room_plan_rad),
            )

        # move the objects into position
        for i, p in enumerate(rect):
            rect[i] = (p[0] + pose["position"][0], p[1] + pose["position"][2])

        
        objects.append(
            {
                "object_type": object_type_name,
                "bbox": pose["bbox"],
                "rotation": unity_rotation,
                "top_down_rect": rect,
                "position": pose["position"].copy(),
                "file": file,
                "mesh_id": file.split("/")[-1].split(".")[0],
                "room_plan_rotation": room_plan_rotation,
            }
        )

    return objects


def extract_rgbd(scan_path, folder):
    frames = scannerapp_utils.load_scan_frames_no_video(scan_path, scale_intrinsics_to_depth=True)

    subfolders = ["image", "depth", "intrinsic", "extrinsic"]

    # Create main folder and subfolders
    os.makedirs(folder, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # Process frames
    for i, frame in enumerate(frames):
        image_path = frame.image_path_full
        depth_path = frame.depth_path
        pose = frame.pose
        intrinsic = frame.intrinsics

        # Construct new file paths
        image_new_path = os.path.join(folder, "image", f"frame_{i}.jpg")
        depth_new_path = os.path.join(folder, "depth", f"frame_{i}.jpg")
        intrinsic_name = os.path.join(folder, "intrinsic", f"intrinsic_{i}.npy")
        extrinsic_name = os.path.join(folder, "extrinsic", f"extrinsic_{i}.npy")

        # Save intrinsic and extrinsic data
        np.save(intrinsic_name, intrinsic)
        np.save(extrinsic_name, pose)

        # Copy image and depth files
        os.system(f"cp {image_path} {image_new_path}")
        os.system(f"cp {depth_path} {depth_new_path}")
