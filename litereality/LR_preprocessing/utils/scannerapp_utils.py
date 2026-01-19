import os


try:
    import cv2
except Exception as e:
    cv2 = None

import json
import numpy as np
from importlib import reload

import glob
import colorsys

import open3d as o3d

has_open3d = True


def filter_depth(depth_img, threshold=2.77):
    """
    Filters an open3d Image with edge filter
    Created / tuned on 640x480 TrueDepth
    Sobel kernel size = 5,  for a 3x3 you must change threshold
    See 'True depth Hough matcher.ipynb'
    """
    # convert to numpy
    depth_arr = np.array(depth_img)

    assert depth_arr.dtype == np.uint16

    depth_arr = depth_arr.astype(np.float64) / 1000.0

    # assert(False)

    # Do sobel filter
    sobelx = cv2.Sobel(depth_arr, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(depth_arr, cv2.CV_64F, 0, 1, ksize=5)

    edges = np.maximum(np.abs(sobelx), np.abs(sobely))
    print(" edges: {:.2f}  max: {:.2f} ".format(edges.min(), edges.max()))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel)

    depth_arr[edges_dilated > threshold] = 0

    # Convert back
    depth_arr = (depth_arr * 1000.0).astype(np.uint16)
    depth_img = o3d.geometry.Image(depth_arr)
    return depth_img


def project_point_k(_v, K_, R_, cam_position_, w, h, flip_y=False):
    # not quite the same results as using the projection matrix
    # not sure why -  distortion?

    aux = _v - cam_position_
    aux = np.dot(R_, aux)
    aux = np.dot(K_, aux)

    x = aux[0] / aux[2]
    y = aux[1] / aux[2]
    z = aux[2]

    if flip_y:
        y = h - y

    return np.array([x, y, z])


def project_point_mvp(p_in, mvp, image_width, image_height):
    # NOTE: only for experimenting - think there's a bug
    p0 = np.append(p_in, [1])
    e0 = np.dot(mvp, p0)
    e0[:3] /= e0[3]
    pos_x = e0[0]
    pos_y = e0[1]
    px = (0.5 + (pos_x) * 0.5) * (image_width)
    py = (1.0 - (0.5 + (pos_y) * 0.5)) * (image_height)
    pz = e0[2]
    return px, py, pz


class CameraFrame:
    depth_path = None
    image_path = None  # image with same dims as depth image
    image_path_full = None  # non-resized image
    image_size = None  # ( w, h )
    conf_path = None
    pose = None
    intrinsics = None
    intrinsics_o3d = None  #

    def to_cloud(self, **kwargs):
        return get_cloud_for_frame(self, **kwargs)

    def to_rgbd(self, **kwargs):
        return get_rgbd_for_frame(self, **kwargs)

    def load_image(self, full=False):
        if full and self.image_path_full is not None:
            image = cv2.imread(self.image_path_full)
        else:
            image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, resize_to_image=True):
        depth = (
            cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            / 1000.0
        )
        if resize_to_image:
            depth = cv2.resize(depth, self.image_size, None, 0, 0, cv2.INTER_NEAREST)

        return depth

    def load_conf(self, resize_to_image=True):
        conf = cv2.imread(self.conf_path, cv2.IMREAD_UNCHANGED)
        if resize_to_image:
            conf = cv2.resize(conf, self.image_size, None, 0, 0, cv2.INTER_NEAREST)
        return conf

    def project_point(
        self, pt_3d, use_k=True, only_valid=True, valid_pad_percentage=0.01
    ):
        """
        valid_pad_percentage - percent of image size to allow projections to fall outside of image
                               and still be considered valid
        """
        x, y, z = -1, -1, -1
        if use_k:
            # R = self.pose[:3,:3]
            R = np.linalg.inv(self.pose[:3, :3])  # .copy()
            # R[:,0] *= -1
            # R[:,1] *= -1
            # R[:,2] *= -1

            K = self.intrinsics
            cam_pos = self.pose[:3, 3].copy()
            x, y, z = project_point_k(
                pt_3d, K, R, cam_pos, self.image_size[0], self.image_size[1]
            )
        else:
            x, y, z = project_point_mvp(
                pt_3d, self.mvp, self.image_size[0], self.image_size[1]
            )

        if only_valid:
            img_w, img_h = self.image_size
            pad_x = self.image_size[0] * valid_pad_percentage
            pad_y = self.image_size[1] * valid_pad_percentage

            if (
                z <= 0.0001
                or x < -pad_x
                or y < -pad_x
                or x >= (img_w + pad_x)
                or y >= (img_h + pad_y)
            ):
                return None

        return np.array([x, y, z])

    def unproject_point(self, pt_2d, depth=1.0):

        cam_pos = self.pose[:3, 3]

        Kinv = np.linalg.inv(self.intrinsics)

        # Matrix3f Kinv = K.inverse();

        Rinv = self.pose[:3, :3]  # np.linalg.inv(self.pose[:3,:3])
        # Matrix3f Rinv = pose_yz_flip.block(0,0,3,3);//.inverse();

        # Vector3f proj3D (_p[0], _p[1], 1);
        proj3D = np.array([pt_2d[0], pt_2d[1], 1.0])

        p3D = Rinv @ (Kinv @ proj3D) * depth + cam_pos

        # p3D = Rinv @ np.dot(Kinv , proj3D) * depth + cam_pos;

        return p3D

    @staticmethod
    def scale_intrinsics(cam, sx, sy):
        """
        Return copy of camera with scaled intrinsics.
        NOTE: view_matrix / projection_matrix are not modified
        """
        cam_out = copy.deepcopy(cam)

        intrinsics_o3d = cam_out.intrinsics_o3d

        new_w = round(intrinsics_o3d.width * sx)
        new_h = round(intrinsics_o3d.height * sy)

        fx, fy = intrinsics_o3d.get_focal_length()
        fx *= sx
        fy *= sy

        cx, cy = intrinsics_o3d.get_principal_point()
        cx *= sx
        cy *= sy

        new_in = o3d.camera.PinholeCameraIntrinsic(
            width=new_w, height=new_h, fx=fx, fy=fy, cx=cx, cy=cy
        )
        cam_out.intrinsics_o3d = new_in

        new_k = np.eye(3)
        new_k[0, 0] = fx
        new_k[1, 1] = fy
        new_k[0, 2] = cx
        new_k[1, 2] = cy
        cam_out.intrinsics = new_k

        # zero out to be safe
        cam_out.view_matrix = None
        cam_out.projection_matrix = None
        cam_out.mvp = None

        return cam_out


def load_and_extract_frames(data_path):
    """
    Load full frames, possibly extract video frames
    NO SCALING of depth images so there would be mismatch between K
    """

    first_image = cv2.imread(os.path.join(data_path, "frame_00000.jpg"))
    image_height, image_width, _ = first_image.shape
    original_image_size = (image_width, image_height)

    images_out_path = os.path.join(data_path, "video_frames")
    # images_out_path_resized = os.path.join(data_path, "video_frames_resized")

    if not os.path.exists(images_out_path):
        video_path = os.path.join(data_path, "frames.mp4")
        convert_video_to_frames(video_path, images_out_path)
    else:
        print("Video frames already exists")

    # scale_intrinsics = (depth_size[0] / original_image_size[0], depth_size[1] / original_image_size[1] )
    cams = load_frames(
        data_path,
        image_base=images_out_path,
        image_base_full=images_out_path,
        image_size=original_image_size,
    )
    print(f"Loaded {len(cams)} frames ")

    return cams


def load_scan_frames_no_video(
    data_path, require_images=True, scale_intrinsics_to_depth=False
):

    first_image = cv2.imread(os.path.join(data_path, "frame_00000.jpg"))
    image_height, image_width, _ = first_image.shape
    original_image_size = (image_width, image_height)
    print(f"Original image size: {original_image_size}")

    depth_image = cv2.imread(os.path.join(data_path, "depth_00000.png"))
    depth_height, depth_width, _ = depth_image.shape
    depth_size = (depth_width, depth_height)
    print(f"Depth image size: {depth_size}")

    scale_intrinsics = (1.0, 1.0)

    if scale_intrinsics_to_depth:
        scale_intrinsics = (
            depth_size[0] / original_image_size[0],
            depth_size[1] / original_image_size[1],
        )

    cams = load_frames(
        data_path,
        image_base=data_path,
        image_base_full=data_path,
        image_size=original_image_size,
        scale_intrinsics=scale_intrinsics,
    )

    if require_images:
        cams = list(filter(lambda x: x.image_path is not None, cams))

    print(f"Loaded {len(cams)} frames ")

    return cams


def load_scan_frames_resized(data_path):
    """
    Loads a scan
    Possibly converts 'frames.mp4' into raw images, and resizes them to the
    depth image size.

    This is for the purpose of working with Open3d data structures.

    TODO: options for not resizing frames, etc.

    Returns:
        - camera_frames : [CameraFrame, ..]

    """
    ## Get the depth size to scale all images to
    depth_image = cv2.imread(os.path.join(data_path, "depth_00000.png"))
    depth_height, depth_width, _ = depth_image.shape
    depth_size = (depth_width, depth_height)
    print(f"Depth image size: {depth_size}")

    first_image = cv2.imread(os.path.join(data_path, "frame_00000.jpg"))
    image_height, image_width, _ = first_image.shape
    original_image_size = (image_width, image_height)
    print(f"Original image size: {original_image_size}")

    images_out_path = os.path.join(data_path, "video_frames")
    images_out_path_resized = os.path.join(data_path, "video_frames_resized")

    if not os.path.exists(images_out_path) or not os.path.exists(
        images_out_path_resized
    ):
        print(f"Extracting video frames to images and resizing to {depth_size}")
        video_path = os.path.join(data_path, "frames.mp4")
        convert_video_to_frames(
            video_path,
            images_out_path,
            resized_output_path=images_out_path_resized,
            output_size=depth_size,
        )
    else:
        print("Video frames already exists")

    scale_intrinsics = (
        depth_size[0] / original_image_size[0],
        depth_size[1] / original_image_size[1],
    )
    cams = load_frames(
        data_path,
        image_base=images_out_path_resized,
        image_base_full=images_out_path,
        image_size=depth_size,
        scale_intrinsics=scale_intrinsics,
    )
    print(f"Loaded {len(cams)} frames ")

    return cams


def load_frames(
    json_base_path, image_base, image_base_full, image_size, scale_intrinsics=None
):

    frames = list(
        sorted(
            filter(lambda s: "frame_" in s and ".json" in s, os.listdir(json_base_path))
        )
    )
    frames = list(map(lambda s: os.path.join(json_base_path, s), frames))

    camera_frames = []

    for f in frames:
        cam = load_frame_info(
            f,
            image_size=image_size,
            image_base=image_base,
            image_base_full=image_base_full,
            scale_intrinsics=scale_intrinsics,
        )
        camera_frames.append(cam)

    return camera_frames


def get_rgbd_for_frame(
    cam_frame, min_conf=1, max_depth=None, depth_offset=0.0, filter_depth_edges=False
):
    # TODO: combine with get_cloud_for_frame
    color = o3d.io.read_image(cam_frame.image_path)
    img_w, img_h = color.get_max_bound()
    img_w = int(round(img_w))
    img_h = int(round(img_h))

    depth_image = o3d.io.read_image(cam_frame.depth_path)
    if filter_depth_edges:
        depth_image = filter_depth(depth_image)

    depth = np.array(depth_image)
    # conf = np.array(o3d.io.read_image(cam_frame.conf_path))
    depth_h, depth_w = depth.shape

    conf = None
    has_conf = False
    if cam_frame.conf_path is not None and os.path.exists(cam_frame.conf_path):
        has_conf = True
        conf = np.array(o3d.io.read_image(cam_frame.conf_path))

        depth[conf < min_conf] = 0

    if max_depth is not None:
        max_depth_mm = round(max_depth * 1000)
        depth[depth >= max_depth_mm] = 0

    # TODO: clean up -- store info about resizing / scaling intrinsics?
    if int(img_w) != depth_w:
        focal = cam_frame.intrinsics[0, 0]
        if (focal / depth_w) < 1.0:
            print("[warning] resized color image to depth ")
            color = cv2.resize(np.array(color), (depth_w, depth_h))
            color = o3d.geometry.Image(color)
        else:
            print("[warning] resized depth image to color size ")
            depth = cv2.resize(np.array(depth), (img_w, img_h), 0, 0, cv2.INTER_NEAREST)
            # cv2.resize(depth_mask, (image.shape[1], image.shape[0]), 0,0, cv2.INTER_NEAREST)
            # conf = cv2.resize(np.array(conf), (img_w, img_h))
            # depth = o3d.geometry.Image( color )

    depth_offset_mm = int(round(depth_offset * 1000.0))

    if abs(depth_offset_mm) > 0:
        assert depth.dtype == np.uint16, "assumed uint16 depth"
        old_type = depth.dtype
        depth = depth.astype(np.int)
        depth += depth_offset_mm
        depth = depth.astype(old_type)

    depth = o3d.geometry.Image(depth)

    source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    return source_rgbd


def get_cloud_for_frame(
    cam_frame,
    estimate_normals=True,
    transform=False,
    min_conf=1,
    max_depth=10,
    depth_offset=0.0,
    filter_depth_edges=False,
    normal_radius=0.1,
    voxel_downsample_radius=0.001,
    resize_color_to_depth_if_needed=True,
):

    # MAX_DEPTH = 65535
    color = o3d.io.read_image(cam_frame.image_path)
    img_w, img_h = color.get_max_bound()
    img_w = int(round(img_w))
    img_h = int(round(img_h))

    depth_image = o3d.io.read_image(cam_frame.depth_path)
    if filter_depth_edges:
        depth_image = filter_depth(depth_image)

    depth = np.array(depth_image)

    conf = None
    has_conf = False
    if cam_frame.conf_path is not None and os.path.exists(cam_frame.conf_path):
        has_conf = True
        conf = np.array(o3d.io.read_image(cam_frame.conf_path))

    depth_h, depth_w = depth.shape

    # TODO: clean up -- store info about resizing / scaling intrinsics?
    if int(img_w) != depth_w:
        focal = cam_frame.intrinsics[0, 0]
        if (focal / depth_w) < 1.0 or resize_color_to_depth_if_needed:
            print("[warning] resized color image to depth ")
            color = cv2.resize(np.array(color), (depth_w, depth_h))
            color = o3d.geometry.Image(color)
        else:
            print("[warning] resized depth image to color size ")
            depth = cv2.resize(np.array(depth), (img_w, img_h), 0, 0, cv2.INTER_NEAREST)
            if has_conf:
                conf = cv2.resize(np.array(conf), (img_w, img_h))
            # depth = o3d.geometry.Image( color )

    if min_conf is not None and has_conf:
        depth[conf < min_conf] = 0

    if max_depth is not None:
        max_depth_mm = round(max_depth * 1000)
        depth[depth >= max_depth_mm] = 0

    # else:
    #     max_depth = 10.0 # infinity / 30 ft ?

    # print(f" >> depth 0,0 pixel: {depth[0,0]}    {depth.shape}  {depth.dtype}   ")
    depth_offset_mm = int(round(depth_offset * 1000.0))

    if abs(depth_offset_mm) > 0:
        assert depth.dtype == np.uint16, "assumed uint16 depth"
        old_type = depth.dtype
        depth = depth.astype(np.int)
        depth += depth_offset_mm
        depth = depth.astype(old_type)

    # print("depth", depth.max(), depth.min())

    # save this as the depth image

    np.save("depth_image.npy", depth)

    depth = o3d.geometry.Image(depth)

    source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False
    )
    # source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False, depth_trunc=(MAX_DEPTH/1000.0) - 0.01)
    # source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd, cam_frame.intrinsics_o3d
    )

    # print(f"  > cloud original num points: { len(pcd.points) } ")

    if transform:
        pcd.transform(cam_frame.pose)
    #     else:
    #         # flip Y / Z anyway ?
    #         pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )

    if voxel_downsample_radius is not None:
        pcd = pcd.voxel_down_sample(voxel_downsample_radius)

    return pcd


def load_frame_info(
    json_file,
    image_size=None,
    image_base=None,
    image_base_full=None,
    depth_base=None,
    scale_intrinsics=None,
):
    """
    Document this -- wtf is image_base_full vs image_base ???

    ah right - image_base_full - for having paths to the original full sized image, not resized to depth size
    """

    info = json.load(open(json_file))
    pose = np.array(info["cameraPoseARFrame"]).reshape((4, 4))
    pose_yz_flip = pose.copy()
    # flip yz
    pose_yz_flip[:3, 1] *= -1.0
    pose_yz_flip[:3, 2] *= -1.0

    K = np.array(info["intrinsics"]).reshape((3, 3))

    frame_name = os.path.basename(json_file)

    if depth_base is None:
        depth_base = os.path.dirname(json_file)

    if image_base is None:
        image_base = os.path.dirname(json_file)

    if image_base_full is None:
        image_base_full = os.path.dirname(json_file)

    image_path = os.path.join(image_base, frame_name.replace(".json", ".jpg"))
    image_path_full = os.path.join(image_base_full, frame_name.replace(".json", ".jpg"))
    depth_path = os.path.join(
        depth_base, frame_name.replace(".json", ".png").replace("frame_", "depth_")
    )
    conf_path = depth_path.replace("depth_", "conf_")

    # assert os.path.exists( image_path ), f"one image per frame required for now. Looked at {image_path}"

    if image_size is None:
        im = cv2.imread(image_path)
        image_size = (im.shape[1], im.shape[0])

    if not os.path.exists(image_path):
        image_path = None

    if not os.path.exists(image_path_full):
        image_path_full = None

    if not os.path.exists(depth_path):
        depth_path = None

    if not os.path.exists(conf_path):
        conf_path = None

    if depth_path is not None and conf_path is None:
        raise Exception("Missing confidence image")

    cam_frame = CameraFrame()
    cam_frame.pose = pose_yz_flip

    projection_matrix = np.array(info["projectionMatrix"]).reshape((4, 4))
    view_matrix = np.linalg.inv(pose)
    mvp = np.dot(projection_matrix, view_matrix)
    cam_frame.mvp = mvp
    cam_frame.view_matrix = view_matrix
    cam_frame.projection_matrix = projection_matrix
    # if scale_intrinsics is not None and scale_intrinsics[0] != 1.0 and scale_intrinsics[1] != 1.0:
    #     print("Warning projection_matrix out of sync with K", scale_intrinsics)

    sx, sy = 1.0, 1.0

    if scale_intrinsics is not None:
        sx, sy = scale_intrinsics
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if has_open3d:

        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=round(image_size[0] * sx),
            height=round(image_size[1] * sy),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        intrinsic_matrix = np.array(intrinsics_o3d.intrinsic_matrix)

        cam_frame.intrinsics_o3d = intrinsics_o3d

    cam_frame.intrinsics = K
    cam_frame.image_path = image_path
    cam_frame.image_path_full = image_path_full
    cam_frame.depth_path = depth_path
    cam_frame.conf_path = conf_path
    cam_frame.image_size = image_size

    return cam_frame


def load_true_depth_frames(input_path):
    """
    NOTE:
    - Images are 1280x720
    - Depth should be 640x480
    - intrinsics are scaled so they will match the depth image size
    - when loading the CameraFrame.load_image() , the image will be resized hopefully
    - We leave image full size to use SIFT on the full image rather than just converting
    """
    depth_files = sorted(glob.glob(os.path.join(input_path, "depth*.png")))
    depth_h, depth_w = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED).shape[:2]

    image_size = (depth_w, depth_h)

    # NOTE: i've renamed image_00001 --> frame_00001 etc
    # for older scans it might be image_0000
    image_files_full = sorted(glob.glob(os.path.join(input_path, "frame_*.jpg")))

    video_path = os.path.join(input_path, "frames.mp4")
    if len(image_files_full) != len(depth_files) and os.path.exists(video_path):
        print("Converting frames.mp4 to images")
        convert_video_to_frames(video_path, input_path, name_format="frame_{:05d}.jpg")
        image_files_full = sorted(glob.glob(os.path.join(input_path, "frame_*.jpg")))

    resized_images_path = os.path.join(input_path, "images_resized")
    image_files = sorted(glob.glob(os.path.join(resized_images_path, "frame_*.jpg")))

    if not os.path.exists(resized_images_path) or len(image_files) != len(depth_files):
        print("Resizing images to depth frame dimensions")
        # resize images to depth size and save
        os.makedirs(resized_images_path, exist_ok=1)
        for image_path in image_files_full:
            image_path_full = os.path.join(
                resized_images_path, os.path.basename(image_path)
            )
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (depth_w, depth_h))
            cv2.imwrite(image_path_full, image)

    image_files = sorted(glob.glob(os.path.join(resized_images_path, "frame_*.jpg")))

    json_files = sorted(glob.glob(os.path.join(input_path, "calib*.json")))

    assert (
        len(depth_files) == len(image_files) == len(json_files)
    ), f"got {len(depth_files)}"

    image_h, image_w = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED).shape[:2]

    frames = []

    for image_path, image_path_full, depth_path, json_path in zip(
        image_files, image_files_full, depth_files, json_files
    ):

        info = json.load(open(json_path))

        ver = info.get("version", 0)

        if ver < 2:
            calibration = info["calibration_data"]
        else:
            calibration = info

        pose = np.eye(4)

        pose[:3, :3] = np.array(info["rotation_matrix"]).reshape((3, 3))[:]

        if True:

            pose_copy = pose.copy()

            # CMAttitude to TUM camera
            # X direction is the -Y axis
            pose[:3, 0] = pose_copy[:3, 1] * -1.0

            # Y axis is the X of cmattitude
            pose[:3, 1] = pose_copy[:3, 0]
            # Z = Z

            # CMAttitude world is Z up, so change to Y up by swapping Z & Y
            pose_copy = pose.copy()

            # Y = Z
            pose[1, :3] = pose_copy[2, :3]

            # Z = -Y  -- why negative? no idea
            pose[2, :3] = -pose_copy[1, :3]

            # X - no
            # XZ - no
            # XY - no
            # XYZ - upside down world but does work
            # pose[:3, 0] *= -1.0
            # pose[:3, 1] *= -1.0
            # pose[:3, 2] *= -1.0

        iw, ih = calibration["intrinsic_matrix_reference_dimensions"]
        iw = float(iw)
        ih = float(ih)

        # Scale the intrinsics to the depth size
        K = np.array(calibration["intrinsic_matrix"]).reshape((3, 3))
        if ver < 2:
            K = K.T

        scale_x = float(depth_w) / iw
        scale_y = float(depth_h) / ih
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        cam_frame = CameraFrame()
        cam_frame.pose = pose

        view_matrix = np.linalg.inv(pose)
        cam_frame.view_matrix = view_matrix

        if has_open3d:
            intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
                width=depth_w, height=depth_h, fx=fx, fy=fy, cx=cx, cy=cy
            )

            cam_frame.intrinsics_o3d = intrinsics_o3d

        cam_frame.intrinsics = K

        cam_frame.image_path = image_path
        cam_frame.image_path_full = image_path_full
        cam_frame.depth_path = depth_path
        cam_frame.conf_path = None
        cam_frame.image_size = image_size

        frames.append(cam_frame)

    return frames


def show_frames_3d(optimized_frames, pose_step=10, cloud_step=2, max_depth=5.0):
    geoms = []

    start = 0

    stop = len(optimized_frames)
    # stop = 150

    show_clouds = 1
    voxel_down = 0.01

    # axes_size = 0.06
    # axes_size = 0.025
    axes_size = 0.08

    points = []

    j = 0
    for i in range(start, stop, pose_step):
        j += 1

        frame = optimized_frames[i]
        pose = frame.pose

        points.append(pose[:3, 3])

        axes_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size)
        axes_mesh.transform(pose)

        if i % 10 == 0:
            axes_mesh.paint_uniform_color([0, 0, 0])

        geoms.append(axes_mesh)

        hue = 0.15 + 0.85 * ((i - start) / (stop - start))
        color = colorsys.hsv_to_rgb(hue, 0.85, 0.85)

        boxsize = axes_size / 3.0
        ball = o3d.geometry.TriangleMesh.create_box(boxsize, boxsize, boxsize)
        ball.paint_uniform_color(color)
        ball.transform(pose)
        geoms.append(ball)

        if j % cloud_step == 0 and show_clouds:

            cloud0 = optimized_frames[i].to_cloud(transform=1, max_depth=max_depth)
            cloud0 = cloud0.voxel_down_sample(voxel_down)

            cloud0.paint_uniform_color(color)
            geoms.append(cloud0)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size * 4))

    lines = [[i - 1, i] for i in range(1, len(points))]
    cam_lines = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines)
    )
    cam_lines.paint_uniform_color([0, 1, 0])
    geoms.append(cam_lines)

    o3d.visualization.draw_geometries(geoms)


def convert_video_to_frames(
    video_path,
    output_path,
    resized_output_path=None,
    output_size=None,
    name_format="frame_{:05d}.jpg",
):
    """
    Converts a video to a folder of jpg frames
    Folder is created at 'video_base_path/video_frames'
    """

    base_path = os.path.dirname(video_path)
    os.makedirs(output_path, exist_ok=1)

    assert os.path.exists(video_path), "video does not exist"

    if output_size is not None:
        assert resized_output_path is not None, "Must provide path for resized frames"

        os.makedirs(resized_output_path, exist_ok=1)

    cap = cv2.VideoCapture(video_path)

    idx = 0

    while 1:

        ret, frame = cap.read()

        if frame is not None and frame.size > 0:

            img_out_path = os.path.join(output_path, name_format.format(idx))
            cv2.imwrite(img_out_path, frame)

            if output_size is not None and resized_output_path is not None:
                frame = cv2.resize(frame, output_size)
                img_out_path_resized = os.path.join(
                    resized_output_path, name_format.format(idx)
                )
                cv2.imwrite(img_out_path_resized, frame)

            idx += 1
        else:
            break

    print(f"Wrote {idx} frames")

    cap.release()
