"""
LIDAR Reprojection Visualisation Tool

Projects LIDAR point clouds onto camera images using calibration data for
sensor fusion validation. This tool helps verify that LIDAR and camera
calibrations are correctly aligned by visualising LIDAR points overlaid
on camera video frames.

Features:
- Loads PCD (Point Cloud Data) files and projects them onto video frames
- Supports background subtraction using first PCD as reference
- Color-coded visualisation by intensity or Z-depth
- Interactive playback controls for frame-by-frame inspection

Author: GMIND SDK Development Team
"""

import logging
import os
import struct
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Calibration", "camera_intrinsics")
    )
)
from camera import CameraModel

# === USER CONFIGURATIONS ===
COLOR_BY_Z = False  # Set to True to color points by Z value instead of intensity

# Enable background subtraction using first PCD as background
ENABLE_BG_SUBTRACTION = True
BG_DIST_THRESHOLD = 0.4  # meters, distance threshold for background subtraction

# Set static color map Z range (meters)
COLORMAP_Z_MIN = 5.0
COLORMAP_Z_MAX = 40.0


# --- 1. Parse sensor_calibration.txt and create CameraModel objects ---
def parse_sensor_calibration(calib_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Parse sensor_calibration.txt file and extract camera and LIDAR extrinsics.

    Reads the GMIND calibration file format and extracts camera intrinsics,
    camera extrinsics, and LIDAR extrinsics.

    Args:
        calib_path: Path to sensor_calibration.txt file

    Returns:
        Tuple containing:
            - cameras: Dictionary mapping camera name to CameraModel objects
            - extrinsics: Dictionary mapping camera name to extrinsics dict
            - lidar_extrinsics: Dictionary mapping LIDAR name to extrinsics dict
    """
    cameras = {}
    extrinsics = {}
    lidar_extrinsics = {}
    with open(calib_path, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            # Read intrinsics (if present)
            intrinsics = {}
            is_lidar = name.upper() in ["LIDAR", "VELODYNE", "CEPTON"]
            while i < len(lines) and not lines[i].strip().startswith("Extrinsics"):
                l = lines[i].strip()
                if ":" in l:
                    k, v = l.split(":", 1)
                    intrinsics[k.strip()] = v.strip()
                i += 1
            # Camera matrix and distortion (only for cameras)
            if not is_lidar:
                fx = float(intrinsics.get("Focal_x", 0))
                fy = float(intrinsics.get("Focal_y", 0))
                cx = float(intrinsics.get("COD_x", 0))
                cy = float(intrinsics.get("COD_y", 0))
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = [float(intrinsics.get(f"Dist_{j+1}", 0)) for j in range(4)]
                cameras[name] = CameraModel(camera_matrix, dist_coeffs)
            # Read extrinsics
            extr = {"X": 0, "Y": 0, "Z": 0, "R": np.eye(3, dtype=np.float32)}
            while i < len(lines) and lines[i].strip() != "":
                l = lines[i].strip()
                if ":" in l:
                    k, v = l.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ["X", "Y", "Z"]:
                        try:
                            extr[k] = float(v)
                        except ValueError:
                            extr[k] = 0
                    elif k.startswith("R_"):
                        idx = k[2:]
                        row, col = int(idx[0]), int(idx[1])
                        if "R" not in extr or not isinstance(extr["R"], np.ndarray):
                            extr["R"] = np.eye(3, dtype=np.float32)
                        try:
                            extr["R"][row, col] = float(v)
                        except ValueError:
                            extr["R"][row, col] = 0
                i += 1
            if is_lidar:
                lidar_extrinsics[name] = extr
            else:
                extrinsics[name] = extr
        i += 1
    return cameras, extrinsics, lidar_extrinsics


def apply_extrinsics(points: np.ndarray, extr: Dict) -> np.ndarray:
    """
    Transform points using extrinsics (rotation + translation).

    Applies the extrinsics transformation to convert points from one coordinate
    frame to another using rotation matrix and translation vector.

    Args:
        points: Points in source frame (Nx3 numpy array)
        extr: Extrinsics dict with keys:
            - 'R': 3x3 rotation matrix (numpy array)
            - 'X', 'Y', 'Z': Translation components (floats)

    Returns:
        Transformed points in target frame (Nx3 numpy array)
    """
    R = extr["R"]
    t = np.array([extr["X"], extr["Y"], extr["Z"]], dtype=np.float32).reshape(1, 3)
    points_out = (R @ points.T).T + t
    return points_out


def project_lidar_to_image(
    points_3d: np.ndarray, cam_intrinsics: np.ndarray, cam_dist: np.ndarray, extr: Dict
) -> np.ndarray:
    """
    Project 3D LIDAR points onto 2D image plane using camera calibration.

    Converts 3D points from LIDAR coordinate frame to camera image coordinates
    using camera intrinsics, distortion coefficients, and extrinsics.

    Args:
        points_3d: 3D points in camera coordinate frame (Nx3 numpy array)
        cam_intrinsics: Camera intrinsic matrix (3x3 numpy array)
        cam_dist: Distortion coefficients (4-element numpy array)
        extr: Extrinsics dict with 'R' (rotation) and 'X', 'Y', 'Z' (translation)

    Returns:
        2D image points (Nx2 numpy array) with pixel coordinates [u, v]
    """
    # Convert rotation matrix to rotation vector
    R = extr["R"]
    t = np.array([extr["X"], extr["Y"], extr["Z"]], dtype=np.float32).reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    # Project using OpenCV
    img_pts, _ = cv2.projectPoints(points_3d, rvec, t, cam_intrinsics, cam_dist)
    return img_pts.reshape(-1, 2)


def transform_points(points: np.ndarray, extr: Dict) -> np.ndarray:
    """
    Transform points using extrinsics (rotation + translation).

    Applies the extrinsics transformation to convert points from one coordinate
    frame to another.

    Args:
        points: Points in source frame (Nx3 numpy array)
        extr: Extrinsics dict with 'R' (rotation matrix) and 'X', 'Y', 'Z' (translation)

    Returns:
        Transformed points in target frame (Nx3 numpy array)
    """
    R = extr["R"]
    t = np.array([extr["X"], extr["Y"], extr["Z"]], dtype=np.float32)
    return (R @ points.T).T + t


def project_points_cv(
    points_3d: np.ndarray, camera: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D using camera intrinsics and distortion.

    Projects points that are already in camera coordinate frame (no extrinsics
    transformation needed) onto the image plane.

    Args:
        points_3d: 3D points in camera frame (Nx3 numpy array)
        camera: Camera intrinsic matrix (3x3 numpy array)
        dist_coeffs: Distortion coefficients (4-element numpy array)

    Returns:
        2D image points (Nx2 numpy array) with pixel coordinates [u, v]
    """
    if points_3d is None or len(points_3d) == 0:
        return np.empty((0, 2), dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)  # No rotation: already in camera frame
    tvec = np.zeros((3, 1), dtype=np.float32)  # No translation: already in camera frame
    img_pts, _ = cv2.projectPoints(points_3d, rvec, tvec, camera, dist_coeffs)
    return img_pts.reshape(-1, 2)


def parse_pcd_header(f):
    """Parse PCD header, return dict with fields, sizes, types, count, data type, header size, and point count."""
    header = {}
    fields = []
    sizes = []
    types = []
    counts = []
    point_count = 0
    data_type = "ascii"
    header_size = 0
    while True:
        line = f.readline()
        if not line:
            break
        header_size += len(line)
        line = line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else line
        l = line.strip()
        if l.startswith("FIELDS"):
            fields = l.split()[1:]
        elif l.startswith("SIZE"):
            sizes = list(map(int, l.split()[1:]))
        elif l.startswith("TYPE"):
            types = l.split()[1:]
        elif l.startswith("COUNT"):
            counts = list(map(int, l.split()[1:]))
        elif l.startswith("POINTS"):
            point_count = int(l.split()[1])
        elif l.startswith("DATA"):
            data_type = l.split()[1].lower()
            break
    header["fields"] = fields
    header["sizes"] = sizes
    header["types"] = types
    header["counts"] = counts
    header["data_type"] = data_type
    header["header_size"] = header_size
    header["point_count"] = point_count
    return header


def load_pcd_xyz(pcd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PCD file and extract 3D points and intensity values.

    Supports both ASCII and binary PCD formats. Returns point coordinates
    and optional intensity values.

    Args:
        pcd_path: Path to PCD file

    Returns:
        Tuple of (points, intensities):
            - points: Nx3 numpy array of [x, y, z] coordinates (float32)
            - intensities: Nx1 numpy array of intensity values (float32), zeros if not available
    """
    points = []
    intensities = []
    with open(pcd_path, "rb") as f:
        header = parse_pcd_header(f)
        fields = header["fields"]
        data_type = header["data_type"]
        point_count = header["point_count"]
        f.seek(header["header_size"])
        if data_type == "ascii":
            # ASCII loader
            for line in f:
                line = line.decode("utf-8", errors="ignore")
                if line.strip() and not line.startswith("#") and not line[0].isalpha():
                    vals = line.strip().split()
                    if len(vals) >= 3:
                        points.append(
                            [
                                float(vals[fields.index("x")]),
                                float(vals[fields.index("y")]),
                                float(vals[fields.index("z")]),
                            ]
                        )
                        if "intensity" in fields:
                            intensities.append(float(vals[fields.index("intensity")]))
                        else:
                            intensities.append(0.0)
        elif data_type == "binary":
            # Binary loader
            fmt_map = {"F": "f", "U": "I", "I": "i"}
            fmt = ""
            for t, s, c in zip(header["types"], header["sizes"], header["counts"]):
                fmt += fmt_map[t] * c
            fmt = "<" + fmt  # little-endian
            point_struct = struct.Struct(fmt)
            for _ in range(point_count):
                data = f.read(point_struct.size)
                if not data or len(data) < point_struct.size:
                    break
                vals = point_struct.unpack(data)
                x = vals[fields.index("x")]
                y = vals[fields.index("y")]
                z = vals[fields.index("z")]
                points.append([x, y, z])
                if "intensity" in fields:
                    intensities.append(vals[fields.index("intensity")])
                else:
                    intensities.append(0.0)
        else:
            raise ValueError(f"Unsupported PCD DATA type: {data_type}")
    return np.array(points, dtype=np.float32), np.array(intensities, dtype=np.float32)


# --- 2. Load video and PCD folder ---
def get_sorted_pcd_files(pcd_folder: str) -> List[str]:
    """
    Get sorted list of PCD files from a folder.

    Files are sorted by numeric prefix in the filename to ensure temporal ordering.

    Args:
        pcd_folder: Path to folder containing PCD files

    Returns:
        List of full file paths, sorted by frame index
    """
    files = [f for f in os.listdir(pcd_folder) if f.lower().endswith(".pcd")]

    # Sort by integer prefix before first '-' in filename
    def frame_index(filename):
        try:
            return int(filename.split("-")[0])
        except Exception:
            return float("inf")

    files.sort(key=frame_index)
    return [os.path.join(pcd_folder, f) for f in files]


# --- 3. Main visualization loop ---
def main(
    calib_path: str, video_path: str, pcd_folder: str, camera_name: str, lidar_name: str
) -> None:
    """
    Main function to visualise LIDAR point cloud projection onto camera video.

    Loads calibration data, video, and PCD files, then projects LIDAR points
    onto each video frame for interactive visualisation and validation.

    Args:
        calib_path: Path to sensor_calibration.txt file
        video_path: Path to input video file
        pcd_folder: Path to folder containing PCD files (sorted by frame index)
        camera_name: Name of camera from calibration file (e.g., "FLIR 8.9MP")
        lidar_name: Name of LIDAR from calibration file (e.g., "Velodyne", "Cepton")
    """
    # Debug info: log file paths
    logger.debug(f"Calibration file: {calib_path}")
    logger.debug(f"Video file: {video_path}")
    logger.debug(f"PCD folder: {pcd_folder}")
    logger.debug(f"Camera name: {camera_name}")
    logger.debug(f"LIDAR name: {lidar_name}")

    cameras, extrinsics, lidar_extrinsics = parse_sensor_calibration(calib_path)
    if camera_name not in cameras:
        logger.error(f"Camera {camera_name} not found in calibration file.")
        return
    if lidar_name not in lidar_extrinsics:
        logger.error(f"LIDAR {lidar_name} not found in calibration file.")
        return
    camera = cameras[camera_name]
    lidar_extr = lidar_extrinsics[lidar_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    pcd_files = get_sorted_pcd_files(pcd_folder)
    if not pcd_files:
        logger.error(f"No PCD files found in {pcd_folder}")
        return

    # Debug info: log first PCD file
    logger.debug(f"Number of PCD files found: {len(pcd_files)}")
    if pcd_files:
        logger.debug(f"First PCD file: {pcd_files[0]}")

    # --- Load background PCD if enabled ---
    bg_points = None
    if ENABLE_BG_SUBTRACTION and pcd_files:
        bg_points, _ = load_pcd_xyz(pcd_files[0])
        # Transform background points into camera frame
        bg_points = transform_points(bg_points, lidar_extr)

    # Trackbar state
    start_pcd_offset = [0]

    def on_trackbar(val):
        start_pcd_offset[0] = val

    cv2.namedWindow("PCD Sync", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PCD Sync", 1500, 100)
    max_trackbar = min(1000, max(1, len(pcd_files) - 1))
    cv2.createTrackbar("PCD Start Frame", "PCD Sync", 0, max_trackbar, on_trackbar)

    frame_idx = 0
    paused = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            pcd_idx = (frame_idx // 3) + start_pcd_offset[0]
            debug_lines = [
                f"Frame: {frame_idx+1} / {total_frames}",
                f"PCD idx: {pcd_idx} / {len(pcd_files)-1}",
                f"PCD file: {os.path.basename(pcd_files[pcd_idx]) if 0 <= pcd_idx < len(pcd_files) else 'N/A'}",
                f"Paused: {paused}",
            ]
            if 0 <= pcd_idx < len(pcd_files):
                points, intensities = load_pcd_xyz(pcd_files[pcd_idx])
                points_in_cam = transform_points(points, lidar_extr)
                # --- Background subtraction using first PCD as background ---
                if ENABLE_BG_SUBTRACTION and bg_points is not None and len(bg_points) > 0:
                    from scipy.spatial import cKDTree

                    bg_tree = cKDTree(bg_points)
                    # Remove n_jobs argument for compatibility
                    dists, _ = bg_tree.query(points_in_cam, k=1)
                    mask = dists > BG_DIST_THRESHOLD
                    points_in_cam = points_in_cam[mask]
                    intensities = intensities[mask]
                cam_intr = camera.camera_matrix
                cam_dist = np.array(camera.dist_coeffs, dtype=np.float32).reshape(-1, 1)
                img_pts = project_points_cv(points_in_cam, cam_intr, cam_dist)
                # Normalize for color mapping with static Z range
                if COLOR_BY_Z:
                    z_vals = points_in_cam[:, 2]
                    # Clip and normalize to static range
                    z_vals_clipped = np.clip(z_vals, COLORMAP_Z_MIN, COLORMAP_Z_MAX)
                    norm_z = (
                        (z_vals_clipped - COLORMAP_Z_MIN) / (COLORMAP_Z_MAX - COLORMAP_Z_MIN) * 255
                    ).astype(np.uint8)
                    colors = cv2.applyColorMap(norm_z, cv2.COLORMAP_JET)
                elif intensities.size > 0:
                    norm_int = cv2.normalize(intensities, None, 0, 255, cv2.NORM_MINMAX).astype(
                        np.uint8
                    )
                    norm_int = norm_int.flatten()
                    colors = cv2.applyColorMap(norm_int, cv2.COLORMAP_JET)
                else:
                    colors = np.full((img_pts.shape[0], 3), (0, 255, 0), dtype=np.uint8)
                for idx, pt in enumerate(img_pts):
                    x, y = int(pt[0] / 2), int(pt[1] / 2)
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        color = tuple(int(c) for c in colors[idx][:3].flatten())
                        cv2.circle(frame, (x, y), 2, color, -1)
            # Draw debug info
            for i, line in enumerate(debug_lines):
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("LIDAR Reprojection", frame)
            cv2.imshow(
                "PCD Sync", np.zeros((100, 1500, 3), dtype=np.uint8)
            )  # Dummy window for trackbar
            frame_idx += 1
        else:
            # When paused, show the current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            pcd_idx = (frame_idx // 3) + start_pcd_offset[0]
            debug_lines = [
                f"Frame: {frame_idx+1} / {total_frames}",
                f"PCD idx: {pcd_idx} / {len(pcd_files)-1}",
                f"PCD file: {os.path.basename(pcd_files[pcd_idx]) if 0 <= pcd_idx < len(pcd_files) else 'N/A'}",
                f"Paused: {paused}",
            ]
            if 0 <= pcd_idx < len(pcd_files):
                points, intensities = load_pcd_xyz(pcd_files[pcd_idx])
                if np.any(intensities):
                    print(
                        f"[DEBUG] Intensities present in PCD file: {os.path.basename(pcd_files[pcd_idx])}"
                    )
                else:
                    print(
                        f"[DEBUG] No intensities in PCD file: {os.path.basename(pcd_files[pcd_idx])}"
                    )
                points_in_cam = transform_points(points, lidar_extr)
                if ENABLE_BG_SUBTRACTION and bg_points is not None and len(bg_points) > 0:
                    from scipy.spatial import cKDTree

                    bg_tree = cKDTree(bg_points)
                    dists, _ = bg_tree.query(points_in_cam, k=1)
                    mask = dists > BG_DIST_THRESHOLD
                    points_in_cam = points_in_cam[mask]
                    intensities = intensities[mask]
                cam_intr = camera.camera_matrix
                cam_dist = np.array(camera.dist_coeffs, dtype=np.float32).reshape(-1, 1)
                img_pts = project_points_cv(points_in_cam, cam_intr, cam_dist)
                if COLOR_BY_Z:
                    z_vals = points_in_cam[:, 2]
                    z_vals_clipped = np.clip(z_vals, COLORMAP_Z_MIN, COLORMAP_Z_MAX)
                    norm_z = (
                        (z_vals_clipped - COLORMAP_Z_MIN) / (COLORMAP_Z_MAX - COLORMAP_Z_MIN) * 255
                    ).astype(np.uint8)
                    colors = cv2.applyColorMap(norm_z, cv2.COLORMAP_JET)
                elif intensities.size > 0:
                    norm_int = cv2.normalize(intensities, None, 0, 255, cv2.NORM_MINMAX).astype(
                        np.uint8
                    )
                    colors = cv2.applyColorMap(norm_int, cv2.COLORMAP_JET)
                else:
                    colors = np.full((img_pts.shape[0], 3), (0, 255, 0), dtype=np.uint8)
                for idx, pt in enumerate(img_pts):
                    x, y = int(pt[0] / 2), int(pt[1] / 2)
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        color = tuple(int(c) for c in colors[idx][:3].flatten())
                        cv2.circle(frame, (x, y), 2, color, -1)
            # Draw debug info
            for i, line in enumerate(debug_lines):
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("LIDAR Reprojection", frame)
            cv2.imshow("PCD Sync", np.zeros((100, 1500, 3), dtype=np.uint8))
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
        elif key == 32:  # Spacebar
            paused = not paused
        elif paused and (key == 83 or key == ord("d")):  # Right arrow or 'd'
            if frame_idx < total_frames - 1:
                frame_idx += 1
        elif paused and (key == 81 or key == ord("a")):  # Left arrow or 'a'
            if frame_idx > 0:
                frame_idx -= 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Example usage of LIDAR reprojection tool.

    Edit the paths below to match your data locations:
    """
    # Set your parameters here
    calib_path = "path/to/sensor_calibration.txt"
    video_path = "path/to/video.mp4"
    pcd_folder = "path/to/pcd/folder/"
    camera_name = "FLIR 8.9MP"  # Camera name from calibration file
    lidar_name = "Cepton"  # LIDAR name from calibration file

    main(calib_path, video_path, pcd_folder, camera_name, lidar_name)
