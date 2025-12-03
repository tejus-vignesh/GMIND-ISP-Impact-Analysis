"""
LIDAR Object Detection Script

Detects 3D objects in stationary LIDAR point cloud data using background subtraction
and clustering. Processes PCD (Point Cloud Data) files from a folder and returns
3D bounding boxes for detected objects such as pedestrians and vehicles.

The pipeline consists of:
1. Background model construction from sampled frames
2. Background subtraction to identify foreground points
3. Clustering of foreground points using distance-based grouping
4. Object detection and centroid computation
5. Optional 3D visualisation of results

Author: GMIND SDK Development Team
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# Optional matplotlib imports for visualisation
# Using PySide6 (QtAgg backend) for real-time visualisation, or Agg as fallback
# TkAgg is avoided since it requires system packages on Linux/WSL
try:
    # Set default backend BEFORE importing matplotlib to prevent TkAgg fallback
    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    selected_backend = "Agg"  # Default to Agg

    # Check for PySide6 and use QtAgg backend for real-time visualisation
    try:
        import PySide6

        matplotlib.use("QtAgg", force=True)
        selected_backend = "QtAgg"
    except ImportError:
        # PySide6 not available, use Agg backend (saves images)
        matplotlib.use("Agg", force=True)
        selected_backend = "Agg"

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_BACKEND = selected_backend
    MATPLOTLIB_IS_INTERACTIVE = selected_backend == "QtAgg"

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_BACKEND = None
    MATPLOTLIB_IS_INTERACTIVE = False
    Axes3D = None
    Poly3DCollection = None


def parse_pcd_header(f):
    """
    Parse PCD (Point Cloud Data) file header.

    Reads the header section of a PCD file and extracts metadata including field
    definitions, data types, and point count.

    Args:
        f: File handle opened in binary mode ('rb')

    Returns:
        Dictionary containing:
            - fields: List of field names (e.g., ['x', 'y', 'z', 'intensity'])
            - sizes: List of field sizes in bytes
            - types: List of field types (e.g., ['F', 'F', 'F', 'F'])
            - counts: List of field counts (usually 1 for scalar fields)
            - data_type: Data format ('ascii' or 'binary')
            - header_size: Size of header in bytes
            - point_count: Number of points in the file
    """
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
    import struct

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


def parse_sensor_calibration(calib_path: str) -> Dict[str, Dict]:
    """
    Parse sensor_calibration.txt to extract LIDAR extrinsics.

    Args:
        calib_path: Path to sensor_calibration.txt

    Returns:
        Dictionary mapping LIDAR name to extrinsics dict with X, Y, Z, R (3x3 rotation matrix)
    """
    lidar_extrinsics = {}

    with open(calib_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
            # Check if this is a LIDAR
            is_lidar = name.upper() in ["LIDAR", "VELODYNE", "CEPTON"]

            # Skip intrinsics section
            while i < len(lines) and not lines[i].strip().startswith("Extrinsics"):
                i += 1

            # Read extrinsics
            extr = {"X": 0, "Y": 0, "Z": 0, "R": np.eye(3, dtype=np.float32)}
            i += 1  # Skip "Extrinsics" line
            while i < len(lines) and lines[i].strip() != "":
                l = lines[i].strip()
                if ":" in l:
                    k, v = l.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if k in ["X", "Y", "Z"]:
                        try:
                            extr[k] = float(v) if v else 0.0
                        except ValueError:
                            extr[k] = 0.0
                    elif k.startswith("R_"):
                        idx = k[2:]
                        if len(idx) >= 2:
                            row, col = int(idx[0]), int(idx[1])
                            if "R" not in extr or not isinstance(extr["R"], np.ndarray):
                                extr["R"] = np.eye(3, dtype=np.float32)
                            try:
                                extr["R"][row, col] = float(v) if v else 0.0
                            except ValueError:
                                extr["R"][row, col] = 0.0
                i += 1

            if is_lidar:
                lidar_extrinsics[name] = extr
        i += 1

    return lidar_extrinsics


def apply_extrinsics(points: np.ndarray, extr: Dict) -> np.ndarray:
    """
    Transform points from LIDAR frame to reference frame using extrinsics.

    Args:
        points: Points in LIDAR frame (Nx3)
        extr: Extrinsics dict with X, Y, Z, R (3x3 rotation matrix)

    Returns:
        Transformed points in reference frame (Nx3)
    """
    R = extr["R"]
    t = np.array([extr["X"], extr["Y"], extr["Z"]], dtype=np.float32)
    return (R @ points.T).T + t


def apply_extrinsics_inverse(points: np.ndarray, extr: Dict) -> np.ndarray:
    """
    Transform points from reference frame to LIDAR frame using inverse extrinsics.

    Args:
        points: Points in reference frame (Nx3)
        extr: Extrinsics dict with X, Y, Z, R (3x3 rotation matrix)

    Returns:
        Transformed points in LIDAR frame (Nx3)
    """
    R = extr["R"]
    t = np.array([extr["X"], extr["Y"], extr["Z"]], dtype=np.float32)
    R_inv = R.T  # Inverse of rotation matrix is its transpose
    t_inv = -R_inv @ t  # Inverse translation
    return (R_inv @ points.T).T + t_inv


def get_sorted_pcd_files(pcd_folder: str) -> List[str]:
    """
    Get sorted list of PCD files from a folder.

    Files are sorted by numeric prefix in the filename (e.g., "001-frame.pcd" comes
    before "002-frame.pcd"). This ensures temporal ordering for sequential processing.

    Args:
        pcd_folder: Path to folder containing PCD files

    Returns:
        List of full file paths, sorted by frame index
    """
    files = [f for f in os.listdir(pcd_folder) if f.lower().endswith(".pcd")]

    # Sort by integer prefix before first '-' in filename, or by filename
    def frame_index(filename):
        try:
            # Try to extract number from start of filename
            base = os.path.splitext(filename)[0]
            parts = base.split("-")
            if parts:
                return int(parts[0])
            # Fallback: try to extract any leading digits
            import re

            match = re.match(r"(\d+)", base)
            if match:
                return int(match.group(1))
            return float("inf")
        except Exception:
            return float("inf")

    files.sort(key=frame_index)
    return [os.path.join(pcd_folder, f) for f in files]


def build_background_model(pcd_files: List[str], num_frames: int = 50) -> np.ndarray:
    """
    Build robust background model by sampling frames evenly throughout the dataset.

    Uses frames from across the dataset to average out transient objects and noise,
    creating a more stable background model than using only the first N frames.
    This is particularly important for long sequences where lighting or scene
    conditions may change.

    Args:
        pcd_files: List of PCD file paths, sorted by frame index
        num_frames: Number of frames to use for background model (sampled evenly
                   across dataset). Default: 50

    Returns:
        Background point cloud as Nx3 numpy array (float32), containing all
        points from sampled frames combined

    Raises:
        ValueError: If no valid PCD files are found or all files fail to load
    """
    all_background_points = []
    num_frames = min(num_frames, len(pcd_files))

    # Sample frames evenly throughout the dataset (not just first N)
    if len(pcd_files) > num_frames:
        # Calculate step size to sample evenly
        step = len(pcd_files) / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
        # Ensure we get the last frame too
        if frame_indices[-1] != len(pcd_files) - 1:
            frame_indices.append(len(pcd_files) - 1)
        frame_indices = sorted(set(frame_indices))  # Remove duplicates and sort
        print(
            f"Building background model from {len(frame_indices)} frames sampled evenly across {len(pcd_files)} frames..."
        )
    else:
        # Use all frames if we have fewer than requested
        frame_indices = list(range(len(pcd_files)))
        print(f"Building background model from all {len(frame_indices)} frames...")

    for i in tqdm(frame_indices, desc="Loading background frames"):
        try:
            points, _ = load_pcd_xyz(pcd_files[i])
            if len(points) > 0:
                all_background_points.append(points)
        except Exception as e:
            print(f"Warning: Failed to load {pcd_files[i]}: {e}")
            continue

    if not all_background_points:
        raise ValueError("No valid PCD files found for background model")

    # Combine all background points from sampled frames
    background = np.vstack(all_background_points)
    print(
        f"Background model contains {len(background)} points from {len(all_background_points)} frames"
    )
    return background


def background_subtraction(
    points: np.ndarray, background: np.ndarray, distance_threshold: float = 0.35
) -> np.ndarray:
    """
    Perform background subtraction using nearest neighbor distance.

    For each point in the current frame, finds the nearest point in the background
    model. Points that are far from the background (beyond threshold) are
    considered foreground (moving objects).

    Args:
        points: Current frame points (Nx3 numpy array)
        background: Background model points (Mx3 numpy array)
        distance_threshold: Distance threshold in meters. Points further than this
                           from the background are considered foreground.
                           Default: 0.35m (tuned for pedestrians/vehicles)

    Returns:
        Foreground points as Nx3 numpy array (float32), containing only points
        that are far enough from the background model
    """
    if len(points) == 0:
        return points

    if len(background) == 0:
        return points

    # Build KDTree for background
    bg_tree = cKDTree(background)

    # Find nearest neighbor distance for each point
    distances, _ = bg_tree.query(points, k=1)

    # Points far from background are foreground
    mask = distances > distance_threshold
    return points[mask]


def cluster_points(points: np.ndarray, eps: float = 0.6, min_points: int = 8) -> List[np.ndarray]:
    """
    Cluster points using DBSCAN-like algorithm (connected components with distance threshold).

    Groups nearby points into clusters using a distance-based approach. Points within
    `eps` distance of each other are grouped together. Clusters with fewer than
    `min_points` are discarded as noise.

    Args:
        points: Point cloud (Nx3 numpy array)
        eps: Maximum distance in meters between points in the same cluster.
             Default: 0.6m (suitable for pedestrians ~0.5m width and vehicles ~1.5-2m width)
        min_points: Minimum number of points required to form a valid cluster.
                   Default: 8 (tuned to detect pedestrians)

    Returns:
        List of clusters, where each cluster is a numpy array of points (Mx3).
        Empty list if no valid clusters are found.
    """
    if len(points) == 0:
        return []

    # Build KDTree for efficient neighbor search
    tree = cKDTree(points)

    # Find all points within eps distance
    visited = np.zeros(len(points), dtype=bool)
    clusters = []

    for i in range(len(points)):
        if visited[i]:
            continue

        # Start new cluster
        cluster_points_indices = [i]
        visited[i] = True

        # Expand cluster using queue
        queue = [i]
        while queue:
            point_idx = queue.pop(0)
            neighbors = tree.query_ball_point(points[point_idx], eps)

            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    cluster_points_indices.append(neighbor_idx)
                    queue.append(neighbor_idx)

        # Only keep clusters with enough points
        if len(cluster_points_indices) >= min_points:
            clusters.append(points[cluster_points_indices])

    return clusters


def compute_centroid(points: np.ndarray) -> Optional[Dict[str, any]]:
    """
    Compute centroid for a cluster of points.

    Calculates the mean position of all points in the cluster, which represents
    the geometric centre of the detected object.

    Args:
        points: Cluster points (Nx3 numpy array)

    Returns:
        Dictionary with centroid information:
            - centroid: [x, y, z] centroid coordinates as list
            - num_points: Number of points in cluster
        Returns None if points array is empty
    """
    if len(points) == 0:
        return None

    centroid = np.mean(points, axis=0)

    return {"centroid": centroid.tolist(), "num_points": len(points)}


def detect_objects_in_frame(
    points: np.ndarray,
    background: np.ndarray,
    bg_threshold: float = 0.35,
    cluster_eps: float = 0.6,
    min_cluster_points: int = 8,
    bg_k_neighbors: int = 3,
) -> Tuple[List[Dict[str, any]], np.ndarray]:
    """
    Detect objects in a single frame and return cluster centroids.
    Uses robust background subtraction with k-nearest neighbors to reduce noise sensitivity.

    Args:
        points: Current frame points (Nx3)
        background: Background model (Mx3)
        bg_threshold: Background subtraction distance threshold (meters)
        cluster_eps: Clustering distance threshold (meters)
        min_cluster_points: Minimum points per cluster
        bg_k_neighbors: Number of nearest neighbors to check for robust background matching

    Returns:
        Tuple of (list of detected clusters with centroid info, background mask for original points)
    """
    # Background subtraction - get mask of which points are foreground
    if len(points) == 0:
        return [], np.array([], dtype=bool)

    if len(background) == 0:
        # No background, all points are foreground
        is_foreground = np.ones(len(points), dtype=bool)
    else:
        # Build KDTree for background
        bg_tree = cKDTree(background)

        # Use k-nearest neighbors for more robust distance calculation
        # This reduces sensitivity to single noisy background points
        k = min(bg_k_neighbors, len(background))
        distances, _ = bg_tree.query(points, k=k)

        # If multiple neighbors, use median distance (robust to outliers)
        if k > 1:
            if distances.ndim > 1:
                distances = np.median(distances, axis=1)
            else:
                distances = distances.flatten()
        else:
            distances = distances.flatten()

        # Points far from background are foreground
        # Add small margin to threshold to account for noise
        is_foreground = distances > bg_threshold

    foreground_points = points[is_foreground]

    if len(foreground_points) == 0:
        return [], ~is_foreground

    # Cluster foreground points
    clusters = cluster_points(foreground_points, cluster_eps, min_cluster_points)

    # Compute centroids
    objects = []
    for i, cluster in enumerate(clusters):
        centroid_info = compute_centroid(cluster)
        if centroid_info:
            centroid_info["cluster_id"] = i
            objects.append(centroid_info)

    return objects, ~is_foreground  # Return background mask (inverse of foreground)


def draw_3d_box(
    ax,
    min_corner: Tuple[float, float, float],
    max_corner: Tuple[float, float, float],
    colour: str = "red",
    alpha: float = 0.3,
    linewidth: int = 2,
):
    """
    Draw a 3D bounding box on a matplotlib 3D axis.

    Creates a wireframe box with semi-transparent faces to visualise object
    bounding volumes in 3D space.

    Args:
        ax: Matplotlib 3D axis object
        min_corner: [x_min, y_min, z_min] coordinates of box minimum corner
        max_corner: [x_max, y_max, z_max] coordinates of box maximum corner
        colour: Box colour (default: "red")
        alpha: Transparency level 0-1 (default: 0.3)
        linewidth: Edge line width in pixels (default: 2)
    """
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner

    # Define the 8 vertices of the box
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]
    )

    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
    ]

    # Draw faces
    face_collection = Poly3DCollection(
        faces, alpha=alpha, facecolor=colour, edgecolor=colour, linewidth=linewidth
    )
    ax.add_collection3d(face_collection)

    # Draw edges more prominently
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],  # Bottom
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],  # Top
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]],  # Vertical
    ]

    for edge in edges:
        ax.plot3D(*zip(*edge), color=colour, linewidth=linewidth, alpha=0.8)


def visualise_3d_lidar(
    points: np.ndarray,
    objects: List[Dict[str, any]],
    ax3d,
    frame_idx: int,
    max_frames: int,
    view_size: float = 50.0,
    max_points: int = 5000,
    view_elev: Optional[float] = None,
    view_azim: Optional[float] = None,
    background_mask: Optional[np.ndarray] = None,
):
    """
    Visualise LIDAR point cloud in 3D with detected cluster centroids.

    Args:
        points: Point cloud points (Nx3)
        objects: List of detected clusters with centroid info
        ax3d: Matplotlib 3D axis to draw on
        frame_idx: Current frame index
        max_frames: Total number of frames
        view_size: Size of the view in meters (default: 50m)
        max_points: Maximum number of points to display (for performance)
        view_elev: Optional elevation angle to preserve camera view
        view_azim: Optional azimuth angle to preserve camera view
        background_mask: Boolean mask indicating which points are background (True = background)
    """
    # Save current view angles before clearing (for camera position preservation)
    if view_elev is None or view_azim is None:
        try:
            elev, azim = ax3d.elev, ax3d.azim
        except:
            elev, azim = 20, 45
    else:
        elev, azim = view_elev, view_azim

    ax3d.clear()

    # Use points directly (no transformation)
    display_points = points.copy()
    display_bg_mask = background_mask.copy() if background_mask is not None else None

    # Filter points within view bounds first
    half_size = view_size / 2.0
    view_mask = (
        (np.abs(display_points[:, 0]) <= half_size)
        & (np.abs(display_points[:, 1]) <= half_size)
        & (np.abs(display_points[:, 2]) <= half_size)
    )

    display_points = display_points[view_mask]
    if display_bg_mask is not None:
        display_bg_mask = display_bg_mask[view_mask]

    if len(display_points) == 0:
        ax3d.set_xlim(-half_size, half_size)
        ax3d.set_ylim(-half_size, half_size)
        ax3d.set_zlim(-half_size, half_size)
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        ax3d.set_title(f"3D LIDAR View - Frame {frame_idx+1}/{max_frames} - No points in view")
        return elev, azim

    # Downsample points if too many (for performance)
    if len(display_points) > max_points:
        indices = np.random.choice(len(display_points), max_points, replace=False)
        display_points = display_points[indices]
        if display_bg_mask is not None:
            display_bg_mask = display_bg_mask[indices]

    # Identify background points
    is_background = (
        display_bg_mask
        if display_bg_mask is not None
        else np.zeros(len(display_points), dtype=bool)
    )

    # Detect points close to cluster centroids (only for foreground points)
    is_in_cluster = np.zeros(len(display_points), dtype=bool)
    cluster_radius = 2.0  # Consider points within 2m of centroid as part of cluster
    for obj in objects:
        centroid = np.array(obj["centroid"])

        # Calculate distances from all points to this centroid
        distances = np.linalg.norm(display_points - centroid, axis=1)
        close_mask = distances <= cluster_radius
        is_in_cluster |= close_mask

    # Assign colours: background=grey, in clusters=green, others=blue (by height)
    colours = np.zeros((len(display_points), 3))

    # Background points: grey
    colours[is_background] = [0.5, 0.5, 0.5]  # Grey

    # Points in clusters: green (override background if both)
    colours[is_in_cluster] = [0.0, 0.8, 0.0]  # Green

    # Other foreground points: colour by height (blue to cyan gradient)
    other_mask = ~(is_background | is_in_cluster)
    if np.any(other_mask):
        z_vals = display_points[:, 2]
        z_min = np.min(z_vals[other_mask])
        z_max = np.max(z_vals[other_mask])
        if z_max > z_min:
            z_other = display_points[other_mask, 2]
            z_norm = (z_other - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros(np.sum(other_mask))
        # Blue (low) to cyan (high)
        colours[other_mask, 0] = 0.0  # R
        colours[other_mask, 1] = z_norm  # G
        colours[other_mask, 2] = 1.0  # B

    # Plot point cloud with assigned colours
    ax3d.scatter(
        display_points[:, 0], display_points[:, 1], display_points[:, 2], c=colours, s=1, alpha=0.5
    )

    # Draw cluster centroids
    for obj in objects:
        centroid = np.array(obj["centroid"])

        # Only draw centroids that are in view
        if not (
            -half_size <= centroid[0] <= half_size
            and -half_size <= centroid[1] <= half_size
            and -half_size <= centroid[2] <= half_size
        ):
            continue

        # Draw centroid point
        ax3d.scatter(
            [centroid[0]],
            [centroid[1]],
            [centroid[2]],
            color="red",
            s=100,
            marker="o",
            edgecolors="darkred",
            linewidth=2,
        )

        # Draw label
        ax3d.text(
            centroid[0],
            centroid[1],
            centroid[2],
            f"ID:{obj['cluster_id']}",
            fontsize=10,
            color="red",
            weight="bold",
        )

    # Set axis limits
    ax3d.set_xlim(-half_size, half_size)
    ax3d.set_ylim(-half_size, half_size)
    ax3d.set_zlim(-half_size, half_size)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"3D LIDAR View - Frame {frame_idx+1}/{max_frames} - {len(objects)} objects")
    # Restore or set viewing angle (preserve user's camera position)
    ax3d.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    return elev, azim  # Return view angles for next frame


def process_lidar_folder(
    pcd_folder: str,
    output_file: Optional[str] = None,
    num_bg_frames: int = 50,
    bg_threshold: float = 0.35,
    cluster_eps: float = 0.6,
    min_cluster_points: int = 8,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    show_visualisation: bool = False,
) -> Dict[str, any]:
    """
    Process all PCD files in a folder and detect objects.

    Main processing function that:
    1. Loads and sorts PCD files from the specified folder
    2. Builds a background model from sampled frames
    3. Processes each frame to detect foreground objects
    4. Clusters foreground points into objects
    5. Optionally visualises results in 3D
    6. Saves results to JSON file

    Args:
        pcd_folder: Path to folder containing PCD files
        output_file: Optional JSON file path to save results. If None and output
                    is enabled, auto-generates filename in parent folder.
        num_bg_frames: Number of frames to use for background model (sampled evenly).
                      Default: 50
        bg_threshold: Background subtraction distance threshold in meters.
                     Lower values detect smaller movements. Default: 0.35m
        cluster_eps: Clustering distance threshold in meters. Max distance between
                    points in same cluster. Default: 0.6m
        min_cluster_points: Minimum points per cluster. Lower values detect smaller
                          objects. Default: 8
        start_frame: First frame index to process (0-based). Default: 0
        end_frame: Last frame index to process (None = all frames). Default: None
        show_visualisation: If True, show interactive 3D visualisation during
                          processing. Requires matplotlib with QtAgg backend for
                          real-time display, or saves images with Agg backend.

    Returns:
        Dictionary containing:
            - pcd_folder: Input folder path
            - num_frames: Number of frames processed
            - parameters: Dictionary of processing parameters used
            - detections: Dictionary mapping frame filename to detection results
              Each detection contains:
                - frame_index: Frame number
                - num_points: Total points in frame
                - num_objects: Number of objects detected
                - objects: List of detected objects, each with:
                  - centroid: [x, y, z] coordinates
                  - num_points: Points in cluster
                  - cluster_id: Unique cluster identifier

    Raises:
        ValueError: If no PCD files are found in the folder
    """
    # Get PCD files
    pcd_files = get_sorted_pcd_files(pcd_folder)

    if len(pcd_files) == 0:
        raise ValueError(f"No PCD files found in {pcd_folder}")

    print(f"Found {len(pcd_files)} PCD files")

    # Build background model (in LIDAR frame) - samples frames evenly across dataset
    background = build_background_model(pcd_files, num_bg_frames)

    # Process frames
    end_frame = end_frame if end_frame is not None else len(pcd_files)
    pcd_files = pcd_files[start_frame:end_frame]

    # Setup visualisation if enabled
    fig, ax3d = None, None
    viz_output_dir = None
    view_size = 50.0  # Fixed 50m x 50m view
    view_elev, view_azim = 20, 45  # Initial camera angles for 3D view
    paused = False  # Pause state for visualisation

    # Keyboard event handler for pause/play
    def on_key(event):
        nonlocal paused
        if event.key == " " or event.key == "p":  # Space or 'p' to pause/unpause
            paused = not paused
            print(
                f"\n{'PAUSED' if paused else 'PLAYING'} - Press SPACE or 'P' to {'resume' if paused else 'pause'}"
            )
        elif event.key == "q" or event.key == "escape":  # Quit
            print("\nStopping visualisation...")
            return False
        return True

    if show_visualisation:
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Visualisation disabled.")
            show_visualisation = False
        else:
            try:
                # Create 3D view only (full screen)
                fig = plt.figure(figsize=(16, 12))
                ax3d = fig.add_subplot(111, projection="3d")

                if MATPLOTLIB_IS_INTERACTIVE:
                    # Qt backend: real-time interactive visualisation
                    plt.ion()  # Turn on interactive mode
                    # Connect keyboard event handler
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    print(
                        f"Visualisation enabled: Real-time 3D display using {MATPLOTLIB_BACKEND} backend"
                    )
                    print("Controls: SPACE or 'P' = pause/resume, 'Q' or ESC = stop visualisation")
                    print("When paused, you can rotate/zoom the 3D view")
                else:
                    # Agg backend: save images instead
                    viz_output_dir = os.path.join(
                        os.path.dirname(pcd_folder) or ".", "lidar_detection_viz"
                    )
                    os.makedirs(viz_output_dir, exist_ok=True)
                    print(f"Visualisation enabled: Saving frames to {viz_output_dir}/")
                    print(
                        f"Note: Using {MATPLOTLIB_BACKEND} backend (non-interactive). Images will be saved instead of displayed."
                    )
                    print(
                        "  Tip: Install PySide6 for real-time visualisation: uv pip install PySide6"
                    )
            except Exception as e:
                print(f"Warning: Failed to initialise visualisation: {e}")
                print("Continuing without visualisation...")
                show_visualisation = False

    results = {
        "pcd_folder": pcd_folder,
        "num_frames": len(pcd_files),
        "parameters": {
            "num_bg_frames": num_bg_frames,
            "bg_threshold": bg_threshold,
            "cluster_eps": cluster_eps,
            "min_cluster_points": min_cluster_points,
        },
        "detections": {},
    }

    print(f"\nProcessing {len(pcd_files)} frames...")
    for frame_idx, pcd_file in enumerate(
        tqdm(pcd_files, desc="Processing frames", disable=show_visualisation)
    ):
        try:
            points, _ = load_pcd_xyz(pcd_file)

            if len(points) > 0:
                objects, background_mask = detect_objects_in_frame(
                    points, background, bg_threshold, cluster_eps, min_cluster_points
                )

                frame_name = os.path.basename(pcd_file)
                results["detections"][frame_name] = {
                    "frame_index": start_frame + frame_idx,
                    "num_points": len(points),
                    "num_objects": len(objects),
                    "objects": objects,
                }

                # Update visualisation
                if show_visualisation and ax3d is not None:
                    if not paused:
                        # Only update visualisation when not paused
                        view_elev, view_azim = visualise_3d_lidar(
                            points,
                            objects,
                            ax3d,
                            frame_idx,
                            len(pcd_files),
                            view_size,
                            view_elev=view_elev,
                            view_azim=view_azim,
                            background_mask=background_mask,
                        )

                        if MATPLOTLIB_IS_INTERACTIVE:
                            plt.draw()
                            plt.pause(0.01)  # Small pause to allow GUI to update
                    else:
                        # When paused, just process events to allow user interaction without updating
                        if MATPLOTLIB_IS_INTERACTIVE:
                            while paused and plt.get_fignums():
                                plt.pause(0.1)
                                fig.canvas.flush_events()

                    if MATPLOTLIB_IS_INTERACTIVE:
                        # Check if user wants to quit
                        if not plt.get_fignums():
                            break
                    else:
                        # Save frame image for Agg backend
                        if viz_output_dir:
                            frame_name_base = os.path.splitext(frame_name)[0]
                            viz_path = os.path.join(viz_output_dir, f"{frame_name_base}.png")
                            plt.savefig(viz_path, dpi=100, bbox_inches="tight")
                        plt.draw()  # Force redraw
            else:
                frame_name = os.path.basename(pcd_file)
                results["detections"][frame_name] = {
                    "frame_index": start_frame + frame_idx,
                    "num_points": 0,
                    "num_objects": 0,
                    "objects": [],
                }
        except Exception as e:
            print(f"Error processing {pcd_file}: {e}")
            frame_name = os.path.basename(pcd_file)
            results["detections"][frame_name] = {
                "frame_index": start_frame + frame_idx,
                "error": str(e),
            }

    if show_visualisation and fig is not None:
        if MATPLOTLIB_IS_INTERACTIVE:
            # Interactive mode: keep window open
            print("\nVisualization complete. Close the window to continue...")
            plt.ioff()  # Turn off interactive mode
            plt.show(block=True)  # Keep window open until closed
        else:
            # Save final summary visualisation for Agg backend
            if viz_output_dir:
                summary_path = os.path.join(viz_output_dir, "summary_all_detections.png")
                try:
                    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
                    print(
                        f"\nVisualization complete. Saved {len(pcd_files)} frame images to {viz_output_dir}/"
                    )
                    print(f"Summary image saved to: {summary_path}")
                except Exception as e:
                    print(f"Warning: Failed to save summary visualisation: {e}")
            plt.close(fig)

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    # Print summary
    total_objects = sum(len(det.get("objects", [])) for det in results["detections"].values())
    print(f"\nSummary:")
    print(f"  Total frames processed: {len(results['detections'])}")
    print(f"  Total objects detected: {total_objects}")
    print(
        f"  Average objects per frame: {total_objects / len(results['detections']) if results['detections'] else 0:.2f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect 3D objects (pedestrians and vehicles) in LIDAR PCD files using background subtraction and clustering"
    )
    parser.add_argument("pcd_folder", type=str, help="Path to folder containing PCD files")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated in parent folder of PCD directory)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable output file generation (print to stdout instead)",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Show 2D XY visualisation during processing (requires matplotlib)",
    )
    parser.add_argument(
        "--num-bg-frames",
        type=int,
        default=50,
        help="Number of frames to use for background model, sampled evenly across dataset (default: 50)",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.35,
        help="Background subtraction distance threshold in meters. Lower values detect smaller movements. (default: 0.35, tuned for pedestrians/vehicles)",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.6,
        help="Clustering distance threshold in meters. Max distance between points in same cluster. (default: 0.6, suitable for pedestrians ~0.5m width and vehicles ~1.5-2m width)",
    )
    parser.add_argument(
        "--min-cluster-points",
        type=int,
        default=8,
        help="Minimum points per cluster. Lower values detect smaller objects like pedestrians. (default: 8, tuned to detect pedestrians)",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to process (default: 0)"
    )
    parser.add_argument(
        "--end-frame", type=int, default=None, help="Last frame to process (default: all)"
    )

    args = parser.parse_args()

    # Convert Windows path if needed
    pcd_folder = args.pcd_folder.replace("\\", os.sep)

    if not os.path.isdir(pcd_folder):
        print(f"Error: {pcd_folder} is not a valid directory")
        sys.exit(1)

    # Generate default output file path if not specified and output is enabled
    output_file = args.output
    if output_file is None and not args.no_output:
        # Put output in parent folder of PCD directory
        parent_folder = os.path.dirname(os.path.abspath(pcd_folder))
        pcd_folder_name = os.path.basename(os.path.abspath(pcd_folder))
        output_file = os.path.join(parent_folder, f"{pcd_folder_name}_detections.json")

    try:
        results = process_lidar_folder(
            pcd_folder=pcd_folder,
            output_file=output_file,
            num_bg_frames=args.num_bg_frames,
            bg_threshold=args.bg_threshold,
            cluster_eps=args.cluster_eps,
            min_cluster_points=args.min_cluster_points,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            show_visualisation=args.visualize,
        )

        if output_file is None:
            print("\nResults:")
            print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
