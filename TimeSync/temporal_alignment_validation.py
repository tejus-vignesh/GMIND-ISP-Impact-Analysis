"""
Multi-Camera Temporal Alignment and Validation Tool.

Interactive visualization tool for time synchronizing and validating 3D object detections
across multiple cameras. Helps align frame timestamps and validate geometric alignment
of objects detected from different camera viewpoints.

Key Features:
- Time synchronization: Adjust frame offsets to align cameras temporally
- Alignment validation: Match objects across cameras and compute alignment metrics
- Multi-camera visualization: Top-down view showing objects from all cameras
- Automatic optimization: Find optimal frame offsets that minimize alignment errors

Usage:
    # Multi-camera temporal alignment (main feature)
    python temporal_alignment_validation.py --plot_xy --annotation_dir "path/to/dir" --cameras FLIR3.2 FLIR8.9

    # Single camera 3D visualization
    python temporal_alignment_validation.py --annotation_file "path/to/anno.json"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib to use PySide6 backend before importing pyplot
import matplotlib
import numpy as np

matplotlib.use("QtAgg")  # QtAgg backend will use PySide6 if available

import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication


def load_coco_annotations(annotation_file: str) -> Dict:
    """
    Load COCO annotation file.

    Args:
        annotation_file: Path to the COCO JSON annotation file
                         Supports both Windows paths (H:\...) and WSL paths (/mnt/h/...)

    Returns:
        Dictionary containing the COCO annotation data
    """
    # Convert Windows path to WSL path if needed
    annotation_file = annotation_file.replace("\\", "/")
    if annotation_file.startswith("H:/") or annotation_file.startswith("H:\\"):
        annotation_file = "/mnt/h/" + annotation_file[3:]

    annotation_path = Path(annotation_file)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_path, "r") as f:
        coco_data = json.load(f)

    return coco_data


def extract_3d_locations(coco_data: Dict) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract 3D locations from COCO annotations.

    Args:
        coco_data: COCO annotation dictionary

    Returns:
        Tuple of (3d_points array, metadata list)
        - 3d_points: numpy array of shape (N, 3) with [X, Y, Z] coordinates
        - metadata: List of dictionaries with annotation metadata (category_id, track_id, image_id, etc.)
    """
    annotations = coco_data.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    images = {img["id"]: img for img in coco_data.get("images", [])}

    points_3d = []
    metadata = []

    for ann in annotations:
        if "location_3d" in ann and ann["location_3d"] is not None:
            location = ann["location_3d"]
            if len(location) == 3:
                points_3d.append(location)

                # Store metadata for this point
                meta = {
                    "category_id": ann.get("category_id", -1),
                    "category_name": categories.get(ann.get("category_id", -1), "unknown"),
                    "track_id": ann.get("track_id", -1),
                    "image_id": ann.get("image_id", -1),
                    "bbox": ann.get("bbox", []),
                    "annotation_id": ann.get("id", -1),
                }

                # Add image filename if available
                if meta["image_id"] in images:
                    meta["image_filename"] = images[meta["image_id"]].get("file_name", "")

                metadata.append(meta)

    if len(points_3d) == 0:
        print("Warning: No 3D locations found in the annotation file.")
        return np.array([]).reshape(0, 3), []

    return np.array(points_3d), metadata


def plot_3d_locations_interactive(
    points_3d: np.ndarray, metadata: List[Dict], title: str = "3D Object Locations"
):
    """
    Create an interactive 3D plot of the 3D locations.

    Args:
        points_3d: numpy array of shape (N, 3) with [X, Y, Z] coordinates
        metadata: List of dictionaries with annotation metadata
        title: Plot title
    """
    if len(points_3d) == 0:
        print("No points to plot.")
        return

    # Group points by category for color coding
    category_colors = {}
    unique_categories = set(m["category_id"] for m in metadata)
    color_map = plt.cm.get_cmap("tab10")
    for i, cat_id in enumerate(sorted(unique_categories)):
        category_colors[cat_id] = color_map(i / max(len(unique_categories), 1))

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points grouped by category
    plotted_categories = set()
    for i, (point, meta) in enumerate(zip(points_3d, metadata)):
        cat_id = meta["category_id"]
        color = category_colors.get(cat_id, "gray")
        label = None
        if cat_id not in plotted_categories:
            label = meta["category_name"]
            plotted_categories.add(cat_id)

        ax.scatter(point[0], point[1], point[2], c=[color], s=20, alpha=0.6, label=label)

    # Set labels and title
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_zlabel("Z (meters)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend
    if len(plotted_categories) > 0:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    # Set equal aspect ratio for metric space (1:1:1 scaling)
    # Calculate centroid of the data
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

    # Set range to up to 120m, centered on data centroid
    max_range = 60.0  # Half of 120m range

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set equal aspect ratio for all axes (metric space)
    ax.set_box_aspect([1, 1, 1])

    # Add grid
    ax.grid(True, alpha=0.3)

    # Plot origin point (camera location)
    ax.scatter([0], [0], [0], c="red", s=100, marker="*", label="Origin (Camera)", alpha=0.8)

    # Print statistics
    print(f"\n3D Location Statistics:")
    print(f"  Total points: {len(points_3d)}")
    print(f"  X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] meters")
    print(f"  Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] meters")
    print(f"  Z range: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] meters")

    # Calculate spread diagnostics
    x_std = points_3d[:, 0].std()
    y_std = points_3d[:, 1].std()
    z_std = points_3d[:, 2].std()
    print(f"\nStandard Deviations (spread):")
    print(f"  X std: {x_std:.2f} m")
    print(f"  Y std: {y_std:.2f} m")
    print(f"  Z std: {z_std:.2f} m")

    # Calculate distances from origin
    distances = np.linalg.norm(points_3d, axis=1)
    print(f"\nDistance from origin:")
    print(f"  Min: {distances.min():.2f} m")
    print(f"  Max: {distances.max():.2f} m")
    print(f"  Mean: {distances.mean():.2f} m")

    # Calculate angular spread (azimuth and elevation)
    # Assuming camera is at origin, calculate angles
    azimuths = np.arctan2(points_3d[:, 1], points_3d[:, 0]) * 180 / np.pi
    elevations = np.arcsin(points_3d[:, 2] / (distances + 1e-6)) * 180 / np.pi
    print(f"\nAngular Spread (from origin):")
    print(
        f"  Azimuth (XY plane): {azimuths.min():.1f}° to {azimuths.max():.1f}° (span: {azimuths.max() - azimuths.min():.1f}°)"
    )
    print(
        f"  Elevation (Z): {elevations.min():.1f}° to {elevations.max():.1f}° (span: {elevations.max() - elevations.min():.1f}°)"
    )

    # Check if points are collinear (low variance in one dimension suggests line)
    variances = np.var(points_3d, axis=0)
    min_var_idx = np.argmin(variances)
    var_ratio = variances.min() / variances.max()
    print(f"\nCollinearity Check:")
    print(f"  Variance ratio (min/max): {var_ratio:.4f}")
    if var_ratio < 0.01:
        print(
            f"  WARNING: Points appear to be collinear (low variance in {'XYZ'[min_var_idx]} dimension)"
        )
        print(f"  This suggests a projection issue - all points are along a line!")

    # Count by category
    category_counts = defaultdict(int)
    for meta in metadata:
        category_counts[meta["category_name"]] += 1

    print(f"\nPoints by category:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"  {cat_name}: {count}")

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_tracks_3d(points_3d: np.ndarray, metadata: List[Dict], title: str = "3D Object Tracks"):
    """
    Plot 3D locations with tracks connected (optional visualization).

    Args:
        points_3d: numpy array of shape (N, 3) with [X, Y, Z] coordinates
        metadata: List of dictionaries with annotation metadata
        title: Plot title
    """
    if len(points_3d) == 0:
        print("No points to plot.")
        return

    # Group points by track_id
    tracks = defaultdict(list)
    for i, (point, meta) in enumerate(zip(points_3d, metadata)):
        track_id = meta.get("track_id", -1)
        if track_id >= 0:
            tracks[track_id].append((point, meta))

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each track
    color_map = plt.cm.get_cmap("tab20")
    for track_idx, (track_id, track_points) in enumerate(tracks.items()):
        if len(track_points) < 2:
            continue  # Skip tracks with only one point

        # Sort by image_id to connect points in order
        track_points.sort(key=lambda x: x[1].get("image_id", 0))

        points = np.array([p[0] for p in track_points])
        color = color_map(track_idx / max(len(tracks), 1))

        # Plot track line
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.4, linewidth=1)

        # Plot track points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[color], s=30, alpha=0.7)

    # Set labels and title
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_zlabel("Z (meters)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set equal aspect ratio for metric space (1:1:1 scaling)
    # Calculate centroid of the data
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

    # Set range to up to 120m, centered on data centroid
    max_range = 60.0  # Half of 120m range

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set equal aspect ratio for all axes (metric space)
    ax.set_box_aspect([1, 1, 1])

    ax.grid(True, alpha=0.3)

    print(f"\nTrack Statistics:")
    print(f"  Total tracks: {len(tracks)}")
    print(f"  Total points: {len(points_3d)}")

    plt.tight_layout()
    plt.show()


def plot_3d_locations_over_time(
    points_3d: np.ndarray, metadata: List[Dict], title: str = "3D Object Locations Over Time"
):
    """
    Create an animated 3D plot showing objects appearing over time.

    Args:
        points_3d: numpy array of shape (N, 3) with [X, Y, Z] coordinates
        metadata: List of dictionaries with annotation metadata
        title: Plot title
    """
    if len(points_3d) == 0:
        print("No points to plot.")
        return

    # Group points by image_id (time/frame)
    points_by_frame = defaultdict(list)
    for i, (point, meta) in enumerate(zip(points_3d, metadata)):
        frame_id = meta.get("image_id", 0)
        points_by_frame[frame_id].append((point, meta))

    # Sort frames
    sorted_frames = sorted(points_by_frame.keys())

    if len(sorted_frames) == 0:
        print("No valid frames found.")
        return

    # Group points by category for color coding
    category_colors = {}
    unique_categories = set(m["category_id"] for m in metadata)
    color_map = plt.cm.get_cmap("tab10")
    for i, cat_id in enumerate(sorted(unique_categories)):
        category_colors[cat_id] = color_map(i / max(len(unique_categories), 1))

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Calculate axis limits
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    max_range = 60.0  # Half of 120m range

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_zlabel("Z (meters)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Store scatter plots for each category (will be created in animate)
    scatter_plots = {}

    # Text annotation for frame number
    frame_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Add legend
    legend_elements = []
    for cat_id in sorted(unique_categories):
        cat_name = next(
            (m["category_name"] for m in metadata if m["category_id"] == cat_id), "unknown"
        )
        color = category_colors[cat_id]
        legend_elements.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=cat_name
            )
        )
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    def animate(frame_idx):
        """Update plot for each frame."""
        # Clear previous scatter plots
        for scatter in scatter_plots.values():
            scatter.remove()
        scatter_plots.clear()

        # Get all points up to and including current frame
        if frame_idx >= len(sorted_frames):
            frame_idx = len(sorted_frames) - 1

        current_frame = sorted_frames[frame_idx]
        total_points = 0

        # Collect all points from frames 0 to current_frame
        for cat_id in unique_categories:
            x_coords = []
            y_coords = []
            z_coords = []

            for f_idx in range(frame_idx + 1):
                frame = sorted_frames[f_idx]
                for point, meta in points_by_frame[frame]:
                    if meta["category_id"] == cat_id:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                        z_coords.append(point[2])
                        total_points += 1

            # Plot points for this category if any exist
            if len(x_coords) > 0:
                color = category_colors[cat_id]
                scatter = ax.scatter(x_coords, y_coords, z_coords, c=[color], s=20, alpha=0.6)
                scatter_plots[cat_id] = scatter

        # Update frame text
        frame_text.set_text(
            f"Frame: {current_frame} / {sorted_frames[-1]}\n"
            f"Objects shown: {total_points}\n"
            f"(Press SPACE to pause/play)"
        )

        return list(scatter_plots.values()) + [frame_text]

    # Create animation
    # Show all frames, with interval in milliseconds (1000ms = 1 fps)
    animation = FuncAnimation(
        fig, animate, frames=len(sorted_frames), interval=1000, blit=False, repeat=True
    )

    # Pause state
    is_paused = False

    def on_key_press(event):
        """Handle key press events for pause/play."""
        nonlocal is_paused
        if event.key == " " or event.key == "space":  # Spacebar to pause/play
            if is_paused:
                animation.resume()
                is_paused = False
            else:
                animation.pause()
                is_paused = True
            fig.canvas.draw()

    # Connect keyboard event (must be after animation is created)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    print(f"\nTime-based Visualization:")
    print(f"  Total frames: {len(sorted_frames)}")
    print(f"  Frame range: {sorted_frames[0]} to {sorted_frames[-1]}")
    print(f"  Total points: {len(points_3d)}")
    print(f"\nControls:")
    print(f"  SPACEBAR - Pause/Play animation")

    plt.tight_layout()
    plt.show()

    return animation


def plot_xy_plane_over_time_multicam(
    points_3d_dict: Dict[str, Tuple[np.ndarray, List[Dict]]],
    title: str = "3D Object Locations (XY Plane - Top-Down View)",
    annotation_dir: Optional[str] = None,
    enable_video: bool = True,
):
    """
    Create a frame-by-frame 2D XY plot (top-down view) showing objects from multiple cameras.

    Args:
        points_3d_dict: Dictionary mapping camera names to (points_3d, metadata) tuples
        title: Plot title
    """
    if len(points_3d_dict) == 0:
        print("No camera data to plot.")
        return

    # Collect all points to determine axis limits
    all_points = []
    for points_3d, _ in points_3d_dict.values():
        if len(points_3d) > 0:
            all_points.append(points_3d)

    if len(all_points) == 0:
        print("No points to plot.")
        return

    all_points = np.vstack(all_points)

    # Group points by image_id (frame) for each camera
    points_by_frame_dict = {}
    sorted_frames_dict = {}

    for camera_name, (points_3d, metadata) in points_3d_dict.items():
        points_by_frame = defaultdict(list)
        for i, (point, meta) in enumerate(zip(points_3d, metadata)):
            frame_id = meta.get("image_id", 0)
            points_by_frame[frame_id].append((point, meta))
        points_by_frame_dict[camera_name] = points_by_frame
        sorted_frames_dict[camera_name] = sorted(points_by_frame.keys())

    # Load video files if requested (after sorted_frames_dict is created)
    video_captures = {}  # Map camera_name -> cv2.VideoCapture
    video_frame_mappings = {}  # Map camera_name -> dict mapping frame_id to video_frame_index
    video_enabled = enable_video

    def find_and_load_video(
        camera_name: str, annotation_dir: Optional[str]
    ) -> Optional[cv2.VideoCapture]:
        """Find and load video file for a camera."""
        if annotation_dir is None:
            return None

        annotation_dir_path = Path(annotation_dir)
        if not annotation_dir_path.exists():
            return None

        # Try to find video file matching camera name patterns
        video_patterns = [
            f"*{camera_name}*.mp4",
            f"*{camera_name}*.avi",
            f"*{camera_name.replace(' ', '')}*.mp4",
            f"*{camera_name.replace(' ', '')}*.avi",
            f"*{camera_name.replace(' ', '')}*.mkv",
        ]

        # Also try with common abbreviations
        if "FLIR 8.9MP" in camera_name or "FLIR8.9" in camera_name:
            video_patterns.extend(["*FLIR8.9*.mp4", "*FLIR8.9*.avi", "*8.9*.mp4"])
        if "FLIR 3.2MP" in camera_name or "FLIR3.2" in camera_name:
            video_patterns.extend(["*FLIR3.2*.mp4", "*FLIR3.2*.avi", "*3.2*.mp4"])

        for pattern in video_patterns:
            matching_files = list(annotation_dir_path.glob(pattern))
            if matching_files:
                video_path = matching_files[0]
                print(f"  Found video for {camera_name}: {video_path.name}")
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    return cap

        return None

    def build_video_frame_mapping(
        camera_name: str, sorted_frames: List[int], metadata: List[Dict]
    ) -> Dict[int, int]:
        """
        Build mapping from annotation frame_id to video frame index.
        Assumes frame_id corresponds directly to video frame number (0-indexed).
        """
        mapping = {}
        # Simple mapping: assume frame_id is the video frame index
        # Could be enhanced with actual video frame metadata if available
        for frame_id in sorted_frames:
            mapping[frame_id] = frame_id  # Direct mapping for now
        return mapping

    if video_enabled and annotation_dir is not None:
        print("\nLoading video files...")
        for camera_name in points_3d_dict.keys():
            cap = find_and_load_video(camera_name, annotation_dir)
            if cap is not None:
                video_captures[camera_name] = cap
                # Build frame mapping
                _, metadata = points_3d_dict[camera_name]
                sorted_frames = sorted_frames_dict[camera_name]
                video_frame_mappings[camera_name] = build_video_frame_mapping(
                    camera_name, sorted_frames, metadata
                )
                print(f"  ✓ Loaded video for {camera_name}")
            else:
                print(f"  ✗ No video found for {camera_name}")

        if len(video_captures) > 0:
            print(f"  Loaded {len(video_captures)} video file(s)")
            # Create OpenCV windows for video display
            for camera_name in video_captures.keys():
                cv2.namedWindow(f"Video - {camera_name}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"Video - {camera_name}", 800, 600)
        else:
            video_enabled = False
            print("  No videos loaded, video display disabled")
    else:
        video_enabled = False

    # Find common frame range across all cameras
    all_frame_ids = set()
    for sorted_frames in sorted_frames_dict.values():
        all_frame_ids.update(sorted_frames)

    if len(all_frame_ids) == 0:
        print("No valid frames found.")
        return

    sorted_frames = sorted(all_frame_ids)

    # Get unique categories across all cameras
    all_categories = set()
    for _, metadata in points_3d_dict.values():
        for meta in metadata:
            all_categories.add((meta["category_id"], meta["category_name"]))

    sorted_categories = sorted(all_categories, key=lambda x: x[0])

    # Map category names to marker symbols
    category_markers = {
        "person": "o",  # circle
        "pedestrian": "o",  # circle
        "people": "o",  # circle
        "bicycle": "s",  # square
        "car": "^",  # triangle up
        "vehicle": "^",  # triangle up
        "truck": "D",  # diamond
        "bus": "v",  # triangle down
    }

    # Create marker map for categories (fallback to 'o' if not found)
    category_marker_map = {}
    for cat_id, cat_name in sorted_categories:
        marker = category_markers.get(cat_name.lower(), "o")
        category_marker_map[cat_id] = marker

    # Create color map for cameras (fixed colors)
    camera_colors = {
        "FLIR 8.9MP": "green",
        "FLIR 3.2MP": "orange",
        "CCTV": "blue",
        "Hikvision Thermal": "purple",
    }

    # Default color for unknown cameras
    default_camera_color = "gray"

    camera_names_sorted = sorted(points_3d_dict.keys())

    # Create figure and 2D axis (XY plane - top-down view)
    fig, ax = plt.subplots(figsize=(14, 12))

    # Calculate axis limits
    margin = 5.0  # meters
    x_min = all_points[:, 0].min() - margin
    x_max = all_points[:, 0].max() + margin
    y_min = all_points[:, 1].min() - margin
    y_max = all_points[:, 1].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.set_xlabel("X (meters) - Right", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y (meters) - Forward", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Plot origin (camera location)
    ax.scatter(
        [0],
        [0],
        c="red",
        s=200,
        marker="*",
        label="Origin (Camera)",
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )

    # Store scatter plots (will be recreated each frame)
    scatter_plots = []
    # Store connection lines for matched objects (will be recreated each frame)
    connection_lines = []

    # Frame number display (large, prominent)
    frame_text = ax.text(
        0.5,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.9, edgecolor="black", linewidth=2),
    )

    # Info text for object counts
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    # Alignment statistics text (top right) - created here, updated in update_frame
    alignment_text_obj = ax.text(
        0.98,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="black"),
    )

    # Alignment region rectangle (drawn once, stays visible)
    alignment_region_x_min = -10.0
    alignment_region_x_max = 20.0
    alignment_region_y_min = 0.0
    alignment_region_y_max = 20.0
    region_rect = plt.Rectangle(
        (alignment_region_x_min, alignment_region_y_min),
        alignment_region_x_max - alignment_region_x_min,
        alignment_region_y_max - alignment_region_y_min,
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )
    ax.add_patch(region_rect)

    # Static object exclusion settings
    exclude_static_objects = False  # Set to True to exclude static objects from alignment metrics
    static_object_threshold_meters = (
        2.0  # Maximum position variance for an object to be considered static
    )

    # Identify static objects by analyzing track variance across all frames
    def identify_static_objects(points_by_frame_dict, threshold=2.0):
        """
        Identify static objects by computing position variance across tracks.
        Returns a set of (camera_name, track_id) tuples that are static.
        """
        static_tracks = set()

        for camera_name, points_by_frame in points_by_frame_dict.items():
            # Group points by track_id
            tracks_by_id = defaultdict(list)
            for frame_id, frame_points in points_by_frame.items():
                for point, meta in frame_points:
                    track_id = meta.get("track_id", -1)
                    if track_id >= 0:  # Valid track ID
                        tracks_by_id[track_id].append((point, meta))

            # For each track, compute position variance
            for track_id, track_points in tracks_by_id.items():
                if len(track_points) < 2:  # Need at least 2 points to compute variance
                    continue

                # Extract XY positions
                positions = np.array([(p[0][0], p[0][1]) for p in track_points])

                # Compute variance in X and Y
                var_x = np.var(positions[:, 0])
                var_y = np.var(positions[:, 1])

                # Use maximum variance as the measure (if either dimension moves, object is not static)
                max_variance = max(var_x, var_y)

                # Convert variance to std dev in meters (assuming positions are in meters)
                std_dev = np.sqrt(max_variance)

                # If standard deviation is below threshold, mark as static
                if std_dev < threshold:
                    static_tracks.add((camera_name, track_id))

        return static_tracks

    # Identify static objects once at initialization
    static_object_tracks = identify_static_objects(
        points_by_frame_dict, static_object_threshold_meters
    )

    if exclude_static_objects and len(static_object_tracks) > 0:
        print(
            f"   🔍 Identified {len(static_object_tracks)} static object tracks (threshold: {static_object_threshold_meters}m)"
        )
        print(f"      (Excluding from alignment metrics)")

    # Current frame index per camera (start at first frame for each)
    camera_frame_indices = {}  # Map camera_name -> current frame index in sorted_frames_dict
    camera_frame_offsets = {}  # Map camera_name -> frame offset (for time synchronization)

    for camera_name in camera_names_sorted:
        sorted_frames_cam = sorted_frames_dict[camera_name]
        if len(sorted_frames_cam) > 0:
            camera_frame_indices[camera_name] = 0  # Start at first frame
            camera_frame_offsets[camera_name] = 0  # No offset initially
        else:
            camera_frame_indices[camera_name] = 0
            camera_frame_offsets[camera_name] = 0

    # Active camera for controls (first camera by default)
    active_camera_idx = [0]  # Index into camera_names_sorted

    # Track single alignment plot figure (reused)
    alignment_plot_figure = None

    def compute_alignment_metrics_from_camera_points(
        camera_points_internal, max_neighbor_dist=8.0, exclude_static=False
    ):
        """
        Compute alignment metrics given a camera_points dictionary.
        Returns (alignment_stats_dict, matched_pairs_list) or (None, [])

        Args:
            camera_points_internal: Dictionary mapping camera names to category->point lists
            max_neighbor_dist: Maximum distance for matching objects
            exclude_static: If True, exclude objects identified as static
        """
        alignment_stats_result = None
        matched_pairs_result = []

        if "FLIR 8.9MP" not in camera_points_internal:
            return alignment_stats_result, matched_pairs_result

        # Collect all FLIR 8.9MP points with metadata in region
        flir89_points = []
        for cat_id in camera_points_internal["FLIR 8.9MP"]:
            for point, meta in camera_points_internal["FLIR 8.9MP"][cat_id]:
                x, y = point[0], point[1]

                # Filter by region
                if not (
                    alignment_region_x_min <= x <= alignment_region_x_max
                    and alignment_region_y_min <= y <= alignment_region_y_max
                ):
                    continue

                # Exclude static objects if requested
                if exclude_static:
                    track_id = meta.get("track_id", -1)
                    if track_id >= 0 and ("FLIR 8.9MP", track_id) in static_object_tracks:
                        continue  # Skip this static object

                flir89_points.append(
                    {
                        "point": np.array([x, y]),
                        "cat_id": cat_id,
                        "cat_name": meta.get("category_name", "unknown"),
                        "meta": meta,
                    }
                )

        # Collect all points from other cameras
        other_camera_points = {}
        for camera_name in camera_names_sorted:
            if camera_name != "FLIR 8.9MP" and camera_name in camera_points_internal:
                other_camera_points[camera_name] = []
                for cat_id in camera_points_internal[camera_name]:
                    for point, meta in camera_points_internal[camera_name][cat_id]:
                        x, y = point[0], point[1]

                        # Exclude static objects if requested
                        if exclude_static:
                            track_id = meta.get("track_id", -1)
                            if track_id >= 0 and (camera_name, track_id) in static_object_tracks:
                                continue  # Skip this static object

                        other_camera_points[camera_name].append(
                            {
                                "point": np.array([x, y]),
                                "cat_id": cat_id,
                                "cat_name": meta.get("category_name", "unknown"),
                                "meta": meta,
                            }
                        )

        # Find matches with one-to-one constraint (same logic as update_frame)
        neighbor_distances = []
        if len(flir89_points) > 0 and len(other_camera_points) > 0:
            potential_matches = []

            for i, flir_obj in enumerate(flir89_points):
                flir_point = flir_obj["point"]
                flir_cat_id = flir_obj["cat_id"]

                if flir_cat_id is None:
                    continue

                for camera_name, other_points in other_camera_points.items():
                    for j, other_obj in enumerate(other_points):
                        if other_obj["cat_id"] is not None and other_obj["cat_id"] == flir_cat_id:
                            other_point = other_obj["point"]
                            distance = np.linalg.norm(flir_point - other_point)
                            if distance <= max_neighbor_dist:
                                potential_matches.append(
                                    {
                                        "flir89_idx": i,
                                        "flir89_point": flir_point,
                                        "flir89_obj": flir_obj,
                                        "other_camera": camera_name,
                                        "other_idx": j,
                                        "other_point": other_point,
                                        "other_obj": other_obj,
                                        "distance": distance,
                                        "cat_id": flir_cat_id,
                                    }
                                )

            potential_matches.sort(key=lambda x: x["distance"])

            matched_flir89_indices = set()
            matched_other_keys = set()

            for match in potential_matches:
                flir_idx = match["flir89_idx"]
                other_key = (match["other_camera"], match["other_idx"])

                if flir_idx in matched_flir89_indices or other_key in matched_other_keys:
                    continue

                matched_flir89_indices.add(flir_idx)
                matched_other_keys.add(other_key)
                neighbor_distances.append(match["distance"])
                matched_pairs_result.append(
                    {
                        "flir89_point": match["flir89_point"],
                        "flir89_obj": match["flir89_obj"],
                        "matched_point": match["other_point"],
                        "matched_obj": match["other_obj"],
                        "distance": match["distance"],
                        "camera": match["other_camera"],
                        "cat_id": match["cat_id"],
                    }
                )

        # Compute statistics
        if len(neighbor_distances) > 0:
            neighbor_distances = np.array(neighbor_distances)

            # Calculate trimmed mean (trim 20% from both ends, i.e., 10% from each end)
            trimmed_mean = None
            if len(neighbor_distances) >= 5:  # Need at least 5 values to trim meaningfully
                sorted_distances = np.sort(neighbor_distances)
                trim_count = max(1, int(len(sorted_distances) * 0.1))  # Trim 10% from each end
                trimmed_distances = (
                    sorted_distances[trim_count:-trim_count] if trim_count > 0 else sorted_distances
                )
                if len(trimmed_distances) > 0:
                    trimmed_mean = np.mean(trimmed_distances)

            alignment_stats_result = {
                "count": len(neighbor_distances),
                "mean": np.mean(neighbor_distances),
                "trimmed_mean": trimmed_mean,
                "median": np.median(neighbor_distances),
                "std": np.std(neighbor_distances),
                "p5": np.percentile(neighbor_distances, 5),
                "p25": np.percentile(neighbor_distances, 25),
                "p75": np.percentile(neighbor_distances, 75),
                "p95": np.percentile(neighbor_distances, 95),
                "min": np.min(neighbor_distances),
                "max": np.max(neighbor_distances),
            }

        return alignment_stats_result, matched_pairs_result

    def update_video_displays(camera_points_current: Dict, matched_pairs_current: List[Dict]):
        """Update video display windows with current frames and bounding boxes."""
        if not video_enabled or len(video_captures) == 0:
            return

        # Build a mapping of matched objects with unique match IDs and colors
        # Map (camera_name, track_id or annotation_id) -> (match_id, match_color)
        matched_objects = {}  # Map (camera_name, 'track'/'ann', id) -> (match_id, color)
        match_colors = [
            (0, 255, 255),  # Yellow (BGR) - high visibility
            (255, 255, 0),  # Cyan (BGR)
            (255, 0, 255),  # Magenta (BGR)
            (0, 165, 255),  # Orange (BGR)
            (128, 0, 255),  # Purple (BGR)
            (0, 255, 128),  # Green-yellow (BGR)
            (255, 192, 203),  # Pink (BGR)
            (255, 20, 147),  # Deep pink (BGR)
        ]
        next_match_id = 0

        # Assign unique match IDs and colors to each matched pair
        for match_pair in matched_pairs_current:
            match_id = next_match_id
            match_color = match_colors[next_match_id % len(match_colors)]
            next_match_id += 1

            # Store FLIR 8.9MP object
            flir89_obj = match_pair.get("flir89_obj", {})
            if flir89_obj:
                meta = flir89_obj.get("meta", {})
                if not meta:
                    meta = flir89_obj
                track_id = meta.get("track_id", -1)
                ann_id = meta.get("annotation_id", -1)
                if track_id >= 0:
                    matched_objects[("FLIR 8.9MP", "track", track_id)] = (match_id, match_color)
                elif ann_id >= 0:
                    matched_objects[("FLIR 8.9MP", "ann", ann_id)] = (match_id, match_color)

            # Store matched object from other camera (same match_id and color)
            matched_obj = match_pair.get("matched_obj", {})
            other_camera = match_pair.get("camera", "")
            if matched_obj:
                meta = matched_obj.get("meta", {})
                if not meta:
                    meta = matched_obj
                track_id = meta.get("track_id", -1)
                ann_id = meta.get("annotation_id", -1)
                if track_id >= 0:
                    matched_objects[(other_camera, "track", track_id)] = (match_id, match_color)
                elif ann_id >= 0:
                    matched_objects[(other_camera, "ann", ann_id)] = (match_id, match_color)

        # Update video for each camera
        for camera_name, cap in video_captures.items():
            if camera_name not in camera_frame_indices:
                continue

            sorted_frames_cam = sorted_frames_dict[camera_name]
            if len(sorted_frames_cam) == 0:
                continue

            frame_idx = camera_frame_indices[camera_name]
            if frame_idx < 0 or frame_idx >= len(sorted_frames_cam):
                continue

            base_frame_id = sorted_frames_cam[frame_idx]
            offset = camera_frame_offsets[camera_name]
            target_frame_id = base_frame_id + offset

            # Get video frame index
            video_frame_idx = video_frame_mappings[camera_name].get(
                target_frame_id, target_frame_id
            )

            # Seek to the correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Draw bounding boxes for current frame objects
            if camera_name in camera_points_current:
                camera_color_bgr = {
                    "FLIR 8.9MP": (0, 255, 0),  # Green
                    "FLIR 3.2MP": (0, 165, 255),  # Orange (BGR)
                    "CCTV": (255, 0, 0),  # Blue
                    "Hikvision Thermal": (128, 0, 128),  # Purple
                }.get(
                    camera_name, (128, 128, 128)
                )  # Gray default

                # Get objects for this camera in current frame
                for cat_id, objects in camera_points_current[camera_name].items():
                    for point, meta in objects:
                        bbox = meta.get("bbox", [])
                        if len(bbox) >= 4:
                            # COCO format: [x, y, width, height] (top-left corner)
                            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                            # Check if this object is matched
                            track_id = meta.get("track_id", -1)
                            ann_id = meta.get("annotation_id", -1)
                            matched = False
                            match_id = None
                            match_color = None
                            box_color = camera_color_bgr
                            line_thickness = 2

                            # Check if this object is in the matched set
                            if (
                                track_id >= 0
                                and (camera_name, "track", track_id) in matched_objects
                            ):
                                matched = True
                                match_id, match_color = matched_objects[
                                    (camera_name, "track", track_id)
                                ]
                            elif ann_id >= 0 and (camera_name, "ann", ann_id) in matched_objects:
                                matched = True
                                match_id, match_color = matched_objects[
                                    (camera_name, "ann", ann_id)
                                ]

                            if matched:
                                # Use match color for matched objects (thicker line)
                                box_color = match_color
                                line_thickness = 4

                            # Draw bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, line_thickness)

                            # Draw footpoint (bottom center of bounding box) - this is where the 3D point is projected from
                            footpoint_x = x + w // 2
                            footpoint_y = y + h

                            # Draw footpoint marker (circle) - more prominent for matched objects
                            footpoint_radius = 8 if matched else 5
                            footpoint_thickness = -1 if matched else 2
                            footpoint_color = match_color if matched else box_color
                            cv2.circle(
                                frame,
                                (footpoint_x, footpoint_y),
                                footpoint_radius,
                                footpoint_color,
                                footpoint_thickness,
                            )

                            # Draw a line from bottom center to bottom of frame (or extend downward) for matched objects
                            if matched:
                                # Draw line extending downward from footpoint
                                line_length = 30
                                line_end_y = min(footpoint_y + line_length, frame.shape[0] - 10)
                                cv2.line(
                                    frame,
                                    (footpoint_x, footpoint_y),
                                    (footpoint_x, line_end_y),
                                    match_color,
                                    3,
                                )

                                # Draw match ID number below the line
                                match_id_text = f"M{match_id}"
                                text_size, _ = cv2.getTextSize(
                                    match_id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                                )
                                text_x = footpoint_x - text_size[0] // 2
                                text_y = line_end_y + text_size[1] + 5
                                # Draw text background
                                cv2.rectangle(
                                    frame,
                                    (text_x - 3, text_y - text_size[1] - 3),
                                    (text_x + text_size[0] + 3, text_y + 3),
                                    match_color,
                                    -1,
                                )
                                cv2.putText(
                                    frame,
                                    match_id_text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2,
                                )

                            # Draw label
                            cat_name = meta.get("category_name", "unknown")
                            track_id_label = meta.get("track_id", -1)
                            label = f"{cat_name}"
                            if track_id_label >= 0:
                                label += f" ID:{track_id_label}"
                            if matched:
                                label += f" [M{match_id}]"  # Add match ID to label

                            # Draw label background
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(
                                frame,
                                (x, y - label_size[1] - 5),
                                (x + label_size[0], y),
                                box_color,
                                -1,
                            )
                            cv2.putText(
                                frame,
                                label,
                                (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2,
                            )

                # Add frame info text
                info_text_video = f"Frame: {target_frame_id}"
                if offset != 0:
                    info_text_video += f" (offset: {offset:+d})"
                cv2.putText(
                    frame,
                    info_text_video,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

            # Display frame
            cv2.imshow(f"Video - {camera_name}", frame)

        # Process OpenCV events (needed for window updates)
        cv2.waitKey(1)

    def update_frame():
        """Update plot with current frames for each camera."""
        # Define max_neighbor_distance here so it's accessible throughout the function
        max_neighbor_distance = (
            8.0  # 8m hard cutoff - objects further apart are not considered matches
        )

        # Clear previous scatter plots
        for scatter in scatter_plots:
            scatter.remove()
        scatter_plots.clear()
        # Clear previous connection lines
        for line in connection_lines:
            line.remove()
        connection_lines.clear()

        # Collect points for current frame for each camera
        camera_points = {}
        total_points = 0
        frame_info = []

        for camera_name in camera_names_sorted:
            sorted_frames_cam = sorted_frames_dict[camera_name]
            if len(sorted_frames_cam) == 0:
                continue

            frame_idx = camera_frame_indices[camera_name]

            # Apply bounds
            if frame_idx < 0:
                frame_idx = 0
            elif frame_idx >= len(sorted_frames_cam):
                frame_idx = len(sorted_frames_cam) - 1

            camera_frame_indices[camera_name] = frame_idx

            # Get frame ID with offset applied
            base_frame_id = sorted_frames_cam[frame_idx]
            offset = camera_frame_offsets[camera_name]
            target_frame_id = base_frame_id + offset

            # Look for the target frame (or closest available)
            points_by_frame = points_by_frame_dict[camera_name]

            # Try exact match first
            actual_frame_id = target_frame_id
            if target_frame_id not in points_by_frame:
                # Find closest available frame
                available_frames = sorted(points_by_frame.keys())
                if available_frames:
                    actual_frame_id = min(available_frames, key=lambda x: abs(x - target_frame_id))

            camera_points[camera_name] = defaultdict(list)

            if actual_frame_id in points_by_frame:
                for point, meta in points_by_frame[actual_frame_id]:
                    cat_id = meta["category_id"]
                    camera_points[camera_name][cat_id].append((point, meta))
                    total_points += 1

            # Store frame info for display (show actual frame ID, not offset)
            frame_info.append(f"{camera_name}: {actual_frame_id}")

        # Compute alignment metrics (FLIR 8.9MP objects in region vs other cameras)
        alignment_stats, matched_pairs = compute_alignment_metrics_from_camera_points(
            camera_points, max_neighbor_distance, exclude_static=exclude_static_objects
        )

        if False:  # Original inline code replaced with function call above
            # Collect all FLIR 8.9MP points with metadata
            flir89_points = []
            for cat_id in camera_points["FLIR 8.9MP"]:
                for point, meta in camera_points["FLIR 8.9MP"][cat_id]:
                    x, y, z = point[0], point[1], point[2] if len(point) > 2 else 0.0
                    # Filter by region
                    if (
                        alignment_region_x_min <= x <= alignment_region_x_max
                        and alignment_region_y_min <= y <= alignment_region_y_max
                    ):
                        flir89_points.append(
                            {
                                "point": np.array([x, y]),
                                "cat_id": cat_id,
                                "cat_name": meta.get("category_name", "unknown"),
                                "meta": meta,
                            }
                        )

            # Collect all points from other cameras
            other_camera_points = {}
            for camera_name in camera_names_sorted:
                if camera_name != "FLIR 8.9MP" and camera_name in camera_points:
                    other_camera_points[camera_name] = []
                    for cat_id in camera_points[camera_name]:
                        for point, meta in camera_points[camera_name][cat_id]:
                            x, y = point[0], point[1]
                            other_camera_points[camera_name].append(
                                {
                                    "point": np.array([x, y]),
                                    "cat_id": cat_id,
                                    "cat_name": meta.get("category_name", "unknown"),
                                    "meta": meta,
                                }
                            )

            # For each FLIR 8.9MP point, find closest neighbor of same class (category_id)
            # Enforce one-to-one matching: each object can only be matched once
            neighbor_distances = []

            if len(flir89_points) > 0 and len(other_camera_points) > 0:
                # Build a list of all potential matches with distances, then sort by distance
                # This ensures we match closest pairs first (greedy optimal matching)
                potential_matches = []

                for i, flir_obj in enumerate(flir89_points):
                    flir_point = flir_obj["point"]
                    flir_cat_id = flir_obj["cat_id"]

                    # Skip if no category ID (can't match without class information)
                    if flir_cat_id is None:
                        continue

                    # Search in all other cameras for objects of the SAME class only
                    for camera_name, other_points in other_camera_points.items():
                        for j, other_obj in enumerate(other_points):
                            # Only consider objects with the same category_id (class)
                            if (
                                other_obj["cat_id"] is not None
                                and other_obj["cat_id"] == flir_cat_id
                            ):
                                other_point = other_obj["point"]
                                distance = np.linalg.norm(flir_point - other_point)

                                # Only consider matches within cutoff distance
                                if distance <= max_neighbor_distance:
                                    potential_matches.append(
                                        {
                                            "flir89_idx": i,
                                            "flir89_point": flir_point,
                                            "flir89_obj": flir_obj,
                                            "other_camera": camera_name,
                                            "other_idx": j,
                                            "other_point": other_point,
                                            "other_obj": other_obj,
                                            "distance": distance,
                                            "cat_id": flir_cat_id,
                                        }
                                    )

                # Sort potential matches by distance (closest first)
                potential_matches.sort(key=lambda x: x["distance"])

                # Track which objects have been matched (one-to-one constraint)
                matched_flir89_indices = set()  # Track which FLIR 8.9MP objects are matched
                matched_other_keys = (
                    set()
                )  # Track which other camera objects are matched: (camera_name, index)

                # Greedily match closest pairs, ensuring strict one-to-one constraint
                for match in potential_matches:
                    flir_idx = match["flir89_idx"]
                    other_key = (match["other_camera"], match["other_idx"])

                    # Skip if either object is already matched (enforce one-to-one bidirectionally)
                    if flir_idx in matched_flir89_indices:
                        continue
                    if other_key in matched_other_keys:
                        continue

                    # This is a valid one-to-one match - mark both as matched
                    matched_flir89_indices.add(flir_idx)
                    matched_other_keys.add(other_key)

                    # Record the match
                    neighbor_distances.append(match["distance"])
                    matched_pairs.append(
                        {
                            "flir89_point": match["flir89_point"],
                            "matched_point": match["other_point"],
                            "distance": match["distance"],
                            "camera": match["other_camera"],
                            "cat_id": match["cat_id"],
                        }
                    )

            # Compute statistics
            if len(neighbor_distances) > 0:
                neighbor_distances = np.array(neighbor_distances)
                alignment_stats = {
                    "count": len(neighbor_distances),
                    "mean": np.mean(neighbor_distances),
                    "median": np.median(neighbor_distances),
                    "std": np.std(neighbor_distances),
                    "p5": np.percentile(neighbor_distances, 5),
                    "p25": np.percentile(neighbor_distances, 25),
                    "p75": np.percentile(neighbor_distances, 75),
                    "p95": np.percentile(neighbor_distances, 95),
                    "min": np.min(neighbor_distances),
                    "max": np.max(neighbor_distances),
                }

        # Plot points for current frames
        for camera_name in camera_names_sorted:
            if camera_name not in camera_points:
                continue

            cam_color = camera_colors.get(camera_name, default_camera_color)

            for cat_id, cat_name in sorted_categories:
                if (
                    cat_id in camera_points[camera_name]
                    and len(camera_points[camera_name][cat_id]) > 0
                ):
                    points_and_meta = camera_points[camera_name][cat_id]
                    points = np.array([p[0] for p in points_and_meta])
                    marker = category_marker_map[cat_id]

                    scatter = ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=cam_color,
                        marker=marker,
                        s=80,
                        alpha=0.8,
                        edgecolors="black",
                        linewidths=0.5,
                        label=None,
                    )

                    scatter_plots.append(scatter)

        # Draw connection lines for matched pairs
        for pair in matched_pairs:
            flir89_pt = pair["flir89_point"]
            matched_pt = pair["matched_point"]
            distance = pair["distance"]

            # Color based on distance (green for close, yellow/orange for far)
            # Normalize distance to 0-1 range (adaptive to max_neighbor_distance)
            normalized_dist = min(distance / max_neighbor_distance, 1.0)
            if normalized_dist < 0.5:
                line_color = "green"  # < 50% of max distance
                line_alpha = 0.6
            elif normalized_dist < 0.75:
                line_color = "yellow"  # 50-75% of max distance
                line_alpha = 0.5
            else:
                line_color = "orange"  # >= 75% of max distance
                line_alpha = 0.4

            # Draw line connecting the matched points
            line = ax.plot(
                [flir89_pt[0], matched_pt[0]],
                [flir89_pt[1], matched_pt[1]],
                color=line_color,
                alpha=line_alpha,
                linewidth=1.5,
                linestyle="--",
                zorder=0,
            )
            connection_lines.extend(line)

        # Update frame text (show all cameras)
        frame_text_lines = []
        for info in frame_info:
            frame_text_lines.append(info)
        frame_text.set_text("\n".join(frame_text_lines))

        # Update alignment statistics text (top right)
        if alignment_stats:
            stats_lines = [
                "Alignment Metrics (FLIR8.9MP)",
                f"Matched objects: {alignment_stats['count']}",
                f"Median: {alignment_stats['median']:.2f}m",
                f"Mean: {alignment_stats['mean']:.2f}m",
                f"Mean: {alignment_stats['mean']:.2f}m",
                f"95th pct: {alignment_stats['p95']:.2f}m",
                f"Std: {alignment_stats['std']:.2f}m",
            ]
            alignment_text_obj.set_bbox(
                dict(boxstyle="round", facecolor="lightgreen", alpha=0.9, edgecolor="black")
            )
            alignment_text_obj.set_text("\n".join(stats_lines))
        else:
            alignment_text_obj.set_bbox(
                dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="black")
            )
            alignment_text_obj.set_text("Alignment: No matches\nin region")

        # Update info text
        active_cam = camera_names_sorted[active_camera_idx[0]]
        sorted_frames_cam = sorted_frames_dict[active_cam]
        max_frame_cam = sorted_frames_cam[-1] if sorted_frames_cam else 0

        info_lines = [
            f"Objects in frame: {total_points}",
            f"Active camera: {active_cam}",
            f"Frame range: 0-{max_frame_cam}",
            f"Offset: {camera_frame_offsets[active_cam]:+d}",
        ]
        info_text.set_text("\n".join(info_lines))

        # Update video displays
        update_video_displays(camera_points, matched_pairs)

        fig.canvas.draw_idle()

    # Add legend
    legend_elements = []
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=2,
            label="Origin (Camera)",
            linestyle="None",
        )
    )

    # Add camera entries to legend
    for cam_name in camera_names_sorted:
        color = camera_colors.get(cam_name, default_camera_color)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
                label=f"{cam_name}",
                linestyle="None",
            )
        )

    # Add category entries to legend (with their symbols)
    for cat_id, cat_name in sorted_categories:
        marker = category_marker_map[cat_id]
        # Use a neutral gray color for legend (actual color comes from camera)
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
                label=f"{cat_name}",
                linestyle="None",
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    # Key press handler for frame navigation (per-camera with offset support)
    def on_key_press(event):
        """Handle key press events for frame navigation."""
        active_cam = camera_names_sorted[active_camera_idx[0]]
        sorted_frames_cam = sorted_frames_dict[active_cam]

        if len(sorted_frames_cam) == 0:
            return

        # Camera selection
        if event.key == "tab":
            # Switch active camera
            active_camera_idx[0] = (active_camera_idx[0] + 1) % len(camera_names_sorted)
            update_frame()
            return

        # Frame navigation for active camera
        if event.key == "right" or event.key == "n" or event.key == "next":
            # Next frame (active camera)
            camera_frame_indices[active_cam] += 1
            update_frame()
        elif event.key == "left" or event.key == "p" or event.key == "prev":
            # Previous frame (active camera)
            camera_frame_indices[active_cam] -= 1
            update_frame()
        elif event.key == "home":
            # First frame (active camera)
            camera_frame_indices[active_cam] = 0
            update_frame()
        elif event.key == "end":
            # Last frame (active camera)
            camera_frame_indices[active_cam] = len(sorted_frames_cam) - 1
            update_frame()

        # Offset adjustment (active camera)
        elif event.key == "up" or event.key == "u":
            # Increase offset
            camera_frame_offsets[active_cam] += 1
            update_frame()
        elif event.key == "down" or event.key == "d":
            # Decrease offset
            camera_frame_offsets[active_cam] -= 1
            update_frame()
        elif event.key == "r" or event.key == "R":
            # Reset offset to 0
            camera_frame_offsets[active_cam] = 0
            update_frame()

        # All cameras navigation (use uppercase for all cameras)
        elif event.key == "N":
            # Next frame for all cameras
            for cam_name in camera_names_sorted:
                sorted_frames_cam2 = sorted_frames_dict[cam_name]
                if len(sorted_frames_cam2) > 0:
                    camera_frame_indices[cam_name] += 1
            update_frame()
        elif event.key == "P":
            # Previous frame for all cameras
            for cam_name in camera_names_sorted:
                sorted_frames_cam2 = sorted_frames_dict[cam_name]
                if len(sorted_frames_cam2) > 0:
                    camera_frame_indices[cam_name] -= 1
            update_frame()

        # Jump to percentage
        elif event.key.isdigit():
            digit = int(event.key)
            target_idx = int(digit / 9.0 * (len(sorted_frames_cam) - 1))
            camera_frame_indices[active_cam] = target_idx
            update_frame()

        # Toggle static object exclusion
        elif event.key == "m" or event.key == "M":
            nonlocal exclude_static_objects
            exclude_static_objects = not exclude_static_objects
            status_msg = "ENABLED" if exclude_static_objects else "DISABLED"
            print(f"\n{'='*60}")
            print(f"   🔄 Static object exclusion: {status_msg}")
            if exclude_static_objects:
                print(f"      Threshold: {static_object_threshold_meters}m")
                print(f"      Excluding {len(static_object_tracks)} static tracks")
            print(f"{'='*60}\n")
            # Trigger frame update to recompute metrics
            update_frame()
            fig.canvas.draw()

        # Find best alignment (search +/- 100 frames for lowest median)
        # FLIR 8.9MP is always the reference camera (offset unchanged)
        # Search adjusts the active camera's offset (or all other cameras if FLIR 8.9MP is active)
        elif event.key == "f" or event.key == "F":
            # Check if FLIR 8.9MP is available as reference
            if "FLIR 8.9MP" not in camera_names_sorted:
                print("\n⚠️  FLIR 8.9MP not found. Cannot perform alignment search.")
                return

            # Determine which camera(s) to search
            if active_cam == "FLIR 8.9MP":
                # If FLIR 8.9MP is active, search all other cameras
                cameras_to_search = [cam for cam in camera_names_sorted if cam != "FLIR 8.9MP"]
                if len(cameras_to_search) == 0:
                    print("\n⚠️  No other cameras to search.")
                    return
                print(
                    f"\n🔍 Searching for best alignment (FLIR 8.9MP = reference, adjusting: {', '.join(cameras_to_search)})..."
                )
                search_camera = cameras_to_search[0]  # Search first other camera
            else:
                # Search the active camera (FLIR 8.9MP stays fixed)
                search_camera = active_cam
                print(
                    f"\n🔍 Searching for best alignment (FLIR 8.9MP = reference, adjusting: {search_camera})..."
                )

            print(f"   Current {search_camera} offset: {camera_frame_offsets[search_camera]:+d}")

            # Store original offset
            original_offset = camera_frame_offsets[search_camera]
            current_base_frame_idx = camera_frame_indices[search_camera]

            # Determine valid search range based on available frames
            # We need to ensure the resulting frame indices don't go out of bounds
            search_camera_sorted_frames = sorted_frames_dict[search_camera]
            max_search_range = 50  # Search +/- 100 frames

            if len(search_camera_sorted_frames) == 0:
                print(f"   ⚠️  No frames available for {search_camera}")
                return

            # Calculate how far we can search backward and forward
            # Backward: limited by how many frames we have before the start (can't go below index 0)
            max_backward = min(max_search_range, current_base_frame_idx)
            # Forward: limited by how many frames we have after current position
            max_forward = min(
                max_search_range, len(search_camera_sorted_frames) - 1 - current_base_frame_idx
            )

            # If we're at the beginning (low index), only search forward
            # If we're near the end, only search backward
            # Otherwise, search both directions
            if current_base_frame_idx < max_search_range:
                # At the beginning - only search forward
                search_range = range(0, max_forward + 1)
                print(
                    f"   At beginning (index {current_base_frame_idx}), searching forward: 0 to +{max_forward} frames"
                )
            elif current_base_frame_idx >= len(search_camera_sorted_frames) - max_search_range:
                # Near the end - only search backward
                search_range = range(-max_backward, 1)
                print(
                    f"   Near end (index {current_base_frame_idx}), searching backward: -{max_backward} to 0 frames"
                )
            else:
                # In the middle - search both directions
                search_range = range(-max_backward, max_forward + 1)
                print(
                    f"   Searching range: -{max_backward} to +{max_forward} frames (from index {current_base_frame_idx})"
                )

            if len(search_range) == 0:
                print(f"   ⚠️  Cannot search: already at frame boundary")
                return

            best_offset = original_offset
            best_metric_value = float("inf")
            best_metric_type = None

            frame_results = []
            offset_values = []
            median_values = []
            match_counts = []  # Track number of matches per offset

            for offset_delta in search_range:
                test_offset = original_offset + offset_delta

                # Build camera_points for this test offset
                # FLIR 8.9MP offset stays unchanged (reference), search_camera offset is adjusted
                test_camera_points = {}
                for camera_name in camera_names_sorted:
                    sorted_frames_cam_test = sorted_frames_dict[camera_name]
                    if len(sorted_frames_cam_test) == 0:
                        continue

                    frame_idx_test = camera_frame_indices[camera_name]
                    base_frame_id_test = sorted_frames_cam_test[frame_idx_test]

                    # Apply test offset only for the camera being searched
                    # FLIR 8.9MP always keeps its current offset (reference)
                    if camera_name == search_camera:
                        offset_test = test_offset
                    else:
                        offset_test = camera_frame_offsets[camera_name]  # Keep current offset

                    target_frame_id_test = base_frame_id_test + offset_test
                    points_by_frame_test = points_by_frame_dict[camera_name]

                    actual_frame_id_test = target_frame_id_test
                    if target_frame_id_test not in points_by_frame_test:
                        available_frames_test = sorted(points_by_frame_test.keys())
                        if available_frames_test:
                            actual_frame_id_test = min(
                                available_frames_test, key=lambda x: abs(x - target_frame_id_test)
                            )

                    test_camera_points[camera_name] = defaultdict(list)
                    if actual_frame_id_test in points_by_frame_test:
                        for point, meta in points_by_frame_test[actual_frame_id_test]:
                            cat_id = meta["category_id"]
                            test_camera_points[camera_name][cat_id].append((point, meta))

                # Compute alignment metrics for this offset
                test_stats, _ = compute_alignment_metrics_from_camera_points(test_camera_points)

                if test_stats:
                    # Use median as primary metric
                    metric_value = (
                        test_stats.get("median")
                        if test_stats.get("median") is not None
                        else float("inf")
                    )

                    frame_results.append(
                        {
                            "offset": test_offset,
                            "median": test_stats.get("median", None),
                            "count": test_stats.get("count", 0),
                        }
                    )

                    # Store values for plotting
                    offset_values.append(test_offset)
                    median_values.append(test_stats.get("median"))
                    match_counts.append(test_stats.get("count", 0))

                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_offset = test_offset
                        best_metric_type = "median"
                else:
                    # No matches found for this offset
                    offset_values.append(test_offset)
                    median_values.append(None)
                    match_counts.append(0)

            # Plot distance metrics over offset range
            if len(offset_values) > 0:
                # Reuse existing plot figure if available, otherwise create new one
                nonlocal alignment_plot_figure
                if alignment_plot_figure is not None:
                    # Clear and reuse existing figure
                    try:
                        alignment_plot_figure.clf()
                        plot_ax = alignment_plot_figure.add_subplot(111)
                    except:
                        # If figure was closed, create a new one
                        alignment_plot_figure = None
                        plot_fig, plot_ax = plt.subplots(figsize=(10, 6))
                        alignment_plot_figure = plot_fig
                else:
                    # Create new figure
                    plot_fig, plot_ax = plt.subplots(figsize=(10, 6))
                    alignment_plot_figure = plot_fig

                # Plot median (primary metric)
                valid_median_indices = [i for i, v in enumerate(median_values) if v is not None]
                if valid_median_indices:
                    valid_offsets_median = [offset_values[i] for i in valid_median_indices]
                    valid_median_vals = [median_values[i] for i in valid_median_indices]
                    plot_ax.plot(
                        valid_offsets_median,
                        valid_median_vals,
                        "o-",
                        color="blue",
                        label="Median",
                        linewidth=2,
                        markersize=6,
                    )
                    # Mark the best offset
                    if best_metric_type == "median" and best_offset in valid_offsets_median:
                        best_idx = valid_offsets_median.index(best_offset)
                        plot_ax.plot(
                            best_offset,
                            valid_median_vals[best_idx],
                            "rs",
                            markersize=12,
                            label=f"Best (median={valid_median_vals[best_idx]:.3f}m)",
                            zorder=10,
                        )

                # Create secondary y-axis for match counts
                plot_ax2 = plot_ax.twinx()

                # Plot match counts on secondary axis
                valid_match_indices = [i for i, count in enumerate(match_counts) if count > 0]
                if valid_match_indices:
                    valid_offsets_match = [offset_values[i] for i in valid_match_indices]
                    valid_match_counts = [match_counts[i] for i in valid_match_indices]
                    plot_ax2.plot(
                        valid_offsets_match,
                        valid_match_counts,
                        "^-",
                        color="orange",
                        label="Match count",
                        linewidth=2,
                        markersize=6,
                        alpha=0.7,
                    )

                # Set labels and styling
                plot_ax.set_xlabel("Offset (frames)", fontsize=12, fontweight="bold")
                plot_ax.set_ylabel(
                    "Distance (meters)", fontsize=12, fontweight="bold", color="black"
                )
                plot_ax2.set_ylabel(
                    "Number of Matches", fontsize=12, fontweight="bold", color="orange"
                )
                plot_ax2.tick_params(axis="y", labelcolor="orange")
                plot_ax.set_title(
                    f"Alignment Metrics vs Offset ({search_camera})", fontsize=14, fontweight="bold"
                )
                plot_ax.grid(True, alpha=0.3, linestyle="--")

                # Combine legends from both axes
                lines1, labels1 = plot_ax.get_legend_handles_labels()
                lines2, labels2 = plot_ax2.get_legend_handles_labels()
                plot_ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

                # Add vertical line at best offset
                plot_ax.axvline(x=best_offset, color="red", linestyle="--", alpha=0.5, linewidth=1)

                # Add text annotation for best offset
                best_match_count = (
                    match_counts[offset_values.index(best_offset)]
                    if best_offset in offset_values
                    else 0
                )
                plot_ax.text(
                    0.02,
                    0.98,
                    f"Best offset: {best_offset:+d}\nMetric: {best_metric_type} = {best_metric_value:.3f}m\nMatches: {best_match_count}",
                    transform=plot_ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
                )

                plt.tight_layout()

                # Force the plot to update/redraw
                if alignment_plot_figure is not None:
                    alignment_plot_figure.canvas.draw()
                    alignment_plot_figure.canvas.flush_events()

                plt.show(block=False)  # Non-blocking so main window stays responsive
                print(f"   📈 Opened/updated distance metrics plot window")

            # Update to best offset found
            if best_offset != original_offset:
                camera_frame_offsets[search_camera] = best_offset
                print(
                    f"✅ Found best alignment for {search_camera} at offset: {best_offset:+d} (metric: {best_metric_type} = {best_metric_value:.3f}m)"
                )

                # Print which frame indices are now aligned
                flir89_sorted_frames = sorted_frames_dict["FLIR 8.9MP"]
                search_sorted_frames = sorted_frames_dict[search_camera]

                if len(flir89_sorted_frames) > 0 and len(search_sorted_frames) > 0:
                    flir89_frame_idx = camera_frame_indices["FLIR 8.9MP"]
                    search_frame_idx = camera_frame_indices[search_camera]

                    # Get the actual frame IDs
                    flir89_base_frame_id = (
                        flir89_sorted_frames[flir89_frame_idx]
                        if flir89_frame_idx < len(flir89_sorted_frames)
                        else flir89_sorted_frames[0]
                    )
                    search_base_frame_id = (
                        search_sorted_frames[search_frame_idx]
                        if search_frame_idx < len(search_sorted_frames)
                        else search_sorted_frames[0]
                    )

                    # Apply offsets to get actual aligned frames
                    flir89_actual_frame = flir89_base_frame_id + camera_frame_offsets["FLIR 8.9MP"]
                    search_actual_frame = search_base_frame_id + best_offset

                    print(f"\n📊 Frame Alignment:")
                    print(
                        f"   FLIR 8.9MP: frame index {flir89_frame_idx} → frame ID {flir89_actual_frame}"
                    )
                    print(
                        f"   {search_camera}: frame index {search_frame_idx} → frame ID {search_actual_frame}"
                    )

                if frame_results:
                    print(f"\n   Top 3 offsets:")
                    sorted_results = sorted(
                        [r for r in frame_results if r["median"] is not None],
                        key=lambda x: x.get("median", float("inf")),
                    )
                    for i, result in enumerate(sorted_results[:3]):
                        metric_val = result.get("median")
                        print(
                            f"     {i+1}. Offset {result['offset']:+d}: median={metric_val:.3f}m ({result['count']} matches)"
                        )
            else:
                camera_frame_offsets[search_camera] = original_offset  # Restore if no change
                print(
                    f"✓ Current {search_camera} offset is already best: {best_offset:+d} (metric: {best_metric_type} = {best_metric_value:.3f}m)"
                )

                # Still print the alignment info
                flir89_sorted_frames = sorted_frames_dict["FLIR 8.9MP"]
                search_sorted_frames = sorted_frames_dict[search_camera]

                if len(flir89_sorted_frames) > 0 and len(search_sorted_frames) > 0:
                    flir89_frame_idx = camera_frame_indices["FLIR 8.9MP"]
                    search_frame_idx = camera_frame_indices[search_camera]

                    flir89_base_frame_id = (
                        flir89_sorted_frames[flir89_frame_idx]
                        if flir89_frame_idx < len(flir89_sorted_frames)
                        else flir89_sorted_frames[0]
                    )
                    search_base_frame_id = (
                        search_sorted_frames[search_frame_idx]
                        if search_frame_idx < len(search_sorted_frames)
                        else search_sorted_frames[0]
                    )

                    flir89_actual_frame = flir89_base_frame_id + camera_frame_offsets["FLIR 8.9MP"]
                    search_actual_frame = search_base_frame_id + best_offset

                    print(f"\n📊 Current Frame Alignment:")
                    print(
                        f"   FLIR 8.9MP: frame index {flir89_frame_idx} → frame ID {flir89_actual_frame}"
                    )
                    print(
                        f"   {search_camera}: frame index {search_frame_idx} → frame ID {search_actual_frame}"
                    )

            update_frame()

    # Connect keyboard event
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    # Close event handler: close alignment plot and video windows when main window closes
    def on_close(event):
        """Close alignment plot figure and video windows when main window is closed."""
        nonlocal alignment_plot_figure
        if alignment_plot_figure is not None:
            try:
                plt.close(alignment_plot_figure)
            except:
                pass
            alignment_plot_figure = None

        # Close all video windows and release resources
        if video_enabled:
            for camera_name in video_captures.keys():
                try:
                    cv2.destroyWindow(f"Video - {camera_name}")
                except:
                    pass
            # Release video captures
            for cap in video_captures.values():
                try:
                    cap.release()
                except:
                    pass
            # Destroy all OpenCV windows to ensure cleanup
            try:
                cv2.destroyAllWindows()
            except:
                pass

        # Exit the Qt application to properly terminate the program
        # Use QTimer to quit after a brief delay to ensure cleanup completes
        try:
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(100, app.quit)  # Quit after 100ms to allow cleanup
        except:
            # Fallback: try immediate quit
            try:
                app = QApplication.instance()
                if app is not None:
                    app.quit()
            except:
                pass

    fig.canvas.mpl_connect("close_event", on_close)

    # Initial frame update
    update_frame()

    print(f"\nXY Plane Visualization (Top-Down View) - Frame by Frame:")
    print(f"  Cameras: {', '.join(camera_names_sorted)}")
    for camera_name in camera_names_sorted:
        points_3d, _ = points_3d_dict[camera_name]
        sorted_frames_cam = sorted_frames_dict[camera_name]
        frame_range = (
            f"{sorted_frames_cam[0]}-{sorted_frames_cam[-1]}" if sorted_frames_cam else "N/A"
        )
        print(f"    {camera_name}: {len(points_3d)} points, frames {frame_range}")
    print(f"\nControls (Active Camera Controls):")
    print(f"  TAB - Switch active camera")
    print(f"  RIGHT ARROW / 'n' - Next frame (active camera)")
    print(f"  LEFT ARROW / 'p' - Previous frame (active camera)")
    print(f"  HOME - First frame (active camera)")
    print(f"  END - Last frame (active camera)")
    print(f"  UP ARROW / 'u' - Increase frame offset (active camera)")
    print(f"  DOWN ARROW / 'd' - Decrease frame offset (active camera)")
    print(f"  'r' - Reset offset to 0 (active camera)")
    print(f"  'f' - Find best alignment (search +/-100 frames for lowest median)")
    print(
        f"  'm' - Toggle static object exclusion (currently: {'ON' if exclude_static_objects else 'OFF'})"
    )
    print(f"  0-9 - Jump to frame at percentage (active camera)")
    print(f"\nControls (All Cameras):")
    print(f"  'N' (uppercase) - Next frame (all cameras)")
    print(f"  'P' (uppercase) - Previous frame (all cameras)")
    print(f"\nAxis ranges:")
    print(f"  X: [{x_min:.1f}, {x_max:.1f}] m")
    print(f"  Y: [{y_min:.1f}, {y_max:.1f}] m")
    print(f"\nColor scheme:")
    print(f"  FLIR 8.9MP: Green")
    print(f"  FLIR 3.2MP: Orange")
    print(f"\nMarker symbols:")
    for cat_id, cat_name in sorted_categories:
        marker = category_marker_map[cat_id]
        print(f"  {cat_name}: {marker}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera temporal alignment and validation tool. Synchronizes frame timestamps and validates 3D object alignment across multiple cameras."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=None,
        help="Path to COCO annotation JSON file (single file)",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=r"H:\GMIND\UrbanJunctionSet\1",
        help="Directory containing annotation files (for multi-camera plotting)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["FLIR3.2", "FLIR8.9"],
        help="Camera names to plot (e.g., FLIR3.2 FLIR8.9)",
    )
    parser.add_argument(
        "--plot_tracks",
        action="store_true",
        help="Plot tracks (connect points by track_id) instead of scatter plot",
    )
    parser.add_argument(
        "--plot_time", action="store_true", help="Plot objects appearing over time (animated)"
    )
    parser.add_argument(
        "--plot_xy",
        action="store_true",
        help="Plot XY plane (top-down view) over time (animated, multi-camera)",
    )
    parser.add_argument("--title", type=str, default=None, help="Custom plot title")

    args = parser.parse_args()

    # Initialize QApplication for PySide6 backend
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Handle multi-camera XY plot
    if args.plot_xy:
        # Convert Windows path to WSL path if needed
        annotation_dir_str = args.annotation_dir.replace("\\", "/")
        if annotation_dir_str.startswith("H:/") or annotation_dir_str.startswith("H:\\"):
            annotation_dir_str = "/mnt/h/" + annotation_dir_str[3:]

        annotation_dir = Path(annotation_dir_str)

        if not annotation_dir.exists():
            print(f"Error: Directory not found: {annotation_dir}")
            return

        # Find annotation files with 3D positions
        points_3d_dict = {}

        for camera_name in args.cameras:
            # Look for files matching the camera name pattern
            pattern = f"*{camera_name}*_with_3d.json"
            matching_files = list(annotation_dir.glob(pattern))

            if not matching_files:
                # Try without the "_with_3d" suffix
                pattern = f"*{camera_name}*_anno.json"
                matching_files = list(annotation_dir.glob(pattern))

            if matching_files:
                anno_file = matching_files[0]
                print(f"Loading {camera_name} from: {anno_file.name}")

                coco_data = load_coco_annotations(str(anno_file))
                points_3d, metadata = extract_3d_locations(coco_data)

                if len(points_3d) > 0:
                    # Use proper camera name for display
                    display_name = camera_name
                    if "FLIR3.2" in str(anno_file):
                        display_name = "FLIR 3.2MP"
                    elif "FLIR8.9" in str(anno_file):
                        display_name = "FLIR 8.9MP"
                    elif "CCTV" in str(anno_file):
                        display_name = "CCTV"
                    elif "Thermal" in str(anno_file):
                        display_name = "Hikvision Thermal"

                    points_3d_dict[display_name] = (points_3d, metadata)
                    print(f"  Loaded {len(points_3d)} 3D positions")
                else:
                    print(f"  Warning: No 3D positions found")
            else:
                print(f"  Warning: No annotation file found for {camera_name}")

        if len(points_3d_dict) == 0:
            print("Error: No annotation files with 3D positions found.")
            return

        # Determine title
        title = args.title
        if title is None:
            title = f"3D Object Locations Over Time (XY Plane) - {', '.join(points_3d_dict.keys())}"

        print(f"\nPlotting XY plane (top-down view) over time...")
        plot_xy_plane_over_time_multicam(
            points_3d_dict, title, annotation_dir=annotation_dir_str, enable_video=True
        )

    else:
        # Original single-file plotting
        if args.annotation_file is None:
            print("Error: --annotation_file is required when not using --plot_xy")
            return

        # Load annotations
        print(f"Loading annotations from: {args.annotation_file}")
        coco_data = load_coco_annotations(args.annotation_file)

        # Extract 3D locations
        print("Extracting 3D locations...")
        points_3d, metadata = extract_3d_locations(coco_data)

        if len(points_3d) == 0:
            print("No 3D locations found in the annotation file.")
            return

        # Determine title
        title = args.title
        if title is None:
            annotation_path = Path(args.annotation_file)
            if args.plot_time:
                title = f"3D Object Locations Over Time - {annotation_path.stem}"
            elif args.plot_tracks:
                title = f"3D Object Tracks - {annotation_path.stem}"
            else:
                title = f"3D Object Locations - {annotation_path.stem}"

        # Plot
        if args.plot_time:
            print("Plotting 3D locations over time (animated)...")
            plot_3d_locations_over_time(points_3d, metadata, title)
        elif args.plot_tracks:
            print("Plotting 3D tracks...")
            plot_tracks_3d(points_3d, metadata, title)
        else:
            print("Plotting 3D locations...")
            plot_3d_locations_interactive(points_3d, metadata, title)

    # Run the Qt event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
