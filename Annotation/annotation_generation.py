"""
Video annotation generator using object detection and multi-object tracking.

This module provides a simplified video annotation pipeline that automatically
detects, tracks, and annotates objects in videos. It supports multiple detection
models (Dome-DETR, YOLOv12x) and generates COCO-format annotations with optional
3D location information using geometric ground plane intersection.

Main Components:
    - Config: Configuration dataclass for pipeline parameters
    - ObjectDetector: Handles object detection using deep learning models
    - Tracker: Multi-object tracking using OC-SORT
    - TrackedObject: Represents tracked objects with 2D/3D information
    - process_video: Main pipeline function

Example:
    >>> from Annotation.annotation_generation import Config, process_video
    >>> config = Config()
    >>> config.video_path = "path/to/video.mp4"
    >>> config.enable_depth_estimation = True
    >>> process_video(config.video_path, config)
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class Config:
    """Configuration parameters for the video annotation pipeline."""

    # Video input
    video_path: str = ""  # Path to input video file
    process_every_n_frames: int = 1  # Will be auto-calculated based on video FPS to target ~5fps
    target_processing_fps: float = 5.0  # Target FPS for processing (interpolation fills gaps)

    # Object Detection
    detector_model: str = "dome-detr"  # Model to use: "dome-detr" or "yolo12x"
    detector_config_file: str = (
        "../Dome-DETR/configs/dome/Dome-L-VisDrone.yml"  # Path to Dome-DETR config file (relative to GMIND-sdk)
    )
    detector_checkpoint: str = (
        "../Dome-DETR/Dome-L-VisDrone-best.pth"  # Path to Dome-DETR checkpoint (relative to GMIND-sdk)
    )
    detector_conf_threshold: float = 0.28
    min_box_size: int = 5  # Minimum width or height in pixels

    # Depth Estimation (optional, for 3D locations)
    enable_depth_estimation: bool = (
        False  # Enable 3D location computation using geometric ground plane intersection
    )
    # Geometric depth parameters
    camera_height: float = 4.0  # Camera height above ground in meters
    camera_pitch_deg: float = 20.0  # Camera pitch angle (degrees, positive = downward)
    camera_roll_deg: float = 0.0  # Camera roll angle (degrees, positive = camera tilted right)
    camera_yaw_deg: float = 0.0  # Camera yaw angle (degrees, positive = camera rotated left)
    ground_height: float = 0.0  # Ground plane height in world coordinates (meters), typically 0.0
    calibration_file: Optional[str] = (
        "sensor_calibration.txt"  # Path to sensor_calibration.txt file (relative to Annotation/ directory, resolves to GMIND-sdk root)
    )
    camera_name: Optional[str] = (
        None  # Camera name in calibration file (e.g., "FLIR8.9mm"). If None, auto-extracted from video filename
    )
    camera_matrix: Optional[np.ndarray] = (
        None  # Camera intrinsics (3x3). If None and calibration_file is provided, will be parsed automatically
    )
    dist_coeffs: Optional[np.ndarray] = (
        None  # Distortion coefficients (5x1). If None and calibration_file is provided, will be parsed automatically
    )

    # Tracking (OC-SORT)
    tracking_method: str = "OC-SORT"
    tracking_max_age: int = 15
    tracking_min_hits: int = 4
    tracking_iou_threshold: float = 0.2
    tracking_det_thresh: float = 0.28
    tracking_delta_t: int = 3
    tracking_inertia: float = 0.2
    tracking_asso_func: str = "giou"

    # Output
    save_annotations: bool = True
    headless: bool = False


# COCO class names (only using first 3: person, bicycle, car)
COCO_CLASSES = ["person", "bicycle", "car"]

# VisDrone class names (for Dome-DETR model)
VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car"]


def extract_camera_name_from_video(video_path: str) -> Optional[str]:
    """
    Extract camera name from video filename.
    Examples:
        - "FLIR8.9-Urban1.mp4" -> "FLIR8.9"
        - "FLIR3.2-Ped2.mp4" -> "FLIR3.2"
        - "CCTV-Urban1.mp4" -> "CCTV"
        - "Thermal-Ped2.mp4" -> "Thermal"
        - "HikvisionThermal-Urban1.mp4" -> "Hikvision Thermal"

    Returns:
        Camera identifier string or None if not found
    """
    video_path = Path(video_path)
    filename = video_path.stem  # Get filename without extension

    # Pattern 1: FLIR cameras (e.g., FLIR8.9, FLIR3.2)
    flir_pattern = r"FLIR\s*(\d+\.\d+)"
    match = re.search(flir_pattern, filename, re.IGNORECASE)
    if match:
        version = match.group(1)
        return f"FLIR{version}"

    # Pattern 2: Hikvision Thermal (full name, matches calibration)
    hikvision_pattern = r"[Hh]ikvision\s*[Tt]hermal"
    match = re.search(hikvision_pattern, filename, re.IGNORECASE)
    if match:
        return "Hikvision Thermal"

    # Pattern 3: Thermal (generic, will match "Hikvision Thermal" in calibration)
    thermal_pattern = r"\b[Tt]hermal\b"
    match = re.search(thermal_pattern, filename, re.IGNORECASE)
    if match:
        return "Thermal"

    # Pattern 4: CCTV
    cctv_pattern = r"\bCCTV\b"
    match = re.search(cctv_pattern, filename, re.IGNORECASE)
    if match:
        return "CCTV"

    return None


def match_camera_name(search_name: str, calibration_camera_name: str) -> bool:
    """
    Match camera name from video to calibration file name.
    Handles variations like:
        - "FLIR8.9" <-> "FLIR 8.9MP"
        - "FLIR3.2" <-> "FLIR 3.2MP"
        - "CCTV" <-> "CCTV"
        - "Thermal" <-> "Hikvision Thermal"
        - Case insensitive matching
        - Handles spaces and suffixes like "MP"

    Args:
        search_name: Camera name from video (e.g., "FLIR8.9", "CCTV", "Thermal")
        calibration_camera_name: Camera name in calibration file (e.g., "FLIR 8.9MP", "CCTV", "Hikvision Thermal")

    Returns:
        True if they match, False otherwise
    """
    if search_name is None or calibration_camera_name is None:
        return False

    # Special handling for Thermal cameras
    # "Thermal" from video should match "Hikvision Thermal" in calibration
    if search_name.lower() == "thermal" and "thermal" in calibration_camera_name.lower():
        return True
    if calibration_camera_name.lower() == "thermal" and "thermal" in search_name.lower():
        return True

    # Normalize both names: remove spaces, convert to lowercase, remove common suffixes
    def normalize(name):
        normalized = name.lower().replace(" ", "").replace("_", "-")
        # Remove common suffixes
        for suffix in ["mp", "mm", "p"]:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
        return normalized

    normalized_search = normalize(search_name)
    normalized_calib = normalize(calibration_camera_name)

    # Check if they match after normalization
    if normalized_search == normalized_calib:
        return True

    # Also check if one is a substring of the other (for partial matches)
    if normalized_search in normalized_calib or normalized_calib in normalized_search:
        return True

    # Special case: "hikvisionthermal" should match "hikvision thermal" or vice versa
    if "hikvision" in normalized_search and "thermal" in normalized_search:
        if "hikvision" in normalized_calib and "thermal" in normalized_calib:
            return True

    return False


def parse_camera_intrinsics_from_calibration(
    calib_path: str, camera_name: Optional[str] = None, video_path: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse camera intrinsics and distortion coefficients from sensor_calibration.txt file.

    Args:
        calib_path: Path to sensor_calibration.txt file
        camera_name: Name of camera in calibration file (e.g., "FLIR 8.9MP").
                     If None, will try to extract from video_path if provided.
        video_path: Optional path to video file. If camera_name is None, will
                    try to extract camera name from video filename.

    Returns:
        Tuple of (camera_matrix, dist_coeffs) where:
        - camera_matrix: Camera intrinsics (3x3) or None if not found
        - dist_coeffs: Distortion coefficients (5x1) or None if not found
    """
    import os

    calib_path = Path(calib_path)
    if not calib_path.exists():
        print(f"⚠️  Warning: Calibration file not found: {calib_path}", flush=True)
        return None

    # Try to extract camera name from video if not provided
    search_camera_name = camera_name
    if search_camera_name is None and video_path is not None:
        extracted = extract_camera_name_from_video(video_path)
        if extracted:
            search_camera_name = extracted
            print(f"📹 Extracted camera name from video: '{extracted}'", flush=True)

    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()

        # First pass: collect all cameras in calibration file
        all_cameras = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
                is_camera = name.upper() not in ["LIDAR", "VELODYNE", "CEPTON", "RADAR"]
                if is_camera:
                    all_cameras.append(name)
            i += 1

        print(f"📋 Found cameras in calibration file: {all_cameras}", flush=True)

        # Second pass: find matching camera and parse intrinsics
        i = 0
        matched_camera = None
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()

                # Check if this is the camera we want
                if search_camera_name is None:
                    # Use first camera found
                    if matched_camera is None:
                        matched_camera = name
                        print(f"📷 Using first camera found: '{name}'", flush=True)
                elif match_camera_name(search_camera_name, name):
                    matched_camera = name
                    print(f"✓ Matched camera: '{search_camera_name}' -> '{name}'", flush=True)

                # Read intrinsics
                intrinsics = {}
                is_camera = name.upper() not in ["LIDAR", "VELODYNE", "CEPTON", "RADAR"]

                if is_camera and (search_camera_name is None or matched_camera == name):
                    while i < len(lines) and not lines[i].strip().startswith("Extrinsics"):
                        l = lines[i].strip()
                        if ":" in l:
                            k, v = l.split(":", 1)
                            intrinsics[k.strip()] = v.strip()
                        i += 1

                    # Parse camera matrix
                    if "Focal_x" in intrinsics and "Focal_y" in intrinsics:
                        fx = float(intrinsics.get("Focal_x", 0))
                        fy = float(intrinsics.get("Focal_y", 0))
                        cx = float(intrinsics.get("COD_x", 0))
                        cy = float(intrinsics.get("COD_y", 0))

                        if fx > 0 and fy > 0:  # Valid intrinsics
                            camera_matrix = np.array(
                                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                            )

                            # Parse distortion coefficients if available
                            dist_coeffs = None
                            if "Dist_1" in intrinsics or "Distortion_1" in intrinsics:
                                # Try both naming conventions
                                dist_keys = ["Dist_1", "Dist_2", "Dist_3", "Dist_4", "Dist_5"]
                                alt_keys = [
                                    "Distortion_1",
                                    "Distortion_2",
                                    "Distortion_3",
                                    "Distortion_4",
                                    "Distortion_5",
                                ]
                                dist_values = []
                                for dk, ak in zip(dist_keys, alt_keys):
                                    val = intrinsics.get(dk, intrinsics.get(ak, "0"))
                                    try:
                                        dist_values.append(float(val))
                                    except (ValueError, TypeError):
                                        dist_values.append(0.0)

                                # Only use if at least first coefficient is non-zero
                                if any(abs(v) > 1e-6 for v in dist_values):
                                    dist_coeffs = np.array(dist_values[:5], dtype=np.float32)
                                    print(
                                        f"✓ Parsed distortion coefficients: {dist_coeffs}",
                                        flush=True,
                                    )

                            print(
                                f"✓ Parsed camera intrinsics for '{name}' from {calib_path}",
                                flush=True,
                            )
                            print(
                                f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}", flush=True
                            )
                            return camera_matrix, dist_coeffs
                    else:
                        # Continue searching if intrinsics not found for this camera
                        i += 1
                        continue
                else:
                    i += 1
                    continue
            i += 1

        if matched_camera is None:
            print(
                f"⚠️  Warning: No matching camera found in calibration file: {calib_path}",
                flush=True,
            )
            if search_camera_name:
                print(f"  Searched for: '{search_camera_name}'", flush=True)
            print(f"  Available cameras: {all_cameras}", flush=True)
        else:
            print(
                f"⚠️  Warning: Could not parse intrinsics for camera '{matched_camera}' from {calib_path}",
                flush=True,
            )
        return None, None

    except Exception as e:
        print(f"⚠️  Warning: Failed to parse calibration file {calib_path}: {e}", flush=True)
        return None


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    frame_id: int
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "person"
    time_since_update: int = 0
    first_frame: int = -1
    hit_streak: int = 0
    class_history: List[int] = field(default_factory=list)
    # Interpolation state
    last_known_bbox: Optional[np.ndarray] = None
    last_known_frame: int = -1
    frame_positions: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # {frame_id: bbox} for detected positions
    interpolated_positions: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # {frame_id: bbox} for interpolated positions
    # 3D locations (optional, if depth estimation is enabled)
    # Coordinates are in world frame with origin at ground level below camera (Z=0 for all ground intersections)
    frame_3d_locations: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # {frame_id: [X, Y, Z]} for detected 3D positions (world coords)
    interpolated_3d_locations: Dict[int, np.ndarray] = field(
        default_factory=dict
    )  # {frame_id: [X, Y, Z]} for interpolated 3D positions (world coords)
    last_known_3d_location: Optional[np.ndarray] = (
        None  # Last known 3D location for interpolation (world coords)
    )

    def __post_init__(self):
        if len(self.class_history) == 0 and self.class_id is not None:
            self.class_history.append(self.class_id)
        if self.first_frame < 0:
            self.first_frame = self.frame_id
        if self.class_id < len(COCO_CLASSES):
            self.class_name = COCO_CLASSES[self.class_id]
        # Store initial position
        self.frame_positions[self.frame_id] = self.bbox.copy()
        self.last_known_bbox = self.bbox.copy()
        self.last_known_frame = self.frame_id

    def update(
        self,
        bbox: np.ndarray,
        frame_id: int,
        confidence: float = 1.0,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        location_3d: Optional[np.ndarray] = None,
    ):
        """Update track with new detection.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_id: Current frame ID
            confidence: Detection confidence
            class_id: Optional class ID
            class_name: Optional class name
            location_3d: Optional 3D location [X, Y, Z] in world coordinates (origin at ground level below camera)
        """
        # Always interpolate between last known position and current position if there's a gap
        # This handles both "lost and regained" tracks and skipped frames when processing every Nth frame
        if self.last_known_frame >= 0 and frame_id > self.last_known_frame + 1:
            # There are frames between last_known_frame and frame_id - interpolate them
            self._interpolate_between_frames(
                self.last_known_frame,
                self.last_known_bbox,
                frame_id,
                bbox,
                self.last_known_3d_location,
                location_3d,  # Also interpolate 3D locations
            )

        if self.first_frame < 0:
            self.first_frame = frame_id
        self.hit_streak += 1

        # Update class based on majority vote
        if class_id is not None:
            self.class_history.append(class_id)
            if len(self.class_history) > 100:
                self.class_history.pop(0)
            if len(self.class_history) > 0:
                class_counts = Counter(self.class_history)
                most_common_class_id, _ = class_counts.most_common(1)[0]
                self.class_id = most_common_class_id
                # Use provided class_name if available, otherwise look up from COCO_CLASSES
                if class_name is not None:
                    self.class_name = class_name
                elif self.class_id < len(COCO_CLASSES):
                    self.class_name = COCO_CLASSES[self.class_id]
                else:
                    self.class_name = f"class_{self.class_id}"

        self.bbox = bbox.copy()
        self.frame_id = frame_id
        self.confidence = confidence
        self.time_since_update = 0

        # Store position for this frame
        self.frame_positions[frame_id] = bbox.copy()
        self.last_known_bbox = bbox.copy()
        self.last_known_frame = frame_id

        # Store 3D location if provided
        if location_3d is not None:
            self.frame_3d_locations[frame_id] = location_3d.copy()
            self.last_known_3d_location = location_3d.copy()

    def get_3d_location_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get 3D location for a specific frame, using detected location or interpolated location."""
        # Priority 1: Check if we have a detected 3D location for this frame
        if frame_id in self.frame_3d_locations:
            loc_3d = self.frame_3d_locations[frame_id].copy()
            if (
                loc_3d is not None
                and len(loc_3d) == 3
                and not np.any(np.isnan(loc_3d))
                and not np.any(np.isinf(loc_3d))
            ):
                return loc_3d

        # Priority 2: Check if we have an interpolated 3D location for this frame
        if frame_id in self.interpolated_3d_locations:
            loc_3d = self.interpolated_3d_locations[frame_id].copy()
            if (
                loc_3d is not None
                and len(loc_3d) == 3
                and not np.any(np.isnan(loc_3d))
                and not np.any(np.isinf(loc_3d))
            ):
                return loc_3d

        # No 3D location available for this frame
        return None

    def _interpolate_between_frames(
        self,
        start_frame: int,
        start_bbox: np.ndarray,
        end_frame: int,
        end_bbox: np.ndarray,
        start_3d: Optional[np.ndarray] = None,
        end_3d: Optional[np.ndarray] = None,
    ):
        """Interpolate positions between two frames (both 2D bbox and 3D locations)."""
        if start_frame >= end_frame:
            return

        # Interpolate for all frames between start and end
        for frame_id in range(start_frame + 1, end_frame):
            alpha = (frame_id - start_frame) / (end_frame - start_frame)
            alpha = np.clip(alpha, 0.0, 1.0)

            # Linear interpolation for 2D bbox
            interpolated_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
            self.interpolated_positions[frame_id] = interpolated_bbox.copy()

            # Linear interpolation for 3D location if both start and end are available
            if start_3d is not None and end_3d is not None:
                if len(start_3d) == 3 and len(end_3d) == 3:
                    interpolated_3d = (1 - alpha) * start_3d + alpha * end_3d
                    self.interpolated_3d_locations[frame_id] = interpolated_3d.copy()

    def get_bbox_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get bbox for a specific frame, using detected position or interpolated position."""
        # Priority 1: Check if we have a detected position for this frame
        if frame_id in self.frame_positions:
            bbox = self.frame_positions[frame_id].copy()
            if (
                bbox is not None
                and len(bbox) == 4
                and not np.any(np.isnan(bbox))
                and not np.any(np.isinf(bbox))
            ):
                return bbox

        # Priority 2: Check if we have an interpolated position for this frame
        if frame_id in self.interpolated_positions:
            bbox = self.interpolated_positions[frame_id].copy()
            if (
                bbox is not None
                and len(bbox) == 4
                and not np.any(np.isnan(bbox))
                and not np.any(np.isinf(bbox))
            ):
                return bbox

        # No position available for this frame
        return None

    def get_track_duration(self) -> int:
        """Get number of frames this track has been active."""
        return self.hit_streak if self.first_frame >= 0 else 0


@dataclass
class Detection:
    """Represents a single object detection."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class ObjectDetector:
    """Run object detector on full frames (supports Dome-DETR or YOLOv12x)."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.postprocessor = None  # For Dome-DETR
        self.backend = None  # 'mmdetection' or 'ultralytics' or 'dome-detr'
        self._load_model()

    def _load_model(self):
        """Load detection model (Dome-DETR or YOLOv12x)."""
        print(f"Loading object detection model: {self.config.detector_model}...")

        if (
            self.config.detector_model.lower() == "dome-detr"
            or self.config.detector_model.lower().startswith("dome")
        ):
            # Load Dome-DETR using MMDetection init_detector (handles YAML configs properly)
            if not self.config.detector_config_file or not self.config.detector_checkpoint:
                raise RuntimeError(
                    "Dome-DETR requires both config_file and checkpoint paths.\n"
                    "Set detector_config_file and detector_checkpoint in Config."
                )

            # Resolve paths (handle relative paths)
            import os

            config_file = self.config.detector_config_file
            checkpoint = self.config.detector_checkpoint

            # If relative path, try resolving from current working directory or script directory
            if not os.path.isabs(config_file):
                # Try current working directory first
                if os.path.exists(config_file):
                    config_file = os.path.abspath(config_file)
                else:
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(script_dir)  # Go up one level from Annotation/
                    potential_path = os.path.join(parent_dir, config_file)
                    if os.path.exists(potential_path):
                        config_file = os.path.abspath(potential_path)

            if not os.path.isabs(checkpoint):
                # Try current working directory first
                if os.path.exists(checkpoint):
                    checkpoint = os.path.abspath(checkpoint)
                else:
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(script_dir)  # Go up one level from Annotation/
                    potential_path = os.path.join(parent_dir, checkpoint)
                    if os.path.exists(potential_path):
                        checkpoint = os.path.abspath(potential_path)

            try:
                # Use Dome-DETR's custom YAMLConfig loading mechanism
                # Add Dome-DETR root to path and change working directory
                import sys

                # Ensure paths are absolute first
                if not os.path.isabs(config_file):
                    config_file = os.path.abspath(config_file)

                dome_detr_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(config_file))
                )  # Go up from configs/dome/Dome-L-VisDrone.yml to Dome-DETR root

                # Add Dome-DETR root to path (not just src, so relative imports work)
                if dome_detr_dir not in sys.path:
                    sys.path.insert(0, dome_detr_dir)

                # Change to Dome-DETR directory so relative imports work
                original_cwd = os.getcwd()
                os.chdir(dome_detr_dir)

                try:
                    import torch
                    from src.core import YAMLConfig

                    # Import Dome-DETR modules to register them
                    # This is necessary for the config system to work
                    from src.zoo import dome  # Registers all Dome-DETR modules

                    try:
                        from src.zoo import backbone  # Registers backbone modules
                    except ImportError:
                        pass
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)

                # Ensure paths are absolute
                if not os.path.isabs(config_file):
                    config_file = os.path.abspath(config_file)
                if not os.path.isabs(checkpoint):
                    checkpoint = os.path.abspath(checkpoint)

                # Load config using Dome-DETR's YAMLConfig
                cfg = YAMLConfig(config_file, resume=checkpoint)

                # Disable pretrained backbone loading (we're loading from checkpoint)
                if "HGNetv2" in cfg.yaml_cfg:
                    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

                # Load checkpoint
                checkpoint_data = torch.load(checkpoint, map_location="cpu")
                if "ema" in checkpoint_data:
                    state = checkpoint_data["ema"]["module"]
                else:
                    state = checkpoint_data["model"]

                # Load state dict into model
                cfg.model.load_state_dict(state)

                # Deploy the model for inference
                deployed_model = cfg.model.deploy()
                deployed_postprocessor = cfg.postprocessor.deploy()

                # Wrap model and postprocessor together (like in their inference script)
                class DomeDETRModel(torch.nn.Module):
                    def __init__(self, model, postprocessor):
                        super().__init__()
                        self.model = model
                        self.postprocessor = postprocessor

                    def forward(self, images, orig_target_sizes, targets=None):
                        outputs = self.model(images, targets=targets)
                        outputs = self.postprocessor(outputs, orig_target_sizes)
                        return outputs

                self.model = DomeDETRModel(deployed_model, deployed_postprocessor)

                # Move to device
                if hasattr(self.model, "to"):
                    self.model.to(self.device)

                if hasattr(self.model, "eval"):
                    self.model.eval()
                self.backend = "dome-detr"
                print(f"✓ Loaded Dome-DETR from {checkpoint}")
            except ImportError as e:
                raise RuntimeError(
                    f"MMDetection is not installed or init_detector is not available: {e}\n"
                    "Install with: pip install mmdet mmcv mmengine"
                )
            except Exception as e:
                import traceback

                error_msg = f"Failed to load Dome-DETR model: {e}\n"
                error_msg += f"Config: {config_file}\n"
                error_msg += f"Checkpoint: {checkpoint}\n"
                error_msg += f"\nTraceback:\n{traceback.format_exc()}"
                raise RuntimeError(error_msg)
        else:
            # Load YOLOv12x using Ultralytics
            try:
                from DeepLearning.adapters import get_model

                self.model = get_model(
                    "yolo12x", num_classes=80, pretrained=True, backend="ultralytics"
                )
                if hasattr(self.model, "to"):
                    self.model.to(self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
                self.backend = "ultralytics"
                print("✓ Loaded YOLOv12x")
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLOv12x model: {e}")

    def _parse_ultralytics_results(self, results, conf_threshold: float) -> List[Detection]:
        """Parse Ultralytics model results into Detection objects."""
        detections = []

        if len(results) > 0 and hasattr(results[0], "boxes"):
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                boxes_data = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes_data)):
                    if scores[i] < conf_threshold:
                        continue

                    x1, y1, x2, y2 = (
                        int(boxes_data[i][0]),
                        int(boxes_data[i][1]),
                        int(boxes_data[i][2]),
                        int(boxes_data[i][3]),
                    )
                    class_id = int(class_ids[i])

                    # Filter to only person (0), bicycle (1), and car (2)
                    if class_id not in [0, 1, 2]:
                        continue

                    # Validate box size
                    if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                        continue

                    class_name = COCO_CLASSES[class_id]

                    detections.append(
                        Detection(
                            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                            confidence=float(scores[i]),
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )

        return detections

    def _parse_mmdetection_results(self, results, conf_threshold: float) -> List[Detection]:
        """Parse MMDetection model results into Detection objects."""
        detections = []

        # Use VisDrone classes if using Dome-DETR, otherwise COCO
        use_visdrone = (
            self.config.detector_model.lower() == "dome-detr"
            or self.config.detector_model.lower().startswith("dome")
        )
        class_names = VISDRONE_CLASSES if use_visdrone else COCO_CLASSES
        valid_class_ids = (
            [0, 1, 2, 3] if use_visdrone else [0, 1, 2]
        )  # VisDrone: pedestrian, people, bicycle, car

        # MMDetection returns a list of arrays, one per class
        # Each array has shape [N, 5] where columns are [x1, y1, x2, y2, score]
        if isinstance(results, (list, tuple)):
            for class_id, class_results in enumerate(results):
                if class_id not in valid_class_ids:
                    continue

                if class_results is None or len(class_results) == 0:
                    continue

                # Handle both numpy arrays and torch tensors
                if hasattr(class_results, "cpu"):
                    class_results = class_results.cpu().numpy()
                class_results = np.array(class_results)

                if len(class_results.shape) == 1:
                    class_results = class_results.reshape(1, -1)

                for det in class_results:
                    if len(det) < 5:
                        continue

                    x1, y1, x2, y2, score = det[0], det[1], det[2], det[3], det[4]

                    if score < conf_threshold:
                        continue

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Validate box size
                    if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                        continue

                    # Get class name
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"

                    # Map VisDrone classes to COCO-equivalent for export
                    # VisDrone: 0=pedestrian, 1=people, 2=bicycle, 3=car
                    # COCO: 0=person, 1=bicycle, 2=car
                    # Map: pedestrian/people -> person (0), bicycle -> bicycle (1), car -> car (2)
                    export_class_id = class_id
                    if use_visdrone:
                        if class_id == 0 or class_id == 1:  # pedestrian or people -> person
                            export_class_id = 0
                        elif class_id == 2:  # bicycle -> bicycle
                            export_class_id = 1
                        elif class_id == 3:  # car -> car
                            export_class_id = 2

                    detections.append(
                        Detection(
                            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                            confidence=float(score),
                            class_id=export_class_id,  # Use mapped class_id for COCO export
                            class_name=class_name,
                        )
                    )

        return detections

    def _parse_dome_detr_results(
        self, labels, boxes, scores, conf_threshold: float
    ) -> List[Detection]:
        """Parse Dome-DETR model results into Detection objects."""
        detections = []

        try:
            # Handle different output formats - Dome-DETR returns tensors directly
            # Convert to numpy if tensors
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu().numpy()
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            # Handle batch dimension - get first batch item if needed
            if isinstance(labels, np.ndarray) and len(labels.shape) > 1:
                labels_batch = labels[0] if labels.shape[0] > 0 else labels
            elif isinstance(labels, (list, tuple)):
                labels_batch = labels[0] if len(labels) > 0 else labels
            else:
                labels_batch = labels

            if isinstance(boxes, np.ndarray) and len(boxes.shape) > 2:
                boxes_batch = boxes[0] if boxes.shape[0] > 0 else boxes
            elif isinstance(boxes, (list, tuple)):
                boxes_batch = boxes[0] if len(boxes) > 0 else boxes
            else:
                boxes_batch = boxes

            if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
                scores_batch = scores[0] if scores.shape[0] > 0 else scores
            elif isinstance(scores, (list, tuple)):
                scores_batch = scores[0] if len(scores) > 0 else scores
            else:
                scores_batch = scores

            # Convert to numpy arrays if not already
            labels_batch = (
                np.array(labels_batch) if not isinstance(labels_batch, np.ndarray) else labels_batch
            )
            boxes_batch = (
                np.array(boxes_batch) if not isinstance(boxes_batch, np.ndarray) else boxes_batch
            )
            scores_batch = (
                np.array(scores_batch) if not isinstance(scores_batch, np.ndarray) else scores_batch
            )

            if len(scores_batch) == 0:
                return detections

            # VisDrone classes are 1-indexed in the model output:
            # 1=pedestrian, 2=people, 3=bicycle, 4=car
            # We want to accept these and map them to 0-indexed for COCO export
            valid_class_ids = [1, 2, 3, 4]  # 1-indexed VisDrone classes

            # Handle 2D scores array - if scores is (num_det, num_classes), we need to get max per detection
            if len(scores_batch.shape) > 1 and scores_batch.shape[1] > 1:
                # Scores is per-class, get max score and corresponding class
                max_scores = np.max(scores_batch, axis=1)
                max_classes = np.argmax(scores_batch, axis=1)
            else:
                # Scores is already per-detection
                max_scores = scores_batch.flatten()
                max_classes = (
                    labels_batch.flatten() if len(labels_batch.shape) > 0 else labels_batch
                )

            num_detections = len(max_scores)

            for i in range(num_detections):
                score = float(max_scores[i])

                if score < conf_threshold:
                    continue

                class_id = int(max_classes[i])

                if class_id not in valid_class_ids:
                    continue

                # Get box - handle different box formats
                if len(boxes_batch.shape) == 2:
                    box = boxes_batch[i]
                elif len(boxes_batch.shape) == 3:
                    box = boxes_batch[0][i] if boxes_batch.shape[0] > 0 else boxes_batch[i]
                else:
                    box = boxes_batch[i]

                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

                # Validate box size
                if (x2 - x1) < self.config.min_box_size or (y2 - y1) < self.config.min_box_size:
                    continue

                # Get class name - VisDrone uses 1-indexed: 1=pedestrian, 2=people, 3=bicycle, 4=car
                # Map to 0-indexed for VISDRONE_CLASSES array
                visdrone_idx = class_id - 1  # Convert from 1-indexed to 0-indexed
                if 0 <= visdrone_idx < len(VISDRONE_CLASSES):
                    class_name = VISDRONE_CLASSES[visdrone_idx]
                else:
                    class_name = f"class_{class_id}"

                # Map VisDrone classes (1-indexed) to COCO (0-indexed) for export
                # VisDrone: 1=pedestrian, 2=people, 3=bicycle, 4=car
                # COCO: 0=person, 1=bicycle, 2=car
                export_class_id = class_id
                if class_id == 1 or class_id == 2:  # pedestrian or people -> person (0)
                    export_class_id = 0
                elif class_id == 3:  # bicycle -> bicycle (1)
                    export_class_id = 1
                elif class_id == 4:  # car -> car (2)
                    export_class_id = 2
                else:
                    continue  # Skip other classes

                detections.append(
                    Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        confidence=score,
                        class_id=export_class_id,
                        class_name=class_name,
                    )
                )
        except Exception as e:
            print(f"Warning: Error parsing Dome-DETR results: {e}")
            import traceback

            traceback.print_exc()

        return detections

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Detection]:
        """Run object detection on full frame."""
        if self.model is None:
            return []

        try:
            if self.backend == "dome-detr":
                # Use Dome-DETR inference
                import torchvision.transforms as T
                from PIL import Image

                # Convert frame to PIL Image
                im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                h, w = frame.shape[:2]
                orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(self.device)

                # Resize to 800x800 (Dome-DETR input size)
                transforms = T.Compose(
                    [
                        T.Resize((800, 800)),
                        T.ToTensor(),
                    ]
                )
                im_data = transforms(im_pil).unsqueeze(0).to(self.device)

                # Run inference
                with torch.no_grad():
                    output = self.model(im_data, orig_size, targets=None)
                    labels, boxes, scores = output

                # Parse results
                return self._parse_dome_detr_results(labels, boxes, scores, conf_threshold)
            elif self.backend == "mmdetection":
                # Use MMDetection inference API
                from mmdet.apis import inference_detector

                results = inference_detector(self.model, frame)
                return self._parse_mmdetection_results(results, conf_threshold)
            else:
                # Use Ultralytics inference
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)
                return self._parse_ultralytics_results(results, conf_threshold)
        except Exception as e:
            print(f"Warning: Detection failed: {e}")
            import traceback

            traceback.print_exc()
            return []


class Tracker:
    """Object tracker using OC-SORT."""

    def __init__(self, config: Config):
        self.config = config
        self.tracks: List[TrackedObject] = []
        self.frame_count = 0
        self.tracker = None

        if config.tracking_method == "OC-SORT":
            ocsort_loaded = False

            # Try import path 1: from OC_SORT directory
            try:
                import os
                import sys

                oc_sort_path = os.path.join(os.path.dirname(__file__), "OC_SORT")
                oc_sort_path = os.path.abspath(oc_sort_path)
                if os.path.exists(oc_sort_path) and oc_sort_path not in sys.path:
                    sys.path.insert(0, oc_sort_path)

                oc_sort_path_parent = os.path.join(os.path.dirname(__file__), "..", "OC_SORT")
                oc_sort_path_parent = os.path.abspath(oc_sort_path_parent)
                if os.path.exists(oc_sort_path_parent) and oc_sort_path_parent not in sys.path:
                    sys.path.insert(0, oc_sort_path_parent)

                if "./OC_SORT" not in sys.path and os.path.exists("./OC_SORT"):
                    sys.path.insert(0, os.path.abspath("./OC_SORT"))

                from trackers.ocsort_tracker.ocsort import OCSort

                self.tracker = OCSort(
                    det_thresh=config.tracking_det_thresh,
                    max_age=config.tracking_max_age,
                    min_hits=config.tracking_min_hits,
                    iou_threshold=config.tracking_iou_threshold,
                    delta_t=config.tracking_delta_t,
                    asso_func=config.tracking_asso_func,
                    inertia=config.tracking_inertia,
                )
                ocsort_loaded = True
                print("✓ Loaded OC-SORT from local directory")
            except ImportError:
                pass

            # Try import path 2: direct import (pip installed)
            if not ocsort_loaded:
                try:
                    from ocsort import OCSort

                    self.tracker = OCSort(
                        det_thresh=config.tracking_det_thresh,
                        max_age=config.tracking_max_age,
                        min_hits=config.tracking_min_hits,
                        iou_threshold=config.tracking_iou_threshold,
                        delta_t=config.tracking_delta_t,
                        asso_func=config.tracking_asso_func,
                        inertia=config.tracking_inertia,
                    )
                    ocsort_loaded = True
                    print("✓ Loaded OC-SORT")
                except ImportError:
                    pass

            if not ocsort_loaded:
                raise ImportError(
                    "OC-SORT not found. Please install it using:\n"
                    "  1. pip install ocsort\n"
                    "  2. pip install git+https://github.com/noahcao/OC_SORT.git"
                )

    def update(
        self, detections: List[Detection], frame_id: int, frame: Optional[np.ndarray] = None
    ) -> List[TrackedObject]:
        """Update tracker with new detections."""
        self.frame_count = frame_id

        if self.tracker is None:
            raise RuntimeError("OC-SORT tracker not initialized")

        # Convert detections to [x1, y1, x2, y2, score] format
        tracker_input = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0 or w < self.config.min_box_size or h < self.config.min_box_size:
                continue

            tracker_input.append([x1, y1, x2, y2, det.confidence])

        if len(tracker_input) == 0:
            tracker_input = np.empty((0, 5))
        else:
            tracker_input = np.array(tracker_input, dtype=np.float32)

        # Update OC-SORT tracker
        try:
            if frame is not None:
                img_h, img_w = frame.shape[:2]
                img_info = [img_h, img_w]
                img_size = [img_h, img_w]
            else:
                img_info = [1080, 1920]
                img_size = [1080, 1920]

            tracked = self.tracker.update(tracker_input, img_info, img_size)

            # Convert OC-SORT results to TrackedObject list
            tracked_objects = []
            tracked_ids = set()

            for track in tracked:
                if len(track) < 5:
                    continue

                x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
                track_id = int(track[4])
                score = float(track[5]) if len(track) > 5 else 1.0
                tracked_ids.add(track_id)

                # Find matching detection for class info
                class_id, class_name = 0, "person"
                for det in detections:
                    iou = self._compute_iou(np.array([x1, y1, x2, y2]), det.bbox)
                    if iou > 0.5:
                        class_id = det.class_id
                        class_name = det.class_name
                        break

                # Find or create track
                existing_track = None
                for existing in self.tracks:
                    if existing.track_id == track_id:
                        existing_track = existing
                        break

                if existing_track is not None:
                    existing_track.update(
                        np.array([x1, y1, x2, y2], dtype=np.float32),
                        frame_id,
                        score,
                        class_id=class_id,
                        class_name=class_name,
                    )
                    tracked_objects.append(existing_track)
                else:
                    new_track = TrackedObject(
                        track_id=track_id,
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        frame_id=frame_id,
                        confidence=score,
                        class_id=class_id,
                        class_name=class_name,
                    )
                    tracked_objects.append(new_track)

            # Handle lost tracks - mark them as lost but keep for interpolation
            # Update time_since_update for lost tracks and filter out old ones
            lost_tracks = []
            for existing_track in self.tracks:
                if existing_track.track_id not in tracked_ids:
                    existing_track.time_since_update += 1
                    # Keep track if it hasn't exceeded max_age (for potential re-identification)
                    if existing_track.time_since_update < self.config.tracking_max_age:
                        lost_tracks.append(existing_track)

            # Update tracks: current tracked objects + lost tracks that haven't expired
            self.tracks = tracked_objects + lost_tracks

            return tracked_objects
        except Exception as e:
            print(f"Warning: OC-SORT update failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def save_coco_annotations(
    video_path: Path,
    tracked_objects_by_frame: Dict[int, List[TrackedObject]],
    width: int,
    height: int,
    output_file: Path,
    config: Config,
    tracker=None,  # Tracker object to access all tracks with interpolated positions
    total_frames: int = None,
):
    """Save tracked objects as COCO format annotations."""
    # Collect all unique tracks from all frames and tracker
    all_unique_tracks: Dict[int, TrackedObject] = {}

    # First, collect from tracked_objects_by_frame
    for frame_objs in tracked_objects_by_frame.values():
        for obj in frame_objs:
            if obj.get_track_duration() >= config.tracking_min_hits:
                # Use most recent instance of each track
                if obj.track_id not in all_unique_tracks:
                    all_unique_tracks[obj.track_id] = obj
                elif obj.frame_id > all_unique_tracks[obj.track_id].frame_id:
                    all_unique_tracks[obj.track_id] = obj

    # Also collect from tracker.tracks (includes tracks with interpolated positions)
    if tracker is not None and hasattr(tracker, "tracks"):
        for track in tracker.tracks:
            if track.get_track_duration() >= config.tracking_min_hits:
                # Use most recent instance of each track
                if track.track_id not in all_unique_tracks:
                    all_unique_tracks[track.track_id] = track
                elif track.frame_id > all_unique_tracks[track.track_id].frame_id:
                    all_unique_tracks[track.track_id] = track

    # Create categories from detected classes
    class_ids_seen = set()
    for track in all_unique_tracks.values():
        class_ids_seen.add(track.class_id)

    categories = [
        {"id": class_id + 1, "name": COCO_CLASSES[class_id], "supercategory": "none"}
        for class_id in sorted(class_ids_seen)
        if class_id < len(COCO_CLASSES)
    ]

    if len(categories) == 0:
        categories = [{"id": 1, "name": "person", "supercategory": "none"}]

    # Create images and annotations
    images = []
    annotations = []
    annotation_id = 1

    # Get all frame indices - include all frames up to total_frames
    all_frame_indices = sorted(tracked_objects_by_frame.keys())
    if total_frames is not None:
        # Add all frames up to total_frames for complete image list
        all_frame_indices = sorted(set(all_frame_indices) | set(range(total_frames)))

    for frame_idx in all_frame_indices:
        image_id = frame_idx + 1
        images.append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": f"{video_path.stem}_frame_{frame_idx:06d}.jpg",
            }
        )

        # Check ALL tracks for this frame (detected or interpolated)
        for track_id, track in all_unique_tracks.items():
            # Get bbox for this frame (detected or interpolated)
            bbox = track.get_bbox_for_frame(frame_idx)
            if bbox is None:
                continue

            if len(bbox) != 4:
                continue
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                continue

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            if x2 <= x1 or y2 <= y1:
                continue

            coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            area = float((x2 - x1) * (y2 - y1))
            category_id = track.class_id + 1 if 0 <= track.class_id < len(COCO_CLASSES) else 1

            # Get 3D location if available
            location_3d = track.get_3d_location_for_frame(frame_idx)

            annotation_data = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "track_id": track.track_id,
            }

            # Add 3D location if available
            if location_3d is not None and len(location_3d) == 3:
                annotation_data["location_3d"] = [
                    float(location_3d[0]),
                    float(location_3d[1]),
                    float(location_3d[2]),
                ]

            annotations.append(annotation_data)
            annotation_id += 1

    coco_data = {
        "info": {
            "description": f"GMIND Video Annotations - {video_path.name}",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "GMIND SDK",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✓ Saved COCO annotations to: {output_file}", flush=True)
    print(f"  Images: {len(images)}", flush=True)
    print(f"  Annotations: {len(annotations)}", flush=True)


def process_video(video_path: str, config: Config):
    """Process video with object detection (Dome-DETR or YOLOv12x) + OC-SORT tracking."""
    # Convert Windows path to WSL path if needed
    if video_path.startswith("H:\\") or video_path.startswith("H:/"):
        video_path = (
            video_path.replace("H:\\", "/mnt/h/").replace("H:/", "/mnt/h/").replace("\\", "/")
        )

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-calculate frame skip to achieve target processing FPS
    if config.process_every_n_frames == 1 and config.target_processing_fps > 0 and fps > 0:
        calculated_skip = max(1, round(fps / config.target_processing_fps))
        config.process_every_n_frames = calculated_skip
        actual_processing_fps = fps / calculated_skip
        print(
            f"Auto-calculated frame skip: {calculated_skip} (targeting {config.target_processing_fps} fps)",
            flush=True,
        )
        print(f"Actual processing rate: {actual_processing_fps:.2f} fps", flush=True)

    print(f"\nVideo: {video_path.name}", flush=True)
    print(f"Resolution: {width}x{height}", flush=True)
    print(f"Video FPS: {fps:.2f}", flush=True)
    print(f"Processing: every {config.process_every_n_frames} frame(s)", flush=True)
    print(f"Total frames: {total_frames}", flush=True)
    print(f"Detection: {config.detector_model}", flush=True)
    if config.detector_model.lower() == "dome-detr":
        print(f"  Config: {config.detector_config_file}", flush=True)
        print(f"  Checkpoint: {config.detector_checkpoint}", flush=True)
    print(f"Tracking: {config.tracking_method}", flush=True)

    # Parse camera intrinsics and distortion coefficients from calibration file if provided
    camera_matrix = config.camera_matrix
    dist_coeffs = config.dist_coeffs
    if (
        config.enable_depth_estimation
        and camera_matrix is None
        and config.calibration_file is not None
    ):
        # Try to parse from calibration file
        import os

        calib_path = config.calibration_file
        # Convert Windows path to WSL path if needed
        if calib_path.startswith("H:\\") or calib_path.startswith("H:/"):
            calib_path = (
                calib_path.replace("H:\\", "/mnt/h/").replace("H:/", "/mnt/h/").replace("\\", "/")
            )

        # If relative path, try resolving from current working directory or script directory
        if not os.path.isabs(calib_path):
            if os.path.exists(calib_path):
                calib_path = os.path.abspath(calib_path)
            else:
                # Try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(script_dir)  # Go up one level from Annotation/
                potential_path = os.path.join(parent_dir, calib_path)
                if os.path.exists(potential_path):
                    calib_path = os.path.abspath(potential_path)

        camera_matrix, dist_coeffs = parse_camera_intrinsics_from_calibration(
            calib_path,
            config.camera_name,
            video_path=str(video_path),  # Pass video path for automatic name extraction
        )
        if camera_matrix is not None:
            config.camera_matrix = camera_matrix
        if dist_coeffs is not None:
            config.dist_coeffs = dist_coeffs

    # Print depth estimation method
    if config.enable_depth_estimation:
        print(
            f"✓ 3D location computation enabled: Geometric (ground plane intersection)", flush=True
        )
        print(
            f"  Camera height: {config.camera_height}m, Pitch: {config.camera_pitch_deg}°",
            flush=True,
        )
        if config.camera_matrix is not None:
            print(f"  Using camera intrinsics for geometric projection", flush=True)
        else:
            print(
                f"  ⚠️  Warning: Camera intrinsics not available, geometric depth will fail",
                flush=True,
            )

    object_detector = ObjectDetector(config)
    tracker = Tracker(config)

    tracked_objects_by_frame: Dict[int, List[TrackedObject]] = {}
    frame_idx = 0
    inference_times = []

    if not config.headless:
        cv2.namedWindow("Video with Tracking", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if config.process_every_n_frames > 1 and frame_idx % config.process_every_n_frames != 0:
            frame_idx += 1
            continue

        start_time = time.time()

        detections = object_detector.detect(frame, conf_threshold=config.detector_conf_threshold)

        # Compute 3D locations for detections first (before tracking)
        # This allows us to pass 3D locations when updating tracks for proper interpolation
        detection_3d_locations = {}  # Map: detection_idx -> 3D location
        if config.enable_depth_estimation:
            if config.camera_matrix is not None:
                # Import robust geometric 3D projection function (only when needed)
                try:
                    from .footpoint_to_ground import (
                        bbox_to_3d_geometric_robust as bbox_to_3d_geometric,
                    )
                except ImportError:
                    from footpoint_to_ground import (
                        bbox_to_3d_geometric_robust as bbox_to_3d_geometric,
                    )

                # Use robust geometric ground plane intersection method with distortion correction
                for i, det in enumerate(detections):
                    location_3d = bbox_to_3d_geometric(
                        bbox=det.bbox,
                        camera_matrix=config.camera_matrix,
                        camera_height=config.camera_height,
                        camera_pitch_deg=config.camera_pitch_deg,
                        ground_height=config.ground_height,
                        dist_coeffs=config.dist_coeffs,
                        camera_roll_deg=config.camera_roll_deg,
                        camera_yaw_deg=config.camera_yaw_deg,
                    )
                    if location_3d is not None:
                        detection_3d_locations[i] = location_3d

        # Update tracker with detections
        tracked_objects = tracker.update(detections, frame_idx, frame)

        # Update tracks with 3D locations (matching detections to tracks)
        if config.enable_depth_estimation:
            # Match detections to tracked objects and assign 3D locations
            for track in tracked_objects:
                bbox = track.bbox

                # Find matching detection by IoU
                best_match_idx = None
                best_iou = 0.0
                for i, det in enumerate(detections):
                    iou = tracker._compute_iou(bbox, det.bbox)
                    if iou > 0.5 and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i

                # Get 3D location from matched detection, or compute it
                location_3d = None
                if best_match_idx is not None and best_match_idx in detection_3d_locations:
                    location_3d = detection_3d_locations[best_match_idx]
                else:
                    # Compute directly from track bbox
                    if config.camera_matrix is not None:
                        location_3d = bbox_to_3d_geometric(
                            bbox=bbox,
                            camera_matrix=config.camera_matrix,
                            camera_height=config.camera_height,
                            camera_pitch_deg=config.camera_pitch_deg,
                            ground_height=config.ground_height,
                            dist_coeffs=config.dist_coeffs,
                            camera_roll_deg=config.camera_roll_deg,
                            camera_yaw_deg=config.camera_yaw_deg,
                        )

                # Update track with 3D location
                if location_3d is not None:
                    # Retroactively update the track (this will trigger interpolation if needed)
                    track.frame_3d_locations[frame_idx] = location_3d.copy()
                    track.last_known_3d_location = location_3d.copy()

                    # If there was a gap, we need to interpolate 3D locations retroactively
                    if track.last_known_frame >= 0 and frame_idx > track.last_known_frame + 1:
                        # We need to interpolate between the previous 3D location and this one
                        prev_3d = track.frame_3d_locations.get(track.last_known_frame)
                        if prev_3d is not None and len(prev_3d) == 3:
                            # Interpolate 3D locations for the gap
                            for gap_frame in range(track.last_known_frame + 1, frame_idx):
                                alpha = (gap_frame - track.last_known_frame) / (
                                    frame_idx - track.last_known_frame
                                )
                                alpha = np.clip(alpha, 0.0, 1.0)
                                interpolated_3d = (1 - alpha) * prev_3d + alpha * location_3d
                                track.interpolated_3d_locations[gap_frame] = interpolated_3d.copy()

            # Also compute 3D locations for all tracks in tracker (including pre-tracking ones)
            # This ensures we get depth for objects that haven't reached min_hits yet
            for track in tracker.tracks:
                # Skip if we already processed this track above
                if track.track_id in [t.track_id for t in tracked_objects]:
                    continue

                # Get bbox for this frame (detected or interpolated)
                bbox = track.get_bbox_for_frame(frame_idx)
                if bbox is not None:
                    # Compute 3D location
                    if config.camera_matrix is not None:
                        location_3d = bbox_to_3d_geometric(
                            bbox=bbox,
                            camera_matrix=config.camera_matrix,
                            camera_height=config.camera_height,
                            camera_pitch_deg=config.camera_pitch_deg,
                            ground_height=config.ground_height,
                            dist_coeffs=config.dist_coeffs,
                            camera_roll_deg=config.camera_roll_deg,
                            camera_yaw_deg=config.camera_yaw_deg,
                        )
                    else:
                        location_3d = None

                    # Update track with 3D location
                    if location_3d is not None:
                        track.frame_3d_locations[frame_idx] = location_3d.copy()
                        track.last_known_3d_location = location_3d.copy()

                        # Retroactively interpolate if there was a gap
                        if track.last_known_frame >= 0 and frame_idx > track.last_known_frame + 1:
                            prev_3d = track.frame_3d_locations.get(track.last_known_frame)
                            if prev_3d is not None and len(prev_3d) == 3:
                                for gap_frame in range(track.last_known_frame + 1, frame_idx):
                                    alpha = (gap_frame - track.last_known_frame) / (
                                        frame_idx - track.last_known_frame
                                    )
                                    alpha = np.clip(alpha, 0.0, 1.0)
                                    interpolated_3d = (1 - alpha) * prev_3d + alpha * location_3d
                                    track.interpolated_3d_locations[gap_frame] = (
                                        interpolated_3d.copy()
                                    )

        valid_tracked_objects = [
            obj for obj in tracked_objects if obj.get_track_duration() >= config.tracking_min_hits
        ]

        if config.save_annotations:
            # Store all valid tracks for this frame
            # Also include tracks that might have interpolated positions (from tracker.tracks)
            all_tracks_for_frame = valid_tracked_objects.copy()
            # Add any tracks that have interpolated positions for this frame
            for track in tracker.tracks:
                if track.get_track_duration() >= config.tracking_min_hits:
                    if track.track_id not in [t.track_id for t in all_tracks_for_frame]:
                        # Check if this track has a position (detected or interpolated) for this frame
                        if track.get_bbox_for_frame(frame_idx) is not None:
                            all_tracks_for_frame.append(track)
            tracked_objects_by_frame[frame_idx] = all_tracks_for_frame

        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        if len(inference_times) > 1000:
            inference_times.pop(0)

        if not config.headless:
            vis_frame = frame.copy()

            # Get ALL tracks to visualize (including those with < min_hits)
            # Show all tracks, but color them green when they become valid (>= min_hits)
            tracks_to_visualize = []
            track_ids_seen = set()

            # First add all valid tracks
            for track in valid_tracked_objects:
                tracks_to_visualize.append(track)
                track_ids_seen.add(track.track_id)

            # Add tracks with interpolated positions for this frame (valid ones)
            for track in tracker.tracks:
                if track.get_track_duration() >= config.tracking_min_hits:
                    if track.track_id not in track_ids_seen:
                        bbox = track.get_bbox_for_frame(frame_idx)
                        if bbox is not None:
                            tracks_to_visualize.append(track)
                            track_ids_seen.add(track.track_id)

            # Now add ALL tracks (including those with < min_hits) that have positions for this frame
            for track in tracker.tracks:
                if track.track_id not in track_ids_seen:
                    bbox = track.get_bbox_for_frame(frame_idx)
                    if bbox is not None:
                        tracks_to_visualize.append(track)
                        track_ids_seen.add(track.track_id)

            for track in tracks_to_visualize:
                # Get bbox for this frame (detected or interpolated)
                bbox = track.get_bbox_for_frame(frame_idx)
                if bbox is None or len(bbox) != 4:
                    continue
                if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                    continue

                # Check if this is an interpolated position
                is_interpolated = frame_idx in track.interpolated_positions

                # Check if track is valid (>= min_hits) - make it green when valid
                is_valid = track.get_track_duration() >= config.tracking_min_hits

                if is_valid:
                    # Green color for valid tracks
                    color = (0, 255, 0)  # Green in BGR
                else:
                    # Use track ID-based color for invalid tracks (before they become valid)
                    track_id_hash = track.track_id * 137
                    color = (
                        int((track_id_hash * 50) % 200) + 55,
                        int((track_id_hash * 100) % 200) + 55,
                        int((track_id_hash * 150) % 200) + 55,
                    )

                # Draw bounding box (dashed for interpolated, solid for detected)
                if is_interpolated:
                    # Draw dashed rectangle
                    dash_length = 10
                    gap_length = 5
                    for x in range(x1, x2, dash_length + gap_length):
                        cv2.line(vis_frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
                        cv2.line(vis_frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
                    for y in range(y1, y2, dash_length + gap_length):
                        cv2.line(vis_frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
                        cv2.line(vis_frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
                else:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Get 3D location if available
                location_3d = track.get_3d_location_for_frame(frame_idx)

                label = f"ID:{track.track_id} {track.class_name}"
                if is_interpolated:
                    label = "[INT] " + label
                if not is_valid:
                    label = f"[{track.get_track_duration()}/{config.tracking_min_hits}] " + label
                if track.confidence < 1.0:
                    label += f" ({track.confidence:.2f})"

                # Add XYZ position if available (always show if 3D is enabled)
                if config.enable_depth_estimation:
                    if location_3d is not None and len(location_3d) == 3:
                        # Check for invalid values
                        if not (np.any(np.isnan(location_3d)) or np.any(np.isinf(location_3d))):
                            X, Y, Z = location_3d[0], location_3d[1], location_3d[2]
                            label += f" | XYZ:({X:.1f},{Y:.1f},{Z:.1f})m"
                        else:
                            label += " | XYZ:---"
                    else:
                        # Show that 3D location is missing
                        label += " | XYZ:---"

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                cv2.putText(
                    vis_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            info_text = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(valid_tracked_objects)} | FPS: {1.0/inference_time:.1f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                info_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            cv2.rectangle(
                vis_frame, (10, 10), (10 + text_width, 10 + text_height + baseline), (0, 0, 0), -1
            )
            cv2.putText(
                vis_frame,
                info_text,
                (10, 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Video with Tracking", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        frame_idx += 1

        if frame_idx % 30 == 0:
            avg_fps = 1.0 / np.mean(inference_times[-30:]) if len(inference_times) > 0 else 0
            progress_msg = f"[{video_path.name}] Frame {frame_idx}/{total_frames}: {len(valid_tracked_objects)} tracks, {avg_fps:.1f} FPS"

            # Add 3D location info if available
            if config.enable_depth_estimation and len(valid_tracked_objects) > 0:
                # Show 3D locations for first few tracks
                xyz_info = []
                for track in valid_tracked_objects[:3]:  # Show first 3 tracks
                    loc_3d = track.get_3d_location_for_frame(frame_idx)
                    if loc_3d is not None and len(loc_3d) == 3:
                        X, Y, Z = loc_3d[0], loc_3d[1], loc_3d[2]
                        xyz_info.append(f"ID{track.track_id}:({X:.1f},{Y:.1f},{Z:.1f})m")
                if xyz_info:
                    progress_msg += f" | 3D: {', '.join(xyz_info)}"

            print(progress_msg, flush=True)

    cap.release()
    if not config.headless:
        cv2.destroyAllWindows()

    if config.save_annotations and len(tracked_objects_by_frame) > 0:
        output_file = video_path.parent / f"{video_path.stem}_anno.json"
        save_coco_annotations(
            video_path,
            tracked_objects_by_frame,
            width,
            height,
            output_file,
            config,
            tracker=tracker,
            total_frames=total_frames,
        )

    avg_inference_time = np.mean(inference_times) if inference_times else 0
    if avg_inference_time > 0:
        print(
            f"\nAverage inference time: {avg_inference_time*1000:.2f}ms ({1.0/avg_inference_time:.1f} FPS)",
            flush=True,
        )
    print(f"Total frames processed: {frame_idx}", flush=True)


if __name__ == "__main__":
    # Example usage
    config = Config()
    config.video_path = "path/to/your/video.mp4"  # Set video path
    # Optionally enable 3D location computation
    # config.enable_depth_estimation = True
    # config.camera_height = 4.0
    # config.camera_pitch_deg = 20.0

    process_video(config.video_path, config)
