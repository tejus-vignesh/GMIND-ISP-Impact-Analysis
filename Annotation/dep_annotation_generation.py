"""
Advanced video annotation generator using foreground segmentation and dual detection.

This module provides a comprehensive video annotation pipeline optimized for static
camera setups with static backgrounds. It combines foreground segmentation, dual
detector systems, and advanced tracking to generate high-quality COCO-format
annotations.

Pipeline:
    1. MOG2 (BGS) → Foreground segmentation
    2. Connected components → Object blobs
    3. OC-SORT or ByteTrack → Multi-object tracking
    4. Dual detector: YOLOv11x (full frame) + RT-DETR-X (regions) with NMS merging
    5. Static object detection via random frame sampling

Features:
    - Foreground segmentation for moving object detection
    - Dual detector system (full frame + region-based)
    - Static object detection
    - Kalman filter extrapolation for occlusions
    - Bicycle-pedestrian association for robust tracking

Example:
    >>> from Annotation.dep_annotation_generation import Config, process_video
    >>> config = Config()
    >>> config.video_path = "path/to/video.mp4"
    >>> config.use_dual_detector = True
    >>> process_video(config.video_path, config)
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import time
from dataclasses import dataclass, field
import json
from datetime import datetime
from collections import Counter
# Using OpenCV for connected components instead of skimage


@dataclass
class Config:
    """Configuration parameters for the video annotation pipeline."""
    
    # Video input
    video_path: str = ""  # Path to input video file
    process_every_n_frames: int = 5  # Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
    
    # Foreground Segmentation
    fg_seg_method: str = "BGS"  # Only MOG2 (BGS) is supported
    fg_seg_history: int = 2000  # History for BGS
    fg_seg_var_threshold: int = 16  # Variance threshold for BGS
    fg_seg_detect_shadows: bool = True
    fg_seg_learning_rate: float = -1  # Learning rate (-1 = auto, 0-1 = manual)
    fg_seg_morph_kernel_size: int = 5  # Morphological operations kernel size
    
    # Connected Components
    cc_min_area: int = 100  # Minimum blob area in pixels
    cc_connectivity: int = 8  # 4 or 8 connectivity
    blob_expansion_ratio: float = 0.15  # Expand blob bounding boxes by this ratio (15% padding)
    
    # Object Detection
    detector_model: str = "rtdetr-x"  # Model to use: "yolo11x", "yolov8x", "rtdetr-l", "rtdetr-x", "maskrcnn_resnet50_fpn_v2"
    detector_conf_threshold: float = 0.3  # Confidence threshold for detections
    detect_static_objects: bool = True  # Run detector on full frame to catch static objects
    static_detection_sample_frames: int = 20  # Number of random frames to sample for static object detection
    static_detection_iou_threshold: float = 0.5  # IoU threshold for grouping static objects across frames
    use_dual_detector: bool = True  # Use YOLOv11x for full frame + RT-DETR-X for regions, then merge with NMS
    full_frame_detection_frequency: int = 0  # DEPRECATED: Static objects are detected via static_detection_sample_frames. Set to 0 to disable.
    nms_threshold: float = 0.2  # IoU threshold for NMS when merging detections (lower = more aggressive)
    nms_nested_ratio: float = 0.25  # If smaller box area is < this ratio of larger box, merge nested detections
    nms_nested_iou: float = 0.5  # If IoU > this and one box is much smaller, merge (for partial nesting)
    filter_by_aspect_ratio: bool = True  # Filter detections by aspect ratio based on expected object shapes
    # Aspect ratio bounds (width/height) for each class - kept lenient to account for lens distortion
    # Person: typically tall (width < height), aspect ratio ~0.2-2.0 (lenient for distortion)
    # Bicycle: typically wider (width > height), aspect ratio ~0.5-4.0 (lenient for angle and distortion)
    # Car: typically much wider (width > height), aspect ratio ~0.8-5.0 (lenient for distortion)
    aspect_ratio_bounds: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        0: (0.2, 2.0),    # Person: width/height between 0.2 and 2.0 (tall, lenient for distortion/poses)
        1: (0.5, 4.0),    # Bicycle: width/height between 0.5 and 4.0 (varies by angle, lenient for distortion)
        2: (0.8, 5.0),    # Car: width/height between 0.8 and 5.0 (wider than tall, lenient for distortion)
    })
    min_box_size: int = 5  # Minimum width or height in pixels (filters invalid/suspiciously small boxes)
    
    # Tracking (OC-SORT/ByteTrack)
    tracking_method: str = "OC-SORT"  # Options: "OC-SORT", "ByteTrack"
    tracking_max_age: int = 60  # Maximum frames to keep lost tracks
    tracking_min_hits: int = 4  # Minimum hits to confirm track 
    tracking_iou_threshold: float = 0.2  # IoU threshold for matching
    tracking_det_thresh: float = detector_conf_threshold  # Detection confidence threshold for OC-SORT (should match detector_conf_threshold)
    tracking_delta_t: int = 3  # Delta time for observation-centric re-update (OC-SORT)
    tracking_inertia: float = 0.2  # Inertia weight for OC-SORT (0.0-1.0)
    tracking_asso_func: str = "giou"  # Association function: "iou", "giou", "ciou", "diou", "ct_dist" (GIoU handles scale changes better)
    enable_bicycle_pedestrian_association: bool = True  # Link bicycles and pedestrians when nearby for robust tracking
    enable_backward_validation: bool = False  # If True, include early frames of tracks that become valid later (retroactive validation)
    max_speed_pixels_per_frame: float = 200.0  # Maximum allowed movement in pixels per frame (filters unrealistic jumps)
    filter_static_dynamic_objects: bool = True  # Filter out dynamic tracks that are actually static (BGS noise)
    static_dynamic_max_movement: float = 20.0  # Maximum total movement in pixels for a track to be considered static
    static_dynamic_min_frames: int = 10  # Minimum number of frames a track must be detected before checking if it's static
    
    # Output
    save_annotations: bool = True
    headless: bool = False
    debug_visualization: bool = False
    show_extrapolated_boxes: bool = False  # Show extrapolated (lost) tracks in visualization


@dataclass
class TrackedObject:
    """Represents a tracked object across frames with Kalman filter for extrapolation."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    frame_id: int
    confidence: float = 1.0
    class_id: int = 0  # COCO class ID
    class_name: str = "object"  # Class name (e.g., "person", "car")
    # Extrapolation state
    time_since_update: int = 0  # Frames since last detection
    was_lost: bool = False  # Was this track lost and then regained?
    extrapolated_frames: Any = field(default_factory=list)  # Frame IDs where this track was extrapolated
    kalman_filter: Any = None  # Kalman filter for position prediction
    # Interpolation state
    last_known_bbox: Any = None  # Last known position before loss
    last_known_frame: int = -1  # Frame ID of last known position
    extrapolated_positions: Any = field(default_factory=dict)  # {frame_id: bbox} for extrapolated positions
    interpolated_positions: Any = field(default_factory=dict)  # {frame_id: bbox} for interpolated positions
    frame_positions: Any = field(default_factory=dict)  # {frame_id: bbox} for actual detected positions at each frame
    # Tracking statistics
    first_frame: int = -1  # First frame where this track appeared
    hit_streak: int = 0  # Number of consecutive frames with detections
    position_history: Any = field(default_factory=list)  # History of positions for static detection
    class_history: Any = field(default_factory=list)  # History of class_id from all detections for this track
    # Bicycle-pedestrian association
    associated_track_ids: Any = field(default_factory=set)  # Track IDs of associated objects (e.g., bicycle-person pairs)
    
    def __post_init__(self):
        """Initialize Kalman filter after object creation."""
        if self.kalman_filter is None:
            self._init_kalman()
        # Initialize class_history with initial class_id if provided
        if len(self.class_history) == 0 and self.class_id is not None:
            self.class_history.append(self.class_id)
    
    def _init_kalman(self):
        """Initialize Kalman filter for position and velocity tracking."""
        try:
            from filterpy.kalman import KalmanFilter
            # State: [cx, cy, w, h, vx, vy] (center x, center y, width, height, velocity x, velocity y)
            kf = KalmanFilter(dim_x=6, dim_z=4)
            
            # State transition matrix (constant velocity model)
            dt = 1.0  # Time step (1 frame)
            kf.F = np.array([
                [1, 0, 0, 0, dt, 0],   # cx' = cx + vx*dt
                [0, 1, 0, 0, 0, dt],   # cy' = cy + vy*dt
                [0, 0, 1, 0, 0, 0],    # w' = w
                [0, 0, 0, 1, 0, 0],    # h' = h
                [0, 0, 0, 0, 1, 0],    # vx' = vx
                [0, 0, 0, 0, 0, 1]     # vy' = vy
            ], dtype=np.float32)
            
            # Measurement matrix (we observe center and size)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],   # observe cx
                [0, 1, 0, 0, 0, 0],   # observe cy
                [0, 0, 1, 0, 0, 0],   # observe w
                [0, 0, 0, 1, 0, 0]    # observe h
            ], dtype=np.float32)
            
            # Process noise (uncertainty in motion model)
            kf.Q = np.eye(6, dtype=np.float32) * 0.1
            
            # Measurement noise (uncertainty in observations)
            kf.R = np.eye(4, dtype=np.float32) * 10.0
            
            # Initial state uncertainty
            kf.P = np.eye(6, dtype=np.float32) * 100.0
            
            # Initialize state from bbox
            x1, y1, x2, y2 = self.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            kf.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float32)
            
            self.kalman_filter = kf
        except ImportError:
            # Fallback if filterpy not available
            self.kalman_filter = None
    
    def update(self, bbox: np.ndarray, frame_id: int, confidence: float = 1.0, class_id: Optional[int] = None, class_name: Optional[str] = None):
        """Update track with new detection."""
        # Check if this is a re-identification after being lost
        was_lost_before = self.time_since_update > 0
        
        if was_lost_before:
            self.was_lost = True
            # Track was regained - perform interpolation
            self._interpolate_after_regain(bbox, frame_id)
        
        # Store last known position before potential loss (only if we have a current bbox)
        if self.bbox is not None:
            self.last_known_bbox = self.bbox.copy()
            self.last_known_frame = self.frame_id
        
        # Track first frame and hit streak
        if self.first_frame < 0:
            self.first_frame = frame_id
        self.hit_streak += 1
        
        # Store position history for static detection (keep last 10 positions)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.position_history.append((cx, cy))
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Update class based on majority vote (most common class across all detections)
        if class_id is not None:
            self.class_history.append(class_id)
            # Limit class_history to last 100 entries to prevent unbounded growth
            # This is enough for majority vote while preventing memory issues
            if len(self.class_history) > 100:
                self.class_history.pop(0)
            # Use majority vote (mode) for class_id
            if len(self.class_history) > 0:
                # Count occurrences of each class_id
                class_counts = Counter(self.class_history)
                # Get the most common class_id
                most_common_class_id, _ = class_counts.most_common(1)[0]
                self.class_id = most_common_class_id
                # Update class_name based on most common class_id
                # Import COCO_CLASSES from module level (defined later in file)
                import sys
                current_module = sys.modules[__name__]
                if hasattr(current_module, 'COCO_CLASSES'):
                    COCO_CLASSES = getattr(current_module, 'COCO_CLASSES')
                    self.class_name = COCO_CLASSES[self.class_id] if 0 <= self.class_id < len(COCO_CLASSES) else f'class_{self.class_id}'
                else:
                    # Fallback if COCO_CLASSES not yet defined
                    if self.class_id == 0:
                        self.class_name = "person"
                    elif self.class_id == 1:
                        self.class_name = "bicycle"
                    elif self.class_id == 2:
                        self.class_name = "car"
                    else:
                        self.class_name = f'class_{self.class_id}'
        
        # Update bbox
        self.bbox = bbox.copy()
        self.frame_id = frame_id
        self.confidence = confidence
        self.time_since_update = 0
        
        # Store bbox for this frame (for proper retrieval during export)
        # Limit frame_positions to last 1000 frames per track to prevent unbounded growth
        # Older frames should be cleaned up by the tracker when tracks are removed
        self.frame_positions[frame_id] = bbox.copy()
        # Clean up old frame positions (keep only last 1000)
        if len(self.frame_positions) > 1000:
            # Remove oldest entries
            sorted_frames = sorted(self.frame_positions.keys())
            for old_frame in sorted_frames[:-1000]:
                del self.frame_positions[old_frame]
        
        # Clear extrapolated positions (they've been interpolated)
        if was_lost_before:
            self.extrapolated_positions.clear()
        
        # Update Kalman filter
        if self.kalman_filter is not None:
            w = x2 - x1
            h = y2 - y1
            measurement = np.array([cx, cy, w, h], dtype=np.float32)
            self.kalman_filter.update(measurement)
    
    def is_static(self, max_movement: float = 20.0, min_frames: int = 10) -> bool:
        """
        Check if this track has been static (not moving much) over its lifetime.
        
        Args:
            max_movement: Maximum total pixel movement to consider as static
            min_frames: Minimum number of frames required to check if static
        
        Returns:
            True if the object has been static (movement < max_movement) for at least min_frames
        """
        if len(self.position_history) < min_frames:
            return False
        
        # Calculate total movement over the track's lifetime
        if len(self.position_history) < 2:
            return False
        
        # Get first and last positions
        first_pos = self.position_history[0]
        last_pos = self.position_history[-1]
        
        # Calculate total displacement
        total_dx = abs(last_pos[0] - first_pos[0])
        total_dy = abs(last_pos[1] - first_pos[1])
        total_movement = max(total_dx, total_dy)
        
        # Also check maximum single-step movement to catch jittery but stationary objects
        max_single_step = 0
        for i in range(1, len(self.position_history)):
            dx = abs(self.position_history[i][0] - self.position_history[i-1][0])
            dy = abs(self.position_history[i][1] - self.position_history[i-1][1])
            max_single_step = max(max_single_step, max(dx, dy))
        
        # Consider static if both total movement and single-step movement are below threshold
        return total_movement < max_movement and max_single_step < max_movement * 1.5
    
    def get_track_duration(self) -> int:
        """Get number of frames this track has been active."""
        if self.first_frame < 0:
            return 0
        return self.hit_streak
    
    def _interpolate_after_regain(self, regained_bbox: np.ndarray, regained_frame_id: int):
        """Interpolate between last known position, extrapolated positions, and regained position."""
        if self.last_known_frame < 0 or len(self.extrapolated_positions) == 0:
            # No interpolation needed if we don't have history
            return
        
        # Get all frames that need interpolation
        frames_to_interpolate = sorted(self.extrapolated_positions.keys())
        if len(frames_to_interpolate) == 0:
            return
        
        # Start point: last known position before loss
        start_bbox = self.last_known_bbox.copy()
        start_frame = self.last_known_frame
        
        # End point: regained position
        end_bbox = regained_bbox.copy()
        end_frame = regained_frame_id
        
        # Interpolate for each extrapolated frame
        for frame_id in frames_to_interpolate:
            if frame_id <= start_frame or frame_id >= end_frame:
                continue
            
            # Linear interpolation between start and end
            alpha = (frame_id - start_frame) / (end_frame - start_frame)
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # Interpolate each bbox coordinate
            interpolated_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
            
            # Store interpolated position
            self.interpolated_positions[frame_id] = interpolated_bbox.copy()
            # Limit interpolated_positions to last 500 frames to prevent unbounded growth
            if len(self.interpolated_positions) > 500:
                sorted_frames = sorted(self.interpolated_positions.keys())
                for old_frame in sorted_frames[:-500]:
                    del self.interpolated_positions[old_frame]
            
            # Update the extrapolated_frames list to mark as interpolated
            # (we'll use interpolated_positions to check if a frame was interpolated)
    
    def interpolate_between_frames(self, start_frame: int, start_bbox: np.ndarray, 
                                    end_frame: int, end_bbox: np.ndarray, 
                                    skipped_frames: List[int]):
        """Interpolate between two known frames for a list of skipped frames."""
        if start_frame >= end_frame:
            return
        
        # Interpolate for each skipped frame
        for frame_id in skipped_frames:
            if frame_id <= start_frame or frame_id >= end_frame:
                continue
            
            # Linear interpolation between start and end
            alpha = (frame_id - start_frame) / (end_frame - start_frame)
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # Interpolate each bbox coordinate
            interpolated_bbox = (1 - alpha) * start_bbox + alpha * end_bbox
            
            # Store interpolated position (overwrite any extrapolated position)
            self.interpolated_positions[frame_id] = interpolated_bbox.copy()
            # Limit interpolated_positions to last 500 frames to prevent unbounded growth
            if len(self.interpolated_positions) > 500:
                sorted_frames = sorted(self.interpolated_positions.keys())
                for old_frame in sorted_frames[:-500]:
                    del self.interpolated_positions[old_frame]
    
    def predict(self, frame_id: int) -> np.ndarray:
        """Predict next position using Kalman filter."""
        if self.kalman_filter is None:
            # Fallback: return current bbox
            return self.bbox.copy()
        
        # Predict next state
        self.kalman_filter.predict()
        
        # Extract predicted bbox from state
        state = self.kalman_filter.x
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        
        # Check for NaN or invalid values
        if np.any(np.isnan([cx, cy, w, h])) or np.any(np.isinf([cx, cy, w, h])):
            # Return current bbox if prediction is invalid
            return self.bbox.copy()
        
        # Ensure width and height are positive
        w = max(1.0, w)
        h = max(1.0, h)
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        predicted_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        # Validate predicted bbox
        if np.any(np.isnan(predicted_bbox)) or np.any(np.isinf(predicted_bbox)):
            # Return current bbox if prediction is invalid
            return self.bbox.copy()
        
        # Store extrapolated position
        self.extrapolated_positions[frame_id] = predicted_bbox.copy()
        # Limit extrapolated_positions to last 200 frames to prevent unbounded growth
        if len(self.extrapolated_positions) > 200:
            sorted_frames = sorted(self.extrapolated_positions.keys())
            for old_frame in sorted_frames[:-200]:
                del self.extrapolated_positions[old_frame]
        
        # Update time since last update
        self.time_since_update += 1
        
        # Record this as an extrapolated frame
        if frame_id not in self.extrapolated_frames:
            self.extrapolated_frames.append(frame_id)
            # Limit extrapolated_frames list to last 200 frames
            if len(self.extrapolated_frames) > 200:
                self.extrapolated_frames.pop(0)
        
        return predicted_bbox
    
    def get_bbox_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get bbox for a specific frame, using interpolated position if available, then stored frame position, then extrapolated, then interpolation between nearby frames.
        
        Returns None if no valid position is available for this frame (should skip annotation for this frame).
        """
        # Priority 1: Check if we have an interpolated position for this frame
        if frame_id in self.interpolated_positions:
            bbox = self.interpolated_positions[frame_id].copy()
            # Validate interpolated bbox
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        # Priority 2: Check if we have a stored position for this exact frame
        if frame_id in self.frame_positions:
            bbox = self.frame_positions[frame_id].copy()
            # Validate frame position bbox
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        # Priority 3: Check if we have an extrapolated position for this frame
        # Only use extrapolated positions if track was regained (was_lost=True) or has interpolated positions
        # This ensures we don't export pure extrapolations without re-identification
        has_interpolated = frame_id in self.interpolated_positions if self.interpolated_positions else False
        if frame_id in self.extrapolated_positions and (self.was_lost or has_interpolated):
            bbox = self.extrapolated_positions[frame_id].copy()
            # Validate extrapolated bbox
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        # Priority 4: Try interpolation between available frames
        # Collect all available frame positions
        # Only include extrapolated positions if track was regained (was_lost=True or has interpolated positions)
        all_available_frames = set()
        if self.frame_positions:
            all_available_frames.update(self.frame_positions.keys())
        if self.interpolated_positions:
            all_available_frames.update(self.interpolated_positions.keys())
        # Only use extrapolated positions if track was regained (interpolated or was_lost)
        if self.extrapolated_positions and (self.was_lost or len(self.interpolated_positions) > 0):
            all_available_frames.update(self.extrapolated_positions.keys())
        
        if len(all_available_frames) > 0:
            # Find nearest frames before and after
            sorted_frames = sorted(all_available_frames)
            before_frame = None
            after_frame = None
            
            for f in sorted_frames:
                if f < frame_id:
                    before_frame = f
                elif f > frame_id:
                    after_frame = f
                    break
            
            # Try to get bbox from before/after frames
            before_bbox = None
            after_bbox = None
            
            if before_frame is not None:
                before_bbox = self._get_bbox_from_dict(before_frame)
            
            if after_frame is not None:
                after_bbox = self._get_bbox_from_dict(after_frame)
            
            # Interpolate if we have both before and after
            if before_bbox is not None and after_bbox is not None:
                alpha = (frame_id - before_frame) / (after_frame - before_frame)
                alpha = np.clip(alpha, 0.0, 1.0)
                interpolated_bbox = (1 - alpha) * before_bbox + alpha * after_bbox
                if interpolated_bbox is not None and len(interpolated_bbox) == 4 and not np.any(np.isnan(interpolated_bbox)) and not np.any(np.isinf(interpolated_bbox)):
                    return interpolated_bbox
            
            # Extrapolate if we only have before (forward extrapolation)
            if before_bbox is not None and after_frame is None:
                # Use last known position
                return before_bbox.copy()
            
            # Extrapolate if we only have after (backward extrapolation)
            if after_bbox is not None and before_frame is None:
                return after_bbox.copy()  # Use first known position
        
        # Priority 5: Fallback to current bbox (only if frame_id matches current frame or very close)
        if self.frame_id >= 0 and abs(frame_id - self.frame_id) <= 1:
            if self.bbox is not None and len(self.bbox) == 4 and not np.any(np.isnan(self.bbox)) and not np.any(np.isinf(self.bbox)):
                return self.bbox.copy()
        
        # No valid position available - return None to skip this frame
        return None
    
    def _get_bbox_from_dict(self, frame_id: int) -> Optional[np.ndarray]:
        """Helper to get bbox from any available dictionary for a frame."""
        # Try interpolated first (most accurate)
        if frame_id in self.interpolated_positions:
            bbox = self.interpolated_positions[frame_id].copy()
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        # Try stored frame position
        if frame_id in self.frame_positions:
            bbox = self.frame_positions[frame_id].copy()
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        # Try extrapolated position (only if track was regained)
        has_interpolated = frame_id in self.interpolated_positions if self.interpolated_positions else False
        if frame_id in self.extrapolated_positions and (self.was_lost or has_interpolated):
            bbox = self.extrapolated_positions[frame_id].copy()
            if bbox is not None and len(bbox) == 4 and not np.any(np.isnan(bbox)) and not np.any(np.isinf(bbox)):
                return bbox
        
        return None


class ForegroundSegmenter:
    """Foreground segmentation using MOG2 (BGS)."""
    
    def __init__(self, config: Config):
        self.config = config
        
        if config.fg_seg_method != "BGS":
            raise ValueError(f"Only 'BGS' (MOG2) is supported. Got: {config.fg_seg_method}")
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.fg_seg_history,
            varThreshold=config.fg_seg_var_threshold,
            detectShadows=config.fg_seg_detect_shadows
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (config.fg_seg_morph_kernel_size, config.fg_seg_morph_kernel_size)
        )
    
    def segment(self, frame: np.ndarray, video_name: str = "") -> np.ndarray:
        """Get foreground mask for the current frame."""
        if self.config.fg_seg_method == "BGS":
            # For CCTV videos, apply slight blur to reduce Moire patterns
            if "CCTV" in video_name.upper():
                frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
            
            # Apply learning rate if specified
            if self.config.fg_seg_learning_rate >= 0:
                fg_mask = self.bg_subtractor.apply(frame, learningRate=self.config.fg_seg_learning_rate)
            else:
                fg_mask = self.bg_subtractor.apply(frame)
            
            # Apply median filter to reduce noise
            if self.config.fg_seg_morph_kernel_size > 0:
                fg_mask = cv2.medianBlur(fg_mask, 5)
            
            # Apply morphological operations to clean up the mask
            if self.config.fg_seg_morph_kernel_size > 0:
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
                # Additional dilation to fill small gaps and expand regions
                fg_mask = cv2.dilate(fg_mask, self.morph_kernel, iterations=5)
            
            return fg_mask


# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class BlobExtractor:
    """Extract object blobs from foreground mask using connected components."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract(self, fg_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract object blobs from foreground mask using connected components.
    
    Returns:
            List of blob dictionaries with keys: 'bbox', 'area', 'centroid', 'mask'
        """
        # Connected components analysis using OpenCV
        num_labels, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(
            fg_mask, 
            connectivity=self.config.cc_connectivity
        )
        
        blobs = []
        for label_id in range(1, num_labels):  # Skip background (label 0)
            # Get stats for this component
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            # Filter by minimum area
            if area < self.config.cc_min_area:
                continue
            
            # Get bounding box from stats (original, before expansion)
            x_min_orig = stats[label_id, cv2.CC_STAT_LEFT]
            y_min_orig = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            x_max_orig = x_min_orig + width
            y_max_orig = y_min_orig + height
            
            # Reject regions smaller than 20px x 20px
            if width < 20 or height < 20:
                continue
            
            # Store original bbox
            bbox_orig = np.array([x_min_orig, y_min_orig, x_max_orig, y_max_orig], dtype=np.float32)
            
            # Expand blob bounding box with adaptive expansion ratio
            # Smaller objects get disproportionately larger expansion for more context
            x_min = x_min_orig
            y_min = y_min_orig
            x_max = x_max_orig
            y_max = y_max_orig
            if self.config.blob_expansion_ratio > 0:
                # Calculate adaptive expansion ratio based on blob size
                # Smaller blobs get more expansion, larger blobs get less
                # Use area as the size metric
                blob_area = width * height
                
                # Reference sizes for scaling (in pixels)
                small_blob_size = 1000  # Blobs smaller than this get maximum expansion
                large_blob_size = 10000  # Blobs larger than this get minimum expansion
                
                # Calculate adaptive ratio: smaller = more expansion
                if blob_area <= small_blob_size:
                    # Small blobs: use 2x the base expansion ratio
                    adaptive_ratio = self.config.blob_expansion_ratio * 2.0
                elif blob_area >= large_blob_size:
                    # Large blobs: use base expansion ratio
                    adaptive_ratio = self.config.blob_expansion_ratio
                else:
                    # Medium blobs: linear interpolation between 2x and 1x
                    # Smaller blobs get closer to 2x, larger get closer to 1x
                    t = (blob_area - small_blob_size) / (large_blob_size - small_blob_size)
                    adaptive_ratio = self.config.blob_expansion_ratio * (2.0 - t)
                
                # Clamp to reasonable bounds (0.1 to 0.5)
                adaptive_ratio = np.clip(adaptive_ratio, 0.1, 0.5)
                
                expand_w = width * adaptive_ratio
                expand_h = height * adaptive_ratio
                x_min = max(0, x_min - expand_w)
                y_min = max(0, y_min - expand_h)
                x_max = x_max + expand_w
                y_max = y_max + expand_h
            
            bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            
            # Get centroid
            centroid = np.array([centroids[label_id, 0], centroids[label_id, 1]], dtype=np.float32)
            
            # Get mask for this blob
            blob_mask = (labeled_mask == label_id)
            
            blobs.append({
                'bbox': bbox,  # Expanded bbox (used for YOLO)
                'bbox_orig': bbox_orig,  # Original bbox (from connected components)
                'area': area,
                'centroid': centroid,
                'mask': blob_mask
            })
        
        return blobs


@dataclass
class Detection:
    """Represents a single object detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class ObjectDetector:
    """Run object detector on foreground ROIs to get instance-level detections with class names."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Primary model (for regions)
        self.model_full_frame = None  # Full frame model (YOLOv11x)
        self.backend = None
        self.backend_full_frame = None
        self._load_model()
    
    @staticmethod
    def _resize_with_aspect_ratio(image: np.ndarray, target_size: int = 640) -> tuple:
        """
        Resize image to target_size while preserving aspect ratio using letterbox padding.
        
        Args:
            image: Input image (H, W, C)
            target_size: Target size (square)
        
        Returns:
            (resized_image, scale, pad_x, pad_y) where:
            - resized_image: Padded and resized image (target_size, target_size, C)
            - scale: Scale factor applied
            - pad_x: Horizontal padding (left padding)
            - pad_y: Vertical padding (top padding)
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit within target_size
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_size, target_size), dtype=image.dtype)
        
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded, scale, pad_x, pad_y
    
    def _load_model(self):
        """Load the specified object detection model(s)."""
        if self.config.use_dual_detector:
            # Dual detector mode: YOLOv11x for full frame, RT-DETR-X for regions
            print("Loading dual detector setup...")
            
            # Load YOLOv11x for full frame
            try:
                from DeepLearning.adapters import get_model
                self.model_full_frame = get_model('yolo11x', num_classes=80, pretrained=True, backend='ultralytics')
                if hasattr(self.model_full_frame, 'to'):
                    self.model_full_frame.to(self.device)
                if hasattr(self.model_full_frame, 'eval'):
                    self.model_full_frame.eval()
                self.backend_full_frame = 'ultralytics'
                print("✓ Loaded YOLOv11x for full frame detection")
            except Exception as e:
                print(f"  YOLOv11x not available: {e}")
                raise RuntimeError(f"Failed to load YOLOv11x for dual detector mode: {e}")
            
            # Load RT-DETR-X for region detection
            try:
                from DeepLearning.adapters import get_model
                self.model = get_model('rtdetr-x', num_classes=80, pretrained=True, backend='ultralytics')
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.backend = 'ultralytics'
                print("✓ Loaded RT-DETR-X for region detection")
            except Exception as e:
                print(f"  RT-DETR-X not available: {e}")
                # Fallback to RT-DETR-L
                try:
                    self.model = get_model('rtdetr-l', num_classes=80, pretrained=True, backend='ultralytics')
                    if hasattr(self.model, 'to'):
                        self.model.to(self.device)
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    self.backend = 'ultralytics'
                    print("✓ Loaded RT-DETR-L for region detection (fallback)")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load RT-DETR for dual detector mode: {e2}")
            return
        
        # Single model mode (original behavior)
        print(f"Loading object detection model: {self.config.detector_model}...")
        
        # Check if it's an RT-DETR model
        if self.config.detector_model.startswith('rtdetr'):
            try:
                from DeepLearning.adapters import get_model
                self.model = get_model(self.config.detector_model, num_classes=80, pretrained=True, backend='ultralytics')
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.backend = 'ultralytics'
                print(f"✓ Loaded {self.config.detector_model.upper()} (Ultralytics)")
                return
            except Exception as e:
                print(f"  {self.config.detector_model} not available: {e}")
                print("  Falling back to default model...")
        
        # Check if it's a YOLO model
        if 'yolo' in self.config.detector_model.lower():
            try:
                from DeepLearning.adapters import get_model
                self.model = get_model(self.config.detector_model, num_classes=80, pretrained=True, backend='ultralytics')
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.backend = 'ultralytics'
                print(f"✓ Loaded {self.config.detector_model.upper()} (Ultralytics)")
                return
            except Exception as e:
                print(f"  {self.config.detector_model} not available: {e}")
                print("  Falling back to default model...")
        
        # Try YOLOv11x (latest) as default
        try:
            from DeepLearning.adapters import get_model
            self.model = get_model('yolo11x', num_classes=80, pretrained=True, backend='ultralytics')
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            self.backend = 'ultralytics'
            print("✓ Loaded YOLOv11x (Ultralytics) - default")
            return
        except Exception as e:
            print(f"  YOLOv11x not available: {e}")
        
        # Try YOLOv8x
        try:
            from DeepLearning.adapters import get_model
            self.model = get_model('yolov8x', num_classes=80, pretrained=True, backend='ultralytics')
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            self.backend = 'ultralytics'
            print("✓ Loaded YOLOv8x (Ultralytics) - fallback")
            return
        except Exception as e:
            print(f"  YOLOv8x not available: {e}")
        
        # Fallback to TorchVision models
        try:
            from DeepLearning.adapters import get_model
            self.model = get_model('maskrcnn_resnet50_fpn_v2', num_classes=91, pretrained=True, backend='torchvision')
            self.model.to(self.device)
            self.model.eval()
            self.backend = 'torchvision'
            print("✓ Loaded Mask R-CNN ResNet50 FPN v2 (TorchVision) - fallback")
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load any detection model: {e}")
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        # Validate inputs
        if box1 is None or box2 is None or len(box1) != 4 or len(box2) != 4:
            return 0.0
        
        if np.any(np.isnan(box1)) or np.any(np.isinf(box1)) or np.any(np.isnan(box2)) or np.any(np.isinf(box2)):
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
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
    
    @staticmethod
    def _validate_bbox(bbox: np.ndarray, min_box_size: int = 5) -> bool:
        """
        Validate bounding box has valid dimensions and meets minimum size requirements.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            min_box_size: Minimum width or height in pixels
        
        Returns:
            True if bbox is valid, False otherwise
        """
        if bbox is None or len(bbox) != 4:
            return False
        
        if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
            return False
        
        x1, y1, x2, y2 = bbox
        
        # Check coordinates are valid
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check minimum size
        width = x2 - x1
        height = y2 - y1
        
        if width < min_box_size or height < min_box_size:
            return False
        
        # Check non-negative coordinates (can be slightly negative due to rounding, allow small margin)
        if x1 < -10 or y1 < -10:
            return False
        
        return True
    
    @staticmethod
    def _check_aspect_ratio(bbox: np.ndarray, class_id: int, aspect_ratio_bounds: Dict[int, Tuple[float, float]]) -> bool:
        """
        Check if a detection's aspect ratio is within expected bounds for its class.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Object class ID (0=person, 1=bicycle, 2=car)
            aspect_ratio_bounds: Dictionary mapping class_id to (min_aspect_ratio, max_aspect_ratio)
        
        Returns:
            True if aspect ratio is within bounds, False otherwise
        """
        if class_id not in aspect_ratio_bounds:
            return True  # No bounds defined for this class, allow it
        
        if bbox is None or len(bbox) != 4:
            return False
        if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
            return False
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height <= 0 or width <= 0:
            return False
        
        aspect_ratio = width / height  # width/height
        min_ratio, max_ratio = aspect_ratio_bounds[class_id]
        
        return min_ratio <= aspect_ratio <= max_ratio
    
    @staticmethod
    def _is_box_within(box_inner: np.ndarray, box_outer: np.ndarray, min_overlap: float = 0.8) -> bool:
        """Check if inner box is mostly within outer box (at least min_overlap of inner box area)."""
        if box_inner is None or box_outer is None or len(box_inner) != 4 or len(box_outer) != 4:
            return False
        
        if np.any(np.isnan(box_inner)) or np.any(np.isinf(box_inner)) or np.any(np.isnan(box_outer)) or np.any(np.isinf(box_outer)):
            return False
        
        x1_in, y1_in, x2_in, y2_in = box_inner
        x1_out, y1_out, x2_out, y2_out = box_outer
        
        # Intersection
        x1_i = max(x1_in, x1_out)
        y1_i = max(y1_in, y1_out)
        x2_i = min(x2_in, x2_out)
        y2_i = min(y2_in, y2_out)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        inner_area = (x2_in - x1_in) * (y2_in - y1_in)
        
        if inner_area <= 0:
            return False
        
        overlap_ratio = intersection / inner_area
        return overlap_ratio >= min_overlap

    @staticmethod
    def _filter_drivers_in_cars(detections: List[Detection]) -> List[Detection]:
        """Filter out small, square pedestrian detections that are within car bounding boxes.
        
        This removes drivers detected as separate pedestrians inside cars, which are typically:
        - Small (area < 5% of car area or < absolute threshold)
        - Square-ish (aspect ratio close to 1.0, unlike standing pedestrians)
        - Within car bounding boxes (at least 80% overlap)
        """
        if len(detections) == 0:
            return []
        
        # Find all car detections (class_id = 2)
        cars = [det for det in detections if det.class_id == 2]
        if len(cars) == 0:
            return detections  # No cars, nothing to filter
        
        # Find all person detections (class_id = 0)
        persons = [det for det in detections if det.class_id == 0]
        if len(persons) == 0:
            return detections  # No persons, nothing to filter
        
        # Compute car areas for relative size comparison
        car_areas = {}
        for car in cars:
            if car.bbox is not None and len(car.bbox) == 4:
                x1, y1, x2, y2 = car.bbox
                car_areas[id(car)] = (x2 - x1) * (y2 - y1)
        
        filtered_detections = []
        
        for det in detections:
            # Keep all non-person detections
            if det.class_id != 0:
                filtered_detections.append(det)
                continue
            
            # Check if this person is a driver in a car
            person_bbox = det.bbox
            if person_bbox is None or len(person_bbox) != 4:
                filtered_detections.append(det)  # Keep invalid person detections
                continue
            
            x1, y1, x2, y2 = person_bbox
            person_w = x2 - x1
            person_h = y2 - y1
            person_area = person_w * person_h
            
            if person_area <= 0:
                filtered_detections.append(det)
                continue
            
            # Check aspect ratio (square-ness): ratio of width to height
            # Standing pedestrians are typically tall (h > w), so aspect ratio < 1.0
            # Square boxes (like driver upper body) have aspect ratio close to 1.0
            aspect_ratio = person_w / person_h if person_h > 0 else 1.0
            # Normalize to [0, 1] range where 1.0 = perfectly square
            # Closer to 1.0 means more square
            squareness = min(aspect_ratio, 1.0 / aspect_ratio)  # min(w/h, h/w)
            
            # Check if person is within any car
            is_within_car = False
            is_small_relative_to_car = False
            
            for car in cars:
                car_bbox = car.bbox
                if car_bbox is None or len(car_bbox) != 4:
                    continue
                
                # Check if person is mostly within car (80% overlap)
                if ObjectDetector._is_box_within(person_bbox, car_bbox, min_overlap=0.8):
                    is_within_car = True
                    
                    # Check if person is small relative to car (less than 5% of car area)
                    car_area = car_areas.get(id(car), 0)
                    if car_area > 0:
                        relative_size = person_area / car_area
                        if relative_size < 0.05:  # Less than 5% of car area
                            is_small_relative_to_car = True
                            break
            
            # Filter if: within car AND (small relative to car OR square-ish)
            # Use squareness threshold of 0.7 (closer to 1.0 = more square)
            # This catches square/compact driver boxes but allows tall standing pedestrians
            if is_within_car and (is_small_relative_to_car or squareness > 0.7):
                # Skip this detection (driver in car)
                continue
            
            # Keep this person detection
            filtered_detections.append(det)
        
        return filtered_detections

    def _apply_nms(self, detections: List[Detection], nms_threshold: float) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(detections) == 0:
            return []
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while len(sorted_dets) > 0:
            # Keep the highest confidence detection
            keep_det = sorted_dets.pop(0)
            keep.append(keep_det)
            
            # Remove overlapping detections of the same class
            remaining = []
            for det in sorted_dets:
                if det.class_id != keep_det.class_id:
                    # Different class - always keep
                    remaining.append(det)
                else:
                    # Same class - check IoU and nested boxes
                    iou = self._compute_iou(keep_det.bbox, det.bbox)
                    
                    # For high-confidence detections (>0.5), be more conservative with nested removal
                    # High confidence often indicates a real object, even if small
                    high_confidence_threshold = 0.5
                    is_high_confidence = det.confidence > high_confidence_threshold or keep_det.confidence > high_confidence_threshold
                    
                    # Check if one box is nested inside the other (complete nesting)
                    # Be less aggressive for high-confidence detections
                    nested_ratio = self.config.nms_nested_ratio * 0.5 if is_high_confidence else self.config.nms_nested_ratio
                    is_nested = self._is_nested_box(keep_det.bbox, det.bbox, nested_ratio)
                    
                    # Check for partial nesting (high IoU with significant size difference)
                    # Also be less aggressive for high-confidence detections
                    if not is_high_confidence:
                        is_partial_nested = self._is_partial_nested(keep_det.bbox, det.bbox, 
                                                                     self.config.nms_nested_iou, 
                                                                     self.config.nms_nested_ratio)
                    else:
                        # For high confidence, require even higher IoU and smaller size ratio
                        is_partial_nested = self._is_partial_nested(keep_det.bbox, det.bbox, 
                                                                     max(0.7, self.config.nms_nested_iou), 
                                                                     self.config.nms_nested_ratio * 0.5)
                    
                    if iou < nms_threshold and not is_nested and not is_partial_nested:
                        # Low overlap and not nested - keep both
                        remaining.append(det)
                    # High overlap or nested - discard (already kept the higher confidence one)
                    # BUT: if both are high confidence, prefer keeping both if overlap is not too high
                    elif is_high_confidence and iou < 0.6 and not is_nested:
                        # Both high confidence and moderate overlap - keep both (might be separate objects)
                        remaining.append(det)
            
            sorted_dets = remaining
        
        return keep
    
    @staticmethod
    def _is_nested_box(box1: np.ndarray, box2: np.ndarray, area_ratio_threshold: float = 0.5) -> bool:
        """
        Check if one box is nested inside another (for merging small boxes inside large ones).
    
    Args:
            box1: First bbox [x1, y1, x2, y2]
            box2: Second bbox [x1, y1, x2, y2]
            area_ratio_threshold: If smaller box area < this ratio of larger box, consider it nested
    
    Returns:
            True if one box is nested inside the other
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Check if box1 is inside box2 (with some tolerance for edge cases)
        # Allow small boxes to be slightly outside but mostly inside
        center1_x = (x1_1 + x2_1) / 2.0
        center1_y = (y1_1 + y2_1) / 2.0
        center2_x = (x1_2 + x2_2) / 2.0
        center2_y = (y1_2 + y2_2) / 2.0
        
        # Check if center of smaller box is inside larger box
        if area1 < area2 * area_ratio_threshold:
            # box1 is smaller - check if its center is inside box2
            if (x1_2 <= center1_x <= x2_2 and y1_2 <= center1_y <= y2_2):
                # Also check if most of box1 is inside box2 (at least 80% overlap)
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                if x2_i > x1_i and y2_i > y1_i:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    overlap_ratio = intersection / area1 if area1 > 0 else 0
                    if overlap_ratio >= 0.8:  # 80% of smaller box is inside larger
                        return True
        
        if area2 < area1 * area_ratio_threshold:
            # box2 is smaller - check if its center is inside box1
            if (x1_1 <= center2_x <= x2_1 and y1_1 <= center2_y <= y2_1):
                # Also check if most of box2 is inside box1 (at least 80% overlap)
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)
                if x2_i > x1_i and y2_i > y1_i:
                    intersection = (x2_i - x1_i) * (y2_i - y1_i)
                    overlap_ratio = intersection / area2 if area2 > 0 else 0
                    if overlap_ratio >= 0.8:  # 80% of smaller box is inside larger
                        return True
        
        return False
    
    @staticmethod
    def _is_partial_nested(box1: np.ndarray, box2: np.ndarray, 
                           iou_threshold: float = 0.5, 
                           area_ratio_threshold: float = 0.25) -> bool:
        """
        Check if one box is partially nested inside another (high IoU with size difference).
        Useful for cases where boxes aren't completely nested but one is much smaller.
    
    Args:
            box1: First bbox [x1, y1, x2, y2]
            box2: Second bbox [x1, y1, x2, y2]
            iou_threshold: Minimum IoU to consider
            area_ratio_threshold: If smaller box area < this ratio of larger box, consider it nested
    
    Returns:
            True if one box is partially nested inside the other
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Compute IoU
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0.0
        
        # If IoU is high and one box is much smaller, consider it nested
        if iou >= iou_threshold:
            if area1 < area2 * area_ratio_threshold or area2 < area1 * area_ratio_threshold:
                return True
        
        return False
    
    def _parse_ultralytics_results(self, results, conf_threshold: float, 
                                    fg_mask: np.ndarray = None, 
                                    require_fg_overlap: bool = False) -> List[Detection]:
        """Parse Ultralytics model results into Detection objects."""
        detections = []
        
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                boxes_data = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes_data)):
                    if scores[i] < conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = boxes_data[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    class_id = int(class_ids[i])
                    
                    # Filter to only person (0), bicycle (1), and car (2)
                    if class_id not in [0, 1, 2]:
                        continue
                    
                    class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f'class_{class_id}'
                    
                    # Check foreground overlap if required
                    if require_fg_overlap and fg_mask is not None:
                        roi_mask = fg_mask[y1:y2, x1:x2]
                        overlaps_fg = roi_mask.size > 0 and roi_mask.sum() > 0
                        if not overlaps_fg:
                            continue
                    
                    # Check aspect ratio if filtering is enabled
                    bbox_array = np.array([x1, y1, x2, y2], dtype=np.float32)
                    if self.config.filter_by_aspect_ratio:
                        if not ObjectDetector._check_aspect_ratio(
                            bbox_array, class_id, self.config.aspect_ratio_bounds
                        ):
                            continue  # Filter out detection with invalid aspect ratio
                    
                    detections.append(Detection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        confidence=float(scores[i]),
                        class_id=class_id,
                        class_name=class_name
                    ))
        
        return detections

    def detect(self, frame: np.ndarray, fg_mask: np.ndarray, blobs: List[Dict[str, Any]], 
               conf_threshold: float = 0.25, detect_static: bool = True, run_full_frame: bool = False) -> List[Detection]:
        """
        Run object detection on foreground ROIs and optionally full frame for static objects.
    
    Args:
            frame: Full frame image
            fg_mask: Foreground mask
            blobs: List of blob ROIs from BlobExtractor
            conf_threshold: Confidence threshold for detections
            detect_static: If True, also run detector on full frame to catch static objects
    
    Returns:
            List of Detection objects with class names and bounding boxes (merged with NMS)
        """
        all_detections = []
        
        # Dual detector mode: RT-DETR-X on regions (always), YOLOv11x on full frame (optional, but static objects handled separately)
        if self.config.use_dual_detector and self.model is not None:
            # Run RT-DETR-X on blob regions (this is the primary detection method for moving objects)
            if len(blobs) > 0:
                try:
                    for blob in blobs:
                        x1, y1, x2, y2 = map(int, blob['bbox'])
                        # Clip to image bounds
                        x1 = max(0, min(x1, frame.shape[1]))
                        y1 = max(0, min(y1, frame.shape[0]))
                        x2 = max(0, min(x2, frame.shape[1]))
                        y2 = max(0, min(y2, frame.shape[0]))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Extract region
                        roi = frame[y1:y2, x1:x2].copy()
                        if roi.size == 0:
                            continue
            
                        # Resize with aspect ratio preservation (letterbox padding)
                        roi_resized, scale, pad_x, pad_y = self._resize_with_aspect_ratio(roi, target_size=640)
                        
                        # Run RT-DETR-X on region
                        results_region = self.model.predict(roi_resized, conf=conf_threshold, verbose=False)
                        region_dets = self._parse_ultralytics_results(results_region, conf_threshold)
                        
                        # Adjust coordinates back to full frame
                        # First, convert from padded/resized coordinates to original ROI coordinates
                        # Then, add ROI offset to get full frame coordinates
                        for det in region_dets:
                            # Convert from padded/resized (640x640) to original ROI coordinates
                            det_x1 = (det.bbox[0] - pad_x) / scale
                            det_y1 = (det.bbox[1] - pad_y) / scale
                            det_x2 = (det.bbox[2] - pad_x) / scale
                            det_y2 = (det.bbox[3] - pad_y) / scale
                            
                            # Clip to ROI bounds
                            det_x1 = max(0, min(det_x1, x2 - x1))
                            det_y1 = max(0, min(det_y1, y2 - y1))
                            det_x2 = max(0, min(det_x2, x2 - x1))
                            det_y2 = max(0, min(det_y2, y2 - y1))
                            
                            # Add ROI offset to get full frame coordinates
                            det.bbox[0] = det_x1 + x1
                            det.bbox[1] = det_y1 + y1
                            det.bbox[2] = det_x2 + x1
                            det.bbox[3] = det_y2 + y1
                        
                        all_detections.extend(region_dets)
                except Exception as e:
                    print(f"Warning: Region RT-DETR-X detection failed: {e}")
            
            # Merge with NMS
            if len(all_detections) > 0:
                all_detections = self._apply_nms(all_detections, self.config.nms_threshold)
            
            return all_detections
        
        # Single model mode (original behavior)
        if self.model is None:
            return []
    
        # Strategy 1: Run detector on full frame and filter to foreground regions (moving objects)
        # Strategy 2: If detect_static=True, also run on full frame without filtering (static objects)
        
        if self.backend == 'ultralytics':
            # Ultralytics YOLO
            try:
                results = self.model.predict(frame, conf=conf_threshold, verbose=False)
                
                # Parse results
                if len(results) > 0 and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        # Get boxes, scores, and class IDs
                        boxes_data = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                        scores = boxes.conf.cpu().numpy()
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        
                        # Process detections
                        for i in range(len(boxes_data)):
                            x1, y1, x2, y2 = boxes_data[i]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get class name
                            class_id = int(class_ids[i])
                            
                            # Filter to only person (0), bicycle (1), and car (2)
                            if class_id not in [0, 1, 2]:
                                continue
                            
                            class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f'class_{class_id}'
                            
                            # Check if detection overlaps with foreground mask (moving objects)
                            roi_mask = fg_mask[y1:y2, x1:x2]
                            overlaps_fg = roi_mask.size > 0 and roi_mask.sum() > 0
                            
                            # Include if: overlaps foreground OR detect_static is True (for static objects)
                            if overlaps_fg or detect_static:
                                all_detections.append(Detection(
                                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                    confidence=float(scores[i]),
                                    class_id=class_id,
                                    class_name=class_name
                                ))
            except Exception as e:
                print(f"Warning: YOLO detection failed: {e}")
        
        elif self.backend == 'torchvision':
            # TorchVision models
            try:
                from torchvision import transforms
                transform = transforms.Compose([transforms.ToTensor()])
                img_tensor = transform(frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                
                # Parse outputs (TorchVision format)
                if len(outputs) > 0:
                    boxes = outputs[0]['boxes'].cpu().numpy()
                    scores = outputs[0]['scores'].cpu().numpy()
                    labels = outputs[0]['labels'].cpu().numpy().astype(int)
                    
                    # Filter by confidence and foreground overlap
                    for i in range(len(boxes)):
                        if scores[i] < conf_threshold:
                            continue
                        
                        x1, y1, x2, y2 = boxes[i]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class name (TorchVision uses 1-indexed, COCO is 0-indexed)
                        class_id = int(labels[i]) - 1  # Convert to 0-indexed
                        
                        # Filter to only person (0), bicycle (1), and car (2)
                        if class_id not in [0, 1, 2]:
                            continue
                        
                        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f'class_{class_id}'
                        
                        # Check if detection overlaps with foreground mask (moving objects)
                        roi_mask = fg_mask[y1:y2, x1:x2]
                        overlaps_fg = roi_mask.size > 0 and roi_mask.sum() > 0
                        
                        # Include if: overlaps foreground OR detect_static is True (for static objects)
                        if overlaps_fg or detect_static:
                            all_detections.append(Detection(
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                confidence=float(scores[i]),
                                class_id=class_id,
                                class_name=class_name
                            ))
            except Exception as e:
                print(f"Warning: TorchVision detection failed: {e}")
        
        return all_detections


class Tracker:
    """Object tracker using OC-SORT or ByteTrack."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tracks: List[TrackedObject] = []
        self.next_id = 1
        self.frame_count = 0
        self.tracker = None
        self.use_advanced_tracker = False
        
        if config.tracking_method == "OC-SORT":
            try:
                # Try multiple import paths for OC-SORT
                # Installation options:
                # 1. pip install ocsort
                # 2. pip install git+https://github.com/noahcao/OC_SORT.git
                # 3. Clone from https://github.com/noahcao/OC_SORT and add to path
                ocsort_loaded = False
                
                # Try import path 1: from OC_SORT directory (if cloned locally)
                try:
                    import sys
                    import os
                    # Check if OC_SORT is in the same directory (Annotation/OC_SORT)
                    oc_sort_path = os.path.join(os.path.dirname(__file__), 'OC_SORT')
                    oc_sort_path = os.path.abspath(oc_sort_path)
                    if os.path.exists(oc_sort_path) and oc_sort_path not in sys.path:
                        sys.path.insert(0, oc_sort_path)
                    
                    # Also try parent directory (for backwards compatibility)
                    oc_sort_path_parent = os.path.join(os.path.dirname(__file__), '..', 'OC_SORT')
                    oc_sort_path_parent = os.path.abspath(oc_sort_path_parent)
                    if os.path.exists(oc_sort_path_parent) and oc_sort_path_parent not in sys.path:
                        sys.path.insert(0, oc_sort_path_parent)
                    
                    # Also try current directory
                    if './OC_SORT' not in sys.path and os.path.exists('./OC_SORT'):
                        sys.path.insert(0, os.path.abspath('./OC_SORT'))
                    
                    # Try importing from trackers.ocsort_tracker
                    from trackers.ocsort_tracker.ocsort import OCSort
                    self.tracker = OCSort(
                        det_thresh=config.tracking_det_thresh,
                        max_age=config.tracking_max_age,
                        min_hits=config.tracking_min_hits,
                        iou_threshold=config.tracking_iou_threshold,
                        delta_t=config.tracking_delta_t,
                        asso_func=config.tracking_asso_func,
                        inertia=config.tracking_inertia
                    )
                    ocsort_loaded = True
                    print("✓ Loaded OC-SORT from local OC_SORT directory")
                except ImportError:
                    pass
                
                # Try import path 2: direct import (if installed via pip)
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
                            inertia=config.tracking_inertia
                        )
                        ocsort_loaded = True
                    except ImportError:
                        pass
                
                # Try import path 3: ocsort.ocsort
                if not ocsort_loaded:
                    try:
                        from ocsort.ocsort import OCSort
                        self.tracker = OCSort(
                            det_thresh=0.3,
                            max_age=config.tracking_max_age,
                            min_hits=config.tracking_min_hits,
                            iou_threshold=config.tracking_iou_threshold
                        )
                        ocsort_loaded = True
                    except ImportError:
                        pass
                
                # Try import path 4: from other common paths
                if not ocsort_loaded:
                    try:
                        import sys
                        import os
                        possible_paths = [
                            os.path.expanduser('~/OC_SORT'),
                            os.path.expanduser('~/oc_sort'),
                            '../OC_SORT'
                        ]
                        for path in possible_paths:
                            if os.path.exists(path) and path not in sys.path:
                                sys.path.insert(0, path)
                                try:
                                    from trackers.ocsort_tracker.ocsort import OCSort
                                    self.tracker = OCSort(
                                        det_thresh=config.tracking_det_thresh,
                                        max_age=config.tracking_max_age,
                                        min_hits=config.tracking_min_hits,
                                        iou_threshold=config.tracking_iou_threshold,
                                        delta_t=config.tracking_delta_t,
                                        asso_func=config.tracking_asso_func,
                                        inertia=config.tracking_inertia
                                    )
                                    ocsort_loaded = True
                                    print(f"✓ Loaded OC-SORT from {path}")
                                    break
                                except ImportError:
                                    continue
                    except Exception:
                        pass
                
                if ocsort_loaded:
                    self.use_advanced_tracker = True
                    print("✓ Loaded OC-SORT tracker")
                else:
                    raise ImportError(
                        "OC-SORT not found. Please install it using one of:\n"
                        "  1. pip install ocsort\n"
                        "  2. pip install git+https://github.com/noahcao/OC_SORT.git\n"
                        "  3. Clone from https://github.com/noahcao/OC_SORT\n"
                        "     and set PYTHONPATH or add to sys.path"
                    )
            except Exception as e:
                print(f"Warning: Could not load OC-SORT: {e}")
                print("Falling back to simple IoU tracker.")
                self.tracker = None
                self.use_advanced_tracker = False
        elif config.tracking_method == "ByteTrack":
            try:
                # Try multiple import paths for ByteTrack
                # Installation options:
                # 1. pip install byte-track
                # 2. pip install git+https://github.com/ifzhang/ByteTrack.git
                # 3. Clone from https://github.com/ifzhang/ByteTrack and add to path
                bytetrack_loaded = False
                
                # Try import path 1: direct import
                try:
                    from byte_tracker import BYTETracker
                    self.tracker = BYTETracker(
                        track_thresh=config.detector_conf_threshold,  # Use same threshold as detector
                        track_buffer=config.tracking_max_age,
                        match_thresh=config.tracking_iou_threshold,
                        frame_rate=30  # Will be updated per frame
                    )
                    bytetrack_loaded = True
                except ImportError:
                    pass
                
                # Try import path 2: from ByteTrack directory (if cloned)
                if not bytetrack_loaded:
                    try:
                        import sys
                        import os
                        # Try common installation paths
                        possible_paths = [
                            os.path.expanduser('~/ByteTrack'),
                            os.path.expanduser('~/byte-track'),
                            './ByteTrack',
                            '../ByteTrack'
                        ]
                        for path in possible_paths:
                            if os.path.exists(path) and path not in sys.path:
                                sys.path.insert(0, path)
                                try:
                                    from byte_tracker import BYTETracker
                                    self.tracker = BYTETracker(
                                        track_thresh=config.detector_conf_threshold,  # Use same threshold as detector
                                        track_buffer=config.tracking_max_age,
                                        match_thresh=config.tracking_iou_threshold,
                                        frame_rate=30
                                    )
                                    bytetrack_loaded = True
                                    print(f"✓ Loaded ByteTrack from {path}")
                                    break
                                except ImportError:
                                    continue
                    except Exception:
                        pass
                
                if bytetrack_loaded:
                    self.use_advanced_tracker = True
                    print("✓ Loaded ByteTrack tracker")
                else:
                    raise ImportError(
                        "ByteTrack not found. Please install it using one of:\n"
                        "  1. pip install byte-track\n"
                        "  2. pip install git+https://github.com/ifzhang/ByteTrack.git\n"
                        "  3. Clone from https://github.com/ifzhang/ByteTrack\n"
                        "     and set PYTHONPATH or add to sys.path"
                    )
            except Exception as e:
                print(f"Warning: Could not load ByteTrack: {e}")
                print("Falling back to simple IoU tracker.")
                self.tracker = None
                self.use_advanced_tracker = False
    
    def update(self, detections: List[Detection], frame_id: int, frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
        """
        Update tracker with new detections.
    
    Args:
            detections: List of Detection objects from ObjectDetector
            frame_id: Current frame ID
            frame: Optional frame image (needed for some trackers)
    
    Returns:
            List of tracked objects
        """
        self.frame_count = frame_id
        
        if not self.use_advanced_tracker or self.tracker is None:
            # Fallback: Simple IoU-based tracker
            return self._update_simple(detections, frame_id)
        
        # Convert detections to format expected by tracker
        # Format: [x1, y1, w, h, score, class_id] or [x1, y1, w, h, score]
        tracker_input = []
        for det in detections:
            bbox = det.bbox
            
            # Validate bbox before passing to tracker (prevents sqrt warnings)
            if not ObjectDetector._validate_bbox(bbox, self.config.min_box_size):
                continue  # Skip invalid boxes
            
            # Convert to [x1, y1, w, h, score] format
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # Double-check width/height are valid (should be caught by validate_bbox, but safety check)
            if w <= 0 or h <= 0 or w < self.config.min_box_size or h < self.config.min_box_size:
                continue
            
            tracker_input.append([x1, y1, w, h, det.confidence])
        
        if len(tracker_input) == 0:
            tracker_input = np.empty((0, 5))
        else:
            tracker_input = np.array(tracker_input, dtype=np.float32)
        
        # Update tracker based on method
        if self.config.tracking_method == "OC-SORT":
            try:
                # OC-SORT update signature: update(output_results, img_info, img_size)
                # output_results: [[x1, y1, x2, y2, score], ...] format
                # img_info: [height, width] of original image
                # img_size: [height, width] of model input (can be same as img_info for us)
                
                # Convert from [x1, y1, w, h, score] to [x1, y1, x2, y2, score]
                detections_xyxy = tracker_input.copy()
                if len(detections_xyxy) > 0:
                    detections_xyxy[:, 2] = detections_xyxy[:, 0] + detections_xyxy[:, 2]  # x2 = x1 + w
                    detections_xyxy[:, 3] = detections_xyxy[:, 1] + detections_xyxy[:, 3]  # y2 = y1 + h
                
                # Get image dimensions from frame or use default
                if frame is not None:
                    img_h, img_w = frame.shape[:2]
                    img_info = [img_h, img_w]
                    img_size = [img_h, img_w]  # Same for our case
                else:
                    # If no frame, use a default size (OC-SORT needs this)
                    img_info = [1080, 1920]  # Default HD size
                    img_size = [1080, 1920]
                
                # Call OC-SORT update
                tracked = self.tracker.update(detections_xyxy, img_info, img_size)
                
                # OC-SORT returns [[x1, y1, x2, y2, track_id], ...] format
                # Map back to original detections to get class names
                tracked_objects = []
                for track in tracked:
                    if len(track) >= 5:
                        x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
                        track_id = int(track[4])
                        score = float(track[5]) if len(track) > 5 else 1.0
                        
                        # Find matching detection to get class info
                        class_id = 0
                        class_name = "object"
                        for det in detections:
                            det_bbox = det.bbox
                            # Check if this track matches the detection (by IoU)
                            iou = self._compute_iou(np.array([x1, y1, x2, y2]), det_bbox)
                            if iou > 0.5:  # Good match
                                class_id = det.class_id
                                class_name = det.class_name
                                break
                        
                        # Check if this is an existing track (for maintaining state)
                        existing_track = None
                        for existing in self.tracks:
                            if existing.track_id == track_id:
                                existing_track = existing
                                break
                        
                        if existing_track is not None:
                            # Update existing track (class will be updated based on majority vote)
                            existing_track.update(
                                np.array([x1, y1, x2, y2], dtype=np.float32),
                                frame_id,
                                score,
                                class_id=class_id,
                                class_name=class_name
                            )
                            tracked_objects.append(existing_track)
                        else:
                            # New track (class_history initialized in __post_init__)
                            new_track = TrackedObject(
                                track_id=track_id,
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                frame_id=frame_id,
                                confidence=score,
                                class_id=class_id,
                                class_name=class_name
                            )
                            tracked_objects.append(new_track)
                
                # Handle lost tracks: extrapolate using Kalman filter
                tracked_ids = {t.track_id for t in tracked_objects}
                for existing_track in self.tracks:
                    if existing_track.track_id not in tracked_ids:
                        # Track is lost - extrapolate
                        if existing_track.time_since_update < self.config.tracking_max_age:
                            predicted_bbox = existing_track.predict(frame_id)
                            # Use interpolated position if available, otherwise use predicted
                            existing_track.bbox = existing_track.get_bbox_for_frame(frame_id)
                            existing_track.frame_id = frame_id
                            tracked_objects.append(existing_track)
                
                # Update internal tracks list
                self.tracks = tracked_objects
                return tracked_objects
            except Exception as e:
                print(f"Warning: OC-SORT update failed: {e}. Falling back to simple tracker.")
                import traceback
                traceback.print_exc()
                return self._update_simple(detections, frame_id)
        
        elif self.config.tracking_method == "ByteTrack":
            try:
                # ByteTrack update method signature
                # update(detections, img) -> [[x1, y1, x2, y2, track_id, score, class_id], ...]
                # ByteTrack expects [x1, y1, x2, y2, score] format
                
                # Convert from [x1, y1, w, h, score] to [x1, y1, x2, y2, score]
                detections_xyxy = tracker_input.copy()
                if len(detections_xyxy) > 0:
                    detections_xyxy[:, 2] = detections_xyxy[:, 0] + detections_xyxy[:, 2]  # x2 = x1 + w
                    detections_xyxy[:, 3] = detections_xyxy[:, 1] + detections_xyxy[:, 3]  # y2 = y1 + h
                
                # ByteTrack requires frame image
                if frame is None:
                    raise ValueError("ByteTrack requires frame image for update")
                
                tracked = self.tracker.update(detections_xyxy, frame)
                
                # Convert to TrackedObject list
                tracked_objects = []
                for track in tracked:
                    if len(track) >= 5:
                        x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
                        track_id = int(track[4])
                        score = float(track[5]) if len(track) > 5 else 1.0
                        
                        # Find matching detection to get class info
                        class_id = 0
                        class_name = "object"
                        for det in detections:
                            det_bbox = det.bbox
                            iou = self._compute_iou(np.array([x1, y1, x2, y2]), det_bbox)
                            if iou > 0.5:
                                class_id = det.class_id
                                class_name = det.class_name
                                break
                        
                        # Check if this is an existing track (for maintaining state)
                        existing_track = None
                        for existing in self.tracks:
                            if existing.track_id == track_id:
                                existing_track = existing
                                break
                        
                        if existing_track is not None:
                            # Update existing track (class will be updated based on majority vote)
                            existing_track.update(
                                np.array([x1, y1, x2, y2], dtype=np.float32),
                                frame_id,
                                score,
                                class_id=class_id,
                                class_name=class_name
                            )
                            tracked_objects.append(existing_track)
                        else:
                            # New track (class_history initialized in __post_init__)
                            new_track = TrackedObject(
                                track_id=track_id,
                                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                                frame_id=frame_id,
                                confidence=score,
                                class_id=class_id,
                                class_name=class_name
                            )
                            tracked_objects.append(new_track)
                
                # Handle lost tracks: extrapolate using Kalman filter
                tracked_ids = {t.track_id for t in tracked_objects}
                for existing_track in self.tracks:
                    if existing_track.track_id not in tracked_ids:
                        # Track is lost - extrapolate
                        if existing_track.time_since_update < self.config.tracking_max_age:
                            predicted_bbox = existing_track.predict(frame_id)
                            # Use interpolated position if available, otherwise use predicted
                            existing_track.bbox = existing_track.get_bbox_for_frame(frame_id)
                            existing_track.frame_id = frame_id
                            tracked_objects.append(existing_track)
                
                # Update internal tracks list
                self.tracks = tracked_objects
                return tracked_objects
            except Exception as e:
                print(f"Warning: ByteTrack update failed: {e}. Falling back to simple tracker.")
                import traceback
                traceback.print_exc()
                return self._update_simple(detections, frame_id)
        
        # Should not reach here, but fallback just in case
        return self._update_simple(detections, frame_id)
    
    def _update_simple(self, detections: List[Detection], frame_id: int) -> List[TrackedObject]:
        """Simple IoU-based tracker as fallback."""
        if len(detections) == 0:
            # Update existing tracks (mark as lost)
            for track in self.tracks:
                track.frame_id = frame_id
            # Remove old tracks
            self.tracks = [t for t in self.tracks 
                          if (frame_id - t.frame_id) < self.config.tracking_max_age]
            return self.tracks
        
        # Compute IoU matrix
        if len(self.tracks) == 0:
            # Create new tracks for all detections
            for det in detections:
                track = TrackedObject(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    frame_id=frame_id,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name
                )
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks
        
        # Match detections to existing tracks using IoU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            # Skip tracks with invalid bbox
            if track.bbox is None or len(track.bbox) != 4:
                continue
            if np.any(np.isnan(track.bbox)) or np.any(np.isinf(track.bbox)):
                continue
            
            for j, det in enumerate(detections):
                if det.bbox is None or len(det.bbox) != 4:
                    continue
                if np.any(np.isnan(det.bbox)) or np.any(np.isinf(det.bbox)):
                    continue
                
                iou_matrix[i, j] = self._compute_iou(track.bbox, det.bbox)
        
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        
        # Sort by IoU (highest first)
        matches = []
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.config.tracking_iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        
        for _, i, j in matches:
            if i not in matched_tracks and j not in matched_dets:
                # Update track (class will be updated based on majority vote)
                det = detections[j]
                self.tracks[i].update(det.bbox, frame_id, det.confidence, class_id=det.class_id, class_name=det.class_name)
                matched_tracks.add(i)
                matched_dets.add(j)
        
        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track = TrackedObject(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    frame_id=frame_id,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name
                )
                # class_history initialized in __post_init__
                self.tracks.append(track)
                self.next_id += 1
        
        # Handle lost tracks: extrapolate using Kalman filter
        matched_track_indices = set(matched_tracks)
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                # Track is lost - extrapolate
                if track.time_since_update < self.config.tracking_max_age:
                    predicted_bbox = track.predict(frame_id)
                    # Use interpolated position if available, otherwise use predicted
                    track.bbox = track.get_bbox_for_frame(frame_id)
                    track.frame_id = frame_id
        
        # Remove old unmatched tracks that exceeded max_age
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update < self.config.tracking_max_age]
        
        return self.tracks
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
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


def detect_static_objects(
    video_path: Path,
    object_detector: ObjectDetector,
    config: Config,
    total_frames: int,
    width: int,
    height: int
) -> List[Detection]:
    """
    Sample random frames and detect static objects that appear consistently.
    
    Returns:
        List of static object detections (averaged positions)
    """
    if not config.detect_static_objects or object_detector.model_full_frame is None:
        return []
    
    print(f"\nDetecting static objects: sampling {config.static_detection_sample_frames} random frames...", flush=True)
    
    # Sample random frame indices
    import random
    sample_frames = sorted(random.sample(range(total_frames), 
                                         min(config.static_detection_sample_frames, total_frames)))
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    # Collect all detections from sampled frames
    all_static_detections: List[List[Detection]] = []
    
    for sample_frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run full-frame detection only (no foreground filtering)
        try:
            results = object_detector.model_full_frame.predict(
                frame, 
                conf=config.detector_conf_threshold, 
                verbose=False
            )
            detections = object_detector._parse_ultralytics_results(results, config.detector_conf_threshold)
            all_static_detections.append(detections)
        except Exception as e:
            print(f"Warning: Static detection failed on frame {sample_frame_idx}: {e}")
            continue
    
    cap.release()
    
    if len(all_static_detections) == 0:
        return []
    
    # Group detections that appear consistently across frames (static objects)
    # Use IoU to match detections across frames
    static_objects: List[Detection] = []
    
    # Start with first frame's detections as candidates
    for det in all_static_detections[0]:
        # Check if this detection appears in multiple frames
        matches = [det]  # Start with first detection
        matched_frames = [0]
        
        for frame_idx in range(1, len(all_static_detections)):
            for other_det in all_static_detections[frame_idx]:
                if other_det.class_id != det.class_id:
                    continue
                
                iou = object_detector._compute_iou(det.bbox, other_det.bbox)
                if iou >= config.static_detection_iou_threshold:
                    matches.append(other_det)
                    matched_frames.append(frame_idx)
                    break
        
        # If detection appears in at least 50% of sampled frames, consider it static
        min_matches = max(2, len(all_static_detections) // 2)
        if len(matches) >= min_matches:
            # Average the bbox positions
            avg_bbox = np.mean([m.bbox for m in matches], axis=0)
            avg_confidence = np.mean([m.confidence for m in matches])
            
            static_det = Detection(
                bbox=avg_bbox,
                confidence=avg_confidence,
                class_id=det.class_id,
                class_name=det.class_name
            )
            static_objects.append(static_det)
    
    # Apply NMS to remove duplicates
    if len(static_objects) > 0:
        static_objects = object_detector._apply_nms(static_objects, config.nms_threshold)
    
    print(f"✓ Found {len(static_objects)} static objects", flush=True)
    return static_objects


def save_coco_annotations(
    video_path: Path,
    tracked_objects_by_frame: Dict[int, List[TrackedObject]],
    width: int,
    height: int,
    output_file: Path,
    config: Config,
    static_objects: List[Detection] = None,
    total_frames: int = None
):
    """Save tracked objects as COCO format annotations."""
    # Create categories from all detected classes (tracked + static)
    class_ids_seen = set()
    for frame_objs in tracked_objects_by_frame.values():
        for obj in frame_objs:
            class_ids_seen.add(obj.class_id)
    
    # Add static object classes
    if static_objects:
        for static_det in static_objects:
            class_ids_seen.add(static_det.class_id)
    
    categories = []
    for class_id in sorted(class_ids_seen):
        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f'class_{class_id}'
        categories.append({
            "id": class_id + 1,  # COCO uses 1-indexed
            "name": class_name,
            "supercategory": "none"
        })
    
    # If no categories found, add default
    if len(categories) == 0:
        categories = [{"id": 1, "name": "object", "supercategory": "none"}]
    
    # Assign unique track_ids to static objects (start from 100000 to avoid conflicts with dynamic tracks)
    static_object_track_ids: Dict[int, int] = {}
    if static_objects:
        next_static_track_id = 100000
        for i, static_det in enumerate(static_objects):
            static_object_track_ids[i] = next_static_track_id
            next_static_track_id += 1
    
    # Create images and annotations
    images = []
    annotations = []
    annotation_id = 1
    
    # Get all frame indices (including frames with no tracked objects)
    all_frame_indices = set(tracked_objects_by_frame.keys())
    
    # Handle case when there are no dynamic objects but we still need to create annotations
    if total_frames is not None:
        # Use total_frames if provided (covers all frames in video)
        all_frame_indices.update(range(total_frames))
    elif static_objects:
        # If we have static objects but no tracked_objects_by_frame, still need to create frames
        # Fallback: use max frame from tracked objects, or default to 0 if empty
        max_frame = max(all_frame_indices) if all_frame_indices else 0
        all_frame_indices.update(range(max_frame + 1))
    elif len(all_frame_indices) == 0:
        # No dynamic objects and no static objects - still create at least one frame
        all_frame_indices.add(0)
    
    # First pass: Identify all tracks that eventually become valid (duration >= tracking_min_hits)
    # If backward validation is enabled, include early frames of tracks that become valid later
    valid_track_ids = set()
    if config.enable_backward_validation:
        # Backward validation: find tracks that become valid later, include their early frames
        for frame_idx in sorted(all_frame_indices):
            frame_tracked_objects = tracked_objects_by_frame.get(frame_idx, [])
            for obj in frame_tracked_objects:
                track_duration = obj.get_track_duration()
                if track_duration >= config.tracking_min_hits:
                    valid_track_ids.add(obj.track_id)
    else:
        # No backward validation: only include tracks that are valid at each specific frame
        # We'll check track_duration per frame during export
        pass
    
    # Build a comprehensive map: (frame_idx, track_id) -> TrackedObject
    # This ensures we use the correct TrackedObject instance for each frame and track_id
    frame_track_map: Dict[Tuple[int, int], TrackedObject] = {}
    # Also build a track_id -> TrackedObject map for fallback lookup
    # (use the most recent instance of each track for interpolation/extrapolation)
    track_id_to_obj: Dict[int, TrackedObject] = {}
    for frame_idx in sorted(all_frame_indices):
        frame_tracked_objects = tracked_objects_by_frame.get(frame_idx, [])
        for obj in frame_tracked_objects:
            # Store the TrackedObject for this specific frame and track_id
            # If multiple objects have the same track_id at this frame, the last one wins
            # (shouldn't happen, but this ensures consistency)
            frame_track_map[(frame_idx, obj.track_id)] = obj
            # Store the most recent instance of each track for fallback lookup
            track_id_to_obj[obj.track_id] = obj
    
    # Track previous frame bboxes for speed limiting: {track_id: (frame_idx, bbox)}
    prev_frame_bboxes: Dict[int, Tuple[int, np.ndarray]] = {}
    
    for frame_idx in sorted(all_frame_indices):
        image_id = frame_idx + 1
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
        })
        
        # Build list of static object bboxes for matching (to invalidate dynamic tracks)
        static_bboxes: List[Tuple[int, np.ndarray, int]] = []  # (static_idx, bbox, class_id)
        if static_objects:
            for i, static_det in enumerate(static_objects):
                if static_det.bbox is not None and len(static_det.bbox) == 4:
                    static_bboxes.append((i, static_det.bbox, static_det.class_id))
        
        # Collect unique track_ids that appear in this frame and are valid
        frame_track_ids = set()
        frame_tracked_objects = tracked_objects_by_frame.get(frame_idx, [])
        for obj in frame_tracked_objects:
            if config.enable_backward_validation:
                # Backward validation: use pre-computed valid_track_ids
                if obj.track_id in valid_track_ids:
                    frame_track_ids.add(obj.track_id)
            else:
                # No backward validation: check track_duration at this specific frame
                track_duration = obj.get_track_duration()
                if track_duration >= config.tracking_min_hits:
                    frame_track_ids.add(obj.track_id)
        
        # Also check all valid track_ids (from backward validation) that might have interpolated positions
        if config.enable_backward_validation:
            # Add all valid track_ids that might have interpolated/extrapolated positions for this frame
            for track_id in valid_track_ids:
                if track_id not in frame_track_ids:
                    # Check if this track has a valid position for this frame (via interpolation/extrapolation)
                    obj = track_id_to_obj.get(track_id)
                    if obj is not None:
                        # Try to get bbox for this frame - if it returns a valid bbox, include this track
                        test_bbox = obj.get_bbox_for_frame(frame_idx)
                        if test_bbox is not None and len(test_bbox) == 4 and not np.any(np.isnan(test_bbox)) and not np.any(np.isinf(test_bbox)):
                            frame_track_ids.add(track_id)
        
        # Process each valid track_id for this frame using the stored TrackedObject instance
        for track_id in sorted(frame_track_ids):
            # Get the TrackedObject instance stored for this specific frame and track_id
            obj = frame_track_map.get((frame_idx, track_id))
            # Fallback: if not found for this specific frame, use any instance of this track
            # (it should still be able to provide a valid bbox via interpolation/extrapolation)
            if obj is None:
                obj = track_id_to_obj.get(track_id)
                if obj is None:
                    continue
            
            # Check if this frame has an interpolated position (for skipped frames or regained tracking)
            has_interpolated = frame_idx in obj.interpolated_positions if obj.interpolated_positions else False
            
            # Only include extrapolated boxes if the track was lost and then regained, or if it has interpolated positions
            # (i.e., was_lost=True means it successfully regained tracking, or has_interpolated means it was interpolated for skipped frames)
            is_extrapolated = frame_idx in obj.extrapolated_frames if obj.extrapolated_frames else False
            if is_extrapolated and not obj.was_lost and not has_interpolated:
                # Skip extrapolated boxes that never regained tracking and weren't interpolated
                continue
            
            # Use interpolated position if available, otherwise use current bbox
            bbox = obj.get_bbox_for_frame(frame_idx)
            
            # Validate bbox
            if bbox is None or len(bbox) != 4:
                continue
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                continue
            
            # Filter out static dynamic objects (BGS noise that appears as static tracks)
            if config.filter_static_dynamic_objects:
                track_duration = obj.get_track_duration()
                # Only check if track has been detected for minimum number of frames
                if track_duration >= config.static_dynamic_min_frames:
                    if obj.is_static(max_movement=config.static_dynamic_max_movement):
                        # This dynamic track is actually static - skip it (should be caught by static detector)
                        continue
            
            # Check speed limit: calculate movement from previous frame
            if track_id in prev_frame_bboxes:
                prev_frame_idx, prev_bbox = prev_frame_bboxes[track_id]
                frame_diff = frame_idx - prev_frame_idx
                
                if frame_diff > 0:
                    # Calculate bbox centers
                    curr_center_x = (bbox[0] + bbox[2]) / 2.0
                    curr_center_y = (bbox[1] + bbox[3]) / 2.0
                    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2.0
                    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2.0
                    
                    # Calculate distance moved
                    distance = np.sqrt((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)
                    speed_per_frame = distance / frame_diff
                    
                    # Skip if speed exceeds limit
                    if speed_per_frame > config.max_speed_pixels_per_frame:
                        continue
            
            # Update previous frame bbox for this track
            prev_frame_bboxes[track_id] = (frame_idx, bbox.copy())
            
            # Check if this tracked object matches any static object (static objects invalidate dynamic predictions)
            matches_static = False
            if static_objects:
                for static_idx, static_bbox, static_class_id in static_bboxes:
                    # Check if same class
                    if static_class_id == obj.class_id:
                        # Check IoU - static objects override dynamic tracks that match them
                        iou = ObjectDetector._compute_iou(bbox, static_bbox)
                        if iou >= 0.5:  # Overlap threshold
                            matches_static = True
                            break
            
            # Skip this tracked object if it matches a static object (static objects invalidate dynamic predictions)
            if matches_static:
                continue
            
            # Check for cross-class overlaps and resolve conflicts
            # Only filter when there's very high overlap (IoU > 0.75) - this catches false positives
            # but allows valid cases like people on bicycles (moderate overlap)
            skip_due_to_overlap = False
            for other_track_id in sorted(frame_track_ids):
                if other_track_id == track_id:
                    continue
                
                # Get the other track object
                other_obj = frame_track_map.get((frame_idx, other_track_id))
                if other_obj is None:
                    other_obj = track_id_to_obj.get(other_track_id)
                if other_obj is None:
                    continue
                
                # Skip if same class (NMS already handled this)
                if other_obj.class_id == obj.class_id:
                    continue
                
                # Get bbox for the other track
                other_bbox = other_obj.get_bbox_for_frame(frame_idx)
                if other_bbox is None or len(other_bbox) != 4:
                    continue
                if np.any(np.isnan(other_bbox)) or np.any(np.isinf(other_bbox)):
                    continue
                
                # Calculate IoU between this track and other track
                iou = ObjectDetector._compute_iou(bbox, other_bbox)
                
                # Only filter if very high overlap (IoU > 0.75) - likely false positive
                # This allows valid cases like people on bicycles (which have moderate overlap)
                if iou > 0.75:
                    # Use confidence to resolve conflict - prefer higher confidence
                    # Check area as tiebreaker for similar confidence
                    this_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    other_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
                    
                    confidence_diff = abs(obj.confidence - other_obj.confidence)
                    if confidence_diff > 0.2:
                        # Significant confidence difference - prefer higher confidence
                        if obj.confidence < other_obj.confidence:
                            skip_due_to_overlap = True
                            break
                    else:
                        # Similar confidence - prefer larger box (less likely to be false positive)
                        if this_area < other_area * 0.7:  # This box is significantly smaller
                            skip_due_to_overlap = True
                            break
            
            # Skip this tracked object if it overlaps with a higher confidence track
            if skip_due_to_overlap:
                continue
            
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                continue
            
            # COCO format: [x, y, width, height]
            coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            area = float((x2 - x1) * (y2 - y1))
            
            # Map class_id to COCO category_id (1-indexed)
            category_id = obj.class_id + 1 if obj.class_id >= 0 else 1
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "track_id": track_id  # Use the track_id from the iteration, ensuring consistency
            })
            annotation_id += 1
    
        # Add static objects to this frame (they appear in every frame and override matching dynamic tracks)
        if static_objects:
            for i, static_det in enumerate(static_objects):
                # Validate bbox
                if static_det.bbox is None or len(static_det.bbox) != 4:
                    continue
                if np.any(np.isnan(static_det.bbox)) or np.any(np.isinf(static_det.bbox)):
                    continue
                
                x1, y1, x2, y2 = static_det.bbox[0], static_det.bbox[1], static_det.bbox[2], static_det.bbox[3]
                
                # Validate coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # COCO format: [x, y, width, height]
                coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                area = float((x2 - x1) * (y2 - y1))
                
                # Map class_id to COCO category_id (1-indexed)
                category_id = static_det.class_id + 1 if static_det.class_id >= 0 else 1
                
                # Get the assigned track_id for this static object
                static_track_id = static_object_track_ids.get(i, 100000 + i)
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                    "track_id": static_track_id  # Static objects have unique track IDs starting from 100000
                })
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
        "categories": categories
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ Saved COCO annotations to: {output_file}", flush=True)
    print(f"  Images: {len(images)}", flush=True)
    print(f"  Annotations: {len(annotations)}", flush=True)


def process_video(
    video_path: str,
    config: Config
):
    """Process video with foreground segmentation + tracking pipeline."""
    # Convert Windows path to WSL path if needed
    if video_path.startswith('H:\\') or video_path.startswith('H:/'):
        video_path = video_path.replace('H:\\', '/mnt/h/').replace('H:/', '/mnt/h/').replace('\\', '/')
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo: {video_path.name}", flush=True)
    print(f"Resolution: {width}x{height}", flush=True)
    print(f"FPS: {fps:.2f}", flush=True)
    print(f"Total frames: {total_frames}", flush=True)
    print(f"FG Segmentation: {config.fg_seg_method}", flush=True)
    print(f"Tracking: {config.tracking_method}", flush=True)
    
    # Initialize pipeline components
    fg_segmenter = ForegroundSegmenter(config)
    blob_extractor = BlobExtractor(config)
    object_detector = ObjectDetector(config)
    tracker = Tracker(config)
    
    # Detect static objects first (sample random frames)
    static_objects: List[Detection] = []
    if config.detect_static_objects and config.use_dual_detector:
        static_objects = detect_static_objects(
            video_path, object_detector, config, total_frames, width, height
        )
    
    # Static objects are detected via static_detection_sample_frames (pre-processing step)
    # No need for full-frame detection during main loop since static objects are handled separately
    
    # Storage for annotations
    tracked_objects_by_frame: Dict[int, List[TrackedObject]] = {}
    
    # Track history for visualization (trails)
    track_history: Dict[int, List[Tuple[float, float]]] = {}  # track_id -> [(cx, cy), ...]
    max_history_length = 30  # Maximum number of points in trail
    
    frame_idx = 0
    last_processed_frame = -1  # Track last processed frame for interpolation
    inference_times = []
    
    if not config.headless:
        cv2.namedWindow('Video with Tracking', cv2.WINDOW_NORMAL)
        if config.debug_visualization:
            cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
            cv2.namedWindow('DL Regions', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Static Objects', cv2.WINDOW_NORMAL)
            cv2.namedWindow('All Valid Objects', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if process_every_n_frames > 1
        if config.process_every_n_frames > 1 and frame_idx % config.process_every_n_frames != 0:
            # Still update frame index and advance video
            frame_idx += 1
            # Update existing tracks (extrapolate using Kalman filter)
            # Only process if we have active tracks (not all lost)
            if len(tracker.tracks) > 0:
                tracked_objects = []
                for track in tracker.tracks:
                    # Only process tracks that haven't been lost for too long
                    if track.time_since_update < config.tracking_max_age:
                        # Only predict if track is still active (recently updated)
                        # Skip expensive is_static check for skipped frames - just predict
                        predicted_bbox = track.predict(frame_idx)
                        tracked_objects.append(track)
                    # Tracks that exceed max_age will be removed by the tracker
                # Store extrapolated tracks for skipped frames
                if config.save_annotations and len(tracked_objects) > 0:
                    tracked_objects_by_frame[frame_idx] = tracked_objects.copy()
            continue
        
        start_time = time.time()
        
        # Step 1: Foreground segmentation (always needed for background model learning)
        fg_mask = fg_segmenter.segment(frame, video_path.name)
        
        # Step 2: Extract object blobs (ROIs) - always run to detect new objects
        blobs = blob_extractor.extract(fg_mask)
        
        # Early exit optimization: If no blobs and no active tracks, skip expensive detection
        # Check if we have any active tracks (not lost for too long)
        has_active_tracks = any(track.time_since_update < config.tracking_max_age for track in tracker.tracks) if len(tracker.tracks) > 0 else False
        
        # Step 3: Run object detector on foreground ROIs
        # Static objects are detected separately via static_detection_sample_frames (pre-processing)
        # Only run region-based detection during main loop
        detections = []
        if len(blobs) > 0 or has_active_tracks:
            # Only run detection if we have blobs or active tracks (objects might be off-screen but still tracked)
            detections = object_detector.detect(
                frame, fg_mask, blobs, 
                conf_threshold=config.detector_conf_threshold,
                detect_static=False,  # Static objects detected separately via random sampling
                run_full_frame=False  # Static objects handled by static_detection_sample_frames
            )
            
            # Step 3.5: Filter out drivers in cars (small, square pedestrians within cars)
            detections = object_detector._filter_drivers_in_cars(detections)
        
        # Step 4: Track objects (this will handle lost tracks even if detections is empty)
        tracked_objects = tracker.update(detections, frame_idx, frame)
        
        # Step 4.5: Associate bicycles and pedestrians when nearby (for robust tracking)
        if config.enable_bicycle_pedestrian_association and len(tracked_objects) > 0:
            # Find bicycles and pedestrians
            bicycles = [t for t in tracked_objects if t.class_id == 1]  # class_id 1 = bicycle
            pedestrians = [t for t in tracked_objects if t.class_id == 0]  # class_id 0 = person
            
            # Check each bicycle against each pedestrian
            for bicycle in bicycles:
                bike_bbox = bicycle.bbox
                if bike_bbox is None or len(bike_bbox) != 4:
                    continue
                    
                # Compute bicycle center and size
                bike_x1, bike_y1, bike_x2, bike_y2 = bike_bbox
                bike_cx = (bike_x1 + bike_x2) / 2.0
                bike_cy = (bike_y1 + bike_y2) / 2.0
                bike_w = bike_x2 - bike_x1
                bike_h = bike_y2 - bike_y1
                
                for person in pedestrians:
                    person_bbox = person.bbox
                    if person_bbox is None or len(person_bbox) != 4:
                        continue
                    
                    person_x1, person_y1, person_x2, person_y2 = person_bbox
                    person_cx = (person_x1 + person_x2) / 2.0
                    person_cy = (person_y1 + person_y2) / 2.0
                    person_w = person_x2 - person_x1
                    person_h = person_y2 - person_y1
                    
                    # Check if person is above/within bicycle (rider position)
                    # Person should be roughly above the bicycle and overlapping horizontally
                    vertical_overlap = person_y2 > bike_y1 and person_y1 < bike_y2
                    horizontal_overlap = person_x2 > bike_x1 and person_x1 < bike_x2
                    
                    # Check if centers are close (within 1.5x bicycle width/height)
                    center_distance_x = abs(bike_cx - person_cx)
                    center_distance_y = abs(bike_cy - person_cy)
                    
                    # Associate if person is above bicycle and overlapping, or very close
                    if (vertical_overlap and horizontal_overlap and 
                        center_distance_x < max(bike_w, person_w) * 1.5 and
                        center_distance_y < max(bike_h, person_h) * 1.5):
                        # Link them bidirectionally
                        bicycle.associated_track_ids.add(person.track_id)
                        person.associated_track_ids.add(bicycle.track_id)
                        
                        # If one track is lost, use the associated track's position to help recover
                        if bicycle.time_since_update > 0 and person.time_since_update == 0:
                            # Bicycle lost but person found - predict bicycle from person
                            # Assume bicycle is below person
                            offset_y = bike_cy - person_cy  # Try to maintain relative position
                            predicted_bike_cy = person_cy + offset_y
                            # Estimate bicycle bbox from person position
                            estimated_bike_x1 = person_x1 - (bike_w / 2)
                            estimated_bike_x2 = person_x2 + (bike_w / 2)
                            estimated_bike_y1 = predicted_bike_cy - (bike_h / 2)
                            estimated_bike_y2 = predicted_bike_cy + (bike_h / 2)
                            # Note: This is a helper prediction, not a direct update
                            
                        elif person.time_since_update > 0 and bicycle.time_since_update == 0:
                            # Person lost but bicycle found - predict person from bicycle
                            # Person is typically above bicycle
                            offset_y = person_cy - bike_cy
                            predicted_person_cy = bike_cy + offset_y
                            # Similar prediction logic for person
                            # Note: This is a helper prediction, not a direct update
        
        # Filter out static objects (those that haven't moved much for >10 frames)
        # These should be picked up by the static object detector
        filtered_tracked_objects = []
        for track in tracked_objects:
            if track.is_static(max_movement=20.0):
                # Object has been static for >10 frames - skip it (static detector will catch it)
                continue
            filtered_tracked_objects.append(track)
        
        # Interpolate for skipped frames if we've processed frames before
        if config.save_annotations and last_processed_frame >= 0 and config.process_every_n_frames > 1 and len(filtered_tracked_objects) > 0:
            # Get skipped frames between last processed and current frame
            skipped_frames = list(range(last_processed_frame + 1, frame_idx))
            
            if len(skipped_frames) > 0:
                # Get tracks from last processed frame
                last_frame_tracks = tracked_objects_by_frame.get(last_processed_frame, [])
                # Create a mapping of track_id to TrackedObject for last frame
                last_frame_track_map = {track.track_id: track for track in last_frame_tracks}
                
                # Only proceed with interpolation if we have tracks from last frame or current frame
                if len(last_frame_track_map) > 0 or len(filtered_tracked_objects) > 0:
                    # Create a mapping of track_id to TrackedObject for skipped frames
                    # (we need to update the track objects stored for skipped frames with interpolated positions)
                    skipped_frame_track_map: Dict[int, Dict[int, TrackedObject]] = {}
                    for skipped_frame in skipped_frames:
                        skipped_frame_tracks = tracked_objects_by_frame.get(skipped_frame, [])
                        skipped_frame_track_map[skipped_frame] = {track.track_id: track for track in skipped_frame_tracks}
                    
                    # Interpolate for each current track that existed in the last processed frame
                    for current_track in filtered_tracked_objects:
                        track_id = current_track.track_id
                        if track_id in last_frame_track_map:
                            last_track = last_frame_track_map[track_id]
                            # Get bbox from last processed frame (may be interpolated or actual)
                            last_bbox = last_track.get_bbox_for_frame(last_processed_frame)
                            # Get current bbox
                            current_bbox = current_track.bbox
                            
                            # Validate both bboxes
                            if (last_bbox is not None and len(last_bbox) == 4 and 
                                current_bbox is not None and len(current_bbox) == 4 and
                                not np.any(np.isnan(last_bbox)) and not np.any(np.isnan(current_bbox)) and
                                not np.any(np.isinf(last_bbox)) and not np.any(np.isinf(current_bbox))):
                                # Interpolate between last processed frame and current frame
                                current_track.interpolate_between_frames(
                                    last_processed_frame, last_bbox,
                                    frame_idx, current_bbox,
                                    skipped_frames
                                )
                                
                                # Ensure interpolated tracks are stored in tracked_objects_by_frame for skipped frames
                                # This ensures they appear in the export even if they weren't originally in those frames
                                for skipped_frame in skipped_frames:
                                    if skipped_frame not in skipped_frame_track_map:
                                        skipped_frame_track_map[skipped_frame] = {}
                                    
                                    skipped_track_map = skipped_frame_track_map[skipped_frame]
                                    
                                    # If track wasn't in this skipped frame, add it (it will use interpolated position)
                                    if track_id not in skipped_track_map:
                                        skipped_track_map[track_id] = current_track
                                        # Ensure tracked_objects_by_frame has this track for this skipped frame
                                        if skipped_frame not in tracked_objects_by_frame:
                                            tracked_objects_by_frame[skipped_frame] = []
                                        # Add track if not already in the list
                                        if not any(t.track_id == track_id for t in tracked_objects_by_frame[skipped_frame]):
                                            tracked_objects_by_frame[skipped_frame].append(current_track)
                                    else:
                                        # Track was already in skipped frame, update interpolated positions
                                        skipped_track = skipped_track_map[track_id]
                                        # Copy interpolated positions for skipped frames from current_track to skipped_track
                                        # (they should be the same object reference, but we ensure consistency)
                                        for sf in skipped_frames:
                                            if sf in current_track.interpolated_positions:
                                                skipped_track.interpolated_positions[sf] = current_track.interpolated_positions[sf].copy()
        
        # Store for annotation export
        if config.save_annotations:
            tracked_objects_by_frame[frame_idx] = filtered_tracked_objects.copy()
            last_processed_frame = frame_idx
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        # Limit inference_times to last 1000 entries to prevent unbounded growth
        # This is enough for FPS calculation while preventing memory issues
        if len(inference_times) > 1000:
            inference_times.pop(0)
        
        # Visualization
        if not config.headless:
            vis_frame = frame.copy()
            
            # Draw foreground mask (semi-transparent overlay) - optional
            if config.debug_visualization:
                fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                vis_frame = cv2.addWeighted(vis_frame, 0.7, fg_colored, 0.3, 0)
                
                # Show individual blob regions on foreground mask
                # Start with the foreground mask as a 3-channel BGR image
                fg_mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                
                # Draw each blob's mask region and bounding boxes
                if len(blobs) > 0:
                    # Pre-allocate a colored mask image
                    colored_mask = np.zeros_like(fg_mask_vis)
                    
                    for i, blob in enumerate(blobs):
                        # Draw the blob mask region in a colored overlay
                        blob_mask = blob['mask']
                        
                        # Ensure mask is boolean and matches image dimensions
                        if not isinstance(blob_mask, np.ndarray):
                            continue
                        if blob_mask.dtype != bool:
                            blob_mask = blob_mask.astype(bool)
                        if blob_mask.shape != fg_mask_vis.shape[:2]:
                            print(f"Warning: Blob {i} mask shape {blob_mask.shape} != image shape {fg_mask_vis.shape[:2]}")
                            continue
                        
                        # Generate distinct bright color for each blob using HSV
                        hue = int((i * 180 / max(len(blobs), 1)) % 180)
                        blob_color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                        blob_color = tuple(map(int, blob_color_bgr))
                        
                        # Directly set colored pixels where mask is True (no blending)
                        colored_mask[blob_mask] = blob_color
                        
                        # Draw original bounding box (from connected components) in blue
                        x1_orig, y1_orig, x2_orig, y2_orig = map(int, blob['bbox_orig'])
                        cv2.rectangle(fg_mask_vis, (x1_orig, y1_orig), (x2_orig, y2_orig), (255, 0, 0), 3)
                        cv2.putText(fg_mask_vis, f"Orig {i}", (x1_orig, max(15, y1_orig - 5)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Draw expanded bounding box (what's used for YOLO) in green
                        x1, y1, x2, y2 = map(int, blob['bbox'])
                        cv2.rectangle(fg_mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(fg_mask_vis, f"Exp {i} (YOLO)", (x1, min(fg_mask_vis.shape[0] - 5, y2 + 20)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw centroid
                        cx, cy = map(int, blob['centroid'])
                        cv2.circle(fg_mask_vis, (cx, cy), 8, (0, 255, 255), -1)
                        cv2.circle(fg_mask_vis, (cx, cy), 8, (0, 0, 0), 2)
                    
                    # Blend the colored mask with the grayscale mask (where colored_mask has color, use it)
                    mask_has_color = np.any(colored_mask > 0, axis=2)
                    fg_mask_vis[mask_has_color] = colored_mask[mask_has_color]
                
                # Add title and debug info
                cv2.putText(fg_mask_vis, "Foreground Mask - Blob Regions", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(fg_mask_vis, f"Blobs: {len(blobs)} | Frame: {frame_idx}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if len(blobs) > 0:
                    cv2.putText(fg_mask_vis, "Blue: Original | Green: Expanded (YOLO)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Foreground Mask', fg_mask_vis)
                
                # Show DL Regions debug window
                dl_regions_vis = frame.copy()
                
                # Draw expanded blob regions (what's used for YOLO filtering)
                if len(blobs) > 0:
                    for i, blob in enumerate(blobs):
                        # Draw expanded bounding box (the region used for YOLO)
                        x1, y1, x2, y2 = map(int, blob['bbox'])
                        
                        # Generate distinct color for each blob
                        hue = int((i * 180 / max(len(blobs), 1)) % 180)
                        blob_color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                        blob_color = tuple(map(int, blob_color_bgr))
                        
                        # Draw semi-transparent filled rectangle for the region
                        overlay = dl_regions_vis.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), blob_color, -1)
                        dl_regions_vis = cv2.addWeighted(dl_regions_vis, 0.7, overlay, 0.3, 0)
                        
                        # Draw border
                        cv2.rectangle(dl_regions_vis, (x1, y1), (x2, y2), blob_color, 3)
                        cv2.putText(dl_regions_vis, f"Region {i}", (x1, max(20, y1 - 5)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, blob_color, 2)
                
                # Draw YOLO detections
                for det in detections:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    conf = det.confidence
                    class_name = det.class_name
                    
                    # Color based on class
                    if 'person' in class_name.lower():
                        det_color = (0, 255, 255)  # Yellow
                    elif 'bicycle' in class_name.lower():
                        det_color = (255, 165, 0)  # Orange
                    elif 'car' in class_name.lower():
                        det_color = (255, 0, 255)  # Magenta
                    else:
                        det_color = (0, 255, 0)  # Green
                    
                    # Draw detection box
                    cv2.rectangle(dl_regions_vis, (x1, y1), (x2, y2), det_color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(dl_regions_vis, label, (x1, max(20, y1 - 5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_color, 2)
                
                # Add title and info
                cv2.putText(dl_regions_vis, "DL Regions - YOLO Detection Areas", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(dl_regions_vis, f"Blob Regions: {len(blobs)} | Detections: {len(detections)} | Frame: {frame_idx}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if config.detect_static_objects:
                    cv2.putText(dl_regions_vis, "Static Detection: ENABLED (Full Frame)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(dl_regions_vis, "Static Detection: DISABLED (FG Only)", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('DL Regions', dl_regions_vis)
                
                # Show Static Objects debug window
                static_objects_vis = frame.copy()
                if static_objects:
                    for i, static_det in enumerate(static_objects):
                        if static_det.bbox is None or len(static_det.bbox) != 4:
                            continue
                        if np.any(np.isnan(static_det.bbox)) or np.any(np.isinf(static_det.bbox)):
                            continue
                        
                        x1, y1, x2, y2 = map(int, static_det.bbox)
                        
                        # Color based on class
                        if static_det.class_id == 0:  # person
                            static_color = (0, 255, 255)  # Yellow
                        elif static_det.class_id == 1:  # bicycle
                            static_color = (255, 165, 0)  # Orange
                        elif static_det.class_id == 2:  # car
                            static_color = (255, 0, 255)  # Magenta
                        else:
                            static_color = (0, 255, 0)  # Green
                        
                        # Draw static object box with thick border
                        cv2.rectangle(static_objects_vis, (x1, y1), (x2, y2), static_color, 4)
                        
                        # Draw label with static ID
                        label = f"STATIC-{i} {static_det.class_name}"
                        if static_det.confidence < 1.0:
                            label += f" ({static_det.confidence:.2f})"
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        
                        # Draw text background
                        cv2.rectangle(static_objects_vis, 
                                     (x1, y1 - text_height - baseline - 5),
                                     (x1 + text_width, y1),
                                     static_color, -1)
                        
                        # Draw text
                        cv2.putText(static_objects_vis, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add title and info
                cv2.putText(static_objects_vis, "Static Objects (All Frames)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(static_objects_vis, f"Frame: {frame_idx} | Static Objects: {len(static_objects) if static_objects else 0}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Static Objects', static_objects_vis)
                
                # Show All Valid Objects debug window (static + dynamic)
                all_objects_vis = frame.copy()
                
                # Draw static objects first (background layer)
                if static_objects:
                    for i, static_det in enumerate(static_objects):
                        if static_det.bbox is None or len(static_det.bbox) != 4:
                            continue
                        if np.any(np.isnan(static_det.bbox)) or np.any(np.isinf(static_det.bbox)):
                            continue
                        
                        x1, y1, x2, y2 = map(int, static_det.bbox)
                        
                        # Static objects in cyan
                        static_color = (255, 255, 0)  # Cyan
                        cv2.rectangle(all_objects_vis, (x1, y1), (x2, y2), static_color, 3)
                        
                        label = f"STATIC {static_det.class_name}"
                        cv2.putText(all_objects_vis, label, (x1, max(20, y1 - 5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, static_color, 2)
                
                # Draw dynamic tracked objects
                for track in tracked_objects:
                    if track.bbox is None or len(track.bbox) != 4:
                        continue
                    if np.any(np.isnan(track.bbox)) or np.any(np.isinf(track.bbox)):
                        continue
                    
                    x1, y1, x2, y2 = map(int, track.bbox)
                    
                    # Validate coordinates
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                        continue
                    
                    # Generate distinct color based on track ID
                    track_id_hash = track.track_id * 137
                    dynamic_color = (
                        int((track_id_hash * 50) % 200) + 55,
                        int((track_id_hash * 100) % 200) + 55,
                        int((track_id_hash * 150) % 200) + 55
                    )
                    
                    # Draw bounding box
                    cv2.rectangle(all_objects_vis, (x1, y1), (x2, y2), dynamic_color, 2)
                    
                    # Draw label
                    label = f"ID:{track.track_id} {track.class_name}"
                    if track.confidence < 1.0:
                        label += f" ({track.confidence:.2f})"
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw text background
                    cv2.rectangle(all_objects_vis, 
                                 (x1, y1 - text_height - baseline - 5),
                                 (x1 + text_width, y1),
                                 dynamic_color, -1)
                    
                    # Draw text
                    cv2.putText(all_objects_vis, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add title and info
                num_static = len(static_objects) if static_objects else 0
                num_dynamic = len(tracked_objects)
                cv2.putText(all_objects_vis, "All Valid Objects (Static + Dynamic)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(all_objects_vis, 
                           f"Frame: {frame_idx} | Static: {num_static} | Dynamic: {num_dynamic} | Total: {num_static + num_dynamic}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(all_objects_vis, "Cyan = Static | Colored = Dynamic", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('All Valid Objects', all_objects_vis)
            
            # Update track history
            for track in tracked_objects:
                # Validate bbox before using
                if track.bbox is None or len(track.bbox) != 4:
                    continue
                if np.any(np.isnan(track.bbox)) or np.any(np.isinf(track.bbox)):
                    continue
                
                x1, y1, x2, y2 = track.bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                if track.track_id not in track_history:
                    track_history[track.track_id] = []
                track_history[track.track_id].append((cx, cy))
                
                # Limit history length
                if len(track_history[track.track_id]) > max_history_length:
                    track_history[track.track_id].pop(0)
            
            # Draw tracked objects with distinct colors per track ID
            for track in tracked_objects:
                # Validate bbox - skip if NaN or invalid
                if track.bbox is None or len(track.bbox) != 4:
                    continue
                
                # Check for NaN or invalid values
                if np.any(np.isnan(track.bbox)) or np.any(np.isinf(track.bbox)):
                    continue
                
                x1, y1, x2, y2 = map(int, track.bbox)
                
                # Validate coordinates are reasonable
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                    continue
                
                # Generate distinct color based on track ID
                track_id_hash = track.track_id * 137  # Prime number for better distribution
                color = (
                    int((track_id_hash * 50) % 200) + 55,
                    int((track_id_hash * 100) % 200) + 55,
                    int((track_id_hash * 150) % 200) + 55
                )
                
                # Draw track trail (history)
                if track.track_id in track_history and len(track_history[track.track_id]) > 1:
                    trail_points = track_history[track.track_id]
                    for i in range(1, len(trail_points)):
                        pt1 = (int(trail_points[i-1][0]), int(trail_points[i-1][1]))
                        pt2 = (int(trail_points[i][0]), int(trail_points[i][1]))
                        # Fade trail (older points are more transparent)
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(vis_frame, pt1, pt2, trail_color, 2)
                
                # Check if track is extrapolated (lost)
                is_extrapolated = track.time_since_update > 0
                
                # Skip extrapolated boxes if flag is disabled
                if is_extrapolated and not config.show_extrapolated_boxes:
                    continue
                
                # Draw bounding box - dashed for extrapolated, solid for active
                if is_extrapolated:
                    # Draw dashed rectangle for extrapolated tracks
                    dash_length = 10
                    gap_length = 5
                    # Top edge
                    for x in range(x1, x2, dash_length + gap_length):
                        cv2.line(vis_frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
                    # Bottom edge
                    for x in range(x1, x2, dash_length + gap_length):
                        cv2.line(vis_frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
                    # Left edge
                    for y in range(y1, y2, dash_length + gap_length):
                        cv2.line(vis_frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
                    # Right edge
                    for y in range(y1, y2, dash_length + gap_length):
                        cv2.line(vis_frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
                else:
                    # Solid rectangle for active tracks
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID, class name, and confidence
                status_prefix = "[EXTRAP]" if is_extrapolated else ""
                label = f"{status_prefix}ID:{track.track_id} {track.class_name}"
                if track.confidence < 1.0:
                    label += f" ({track.confidence:.2f})"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw text background
                cv2.rectangle(vis_frame, 
                             (x1, y1 - text_height - baseline - 5),
                             (x1 + text_width, y1),
                             color, -1)
                
                # Draw text
                cv2.putText(vis_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Clean up old track history (remove tracks that are no longer active)
            active_track_ids = {track.track_id for track in tracked_objects}
            track_history = {tid: hist for tid, hist in track_history.items() 
                           if tid in active_track_ids}
            
            # Show foreground mask in separate window if debug mode
            if config.debug_visualization:
                cv2.imshow('Foreground Mask', fg_mask)
            
            # Add info text with background
            info_text = f"Frame: {frame_idx}/{total_frames} | Tracks: {len(tracked_objects)} | FPS: {1.0/inference_time:.1f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                info_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            )
            cv2.rectangle(vis_frame, (10, 10), (10 + text_width, 10 + text_height + baseline),
                         (0, 0, 0), -1)
            cv2.putText(vis_frame, info_text, (10, 10 + text_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow('Video with Tracking', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                break
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 30 == 0:
            avg_fps = 1.0 / np.mean(inference_times[-30:])
            video_name = Path(video_path).name
            progress_msg = f"[{video_name}] Frame {frame_idx}/{total_frames}: {len(tracked_objects)} tracks, {avg_fps:.1f} FPS"
            print(progress_msg, flush=True)
    
    cap.release()
    if not config.headless:
        cv2.destroyAllWindows()
    
    # Save annotations
    if config.save_annotations and (len(tracked_objects_by_frame) > 0 or len(static_objects) > 0):
        # Save in same directory as video, with same name but .json extension
        output_file = video_path.parent / f"{video_path.stem}.json"
        save_coco_annotations(video_path, tracked_objects_by_frame, width, height, output_file, config, static_objects, total_frames)
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    if avg_inference_time > 0:
        print(f"\nAverage inference time: {avg_inference_time*1000:.2f}ms ({1.0/avg_inference_time:.1f} FPS)", flush=True)
    else:
        print(f"\nNo frames processed", flush=True)
    print(f"Total frames processed: {frame_idx}", flush=True)


if __name__ == "__main__":
    # Create config with all parameters
    config = Config()
    
    # Process video
    process_video(config.video_path, config)
