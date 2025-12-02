"""
Annotation Generation Module

This module provides video annotation pipelines for automated object detection,
tracking, and 3D location computation. It generates COCO-format annotations
with temporal tracking information.

Main modules:
    - annotation_generation: Simplified pipeline with Dome-DETR/YOLOv12x detection
    - dep_annotation_generation: Advanced pipeline with foreground segmentation
    - footpoint_to_ground: Geometric 3D projection with distortion correction
"""

__version__ = "1.0.0"

from .annotation_generation import Config, process_video, parse_camera_intrinsics_from_calibration
from .footpoint_to_ground import bbox_to_3d_geometric_robust

__all__ = [
    'Config',
    'process_video',
    'parse_camera_intrinsics_from_calibration',
    'bbox_to_3d_geometric_robust',
]
