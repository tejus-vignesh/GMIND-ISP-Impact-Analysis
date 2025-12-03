# GMIND-sdk

![GMIND Overview](misc/GMIND-Overall.png)

GMIND-sdk is a toolkit for working with the GMIND ITS infrastructure node dataset. It provides scripts and utilities for image signal processing (ISP), video generation and compression, calibration, validation, and machine learning data loading.

## Features

- **Image Signal Processing (ISP):**
  - Run ISP on RAW images.
  - Output processed images as video files.
  - Compress videos with customizable settings.

- **Calibration & Validation:**
  - Tools for camera and sensor calibration.
  - Scripts for validating dataset alignment and sensor fusion.

- **LIDAR & Camera Reprojection:**
  - Overlay LIDAR point clouds onto camera images using calibration data.

- **Video Annotation Generation:**
  - Automated object detection and tracking pipeline.
  - Generates COCO-format annotations with temporal tracking.
  - Optional 3D location computation using geometric ground plane intersection.
  - Supports multiple detection models (Dome-DETR, YOLOv12x).

- **PyTorch DataLoader:**
  - Unified DataLoader for all supported data formats.
  - Enables consistent training across work packages and models.
  - Facilitates benchmarking and sensor selection for ITS use cases.

## Getting Started

1. Clone the repository with submodules and install dependencies:
   ```sh
   # Clone with submodules (recommended)
   git clone --recurse-submodules https://github.com/daramolloy/GMIND-sdk
   cd GMIND-sdk
   
   # Or if already cloned, initialize submodules
   git submodule update --init --recursive
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Or install as a package (recommended)
   pip install -e .
   
   # With optional dependencies
   pip install -e ".[ultralytics,eval]"  # YOLO support + evaluation tools
   pip install -e ".[all]"               # All optional dependencies
   ```

   **Note**: This repository uses git submodules for:
   - `Calibration/` - Sensor Calibration Toolbox
   - `OC_SORT/` - OC-SORT tracker for multi-object tracking (used by Annotation and Evaluation modules)

2. **Download Model Files (for Annotation Generation):**
   
   The annotation generation module requires model weights and config files that are too large to commit. Download them separately:
   
   **For Dome-DETR (recommended for VisDrone dataset):**
   - Download `Dome-L-VisDrone-best.pth` (checkpoint) and `Dome-L-VisDrone.yml` (config)
   - Place them in one of these locations:
     - `../Dome-DETR/` directory (sibling to GMIND-sdk): 
       - `../Dome-DETR/configs/dome/Dome-L-VisDrone.yml`
       - `../Dome-DETR/Dome-L-VisDrone-best.pth`
     - Or update the paths in `Config` when using the annotation generator
   - **Download links:** [Add your download links here - e.g., Google Drive, Hugging Face, or your hosting]
   
   **Alternative: YOLOv12x (no download needed):**
   - YOLOv12x models are automatically downloaded by Ultralytics on first use
   - No manual download required

3. Explore the module folders (see [Module Documentation](#module-documentation) below) for scripts and utilities.
4. Use the provided DataLoader for training models with PyTorch.

## Module Documentation

Each module has its own README with detailed documentation:

- **[Annotation/](Annotation/README.md)** - Video annotation generation with object detection and tracking
- **[Calibration/](Calibration/README.md)** - Camera and sensor calibration tools
- **[DataLoader/](DataLoader/README.md)** - PyTorch DataLoader for training
- **[DeepLearning/](DeepLearning/README.md)** - Multi-backend training framework
- **[Evaluation/](Evaluation/README.md)** - Model evaluation and benchmarking tools
- **[ImageSignalProcessing/](ImageSignalProcessing/README.md)** - ISP pipeline for RAW image processing
- **[TimeSync/](TimeSync/README.md)** - Temporal synchronization utilities
- **[Validation/](Validation/README.md)** - Sensor fusion validation and visualization
- **[tests/](tests/README.md)** - Test suite and validation scripts

## Example Workflows

- **Run ISP and Export Video:**
  - Use scripts in `ImageSignalProcessing/` to process RAW images and export videos.
- **Calibrate and Validate Sensors:**
  - Use scripts in `Calibration/` and `Validation/` to calibrate cameras/LIDARs and validate dataset alignment.
- **Generate Video Annotations:**
  - Use `Annotation/annotation_generation.py` to automatically detect, track, and annotate objects in videos.
  - See the [Annotation Generation](#annotation-generation) section below for details.
- **Train Models:**
  - Use the DataLoader to train models on the dataset and compare sensor performance.

## Annotation Generation

### Overview

The annotation generation module (`Annotation/annotation_generation.py`) is a complete video annotation pipeline that processes videos to automatically detect, track, and annotate objects in COCO format. It uses state-of-the-art object detection models combined with multi-object tracking to generate temporal annotations with optional 3D location information.

### Core Functionality

The module processes video files frame-by-frame to:

1. **Detect objects** using deep learning models (Dome-DETR or YOLOv12x)
2. **Track objects** across frames using OC-SORT tracker
3. **Compute 3D locations** using geometric ground plane intersection (optional)
4. **Generate COCO-format annotations** with temporal tracking information
5. **Interpolate positions** for frames between detections

### Main Components

**Config Class:** Configuration dataclass with pipeline parameters including video input, detection model selection, 3D projection settings, tracking parameters, and output options.

**ObjectDetector Class:** Handles object detection using either:
- **Dome-DETR**: Transformer-based detector optimized for VisDrone dataset
- **YOLOv12x**: Ultralytics YOLO model for general object detection

Supports three classes: `person`, `bicycle`, `car`

**Tracker Class:** Implements multi-object tracking using **OC-SORT** with IoU/GIoU association, handling occlusions and temporary disappearances.

**TrackedObject Class:** Represents a tracked object with 2D/3D tracking, class information, and temporal data.

### Key Features

- **Adaptive Frame Processing**: Automatically calculates frame skip to achieve target processing FPS (~5 FPS)
- **3D Location Computation**: Optional geometric ground plane intersection with distortion correction support
- **Camera Calibration**: Automatically extracts camera intrinsics and distortion coefficients from calibration files
- **Track Interpolation**: Linear interpolation for both 2D bboxes and 3D locations across skipped frames
- **COCO Format Output**: Standard JSON format compatible with evaluation tools

### Usage Example

```python
from Annotation.annotation_generation import Config, process_video

# Create configuration
config = Config()
config.video_path = "path/to/video.mp4"

# For Dome-DETR: set paths to downloaded model files (if not in default location)
# config.detector_model = "dome-detr"
# config.detector_config_file = "path/to/Dome-L-VisDrone.yml"
# config.detector_checkpoint = "path/to/Dome-L-VisDrone-best.pth"

# For YOLOv12x: no paths needed, model downloads automatically
# config.detector_model = "yolo12x"

config.enable_depth_estimation = True  # Enable 3D locations
config.camera_height = 4.0  # meters
config.camera_pitch_deg = 20.0  # degrees
config.dist_coeffs = None  # Optional: distortion coefficients from calibration

# Process video
process_video(config.video_path, config)

# Output: video_name_anno.json (COCO format)
```

### Pipeline Workflow

```
Video Input
    ↓
Frame-by-Frame Processing
    ├──→ Object Detection (Dome-DETR/YOLO)
    │       ↓
    │   Bounding Boxes + Classes
    │       ↓
    ├──→ 3D Projection (if enabled)
    │       ├──→ Camera Intrinsics + Distortion
    │       ├──→ Ground Plane Intersection
    │       └──→ 3D Coordinates (X, Y, Z)
    │       ↓
    └──→ OC-SORT Tracking
            ├──→ Track Association
            ├──→ Interpolation (missing frames)
            └──→ Track Management
                ↓
COCO Annotations
    ├──→ Images metadata
    ├──→ Annotations with track IDs
    └──→ 3D locations (optional)
```

### Supported Classes

- **Dome-DETR**: Maps VisDrone classes (`pedestrian/people`, `bicycle`, `car`) to COCO format
- **YOLOv12x**: Uses COCO classes directly (`person`, `bicycle`, `car`)

### 3D Coordinate System

- **Origin**: Ground level directly below camera
- **X**: Right (positive = right from camera)
- **Y**: Forward (positive = forward from camera)
- **Z**: Up (always 0 for ground intersections)

### Configuration Tips

1. **For faster processing**: Increase `process_every_n_frames` or reduce `target_processing_fps`
2. **For better tracking**: Lower `tracking_iou_threshold` (0.2 recommended)
3. **For 3D accuracy**: Ensure camera calibration is correct, including distortion coefficients if available
4. **For detection quality**: Adjust `detector_conf_threshold` based on model performance

### Output Format

The generated JSON file follows COCO annotation format:
- Each image (frame) has unique `image_id`
- Each annotation includes:
  - `bbox`: [x, y, width, height] in pixels
  - `category_id`: Class ID (1=person, 2=bicycle, 3=car)
  - `track_id`: Persistent track ID across frames
  - `location_3d`: [X, Y, Z] in meters (if enabled)

This format is compatible with standard COCO evaluation tools and can be used for training, evaluation, and further analysis.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## Third-Party Code and Attributions

This repository includes third-party code and git submodules:

- **Git submodules:**
  - `OC_SORT/` - OC-SORT tracker for multi-object tracking (used by Annotation and Evaluation modules)
  - `Calibration/` - Sensor Calibration Toolbox
  
- **Heavily modified third-party code:**
  - `ImageSignalProcessing/` - Based on fast-openISP with significant modifications

- **License information:**
  - This project is licensed under the [MIT License](http://opensource.org/licenses/MIT)
  - Third-party components retain their original licenses (see respective submodule directories)
  - OC-SORT: See `OC_SORT/LICENSE` for license details
  - Sensor Calibration Toolbox: See `Calibration/` for license information

---

**GMIND-sdk** aims to provide a complete, consistent, and extensible toolkit for research and development with the GMIND ITS infrastructure node dataset.
