# Video Annotation Generator

A comprehensive video processing tool that performs foreground segmentation, object detection, and tracking on video files, generating COCO-format annotations. The system uses a sophisticated processing pipeline optimized for static camera, static background scenarios with moving objects.

## Features

- **Foreground Segmentation**: Uses MOG2 (BGS) background subtraction to identify moving regions
- **Connected Components**: Extracts object blobs from foreground masks with adaptive bounding box expansion
- **Dual Detector System**: Optionally uses YOLOv11x for full-frame detection and RT-DETR-X for region detection
- **Multi-Model Support**: Supports YOLOv11x, YOLOv8x, RT-DETR-X, RT-DETR-L, and Mask R-CNN models
- **Object Tracking**: Uses OC-SORT or ByteTrack for temporal tracking with Kalman filter extrapolation
- **Static Object Detection**: Samples random frames to detect static objects that appear consistently
- **COCO Annotations**: Exports tracked objects in standard COCO format with track IDs
- **Interpolation/Extrapolation**: Uses Kalman filters for smooth tracking during occlusions
- **Debug Visualization**: Optional 3-window visualization showing tracking, foreground mask, and detection regions
- **Frame Sampling**: Processes every Nth frame (default: 30) for faster processing

## Processing Pipeline

The system follows a structured processing flow optimized for CCTV and surveillance videos:

### Step 1: Foreground Segmentation (BGS)
- Runs MOG2 background subtraction to identify moving regions
- Applies morphological operations (close, open, dilation) to clean up the mask
- Special handling for CCTV videos (blur to reduce Moire patterns)
- Configurable via `fg_seg_*` parameters in the `Config` class

### Step 2: Connected Components
- Extracts object blobs from foreground mask using connected components analysis
- Filters blobs by minimum area threshold
- Expands blob bounding boxes with adaptive expansion (smaller objects get more padding)
- Rejects regions smaller than 20x20 pixels
- Configurable via `cc_*` and `blob_expansion_ratio` parameters

### Step 3: Object Detection
- **Dual Detector Mode** (recommended): 
  - YOLOv11x runs on full frame periodically (configurable frequency) for static objects
  - RT-DETR-X runs on foreground blob regions for moving objects
  - Detections are merged with NMS
- **Single Detector Mode**:
  - Runs detector on full frame and filters to foreground regions
  - Optionally includes static objects detected across entire frame
- Only tracks: Person (0), Bicycle (1), and Car (2)
- Configurable via `detector_*`, `use_dual_detector`, `nms_*` parameters

### Step 4: Static Object Detection (Optional)
- Samples random frames throughout the video
- Groups detections that appear consistently across frames using IoU matching
- Validates static objects that appear in at least 50% of sampled frames
- Averages positions for consistency
- Configurable via `detect_static_objects`, `static_detection_*` parameters

### Step 5: Temporal Tracking
- Uses OC-SORT or ByteTrack for multi-object tracking
- Maintains object IDs across frames with velocity-based prediction
- Handles occlusions using Kalman filter extrapolation
- Interpolates between lost and regained positions for smooth tracks
- Filters short tracks (must be tracked for >2 frames to be saved)
- Configurable via `tracking_*` parameters

### Step 6: COCO Export
- Saves annotations in COCO format with track IDs
- Includes tracked moving objects and detected static objects
- Output saved in same directory as video with `.json` extension

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- NumPy
- Ultralytics (for YOLO/RT-DETR models)
- FilterPy (for Kalman filters, optional but recommended)
- OC-SORT or ByteTrack (for tracking)

### Setup
```bash
# Install core dependencies
pip install opencv-python torch torchvision numpy ultralytics

# Install Kalman filter support (recommended)
pip install filterpy

# Install OC-SORT (for OC-SORT tracking)
pip install ocsort
# OR
pip install git+https://github.com/noahcao/OC_SORT.git

# Install ByteTrack (for ByteTrack tracking)
# Clone from https://github.com/ifzhang/ByteTrack and add to path
```

## Usage

### Configuration

The script uses a `Config` dataclass for all parameters. Edit the `Config` class at the top of `dep_annotation_generation.py` to customize behavior:

```python
@dataclass
class Config:
    # Video input
    video_path: str = "H:\\GMIND\\OutlierSet\\1\\FLIR8.9-Outlier1.mp4"
    process_every_n_frames: int = 30  # Process every Nth frame
    
    # Foreground Segmentation
    fg_seg_method: str = "BGS"  # Only MOG2 (BGS) supported
    fg_seg_history: int = 500
    fg_seg_var_threshold: int = 16
    
    # Object Detection
    detector_model: str = "rtdetr-x"
    use_dual_detector: bool = True  # YOLOv11x + RT-DETR-X
    detector_conf_threshold: float = 0.15
    
    # Tracking
    tracking_method: str = "OC-SORT"  # "OC-SORT" or "ByteTrack"
    
    # Output
    output_dir: str = "/mnt/h/GMIND/Annotations/video_tracking"
    save_annotations: bool = True
    headless: bool = False
    debug_visualization: bool = True
```

### Basic Usage

Process a single video with default settings:
```bash
python dep_annotation_generation.py
```

The script will:
1. Load the video from `Config.video_path` (Windows paths like `H:\GMIND\...` are automatically converted to WSL paths)
2. Process every Nth frame (default: 30)
3. Save annotations to the same directory as the video with `.json` extension

### Configuration Options

#### Video Input
- `video_path`: Path to video file (Windows paths automatically converted to WSL)
- `process_every_n_frames`: Process every Nth frame (1 = all frames, 30 = every 30th frame)

#### Foreground Segmentation
- `fg_seg_method`: Only "BGS" (MOG2) is supported
- `fg_seg_history`: Number of frames for background model history
- `fg_seg_var_threshold`: Variance threshold for BGS
- `fg_seg_detect_shadows`: Enable shadow detection
- `fg_seg_learning_rate`: Learning rate (-1 = auto, 0-1 = manual)
- `fg_seg_morph_kernel_size`: Morphological operations kernel size

#### Connected Components
- `cc_min_area`: Minimum blob area in pixels
- `cc_connectivity`: 4 or 8 connectivity
- `blob_expansion_ratio`: Expand blob bounding boxes by this ratio (default: 0.15 = 15% padding)

#### Object Detection
- `detector_model`: Model to use ("yolo11x", "yolov8x", "rtdetr-l", "rtdetr-x", "maskrcnn_resnet50_fpn_v2")
- `detector_conf_threshold`: Confidence threshold for detections (default: 0.15)
- `use_dual_detector`: Use YOLOv11x (full frame) + RT-DETR-X (regions), then merge with NMS
- `full_frame_detection_frequency`: Run full frame detection N times evenly spaced throughout video
- `detect_static_objects`: Enable static object detection
- `static_detection_sample_frames`: Number of random frames to sample for static detection
- `static_detection_iou_threshold`: IoU threshold for grouping static objects across frames
- `nms_threshold`: IoU threshold for NMS when merging detections (lower = more aggressive)
- `nms_nested_ratio`: If smaller box area < this ratio of larger box, merge nested detections
- `nms_nested_iou`: IoU threshold for partial nesting detection

#### Tracking
- `tracking_method`: "OC-SORT" or "ByteTrack"
- `tracking_max_age`: Maximum frames to keep lost tracks
- `tracking_min_hits`: Minimum hits to confirm track
- `tracking_iou_threshold`: IoU threshold for matching
- `tracking_det_thresh`: Detection confidence threshold (should match detector_conf_threshold)
- `tracking_delta_t`: Delta time for observation-centric re-update (OC-SORT)
- `tracking_inertia`: Inertia weight for OC-SORT (0.0-1.0)
- `tracking_asso_func`: Association function ("iou", "giou", "ciou", "diou", "ct_dist")

#### Output
- `output_dir`: Output directory for annotations (not used - saves next to video)
- `save_annotations`: Enable saving COCO annotations
- `headless`: Run without display windows
- `debug_visualization`: Enable debug visualization windows
- `show_extrapolated_boxes`: Show extrapolated (lost) tracks in visualization

### Examples

#### Change Video Path
Edit the `Config` class:
```python
config = Config()
config.video_path = "H:\\GMIND\\UrbanJunctionSet\\1\\FLIR8.9-Urban1.mp4"
```

#### Process All Frames
```python
config = Config()
config.process_every_n_frames = 1  # Process every frame
```

#### Use Single Detector
```python
config = Config()
config.use_dual_detector = False
config.detector_model = "yolo11x"
```

#### Use ByteTrack Instead of OC-SORT
```python
config = Config()
config.tracking_method = "ByteTrack"
```

#### Headless Mode (No Display)
```python
config = Config()
config.headless = True
config.save_annotations = True
```

## Interactive Controls

When running in non-headless mode:

- **'q'** or **ESC**: Quit

## Debug Visualization

When `debug_visualization=True` and `headless=False`, the display shows 3 separate windows:

1. **"Video with Tracking"** window: Main visualization
   - Original frame with tracked objects
   - Colored boxes per track ID (format: "ID:class" or "[EXTRAP]ID:class")
   - Track trails showing movement history
   - Solid boxes: Active tracks
   - Dashed boxes: Extrapolated tracks (lost, if enabled)
   - Semi-transparent foreground mask overlay (if debug enabled)
   - Info text: Frame number, track count, FPS

2. **"Foreground Mask"** window: Background subtraction result
   - White pixels: Detected motion
   - Black pixels: Background
   - Shows cleaned foreground mask after morphological operations

3. **"DL Regions"** window: Detection regions and results
   - Colored regions: Foreground blob regions sent to detector
   - Detection boxes: Raw detections from object detector
   - Color-coded by class (Yellow: person, Orange: bicycle, Magenta: car)
   - Shows static detection status

All windows can be resized independently.

## Tracked Classes

The system tracks the following COCO classes:

- **Person** (class ID: 0)
- **Bicycle** (class ID: 1)
- **Car** (class ID: 2)

Note: Only these 3 classes are tracked. Other detections are filtered out.

## Configuration

All parameters are centralized in the `Config` dataclass at the top of `dep_annotation_generation.py`, organized by processing component:

- `video_path`, `process_every_n_frames`: Video input parameters
- `fg_seg_*`: Foreground segmentation parameters
- `cc_*`, `blob_expansion_ratio`: Connected components parameters
- `detector_*`, `use_dual_detector`, `nms_*`: Detection parameters
- `detect_static_objects`, `static_detection_*`: Static object detection parameters
- `tracking_*`: Tracking parameters
- `output_dir`, `save_annotations`, `headless`, `debug_visualization`: Output parameters

Modify these values directly in the code to adjust behavior.

## Output Format

### COCO Annotations

The generated annotations follow the COCO format and are saved in the same directory as the video file with a `.json` extension:

```json
{
  "info": {
    "description": "GMIND Video Annotations - video_name.mp4",
    "version": "1.0",
    "year": 2024,
    "contributor": "GMIND SDK",
    "date_created": "2024-01-01T00:00:00"
  },
  "images": [
    {
      "id": 1,
      "width": 1920,
      "height": 1080,
      "file_name": "video_name_frame_000000.jpg"
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345.0,
      "iscrowd": 0,
      "score": 0.95,
      "track_id": 42
    },
    ...
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "none"
    },
    ...
  ]
}
```

**Note**: 
- The `track_id` field is a custom extension to the COCO format, allowing downstream tools to link annotations across frames
- Bounding boxes are in COCO format: `[x, y, width, height]` (top-left corner + size)
- Only tracks with duration > 2 frames are saved
- Extrapolated boxes are only saved if the track was later regained (`was_lost=True`)

## Performance

- **Frame Sampling**: Processes every Nth frame (default: 30) for faster processing
- **Dual Detector**: Efficiently uses YOLOv11x for full frame and RT-DETR-X for regions
- **GPU Acceleration**: Uses GPU when available for model inference
- **Progress Reporting**: Prints progress every 30 frames

Typical performance: 5-30 FPS depending on video resolution, frame sampling rate, model size, and hardware.

## Architecture

### Key Components

1. **ForegroundSegmenter**: MOG2 background subtraction with morphological operations
2. **BlobExtractor**: Connected components analysis with adaptive bounding box expansion
3. **ObjectDetector**: Multi-model detector with dual detector support (YOLOv11x + RT-DETR-X)
4. **Tracker**: OC-SORT or ByteTrack wrapper for temporal tracking
5. **TrackedObject**: Represents tracked objects with Kalman filter for extrapolation/interpolation
6. **Static Object Detection**: Samples and groups detections across frames

### Data Structures

- `Detection`: Represents a single detection with box, score, class ID, and class name
- `TrackedObject`: Represents a tracked object with Kalman filter, history, and interpolation/extrapolation state
- `Config`: Centralized configuration dataclass

### Tracking Features

- **Kalman Filter Extrapolation**: Predicts position during occlusions
- **Interpolation**: Smoothly interpolates between lost and regained positions
- **Position History**: Tracks last 10 positions for static object detection
- **Track Duration**: Tracks first frame, hit streak, and duration

## Troubleshooting

### Model Loading Issues
- Ensure Ultralytics is installed: `pip install ultralytics`
- Check that model weights are downloaded (first run may download automatically)
- For RT-DETR models, ensure Ultralytics version supports them

### Tracking Import Issues
- **OC-SORT**: Install via `pip install ocsort` or clone from GitHub
- **ByteTrack**: Clone repository and add to Python path
- Check that tracking method matches installed packages

### Path Issues
- Windows paths (`H:\...`) are automatically converted to WSL paths (`/mnt/h/...`)
- Ensure video files exist and are accessible
- Output annotations are saved in the same directory as the video

### Performance Issues
- Use `process_every_n_frames > 1` to process fewer frames (faster)
- Set `headless=True` for batch processing (no display overhead)
- Disable `debug_visualization` if not needed
- Use smaller models (RT-DETR-L instead of RT-DETR-X) for faster inference

### Memory Issues
- Reduce `process_every_n_frames` to process fewer frames at once
- Disable `detect_static_objects` if not needed
- Reduce `static_detection_sample_frames` for static detection

### No Detections
- Check that foreground mask is being generated correctly (use debug visualization)
- Lower `detector_conf_threshold` if detections are too strict
- Check that video has moving objects (BGS requires motion)
- Enable `detect_static_objects` to catch stationary objects

### Tracking Issues
- Lower `tracking_iou_threshold` if tracks are breaking frequently
- Increase `tracking_max_age` to keep lost tracks longer
- Lower `tracking_min_hits` to confirm tracks faster
- Check that detections are consistent (use debug visualization)

## License

Part of the GMIND SDK project.

## Author

GMIND SDK Development Team
