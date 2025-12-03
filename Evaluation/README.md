# Evaluation Package

Comprehensive evaluation tools for object detection and tracking models on the GMIND dataset.

## Overview

This package provides a complete suite of evaluation tools following COCO evaluation protocols. It includes scripts for running evaluations, analysing results, computing metrics, visualising outputs, and preparing data.

## Structure

### **`core/`** - Core Evaluation Scripts
- **`baseline_detector_and_tracker.py`** - Main evaluation script
  - Uses YOLOv11n for detection and OC-SORT for tracking
  - Computes COCO metrics (AP50, AP50-95, APsmall, APlarge, etc.)
  - Per-class and per-video metric breakdowns
  - Real-time visualisation and checkpointing support

### **`analysis/`** - Analysis and Reporting Scripts
- **`analyse_results.py`** - Unified analysis script (combines all analysis functionality)
  - Detection results analysis (objects per frame, class breakdown, GT overlaps)
  - Missed objects per frame analysis
  - MOTA per video breakdown and overall MOTA
  - Size vs performance plotting (miss rate and AP50 by object size)
  - Large object performance debugging
  - Comprehensive evaluation reports
  - Static vs moving object comparison
- **`analysis_utils.py`** - Common utility functions for analysis
  - IoU computation, area categorisation, matching functions
  - COCO metrics computation, per-class AP calculation


### **`visualisation/`** - Visualisation Tools
- **`visualise_gt_and_pred.py`** - Visualise ground truth and predictions in real-time
  - Side-by-side or overlaid visualisation
  - Highlights missed large objects
  - Interactive frame-by-frame navigation
  - Requires access to original video files from the dataset

### **`utils/`** - Utility Scripts
- **`filter_moving_objects.py`** - Filter to moving objects only
  - Identifies objects appearing in multiple consecutive frames
  - Creates filtered GT and results files

## Quick Start

### Running Main Evaluation

```bash
# Run evaluation with default settings
python -m Evaluation.core.baseline_detector_and_tracker \
    --config DeepLearning/gmind_config.yaml \
    --output-dir baseline_evaluation_results

# With real-time visualisation
python -m Evaluation.core.baseline_detector_and_tracker \
    --config DeepLearning/gmind_config.yaml \
    --show-vis \
    --output-dir baseline_evaluation_results
```

### Analysing Results

```bash
# Run all analyses
python -m Evaluation.analysis.analyse_results \
    --all \
    --output-dir baseline_evaluation_results

# Run specific analyses
python -m Evaluation.analysis.analyse_results \
    --detection \
    --missed \
    --mota \
    --overall-mota \
    --size-plot \
    --large \
    --output-dir baseline_evaluation_results

# Generate comprehensive report
python -m Evaluation.analysis.analyse_results \
    --full-report \
    --output-dir baseline_evaluation_results

# Compare static vs moving objects
python -m Evaluation.analysis.analyse_results \
    --compare-static-moving \
    --output-dir baseline_evaluation_results

# Plot size vs performance
python -m Evaluation.analysis.analyse_results \
    --size-plot \
    --output-dir baseline_evaluation_results
```

### Visualisation

```bash
# Visualise GT and predictions
python -m Evaluation.visualisation.visualise_gt_and_pred \
    --gt-file baseline_evaluation_results/coco_gt.json \
    --results baseline_evaluation_results/coco_results.json \
    --output-dir baseline_evaluation_results
```

**Note:** For visualising annotations by themselves (without evaluation), use:
```bash
python -m Annotation.visualise_annotations /path/to/video.mp4 --delay 10
```

### Data Preparation

```bash
# Filter to moving objects only
python -m Evaluation.utils.filter_moving_objects \
    --gt-file baseline_evaluation_results/coco_gt.json \
    --results baseline_evaluation_results/coco_results.json \
    --output-dir baseline_evaluation_results \
    --min-consecutive-frames 3
```

**Note:** 3D positions are automatically added during annotation generation in `Annotation/annotation_generation.py` when `enable_depth_estimation` is enabled.

## Data Formats

### Input Formats

All scripts expect COCO format data:

**Ground Truth (COCO format):**
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5
    }
  ],
  "categories": [...]
}
```

**Results (COCO results format):**
```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.95
  }
]
```

### Output Formats

- **Metrics**: Printed to stdout in human-readable format
- **Plots**: Saved as PNG files (e.g., `size_vs_performance.png`)
- **Processed Data**: Saved as JSON files in COCO format

## Metrics Explained

### COCO Metrics

- **AP50-95**: Average Precision over IoU thresholds 0.5:0.05:0.95
- **AP50**: Average Precision at IoU threshold 0.5
- **AP75**: Average Precision at IoU threshold 0.75
- **APsmall/APmedium/APlarge**: AP for different object sizes
  - Small: area < 32² = 1,024 pixels²
  - Medium: 32² ≤ area < 96² = 9,216 pixels²
  - Large: area ≥ 96² = 9,216 pixels²

### Tracking Metrics

- **MOTA**: Multiple Object Tracking Accuracy
  - Formula: MOTA = 1 - (FN + FP + IDSW) / GT
  - Note: Our implementation provides a detection-based approximation
    without ID switches (requires tracking information for full MOTA)

## Dependencies

- Python 3.8+
- pycocotools (for COCO evaluation)
- ultralytics (for YOLO models)
- OC-SORT (for tracking)
- OpenCV (for visualisation)
- NumPy, Matplotlib (for plotting and analysis)

## Notes

- All scripts use British English spelling (e.g., "visualise" not "visualize")
- COCO format is used throughout for compatibility
- Scripts are designed to work with the GMIND dataset structure
- Most scripts can auto-detect paths and create GT files from datasets if needed

## License

See main repository license.
