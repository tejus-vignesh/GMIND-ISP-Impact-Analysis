# GMIND Dataset DataLoader

Custom PyTorch Dataset and DataLoader for GMIND videos with 2D bounding box annotations.

## Features

- **Video Loading**: Loads video files (.mp4) and extracts frames on-demand
- **2D Bounding Boxes**: Supports COCO-format annotations with 2D bounding boxes
- **Multi-Sensor Support**: Load videos from different sensors (FLIR3.2, FLIR8.9, CCTV, Thermal, EB, Drone)
- **Multiple Dataset Sets**: Support for UrbanJunctionSet, DistanceTestSet, OutlierSet, PedestrianInteractionSet
- **Frame Sampling**: Optional frame stride and max frames per video
- **PyTorch Compatible**: Works seamlessly with PyTorch training pipelines

## Dataset Structure

The GMIND dataset should be organized as follows:

```
GMIND/
├── UrbanJunctionSet/
│   ├── 1/
│   │   ├── FLIR3.2-Urban1.mp4
│   │   ├── FLIR3.2-Urban1.json
│   │   ├── FLIR8.9-Urban1.mp4
│   │   ├── FLIR8.9-Urban1.json
│   │   └── ...
│   └── 2/
│       └── ...
├── DistanceTestSet/
├── OutlierSet/
└── PedestrianInteractionSet/
```

## Usage

### Basic Usage

```python
from DataLoader import get_gmind_dataloader
from DeepLearning.train_models import get_transform

# Create dataloader
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",  # or "H:/GMIND" on Windows
    sets=["UrbanJunctionSet"],
    sensor="FLIR3.2",
    transforms=get_transform(train=True),
    batch_size=4,
    shuffle=True,
    num_workers=4,
)

# Use in training loop
for images, targets in train_loader:
    # images: Tensor of shape [B, C, H, W]
    # targets: List of dictionaries with 'boxes', 'labels', 'area', 'iscrowd', 'image_id'
    #   - boxes: Tensor [N, 4] in [x1, y1, x2, y2] format
    #   - labels: Tensor [N] with category IDs (1-based)
    #   - area: Tensor [N] with bounding box areas
    #   - iscrowd: Tensor [N] with crowd flags
    #   - image_id: Tensor [1] with image ID (required for COCO evaluation)
    ...
```

### Direct Dataset Usage

```python
from DataLoader import GMINDDataset
from DeepLearning.train_models import get_transform

dataset = GMINDDataset(
    data_root="/mnt/h/GMIND",
    sets=["UrbanJunctionSet"],
    sensor="FLIR3.2",
    transforms=get_transform(train=True),
    frame_stride=1,  # Load every frame
    max_frames=None,  # Load all frames
)

# Access individual samples
image, target = dataset[0]
```

### Frame Sampling

You can sample frames to reduce dataset size:

```python
# Load every 10th frame
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",
    sets=["UrbanJunctionSet"],
    sensor="FLIR3.2",
    frame_stride=10,  # Every 10th frame
    max_frames=100,   # Max 100 frames per video
    batch_size=4,
)
```

### Advanced Filtering Options

Filter by specific subdirectories or sets:

```python
# Only load videos from subdirectory 1 and 2
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",
    sets=["UrbanJunctionSet"],
    sensor="FLIR3.2",
    subdirs=[1, 2],  # Only subdirectories 1 and 2
    batch_size=4,
)

# Different subdirectories for different sets
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",
    sets=["UrbanJunctionSet", "PedestrianInteractionSet"],
    sensor="FLIR3.2",
    set_subdirs={
        "UrbanJunctionSet": [1, 2],      # Only subdirs 1, 2 for UrbanJunctionSet
        "PedestrianInteractionSet": [1],  # Only subdir 1 for PedestrianInteractionSet
    },
    batch_size=4,
)
```

### Percentage Splits

Use a percentage of frames from specific videos:

```python
# Use first 60% of frames from PedestrianInteractionSet/2
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",
    sets=["PedestrianInteractionSet"],
    sensor="FLIR3.2",
    percentage_split={"PedestrianInteractionSet/2": 0.6},  # First 60%
    batch_size=4,
)

# Use frames from 40% to 80% of the video
train_loader = get_gmind_dataloader(
    data_root="/mnt/h/GMIND",
    sets=["PedestrianInteractionSet"],
    sensor="FLIR3.2",
    percentage_split={"PedestrianInteractionSet/2": 0.4},      # 40% of frames
    percentage_split_start={"PedestrianInteractionSet/2": 0.4}, # Starting at 40%
    # Result: frames from 40% to 80% of the video
    batch_size=4,
)
```

## Parameters

### GMINDDataset

- `data_root`: Root directory of GMIND dataset (str or Path)
- `sets`: List of dataset sets to include (default: all sets)
- `sensor`: Sensor type to load (e.g., "FLIR3.2", "FLIR8.9", "CCTV", "Thermal", "EB", "Drone")
- `annotation_format`: Format of annotations (currently "coco" supported)
- `transforms`: Optional transforms to apply to images
- `frame_stride`: Load every Nth frame (default: 1, load all frames)
- `max_frames`: Maximum number of frames to load per video (None for all)
- `subdirs`: Optional list of subdirectory numbers to include (e.g., [1, 2])
- `set_subdirs`: Optional dict mapping set names to subdir lists (e.g., `{"UrbanJunctionSet": [1]}`)
- `percentage_split`: Optional dict for percentage splits (e.g., `{"PedestrianInteractionSet/2": 0.6}`). If specified, only uses the first N% of frames from that set/subdir
- `percentage_split_start`: Optional dict for starting point of percentage splits (e.g., `{"PedestrianInteractionSet/2": 0.4}`). If specified with `percentage_split`, uses frames from start% to (start+split)%. If specified alone, uses frames from start% to end

### get_gmind_dataloader

All parameters from `GMINDDataset` plus:
- `batch_size`: Batch size for DataLoader (default: 4)
- `shuffle`: Whether to shuffle the dataset (default: True)
- `num_workers`: Number of worker processes (default: 4)
- Additional DataLoader kwargs (e.g., `pin_memory`, `drop_last`, etc.)

## Annotation Format

The DataLoader expects COCO-format JSON files with the following structure:

```json
{
  "images": [
    {
      "id": 1,
      "width": 2048,
      "height": 1536,
      "file_name": "FLIR3.2-Urban1_frame_000000.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 816.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"}
  ]
}
```

Bounding boxes are returned in `[x1, y1, x2, y2]` format (absolute coordinates).

## Notes

- Videos are loaded on-demand (frames extracted when needed)
- Annotations are matched to frames by frame number extracted from filename
- If no annotations are found for a frame, empty targets are returned
- Currently supports per-sensor loading (synchronized multi-sensor support coming soon)

