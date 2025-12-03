# TimeSync - Multi-Camera Time Synchronization and Alignment Tool

## Overview

The `TimeSync/temporal_alignment_validation.py` script is an **interactive visualization and time synchronization tool** for validating and aligning 3D object detections across multiple cameras. It's designed to help synchronize timestamps and validate the geometric alignment of objects detected from different camera viewpoints.

## Primary Purpose

**Time Synchronization & Alignment Validation**: When you have multiple cameras (e.g., FLIR 8.9MP, FLIR 3.2MP, CCTV) capturing the same scene, their frame timestamps may not be perfectly aligned. This tool helps you:

1. **Visualize 3D object locations** from multiple cameras in a shared coordinate space
2. **Adjust frame offsets** to synchronize cameras temporally
3. **Validate geometric alignment** by matching the same object across cameras
4. **Find optimal synchronization** automatically using alignment metrics

## Key Features

### 1. **Multi-Camera Visualization**
- Loads COCO annotation files with 3D locations from multiple cameras
- Displays objects from all cameras in a **top-down XY view** (bird's eye view)
- Color-coded by camera (FLIR 8.9MP = green, FLIR 3.2MP = orange, etc.)
- Marker shapes indicate object categories (circle = person, square = bicycle, triangle = car)

### 2. **Frame-by-Frame Navigation**
- Navigate through frames independently for each camera
- Compare what each camera sees at different frame indices
- See real-time video playback synchronized with the plot

### 3. **Time Synchronization (Frame Offset Adjustment)**
- **Manually adjust offsets**: Use arrow keys to shift one camera's frames forward/backward
- **Automatic optimization**: Press 'F' to automatically find the best frame offset that minimizes alignment errors
- **Per-camera offsets**: Each camera can have its own frame offset

### 4. **Alignment Metrics & Validation**
- **Matches objects** from different cameras based on:
  - Same category/class (person, bicycle, car)
  - Proximity in 3D space (within 8m threshold)
  - One-to-one matching constraint
- **Computes statistics**:
  - Median alignment distance (primary metric)
  - Mean, std dev, percentiles
  - Number of matched objects
- **Visual feedback**: Connection lines show matched objects with color indicating distance

### 5. **Advanced Features**
- **Static object filtering**: Option to exclude stationary objects from alignment metrics
- **Alignment region**: Focuses on a specific region (e.g., -10m to +20m in X, 0m to 20m in Y)
- **Video overlay**: Displays bounding boxes and footpoints in video windows
- **Match visualization**: Color-coded bounding boxes show which objects are matched across cameras

## Visualization Modes

### 1. **Single Camera - Static 3D Plot**
```bash
python TimeSync/temporal_alignment_validation.py --annotation_file path/to/video_anno.json
```
- Shows all 3D locations in a 3D scatter plot
- Color-coded by object category
- Interactive rotation and zoom

### 2. **Single Camera - Animated Over Time**
```bash
python TimeSync/temporal_alignment_validation.py --annotation_file path/to/video_anno.json --plot_time
```
- Animated visualization showing objects appearing over time
- Press SPACEBAR to pause/play

### 3. **Single Camera - Track Visualization**
```bash
python TimeSync/temporal_alignment_validation.py --annotation_file path/to/video_anno.json --plot_tracks
```
- Shows object tracks (connects points by track_id)
- Useful for seeing object trajectories

### 4. **Multi-Camera Time Synchronization (Main Feature)**
```bash
python TimeSync/temporal_alignment_validation.py \
    --plot_xy \
    --annotation_dir "/mnt/h/GMIND/UrbanJunctionSet/1" \
    --cameras FLIR3.2 FLIR8.9
```
- **Top-down XY view** showing objects from multiple cameras
- **Frame-by-frame navigation** with time synchronization
- **Alignment validation** with automatic optimization
- **Video playback** with bounding box overlay

## Interactive Controls

### Frame Navigation (Active Camera)
- **TAB**: Switch active camera (for controls)
- **RIGHT ARROW / 'n'**: Next frame (active camera)
- **LEFT ARROW / 'p'**: Previous frame (active camera)
- **HOME**: Jump to first frame
- **END**: Jump to last frame
- **0-9**: Jump to frame at percentage (0 = start, 9 = 90%)

### Time Synchronization
- **UP ARROW / 'u'**: Increase frame offset (+1 frame) for active camera
- **DOWN ARROW / 'd'**: Decrease frame offset (-1 frame) for active camera
- **'r'**: Reset offset to 0
- **'f'**: **Find best alignment** - automatically searches ±100 frames for optimal offset

### All Cameras Navigation
- **'N' (uppercase)**: Next frame for ALL cameras simultaneously
- **'P' (uppercase)**: Previous frame for ALL cameras simultaneously

### Other Controls
- **'m'**: Toggle static object exclusion from alignment metrics

## How It Works

### 1. **Loading Data**
- Reads COCO annotation JSON files from multiple cameras
- Extracts 3D locations from the `location_3d` field in annotations
- Groups objects by frame (image_id) for temporal navigation

### 2. **Matching Objects Across Cameras**
For each frame, the tool:
1. Takes FLIR 8.9MP as the reference camera
2. Finds objects from other cameras that are:
   - Same category/class
   - Within 8m distance in 3D space
   - Closest match (greedy one-to-one matching)
3. Computes alignment statistics (median distance, etc.)

### 3. **Finding Optimal Synchronization**
When you press 'F':
- Temporarily adjusts frame offset for the active camera
- Tests offsets from -100 to +100 frames
- Computes alignment metrics for each offset
- Selects the offset with the **lowest median distance**
- Automatically applies the best offset

### 4. **Visual Feedback**
- **Green lines**: Well-aligned matches (< 4m)
- **Yellow lines**: Medium alignment (4-6m)
- **Orange lines**: Poor alignment (6-8m)
- **No line**: Objects too far apart (> 8m) or different categories

## Example Workflow

1. **Load multi-camera annotations**:
   ```bash
   python TimeSync/temporal_alignment_validation.py \
       --plot_xy \
       --annotation_dir "H:/GMIND/UrbanJunctionSet/1" \
       --cameras FLIR3.2 FLIR8.9
   ```

2. **Navigate to a frame** with many objects (using arrow keys)

3. **Check alignment**: Look at the connection lines between cameras
   - Green lines = good alignment
   - Orange/red lines = poor alignment (may need sync)

4. **Adjust frame offset manually**:
   - Select camera (TAB)
   - Use UP/DOWN arrows to shift frames
   - Watch alignment metrics update

5. **Find optimal sync automatically**:
   - Press 'F' to search for best offset
   - Tool will test multiple offsets and pick the best one
   - View the alignment metrics plot to see results

6. **Validate**: Check that matched objects (same color boxes in videos) correspond to the same real-world objects

## Use Cases

1. **Time Synchronization**: Align frame timestamps across cameras that may have slight timing differences
2. **Calibration Validation**: Verify that 3D projections are consistent across cameras
3. **Dataset Quality Check**: Ensure objects are correctly localized in 3D space
4. **Multi-Sensor Fusion**: Validate that detections from different cameras correspond to the same objects
5. **Alignment Debugging**: Diagnose issues with 3D projection or calibration

## Technical Details

### Input Format
- Expects COCO-format JSON files with `location_3d` field in annotations
- Location format: `[X, Y, Z]` in meters (ground plane coordinates)
- Requires frame information (image_id) for temporal alignment

### Alignment Region
- Focuses on a specific region: X = -10m to +20m, Y = 0m to +20m
- Objects outside this region don't contribute to alignment metrics
- Region is shown as a cyan dashed rectangle

### Static Object Detection
- Objects with position variance < 2m are considered static
- Can be excluded from alignment metrics (press 'm')
- Helps focus on moving objects for better synchronization

### Video Integration
- Automatically finds video files matching camera names
- Displays bounding boxes with color coding:
  - Camera color = unmatched objects
  - Match color = matched objects across cameras
- Shows footpoint markers (where 3D projection originates)

## Output

The tool provides real-time feedback:
- **Alignment statistics**: Median distance, mean, std dev, match count
- **Frame information**: Current frame numbers and offsets for each camera
- **Object counts**: Number of objects visible in current frame
- **Best offset results**: Shows top 3 offsets when using automatic search

## Notes

- **FLIR 8.9MP is always the reference**: Other cameras are adjusted relative to it
- **One-to-one matching**: Each object can only match with one object from another camera
- **Category-based matching**: Objects must be the same class (person, bicycle, car) to match
- **Distance threshold**: 8m maximum distance for matching (configurable in code)

This tool is essential for ensuring accurate multi-camera datasets where temporal and spatial alignment between sensors is critical!

