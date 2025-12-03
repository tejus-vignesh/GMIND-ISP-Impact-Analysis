# Validation

This folder contains validation and sensor fusion visualization tools for the GMIND SDK. These scripts help validate calibration accuracy, visualize sensor data overlays, and analyze projection errors.

## Scripts

### `lidar_reprojection.py`
Projects LIDAR point clouds onto camera images using calibration data.

**Features:**
- Loads PCD (Point Cloud Data) files and projects them onto video frames
- Supports background subtraction using first PCD as reference
- Color-coded visualization by intensity or Z-depth
- Interactive playback controls

**Usage:**
```python
# Edit the script to set paths:
calib_path = "path/to/sensor_calibration.txt"
video_path = "path/to/video.mp4"
pcd_folder = "path/to/pcd/folder"
camera_name = "FLIR8.9"  # Camera name from calibration file
lidar_name = "Velodyne"  # LIDAR name from calibration file

# Run:
python lidar_reprojection.py
```

**Configuration:**
- `COLOR_BY_Z`: Color points by Z-depth instead of intensity
- `ENABLE_BG_SUBTRACTION`: Enable background subtraction
- `BG_DIST_THRESHOLD`: Distance threshold for background subtraction (meters)
- `COLORMAP_Z_MIN/MAX`: Z-depth range for colormap (meters)

### `radar_visualisation.py`
Standalone radar data visualization tool.

**Features:**
- Visualizes radar detections in Birds Eye View
- Temporal tracking and visualization
- Range and angle visualization

**Usage:**
```python
# Edit the script to set the radar file path:
radar_file = "path/to/radar.csv"

# Run:
python radar_visualisation.py
```

### `infinity_projection_error.py`
Analyzes and visualizes projection errors at infinity for validation.

**Features:**
- Computes projection errors for points at infinity
- Visualizes error patterns
- Validates calibration accuracy
- Generates error plots and statistics

**Usage:**
```python
# Edit the script to set paths:
calib_path = "path/to/sensor_calibration.txt"
camera_name = "FLIR8.9"

# Run:
python infinity_projection_error.py
```

## Dependencies

All scripts require:
- OpenCV (`cv2`)
- NumPy
- Camera calibration utilities from `../Calibration/camera_intrinsics/`
- Sensor calibration file (`sensor_calibration.txt`) in GMIND format

Additional dependencies:
- `lidar_reprojection.py`: SciPy (for KDTree in background subtraction)
- `radar_visualisation.py`: Pandas (for CSV reading)
- `infinity_projection_error.py`: Matplotlib (for plotting)

## Calibration File Format

All scripts expect a `sensor_calibration.txt` file with the following format:

```
Name: CameraName
Focal_x: 1234.5
Focal_y: 1234.5
COD_x: 960.0
COD_y: 540.0
Dist_1: 0.1
Dist_2: -0.2
...
Extrinsics:
X: 0.0
Y: 0.0
Z: 0.0
R_00: 1.0
R_01: 0.0
...
```

**Calibration file details:**
- **Intrinsics**: Focal length (Focal_x, Focal_y), principal point (COD_x, COD_y), distortion coefficients (Dist_1 through Dist_5)
- **Extrinsics**: Translation (X, Y, Z) and rotation matrix (R_00 through R_22) for sensor-to-world or sensor-to-sensor transformations
- **Format**: Plain text with key-value pairs separated by colons
- **Multiple sensors**: Each sensor has its own section starting with "Name:"

For detailed calibration instructions, see the Calibration module documentation in `../Calibration/README.md`.

## Common Workflow

1. **Calibrate sensors** using tools in `../Calibration/`
2. **Validate calibration** using `infinity_projection_error.py`
3. **Visualize sensor fusion** using `lidar_reprojection.py`
4. **Verify alignment** by checking that projected points align with image features

## Troubleshooting

**No points visible:**
- Check that calibration file contains correct camera and sensor names
- Verify that PCD/radar files are in the correct format
- Ensure coordinate systems match between sensors

**Projection errors:**
- Run `infinity_projection_error.py` to validate calibration
- Check that extrinsics are correctly calibrated
- Verify sensor timestamps are synchronized

**Performance issues:**
- Reduce point cloud density for faster processing
- Disable background subtraction if not needed
- Use lower resolution videos for visualization

