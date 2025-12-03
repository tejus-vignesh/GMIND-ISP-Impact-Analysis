"""
RADAR Visualisation Tool

Standalone radar data visualisation tool that displays radar detections in
Birds Eye View (BEV) format. This tool helps visualise and validate radar
data independently of camera sensors.

Features:
- Visualises radar detections in Birds Eye View
- Temporal tracking and visualisation
- Range and angle visualisation with distance rings and angle lines
- Interactive playback controls (play/pause, step forward/backward)

Author: GMIND SDK Development Team
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === USER CONFIGURATIONS ===
# Image and Drawing settings
IMG_WIDTH = 1000
IMG_HEIGHT = 1000
SCALE = 8.0  # pixels per meter

# Animation settings
ROUND_FPS = (
    False  # Set to True to round FPS to nearest integer, False to use float for smoother playback.
)

# Colors
BG_COLOR = (0, 0, 0)
GRID_COLOR = (50, 50, 50)
TEXT_COLOR = (200, 200, 200)
POINT_COLOR = (0, 255, 255)  # Yellow
EGO_VEHICLE_COLOR = (255, 255, 255)

# Point settings
POINT_SIZE = 3


def load_radar_data(radar_file_path: str) -> pd.DataFrame:
    """
    Load and preprocess RADAR data from a CSV file.

    Reads radar detection data from CSV format, filters out invalid entries,
    and converts timestamps for temporal grouping.

    Args:
        radar_file_path: Path to CSV file containing radar detections

    Returns:
        Pandas DataFrame with columns:
            - Angle: Detection angle in degrees
            - Distance: Detection distance in meters
            - Time: Timestamp string (HH:MM:SS format)
            - datetime: Parsed datetime object for grouping
        Returns empty DataFrame if file cannot be loaded
    """
    logger.info(f"Loading RADAR data from: {os.path.basename(radar_file_path)}")
    try:
        df = pd.read_csv(radar_file_path)

        # Filter out rows where the target has disappeared
        if "TargetDissappear" in df.columns:
            df = df[df["TargetDissappear"] == False]

        # Drop rows where essential data is missing
        df.dropna(subset=["Angle", "Distance", "Time"], inplace=True)

        # Convert time string to datetime objects for grouping
        df["datetime"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)

        logger.info(f"Found {len(df)} valid RADAR detections.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {radar_file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Could not process file {radar_file_path}: {e}")
        return pd.DataFrame()


def estimate_radar_fps(df: pd.DataFrame) -> float:
    """
    Estimate the RADAR's frames per second (FPS) based on timestamps.

    Calculates the average time difference between consecutive timestamps
    and returns the reciprocal as the estimated frame rate.

    Args:
        df: DataFrame with 'datetime' column containing timestamps

    Returns:
        Estimated FPS as float. Returns default values (1.0 or 10.0) if
        insufficient data is available.
    """
    if "datetime" not in df.columns:
        return 1.0  # Default FPS

    # Get unique, sorted timestamps
    unique_times = sorted(df["datetime"].unique())
    if len(unique_times) < 2:
        return 10.0  # Not enough data to estimate, return default

    # Calculate the differences between consecutive timestamps in seconds
    time_diffs = np.diff(unique_times) / np.timedelta64(1, "s")

    # Filter out zero differences which can occur
    time_diffs = time_diffs[time_diffs > 0]
    if len(time_diffs) == 0:
        return 10.0

    # FPS is the reciprocal of the average time difference
    avg_diff = np.mean(time_diffs)
    return 1.0 / avg_diff if avg_diff > 0 else 10.0


def create_background_image(width: int, height: int, scale: float, max_range: float) -> np.ndarray:
    """
    Create a background image with distance rings and angle lines for BEV visualisation.

    Generates a Birds Eye View background with:
    - Distance rings at 10m intervals
    - Angle lines at 15-degree intervals (0-180 degrees, where 90° is forward)
    - Ego vehicle representation at sensor origin

    Args:
        width: Image width in pixels
        height: Image height in pixels
        scale: Pixels per meter for coordinate conversion
        max_range: Maximum range in meters to display

    Returns:
        Background image as numpy array (height x width x 3, uint8)
    """
    # Create a black background
    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    # Define the sensor's origin point on the image (bottom center)
    sensor_origin_px = (width // 2, height - 50)

    # --- Draw Distance Rings ---
    for dist_m in range(10, int(max_range) + 1, 10):
        radius_px = int(dist_m * scale)
        cv2.circle(img, sensor_origin_px, radius_px, GRID_COLOR, 1)
        # Add text label for the distance
        text_pos = (sensor_origin_px[0] + 5, sensor_origin_px[1] - radius_px)
        cv2.putText(img, f"{dist_m}m", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    # --- Draw Angle Lines ---
    # The user specifies that 90 degrees is forward. We draw labels from 0 to 180.
    for angle_deg_label in range(0, 181, 15):
        # Convert label angle (90=fwd) to calculation angle (0=fwd) for trig functions
        calc_angle_deg = angle_deg_label - 90
        angle_rad = np.radians(calc_angle_deg)

        # Calculate the end point of the line at max_range
        # Standard automotive coordinates: X=forward, Y=left
        x_world = max_range * np.cos(angle_rad)
        y_world = max_range * np.sin(angle_rad)

        # Convert world coordinates to image pixel coordinates (Y-axis is inverted)
        end_point_px = (
            int(sensor_origin_px[0] - y_world * scale),
            int(sensor_origin_px[1] - x_world * scale),
        )

        cv2.line(img, sensor_origin_px, end_point_px, GRID_COLOR, 1)
        # Add text label for the angle using the original 0-180 degree value
        label_pos = (end_point_px[0] + 5, end_point_px[1])
        cv2.putText(
            img, f"{angle_deg_label}°", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1
        )

    # --- Draw Ego Vehicle ---
    cv2.rectangle(
        img,
        (sensor_origin_px[0] - 10, sensor_origin_px[1] - 20),
        (sensor_origin_px[0] + 10, sensor_origin_px[1]),
        EGO_VEHICLE_COLOR,
        -1,
    )
    cv2.arrowedLine(
        img,
        (sensor_origin_px[0], sensor_origin_px[1] - 20),
        (sensor_origin_px[0], sensor_origin_px[1] - 40),
        (0, 255, 0),
        2,
    )

    return img


def main(radar_file: str) -> None:
    """
    Main function to generate and display the RADAR visualisation animation.

    Loads radar data, estimates frame rate, and displays interactive BEV
    visualisation with playback controls.

    Args:
        radar_file: Path to CSV file containing radar detections
    """
    # 1. Load the radar data
    radar_df = load_radar_data(radar_file)
    if radar_df.empty:
        return

    # 2. Dynamically set max range based on data, rounded up to nearest 10
    max_dist = radar_df["Distance"].max()
    max_range = np.ceil(max_dist / 10) * 10 if max_dist > 0 else 50.0
    logger.info(
        f"Max distance in data: {max_dist:.2f}m. Setting visualization range to {max_range:.0f}m."
    )

    # 3. Estimate FPS and calculate frame delay
    fps_float = estimate_radar_fps(radar_df)
    if ROUND_FPS:
        fps = int(round(fps_float))
        wait_time_ms = int(1000 / fps) if fps > 0 else 100
        logger.info(
            f"Estimated RADAR FPS: {fps_float:.2f}, rounded to {fps} (frame delay: {wait_time_ms}ms)"
        )
    else:
        fps = fps_float
        wait_time_ms = int(1000 / fps) if fps > 0 else 100
        logger.info(
            f"Estimated RADAR FPS: {fps_float:.2f} (using float, frame delay: {wait_time_ms}ms)"
        )

    # 4. Group data by timestamp and convert to a list for indexed access
    grouped_by_time = radar_df.groupby("datetime")
    time_frames = list(grouped_by_time)
    num_frames = len(time_frames)
    sensor_origin_px = (IMG_WIDTH // 2, IMG_HEIGHT - 50)

    # 5. Setup for interactive loop
    frame_idx = 0
    paused = True
    cv2.namedWindow("RADAR Visualization", cv2.WINDOW_NORMAL)

    def on_trackbar(val):
        pass

    cv2.createTrackbar("Timeline", "RADAR Visualization", 0, num_frames - 1, on_trackbar)

    # 6. Main interactive animation loop
    while True:
        # If paused, user controls the frame via the trackbar. Otherwise, code controls it.
        if paused:
            frame_idx = cv2.getTrackbarPos("Timeline", "RADAR Visualization")
        else:
            cv2.setTrackbarPos("Timeline", "RADAR Visualization", frame_idx)

        # Get data for the current frame
        timestamp, frame_data = time_frames[frame_idx]

        # Create a fresh background for each frame
        vis_image = create_background_image(IMG_WIDTH, IMG_HEIGHT, SCALE, max_range)

        # Iterate through detections for the current timestamp and draw them
        for _, row in frame_data.iterrows():
            angle_deg, distance_m = row["Angle"], row["Distance"]
            if distance_m > max_range:
                continue
            angle_rad = np.radians(angle_deg - 90)
            x_world = distance_m * np.cos(angle_rad)
            y_world = distance_m * np.sin(angle_rad)
            point_px = (
                int(sensor_origin_px[0] - y_world * SCALE),
                int(sensor_origin_px[1] - x_world * SCALE),
            )
            cv2.circle(vis_image, point_px, POINT_SIZE, POINT_COLOR, -1)

        # Add timestamp and info text to the image
        state_text = "(PAUSED)" if paused else "(PLAYING)"
        info_text = f"Time: {timestamp.strftime('%H:%M:%S')} | Frame: {frame_idx}/{num_frames-1} {state_text}"
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

        # Display the frame
        cv2.imshow("RADAR Visualization", vis_image)

        # Wait for key press, delay depends on paused state for UI responsiveness
        delay = wait_time_ms if not paused else 30
        key = cv2.waitKey(delay)

        if key == 27:  # ESC key
            break
        elif key == 32:  # Spacebar to toggle play/pause
            paused = not paused
        elif key == ord("d"):  # 'd' to step forward
            frame_idx = min(frame_idx + 1, num_frames - 1)
            paused = True
        elif key == ord("a"):  # 'a' to step backward
            frame_idx = max(frame_idx - 1, 0)
            paused = True

        # If playing, advance frame index and loop at the end
        if not paused:
            frame_idx = (frame_idx + 1) % num_frames

    cv2.destroyAllWindows()
    logger.info("Exiting.")


if __name__ == "__main__":
    """
    Example usage of RADAR visualisation tool.

    Edit the path below to match your radar data file:
    """
    # Define input path to radar CSV file
    radar_file_path = "path/to/radar_data.csv"

    # Run the main function
    main(radar_file_path)
