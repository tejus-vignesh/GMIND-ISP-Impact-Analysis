"""
Unit test for geometric 3D projection from 2D pixel grid to ground plane.

Projects a grid of 2D pixel points throughout the image to 3D using ground plane intersection
and visualizes the results as a bird's-eye view image.
"""

import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Annotation.annotation_generation import parse_camera_intrinsics_from_calibration
from Annotation.footpoint_to_ground import bbox_to_3d_geometric_robust


def pixel_to_3d_geometric(
    pixel_x: float,
    pixel_y: float,
    camera_matrix: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
    ground_height: float = 0.0,
) -> np.ndarray | None:
    """
    Project a single pixel coordinate to 3D using geometric ground plane intersection.

    This is similar to bbox_to_3d_geometric but for a single pixel point.
    We use the pixel_y as the bottom of a hypothetical bbox (where object touches ground).

    Args:
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate (bottom edge, where object touches ground)
        camera_matrix: Camera intrinsics (3x3) with fx, fy, cx, cy
        camera_height: Camera height above ground in meters
        camera_pitch_deg: Camera pitch angle in degrees (positive = downward)
        ground_height: Ground plane height in world coordinates (meters), default 0.0

    Returns:
        3D point [X, Y, Z] in camera coordinates (meters), or None if invalid
    """
    # Create a small bbox centered at this pixel (with bottom at pixel_y)
    # Use a small width for the bbox
    bbox_width = 10.0  # pixels
    bbox_height = 20.0  # pixels

    x1 = pixel_x - bbox_width / 2
    x2 = pixel_x + bbox_width / 2
    y1 = pixel_y - bbox_height
    y2 = pixel_y  # Bottom edge

    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

    return bbox_to_3d_geometric_robust(
        bbox=bbox,
        camera_matrix=camera_matrix,
        camera_height=camera_height,
        camera_pitch_deg=camera_pitch_deg,
        ground_height=ground_height,
        dist_coeffs=None,  # Optional: can add distortion coefficients here
    )


class TestGeometric3DProjectionGrid:
    """Test suite for geometric 3D projection from pixel grid."""

    @pytest.fixture(scope="class")
    def camera_matrix(self):
        """Load FLIR 8.9MP camera intrinsics from calibration file."""
        calib_path = Path(__file__).parent.parent / "sensor_calibration.txt"
        if not calib_path.exists():
            pytest.skip(f"Calibration file not found: {calib_path}")

        camera_matrix = parse_camera_intrinsics_from_calibration(
            str(calib_path), camera_name="FLIR 8.9MP"
        )

        if camera_matrix is None:
            pytest.skip("Could not load FLIR 8.9MP camera intrinsics")

        logger.info("Loaded camera matrix:")
        logger.info("\n%s", camera_matrix)

        return camera_matrix

    def test_project_pixel_grid_to_3d(self, camera_matrix, tmp_path):
        """
        Project a vertical line of 2D pixels to 3D and visualize the results.

        Creates a line of pixel points down the middle of a real image and projects
        them to 3D using geometric ground plane intersection. Displays pixel coordinates
        and their corresponding 3D positions overlaid on the actual image.
        """
        # Load the actual image
        image_path = Path(__file__).parent.parent / "Annotation" / "FLIR8.9-Urban1.jpg"
        if not image_path.exists():
            pytest.skip(f"Image file not found: {image_path}")

        # Load image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            pytest.skip(f"Could not load image: {image_path}")

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_height, image_width = img_rgb.shape[:2]

        logger.info("Loaded image: %s", image_path)
        logger.info("Image dimensions: %dx%d pixels", image_width, image_height)

        # Camera parameters
        camera_height = 4.0  # meters
        camera_pitch_deg = 20.0  # degrees
        ground_height = 0.0  # meters

        # Create a grid with 3 columns and multiple rows
        center_x = image_width / 2.0
        grid_width = image_width * 0.6  # Use 60% of image width for the grid
        left_x = center_x - grid_width / 2
        right_x = center_x + grid_width / 2
        x_coords = np.array([left_x, center_x, right_x])  # 3 columns

        # Create points from top to bottom with regular spacing
        # Use fewer points so labels don't overlap
        num_rows = 15
        y_coords = np.linspace(100, image_height - 100, num_rows)

        logger.info(
            "Creating grid of %d columns x %d rows = %d points",
            len(x_coords),
            num_rows,
            len(x_coords) * num_rows,
        )
        logger.info(
            "Pixel X: %.1f, %.1f, %.1f (left, center, right)",
            x_coords[0],
            x_coords[1],
            x_coords[2],
        )
        logger.info("Pixel Y range: %.1f - %.1f px", y_coords[0], y_coords[-1])
        logger.info(
            "Pixel X      Pixel Y     3D X (m)     3D Y (m)     3D Z (m)     Distance (m) Status"
        )
        logger.info("-" * 85)

        # Project each pixel to 3D and collect results
        pixel_data = []
        points_3d = []
        valid_pixel_points = []
        pixel_colors = []
        total_pixels = len(x_coords) * len(y_coords)

        for px in x_coords:
            for py in y_coords:
                # Project to 3D
                point_3d = pixel_to_3d_geometric(
                    pixel_x=px,
                    pixel_y=py,
                    camera_matrix=camera_matrix,
                    camera_height=camera_height,
                    camera_pitch_deg=camera_pitch_deg,
                    ground_height=ground_height,
                )

                if point_3d is not None:
                    distance = np.linalg.norm(point_3d)
                    pixel_data.append(
                        (px, py, point_3d[0], point_3d[1], point_3d[2], distance, "Ground")
                    )
                    points_3d.append(point_3d)
                    valid_pixel_points.append((px, py))

                    # Color based on distance from origin (since Z=0 for all points, use distance)
                    # Note: In world coordinates, all points have Z=0, so we color by distance from origin
                    distance = np.linalg.norm(point_3d)
                    distance_normalized = np.clip(distance / 200.0, 0.0, 1.0)  # Scale to 200m max
                    pixel_colors.append((1.0 - distance_normalized, 0.0, distance_normalized))

                    logger.debug(
                        "%12.1f %12.1f %12.2f %12.2f %12.2f %12.2f %-12s",
                        px,
                        py,
                        point_3d[0],
                        point_3d[1],
                        point_3d[2],
                        distance,
                        "Ground",
                    )
                else:
                    pixel_data.append((px, py, None, None, None, None, "Sky"))
                    logger.debug(
                        "%12.1f %12.1f %12s %12s %12s %12s %-12s",
                        px,
                        py,
                        "---",
                        "---",
                        "---",
                        "---",
                        "Sky",
                    )

        logger.info("Successfully projected %d / %d points to 3D", len(points_3d), total_pixels)

        if len(points_3d) == 0:
            pytest.fail("No points were successfully projected to 3D!")

        points_3d = np.array(points_3d)

        # Log statistics
        logger.info("3D Projection Statistics:")
        logger.info("X range: %.2f to %.2f m", points_3d[:, 0].min(), points_3d[:, 0].max())
        logger.info("Y range: %.2f to %.2f m", points_3d[:, 1].min(), points_3d[:, 1].max())
        logger.info("Z range: %.2f to %.2f m", points_3d[:, 2].min(), points_3d[:, 2].max())
        logger.info("Mean depth: %.2f m", points_3d[:, 2].mean())

        # Create visualization: Real image with 3D positions overlaid
        fig, ax_img = plt.subplots(1, 1, figsize=(24, 14))

        # Display the actual image as background (scaled down for better visibility)
        # Use a scale factor to fit the image on screen while maintaining aspect ratio
        max_display_width = 2000  # Maximum display width in pixels
        scale_factor = min(1.0, max_display_width / image_width)
        vis_width = int(image_width * scale_factor)
        vis_height = int(image_height * scale_factor)

        # Resize the image for display
        img_display = cv2.resize(img_rgb, (vis_width, vis_height), interpolation=cv2.INTER_AREA)

        # Display the actual image
        ax_img.imshow(img_display, origin="upper", aspect="auto")
        ax_img.set_xlabel("Pixel X", fontsize=14)
        ax_img.set_ylabel("Pixel Y", fontsize=14)

        # Scale pixel coordinates for visualization (define before use)
        def scale_px(x):
            return x * scale_factor

        def scale_py(y):
            return y * scale_factor

        # Calculate and display theoretical horizon line
        fy = camera_matrix[1, 1]
        cy = camera_matrix[1, 2]
        pitch_rad = np.radians(camera_pitch_deg)
        horizon_y_normalized = -np.tan(pitch_rad)
        horizon_pixel_y = cy + horizon_y_normalized * fy

        # Draw horizon line if it's within image bounds
        if 0 < horizon_pixel_y < image_height:
            horizon_y_scaled = scale_py(horizon_pixel_y)
            ax_img.axhline(
                y=horizon_y_scaled,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Horizon (pitch {camera_pitch_deg}°)",
                zorder=3,
            )
            ax_img.text(
                vis_width * 0.5,
                horizon_y_scaled - 20,
                f"Horizon at Y={horizon_pixel_y:.0f}px (theoretical)",
                fontsize=12,
                fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round", facecolor="orange", alpha=0.8, edgecolor="black"),
            )

        ax_img.set_title(
            f"FLIR 8.9-Urban1.jpg ({image_width}x{image_height}px) - 3D Projections\nCamera Height: {camera_height}m, Pitch: {camera_pitch_deg}°",
            fontsize=16,
            fontweight="bold",
        )

        # Scale pixel coordinates for visualization (define here so it's available for horizon line)
        def scale_px(x):
            return x * scale_factor

        def scale_py(y):
            return y * scale_factor

        # Label each point with just XYZ (larger, clearer)
        for idx, (pixel_pt, point_3d) in enumerate(zip(valid_pixel_points, points_3d)):
            px, py = pixel_pt
            x, y, z = point_3d[0], point_3d[1], point_3d[2]

            # Scale coordinates for visualization
            px_vis = scale_px(px)
            py_vis = scale_py(py)

            # Place text to the right of each point (adjust offset based on scale)
            text_offset = max(50, int(200 * scale_factor))
            text_x = px_vis + text_offset
            text_y = py_vis

            # Create label text - just XYZ, very clear
            label = f"X: {x:7.2f}m Y: {y:7.2f}m Z: {z:7.2f}m"

            # Draw line from point to label
            ax_img.plot(
                [px_vis, text_x - 30], [py_vis, text_y], "k-", linewidth=2, alpha=0.7, zorder=5
            )

            # Add text box with very large font
            ax_img.text(
                text_x,
                text_y,
                label,
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="yellow",
                    alpha=0.95,
                    edgecolor="black",
                    linewidth=2,
                ),
                verticalalignment="center",
                horizontalalignment="left",
                family="monospace",
            )

            # Mark the point (larger, more visible) - scale marker size with image scale
            marker_size = max(50, int(150 * scale_factor))
            ax_img.scatter(
                [px_vis],
                [py_vis],
                c="red",
                s=marker_size,
                marker="o",
                edgecolors="black",
                linewidths=3,
                zorder=10,
            )

        # Mark points that didn't project (looking at sky) - with explanation
        text_offset = max(50, int(200 * scale_factor))
        for px in x_coords:
            for py in y_coords:
                if (px, py) not in valid_pixel_points:
                    px_vis = scale_px(px)
                    py_vis = scale_py(py)
                    px_int, py_int = int(px_vis), int(py_vis)
                    if 0 <= px_int < vis_width and 0 <= py_int < vis_height:
                        ax_img.scatter(
                            [px_vis],
                            [py_vis],
                            c="blue",
                            s=100 * scale_factor,
                            marker="x",
                            linewidths=4,
                            zorder=10,
                        )

        # Add explanation about why some points don't hit ground
        explanation = f"Red circles = Ground hits (with XYZ labels), Blue X = Sky (above horizon)\n"
        explanation += f"Horizon boundary: {len(points_3d)}/{total_pixels} pixels hit ground plane (Camera: {camera_height}m height, {camera_pitch_deg}° pitch)"
        ax_img.text(
            0.02,
            0.98,
            explanation,
            transform=ax_img.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="black"),
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Save the visualization to both tmp_path and a permanent location
        output_path_tmp = tmp_path / "geometric_3d_projection_grid.png"

        # Also save to tests directory for easy access
        tests_dir = Path(__file__).parent
        output_path_permanent = tests_dir / "geometric_3d_projection_grid.png"

        plt.savefig(str(output_path_tmp), dpi=150, bbox_inches="tight")
        plt.savefig(str(output_path_permanent), dpi=150, bbox_inches="tight")

        logger.info("Saved visualization to:")
        logger.info("Temporary: %s", output_path_tmp)
        logger.info("Permanent: %s", output_path_permanent)

        plt.close()

        # Verify the output files exist
        assert output_path_tmp.exists(), "Temporary visualization file was not created!"
        assert output_path_tmp.stat().st_size > 0, "Temporary visualization file is empty!"
        assert output_path_permanent.exists(), "Permanent visualization file was not created!"
        assert output_path_permanent.stat().st_size > 0, "Permanent visualization file is empty!"

        # Additional assertion: Verify that points are in world coordinates
        # All points should have Z = 0 (on ground plane) since origin is at ground level
        assert np.allclose(
            points_3d[:, 2], 0.0, atol=0.1
        ), f"Some points don't have Z=0! Z values: {points_3d[:, 2]}"

        # Verify that points are at reasonable distance from origin (camera position on ground)
        # Note: Some points very close to camera (near principal point) may be very close to origin
        distances = np.linalg.norm(points_3d, axis=1)
        assert np.all(distances > 0.01), "Some points are too close to origin (<1cm)!"
        assert np.all(distances < 500), "Some points are unreasonably far (>500m)!"

        logger.info("All assertions passed!")
        logger.info("Visualization saved to: %s", output_path_permanent)
