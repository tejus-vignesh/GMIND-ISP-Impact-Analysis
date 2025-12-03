"""
Create a sparse 6x6 grid visualization of 3D projections to ground plane.

Shows XYZ positions for grid points throughout the FOV.
"""

import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

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
from Annotation.footpoint_to_ground import bbox_to_3d_geometric_robust as bbox_to_3d_geometric


def create_sparse_grid_visualization():
    """Create a 6x6 grid visualization of 3D projections."""

    # Camera parameters
    camera_matrix = np.array(
        [[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32
    )
    camera_height = 4.0  # meters
    camera_pitch_deg = 20.0  # degrees
    ground_height = 0.0

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    img_width = 4112  # FLIR 8.9MP width
    img_height = 3008  # FLIR 8.9MP height

    # Create 6x6 grid (sparse)
    grid_size = 6
    x_pixels = np.linspace(0.1 * img_width, 0.9 * img_width, grid_size)
    y_pixels = np.linspace(0.1 * img_height, 0.9 * img_height, grid_size)

    # Store results
    grid_points = []
    grid_xyz = []

    logger.info("=" * 70)
    logger.info("SPARSE 6x6 GRID PROJECTION TO GROUND PLANE")
    logger.info("=" * 70)
    logger.info("Camera: FLIR 8.9MP")
    logger.info("Image size: %d x %d", img_width, img_height)
    logger.info("Optical center: (%.1f, %.1f)", cx, cy)
    logger.info("Camera height: %.1fm, pitch: %.1f degrees", camera_height, camera_pitch_deg)
    logger.info("Pixel X      Pixel Y     X (m)         Y (m)         Z (m)       Status")
    logger.info("-" * 70)

    for i, py in enumerate(y_pixels):
        for j, px in enumerate(x_pixels):
            # Create bbox centered at this pixel
            bbox = np.array([px - 10, py - 20, px + 10, py], dtype=np.float32)

            # Project to 3D
            result = bbox_to_3d_geometric(
                bbox, camera_matrix, camera_height, camera_pitch_deg, ground_height
            )

            if result is not None:
                x, y, z = result[0], result[1], result[2]
                grid_points.append((px, py))
                grid_xyz.append((x, y, z))

                # Determine position relative to optical axis
                status = ""
                if abs(px - cx) < 50 and abs(py - cy) < 50:
                    status = "AXIS"
                elif py < cy:
                    status = "ABOVE" if y > 10.99 else "ERROR"
                else:
                    status = "BELOW" if y < 10.99 else "ERROR"
                if px < cx:
                    status += " LEFT" if x < 0 else " ERROR"
                else:
                    status += " RIGHT" if x > 0 else " ERROR"

                logger.debug("%10.1f %10.1f %12.2f %12.2f %10.2f %-10s", px, py, x, y, z, status)
            else:
                logger.debug(
                    "%10.1f %10.1f %12s %12s %10s", px, py, "INVALID", "INVALID", "INVALID"
                )

    # Create visualization image
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw a simple representation of the image space
    # We'll show the projected points on a virtual camera image
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert Y axis (image coordinates)
    ax.set_aspect("equal")
    ax.set_xlabel("Pixel X", fontsize=12)
    ax.set_ylabel("Pixel Y", fontsize=12)
    ax.set_title(
        "3D Projection Grid - Sparse 6x6\nXYZ positions shown for each grid point", fontsize=14
    )

    # Draw grid lines
    for px in x_pixels:
        ax.axvline(px, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
    for py in y_pixels:
        ax.axhline(py, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

    # Draw optical axis
    ax.plot(cx, cy, "r+", markersize=15, markeredgewidth=2, label="Optical Axis")
    ax.axvline(cx, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax.axhline(cy, color="red", linestyle=":", alpha=0.5, linewidth=1)

    # Draw each grid point with XYZ label
    for (px, py), (x, y, z) in zip(grid_points, grid_xyz):
        # Color based on position
        if abs(px - cx) < 50 and abs(py - cy) < 50:
            color = "red"
        elif py < cy:  # Above optical axis
            color = "blue"  # Should be further (Y > 10.99)
        else:  # Below optical axis
            color = "green"  # Should be closer (Y < 10.99)

        # Draw point
        ax.plot(px, py, "o", color=color, markersize=8)

        # Add XYZ label
        label = f"X:{x:.1f}\nY:{y:.1f}\nZ:{z:.1f}"
        ax.text(
            px,
            py - 80,
            label,
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color, linewidth=1
            ),
        )

    # Add legend
    ax.legend(loc="upper right")

    # Add text annotation about expected behavior
    info_text = (
        "Expected behavior:\n"
        "• Optical axis (red +): Y = 10.99m (anchor)\n"
        "• Above axis (blue): Y > 10.99m (further)\n"
        "• Below axis (green): Y < 10.99m (closer)\n"
        "• Left of axis: X < 0\n"
        "• Right of axis: X > 0"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save image
    output_path = Path(__file__).parent / "3d_projection_grid_sparse.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Visualization saved to: %s", output_path)

    plt.show()


if __name__ == "__main__":
    create_sparse_grid_visualization()
