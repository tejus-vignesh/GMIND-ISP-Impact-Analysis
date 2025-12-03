"""
Unit tests for Validation module functions.

Tests validation utilities including sensor calibration parsing, ray transformations,
and projection error calculations.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Add calibration camera_intrinsics to path (needed by Validation module)
    sys.path.append(str(Path(__file__).parent.parent / "Calibration" / "camera_intrinsics"))
    # Import from Validation module
    sys.path.insert(0, str(Path(__file__).parent.parent / "Validation"))
    from infinity_projection_error import (
        parse_sensor_calibration,
        transform_ray_to_camera_coords,
    )
except ImportError as e:
    pytest.skip(f"Validation module not available: {e}", allow_module_level=True)


class TestValidationFunctions:
    """Test suite for Validation module functions."""

    @pytest.fixture
    def sample_calibration_content(self):
        """Create sample sensor calibration file content."""
        return """Name: FLIR 8.9MP
Focal_x: 1805.82
Focal_y: 1808.48
COD_x: 2057.78
COD_y: 1115.85
Dist_1: -0.1
Dist_2: 0.05
Dist_3: 0.0
Dist_4: 0.0
Width: 4112
Height: 3008
Extrinsics:
X: 0.0
Y: 0.0
Z: 0.0
R_00: 1.0
R_01: 0.0
R_02: 0.0
R_10: 0.0
R_11: 1.0
R_12: 0.0
R_20: 0.0
R_21: 0.0
R_22: 1.0

Name: FLIR 3.2MP
Focal_x: 1200.0
Focal_y: 1200.0
COD_x: 960.0
COD_y: 540.0
Dist_1: -0.1
Dist_2: 0.05
Dist_3: 0.0
Dist_4: 0.0
Width: 1920
Height: 1080
Extrinsics:
X: 1.0
Y: 0.5
Z: 0.2
R_00: 1.0
R_01: 0.0
R_02: 0.0
R_10: 0.0
R_11: 1.0
R_12: 0.0
R_20: 0.0
R_21: 0.0
R_22: 1.0
"""

    @pytest.fixture
    def temp_calibration_file(self, tmp_path, sample_calibration_content):
        """Create a temporary calibration file."""
        calib_file = tmp_path / "sensor_calibration.txt"
        calib_file.write_text(sample_calibration_content)
        return str(calib_file)

    def test_parse_sensor_calibration_basic(self, temp_calibration_file):
        """Test parsing sensor calibration file."""
        cameras, extrinsics, image_dimensions = parse_sensor_calibration(temp_calibration_file)

        assert cameras is not None
        assert isinstance(cameras, dict)
        assert len(cameras) > 0

        assert extrinsics is not None
        assert isinstance(extrinsics, dict)

        assert image_dimensions is not None
        assert isinstance(image_dimensions, dict)

    def test_parse_sensor_calibration_camera_names(self, temp_calibration_file):
        """Test that camera names are parsed correctly."""
        cameras, extrinsics, _ = parse_sensor_calibration(temp_calibration_file)

        assert "FLIR 8.9MP" in cameras
        assert "FLIR 3.2MP" in cameras

    def test_parse_sensor_calibration_camera_models(self, temp_calibration_file):
        """Test that CameraModel objects are created correctly."""
        cameras, _, _ = parse_sensor_calibration(temp_calibration_file)

        camera = cameras["FLIR 8.9MP"]
        assert camera is not None
        assert hasattr(camera, "camera_matrix")
        assert hasattr(camera, "dist_coeffs")

        # Check camera matrix values
        matrix = camera.camera_matrix
        assert np.isclose(matrix[0, 0], 1805.82)  # Focal_x
        assert np.isclose(matrix[1, 1], 1808.48)  # Focal_y
        assert np.isclose(matrix[0, 2], 2057.78)  # COD_x
        assert np.isclose(matrix[1, 2], 1115.85)  # COD_y

    def test_parse_sensor_calibration_extrinsics(self, temp_calibration_file):
        """Test that extrinsics are parsed correctly."""
        _, extrinsics, _ = parse_sensor_calibration(temp_calibration_file)

        assert "FLIR 8.9MP" in extrinsics
        assert "FLIR 3.2MP" in extrinsics

        ext = extrinsics["FLIR 8.9MP"]
        assert "X" in ext
        assert "Y" in ext
        assert "Z" in ext
        assert "R" in ext
        assert isinstance(ext["R"], np.ndarray)

    def test_parse_sensor_calibration_image_dimensions(self, temp_calibration_file):
        """Test that image dimensions are parsed correctly."""
        _, _, image_dimensions = parse_sensor_calibration(temp_calibration_file)

        assert "FLIR 8.9MP" in image_dimensions
        assert image_dimensions["FLIR 8.9MP"] == (4112, 3008)
        assert image_dimensions["FLIR 3.2MP"] == (1920, 1080)

    def test_transform_ray_to_camera_coords_identity(self):
        """Test ray transformation with identity transform."""
        # Identity transformation (no rotation, no translation)
        cam1_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
        cam2_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }

        ray_cam1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Forward direction

        ray_cam2 = transform_ray_to_camera_coords(ray_cam1, cam1_extr, cam2_extr)

        assert ray_cam2 is not None
        assert np.allclose(ray_cam1, ray_cam2), "Identity transform should preserve ray"

    def test_transform_ray_to_camera_coords_translation(self):
        """Test ray transformation with translation only (ray direction unchanged)."""
        cam1_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
        cam2_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 1.0,  # Translation
            "Y": 0.5,
            "Z": 0.2,
        }

        ray_cam1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Forward direction

        ray_cam2 = transform_ray_to_camera_coords(ray_cam1, cam1_extr, cam2_extr)

        # Direction should be unchanged (only translation, no rotation)
        assert ray_cam2 is not None
        assert np.allclose(ray_cam1, ray_cam2), "Translation should not change ray direction"

    def test_transform_ray_to_camera_coords_rotation(self):
        """Test ray transformation with rotation."""
        # 90 degree rotation around Y axis
        cam1_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
        cam2_extr = {
            "R": np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),  # 90 deg rotation around Y
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }

        ray_cam1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Forward (Z)

        ray_cam2 = transform_ray_to_camera_coords(ray_cam1, cam1_extr, cam2_extr)

        # Forward in cam1 should become left (-X) in cam2
        expected = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert ray_cam2 is not None
        assert np.allclose(ray_cam2, expected, atol=0.01), f"Expected {expected}, got {ray_cam2}"

    def test_transform_ray_to_camera_coords_normalized(self):
        """Test that transformed ray remains normalized."""
        cam1_extr = {
            "R": np.eye(3, dtype=np.float32),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0,
        }
        cam2_extr = {
            "R": np.array(
                [
                    [0.707, 0.0, 0.707],
                    [0.0, 1.0, 0.0],
                    [-0.707, 0.0, 0.707],
                ],
                dtype=np.float32,
            ),  # 45 deg rotation
            "X": 1.0,
            "Y": 0.5,
            "Z": 0.2,
        }

        ray_cam1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        ray_cam2 = transform_ray_to_camera_coords(ray_cam1, cam1_extr, cam2_extr)

        # Ray should be normalized
        norm = np.linalg.norm(ray_cam2)
        assert np.isclose(norm, 1.0, atol=0.01), f"Ray should be normalized, got norm={norm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
