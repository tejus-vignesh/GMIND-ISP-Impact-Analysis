"""
Unit tests for Calibration CameraModel class.

Tests the CameraModel utility class for 2D/3D mapping and projection operations.
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
    sys.path.insert(0, str(Path(__file__).parent.parent / "Calibration" / "camera_intrinsics"))
    from camera import CameraModel
except ImportError as e:
    pytest.skip(f"Calibration CameraModel not available: {e}", allow_module_level=True)


class TestCameraModel:
    """Test suite for CameraModel class."""

    @pytest.fixture
    def camera_matrix(self):
        """Sample camera intrinsics matrix."""
        return np.array([[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32)

    @pytest.fixture
    def dist_coeffs(self):
        """Sample distortion coefficients."""
        return np.array([-0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

    @pytest.fixture
    def camera_model(self, camera_matrix, dist_coeffs):
        """Create a CameraModel instance."""
        return CameraModel(camera_matrix, dist_coeffs)

    def test_camera_model_initialization(self, camera_matrix):
        """Test CameraModel initialization."""
        model = CameraModel(camera_matrix)

        assert model is not None
        assert hasattr(model, "camera_matrix")
        assert hasattr(model, "dist_coeffs")
        assert np.array_equal(model.camera_matrix, camera_matrix)

    def test_camera_model_initialization_with_distortion(self, camera_matrix, dist_coeffs):
        """Test CameraModel initialization with distortion coefficients."""
        model = CameraModel(camera_matrix, dist_coeffs)

        assert model is not None
        assert hasattr(model, "dist_coeffs")
        assert model.dist_coeffs is not None

    def test_camera_model_initialization_no_distortion(self, camera_matrix):
        """Test CameraModel initialization without distortion coefficients."""
        model = CameraModel(camera_matrix)

        assert model is not None
        assert hasattr(model, "dist_coeffs")
        # Should default to zeros if not provided
        assert model.dist_coeffs is not None

    def test_project_point_3d_to_2d(self, camera_model):
        """Test projecting 3D points to 2D image coordinates."""
        # Create a 3D point in camera coordinates (forward from camera)
        point_3d = np.array([[0.0, 10.0, 5.0]], dtype=np.float32)  # 10m forward, 5m depth

        image_points = camera_model.project_point(point_3d)

        assert image_points is not None
        assert len(image_points.shape) == 2
        assert image_points.shape[1] == 2  # Should have (x, y) coordinates
        assert image_points.shape[0] == 1  # One point

        # Projected point should be within image bounds (assuming reasonable camera)
        x, y = image_points[0]
        assert np.isfinite(x) and np.isfinite(y)

    def test_project_multiple_points(self, camera_model):
        """Test projecting multiple 3D points."""
        points_3d = np.array(
            [
                [0.0, 10.0, 5.0],
                [5.0, 10.0, 5.0],
                [-5.0, 10.0, 5.0],
            ],
            dtype=np.float32,
        )

        image_points = camera_model.project_point(points_3d)

        assert image_points is not None
        assert image_points.shape[0] == 3  # Three points
        assert image_points.shape[1] == 2  # Each with (x, y)

    def test_unproject_pixel(self, camera_model):
        """Test unprojecting a 2D pixel with depth to 3D."""
        # Use principal point (optical center)
        fx = camera_model.camera_matrix[0, 0]
        fy = camera_model.camera_matrix[1, 1]
        cx = camera_model.camera_matrix[0, 2]
        cy = camera_model.camera_matrix[1, 2]

        pixel = np.array([cx, cy], dtype=np.float32)  # Optical center
        depth = 10.0  # 10 meters depth

        point_3d = camera_model.unproject_pixel(pixel, depth)

        assert point_3d is not None
        assert len(point_3d) == 3  # Should be 3D point
        assert np.isclose(point_3d[2], depth, rtol=0.01), "Z should be close to depth"

    def test_pixel_to_ray(self, camera_model):
        """Test converting pixel to normalized ray direction."""
        fx = camera_model.camera_matrix[0, 0]
        cx = camera_model.camera_matrix[0, 2]
        cy = camera_model.camera_matrix[1, 2]

        pixel = np.array([cx, cy], dtype=np.float32)  # Optical center

        ray = camera_model.pixel_to_ray(pixel)

        assert ray is not None
        assert len(ray) == 3  # Should be 3D direction vector

        # Ray should be normalized (unit vector)
        norm = np.linalg.norm(ray)
        assert np.isclose(norm, 1.0, rtol=0.01), f"Ray should be normalized, got norm={norm}"

    def test_get_set_intrinsics(self, camera_model, camera_matrix, dist_coeffs):
        """Test getting and setting intrinsics."""
        # Get intrinsics
        cam_matrix, dist = camera_model.get_intrinsics()

        assert cam_matrix is not None
        assert dist is not None
        assert np.array_equal(cam_matrix, camera_matrix)

        # Set new intrinsics
        new_camera_matrix = camera_matrix * 1.1  # Scale up
        camera_model.set_intrinsics(new_camera_matrix, dist_coeffs)

        cam_matrix_new, dist_new = camera_model.get_intrinsics()
        assert np.array_equal(cam_matrix_new, new_camera_matrix)
        assert np.array_equal(dist_new, dist_coeffs)

    def test_project_unproject_consistency(self, camera_model):
        """Test that project and unproject are consistent."""
        # Start with a 3D point
        point_3d_original = np.array([5.0, 10.0, 15.0], dtype=np.float32)

        # Project to 2D
        image_point = camera_model.project_point(point_3d_original.reshape(1, -1))[0]

        # Extract depth from original point
        depth = point_3d_original[2]

        # Unproject back to 3D
        point_3d_recovered = camera_model.unproject_pixel(image_point, depth)

        # The recovered point should be close to original (allowing for numerical precision)
        # Note: This may not be exact due to distortion, but should be close
        distance = np.linalg.norm(point_3d_original - point_3d_recovered)
        assert distance < 0.5, f"Project-unproject should be consistent, distance={distance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
