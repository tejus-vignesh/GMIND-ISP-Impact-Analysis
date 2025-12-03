"""
Integration tests for annotation generation with geometric 3D projection.

Tests that annotation_generation.py correctly uses geometric ground plane intersection
after removing depth_anything dependencies.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Annotation.annotation_generation import Config, parse_camera_intrinsics_from_calibration


class TestAnnotationGenerationIntegration:
    """Test integration of geometric 3D projection with annotation generation."""

    def test_config_no_depth_anything_params(self):
        """Test that Config no longer has depth_anything related parameters."""
        config = Config()

        # Should have geometric depth parameters
        assert hasattr(config, "enable_depth_estimation")
        assert hasattr(config, "camera_height")
        assert hasattr(config, "camera_pitch_deg")
        assert hasattr(config, "ground_height")
        assert hasattr(config, "calibration_file")
        assert hasattr(config, "camera_matrix")

        # Should NOT have depth_anything parameters
        assert not hasattr(config, "depth_model"), "depth_model should have been removed"
        assert not hasattr(
            config, "depth_calibration_point"
        ), "depth_calibration_point should have been removed"
        assert not hasattr(
            config, "use_geometric_depth"
        ), "use_geometric_depth should have been removed (always geometric now)"

        # Default should be geometric depth enabled (when enabled_depth_estimation is True)
        assert config.camera_height == 4.0
        assert config.camera_pitch_deg == 20.0
        assert config.ground_height == 0.0

    def test_config_geometric_depth_defaults(self):
        """Test that geometric depth parameters have correct defaults."""
        config = Config()

        # Default values should be reasonable
        assert isinstance(config.camera_height, float)
        assert config.camera_height > 0, "Camera height should be positive"

        assert isinstance(config.camera_pitch_deg, float)
        assert -90 < config.camera_pitch_deg < 90, "Pitch should be reasonable angle"

        assert isinstance(config.ground_height, float)

        # enable_depth_estimation defaults to False
        assert config.enable_depth_estimation == False

    def test_import_footpoint_to_ground(self):
        """Test that footpoint_to_ground can be imported."""
        try:
            from Annotation.footpoint_to_ground import (
                bbox_to_3d_geometric_robust as bbox_to_3d_geometric,
            )

            assert callable(bbox_to_3d_geometric), "bbox_to_3d_geometric_robust should be callable"
        except ImportError as e:
            pytest.fail(f"Failed to import bbox_to_3d_geometric_robust: {e}")

    def test_no_depth_anything_import(self):
        """Test that depth_anything module is not imported anywhere."""
        import importlib
        import sys

        # Check that depth_anything is not in sys.modules
        assert "depth_anything" not in sys.modules, "depth_anything should not be imported"

        # Try importing - should fail
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("depth_anything")

    def test_camera_intrinsics_parsing(self):
        """Test that camera intrinsics parsing still works."""
        calib_path = Path(__file__).parent.parent / "sensor_calibration.txt"

        if not calib_path.exists():
            pytest.skip(f"Calibration file not found: {calib_path}")

        # Test parsing for FLIR 8.9MP camera
        camera_matrix = parse_camera_intrinsics_from_calibration(
            str(calib_path), camera_name="FLIR 8.9MP"
        )

        if camera_matrix is not None:
            assert isinstance(camera_matrix, np.ndarray), "Camera matrix should be numpy array"
            assert camera_matrix.shape == (
                3,
                3,
            ), f"Camera matrix should be 3x3, got {camera_matrix.shape}"

            # Check that it's a valid camera matrix
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            assert fx > 0, "Focal length fx should be positive"
            assert fy > 0, "Focal length fy should be positive"
            assert cx > 0, "Principal point cx should be positive"
            assert cy > 0, "Principal point cy should be positive"

    def test_bbox_to_3d_geometric_import_and_usage(self):
        """Test that bbox_to_3d_geometric_robust can be imported and used."""
        from Annotation.footpoint_to_ground import (
            bbox_to_3d_geometric_robust as bbox_to_3d_geometric,
        )

        # Create a test camera matrix
        camera_matrix = np.array(
            [[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32
        )

        # Create a test bbox
        bbox = np.array([2000, 1000, 2100, 1100], dtype=np.float32)

        # Test that function can be called
        result = bbox_to_3d_geometric(
            bbox=bbox,
            camera_matrix=camera_matrix,
            camera_height=4.0,
            camera_pitch_deg=20.0,
            ground_height=0.0,
        )

        # Result should either be None or a valid 3D point
        if result is not None:
            assert isinstance(result, np.ndarray), "Result should be numpy array"
            assert len(result) == 3, "Result should be 3D point"
            assert result.shape == (3,), f"Result shape should be (3,), got {result.shape}"


class TestAnnotationGenerationNoDepthAnything:
    """Test that annotation generation does not reference depth_anything."""

    def test_no_depth_anything_in_code(self):
        """Test that annotation_generation.py doesn't import depth_anything."""
        annotation_gen_path = (
            Path(__file__).parent.parent / "Annotation" / "annotation_generation.py"
        )

        if not annotation_gen_path.exists():
            pytest.skip(f"annotation_generation.py not found: {annotation_gen_path}")

        # Read the file and check for depth_anything references
        content = annotation_gen_path.read_text()

        # Should not contain depth_anything imports
        assert "from depth_anything" not in content, "Should not import depth_anything"
        assert "import depth_anything" not in content, "Should not import depth_anything"
        assert "DepthEstimator" not in content, "Should not reference DepthEstimator"

        # Should contain robust geometric projection import
        assert "footpoint_to_ground" in content, "Should import footpoint_to_ground"
        assert "bbox_to_3d_geometric_robust" in content, "Should use bbox_to_3d_geometric_robust"

    def test_no_depth_estimator_variable(self):
        """Test that annotation_generation.py doesn't use depth_estimator variable."""
        annotation_gen_path = (
            Path(__file__).parent.parent / "Annotation" / "annotation_generation.py"
        )

        if not annotation_gen_path.exists():
            pytest.skip(f"annotation_generation.py not found: {annotation_gen_path}")

        content = annotation_gen_path.read_text()

        # Should not contain depth_estimator variable
        assert "depth_estimator" not in content, "Should not use depth_estimator variable"
        assert "depth_map" not in content, "Should not use depth_map variable"
        assert "get_depth_map" not in content, "Should not call get_depth_map"

    def test_geometric_projection_is_used(self):
        """Test that geometric projection is being used."""
        annotation_gen_path = (
            Path(__file__).parent.parent / "Annotation" / "annotation_generation.py"
        )

        if not annotation_gen_path.exists():
            pytest.skip(f"annotation_generation.py not found: {annotation_gen_path}")

        content = annotation_gen_path.read_text()

        # Should use robust geometric projection
        assert "bbox_to_3d_geometric_robust" in content, "Should use bbox_to_3d_geometric_robust"
        assert (
            "use_geometric_depth" not in content
            or "use_geometric_depth" in content.split("use_geometric_depth")[0]
        ), "use_geometric_depth check should be removed (always geometric now)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
