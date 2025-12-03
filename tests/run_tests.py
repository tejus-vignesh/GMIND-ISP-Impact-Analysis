#!/usr/bin/env python3
"""
Simple test runner to verify geometric 3D projection works after removing depth_anything.

Run this script to verify all tests pass:
    python tests/run_tests.py
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from Annotation.footpoint_to_ground import bbox_to_3d_geometric_robust as bbox_to_3d_geometric


def test_basic_projection():
    """Test basic 3D projection."""
    logger.info("Test 1: Basic 3D projection...")

    camera_matrix = np.array(
        [[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32
    )

    bbox = np.array([2000, 1000, 2100, 1100], dtype=np.float32)

    result = bbox_to_3d_geometric(
        bbox=bbox,
        camera_matrix=camera_matrix,
        camera_height=4.0,
        camera_pitch_deg=20.0,
        ground_height=0.0,
    )

    assert result is not None, "Result should not be None"
    assert len(result) == 3, "Result should be 3D point"
    assert abs(result[2]) < 0.1, "Z should be approximately 0 (on ground plane)"

    logger.info("Test 1: PASSED")


def test_optical_axis():
    """Test projection at optical axis."""
    logger.info("Test 2: Optical axis projection...")

    camera_matrix = np.array(
        [[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32
    )

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    bbox = np.array([cx - 50, cy - 100, cx + 50, cy], dtype=np.float32)

    result = bbox_to_3d_geometric(
        bbox=bbox,
        camera_matrix=camera_matrix,
        camera_height=4.0,
        camera_pitch_deg=20.0,
        ground_height=0.0,
    )

    assert result is not None, "Result should not be None"
    assert result[1] > 0, "Should be forward (Y > 0)"
    assert 5.0 < result[1] < 20.0, f"Forward distance should be reasonable, got {result[1]:.3f}m"

    logger.info("Test 2: PASSED")


def test_consistency():
    """Test that same bbox gives consistent results."""
    logger.info("Test 3: Consistency check...")

    camera_matrix = np.array(
        [[1805.82, 0, 2057.78], [0, 1808.48, 1115.85], [0, 0, 1]], dtype=np.float32
    )

    bbox = np.array([2000, 1000, 2100, 1100], dtype=np.float32)

    results = []
    for _ in range(3):
        result = bbox_to_3d_geometric(
            bbox=bbox,
            camera_matrix=camera_matrix,
            camera_height=4.0,
            camera_pitch_deg=20.0,
            ground_height=0.0,
        )
        assert result is not None
        results.append(result)

    # All results should be identical
    for i in range(1, len(results)):
        assert np.allclose(results[0], results[i], atol=1e-5), "Results should be consistent"

    logger.info("PASSED")


def test_no_depth_anything_import():
    """Test that depth_anything is not imported."""
    logger.info("Test 4: Verify depth_anything is not imported...")

    import sys

    # Check that depth_anything is not in sys.modules
    assert "depth_anything" not in sys.modules, "depth_anything should not be imported"

    # Try importing - should fail
    try:
        import depth_anything

        assert False, "depth_anything should not be importable"
    except (ImportError, ModuleNotFoundError):
        pass  # Expected

    logger.info("PASSED")


def test_annotation_imports():
    """Test that annotation_generation imports work correctly."""
    logger.info("Test 5: Verify annotation generation imports...")

    from Annotation.annotation_generation import Config

    config = Config()

    # Should have geometric depth parameters
    assert hasattr(config, "enable_depth_estimation")
    assert hasattr(config, "camera_height")
    assert hasattr(config, "camera_pitch_deg")

    # Should NOT have depth_anything parameters
    assert not hasattr(config, "depth_model"), "depth_model should have been removed"
    assert not hasattr(
        config, "depth_calibration_point"
    ), "depth_calibration_point should have been removed"

    logger.info("PASSED")


def test_geometric_projection_function_exists():
    """Test that geometric projection function exists and is callable."""
    logger.info("Test 6: Verify geometric projection function exists...")

    from Annotation.footpoint_to_ground import bbox_to_3d_geometric_robust as bbox_to_3d_geometric

    assert callable(bbox_to_3d_geometric), "bbox_to_3d_geometric should be callable"

    logger.info("PASSED")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running tests for geometric 3D projection (after depth_anything removal)")
    logger.info("=" * 60)

    tests = [
        test_no_depth_anything_import,
        test_geometric_projection_function_exists,
        test_annotation_imports,
        test_basic_projection,
        test_optical_axis,
        test_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            logger.error(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"ERROR: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
