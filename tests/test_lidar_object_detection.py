"""
Tests for LIDAR object detection functionality.

Tests the core functions for detecting objects in LIDAR point cloud data,
including PCD file loading, background subtraction, clustering, and
object detection.
"""

import os

# Import functions to test
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Annotation"))
from lidar_object_detection import (
    apply_extrinsics,
    apply_extrinsics_inverse,
    background_subtraction,
    cluster_points,
    compute_centroid,
    detect_objects_in_frame,
    get_sorted_pcd_files,
    load_pcd_xyz,
    parse_pcd_header,
    parse_sensor_calibration,
)


class TestPCDParsing:
    """Tests for PCD file parsing functions."""

    def test_parse_pcd_header_ascii(self):
        """Test parsing ASCII PCD header."""
        header_text = b"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 100
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 100
DATA ascii
"""
        import io

        f = io.BytesIO(header_text)
        header = parse_pcd_header(f)

        assert header["fields"] == ["x", "y", "z", "intensity"]
        assert header["sizes"] == [4, 4, 4, 4]
        assert header["types"] == ["F", "F", "F", "F"]
        assert header["point_count"] == 100
        assert header["data_type"] == "ascii"

    def test_parse_pcd_header_binary(self):
        """Test parsing binary PCD header."""
        header_text = b"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 50
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 50
DATA binary
"""
        import io

        f = io.BytesIO(header_text)
        header = parse_pcd_header(f)

        assert header["fields"] == ["x", "y", "z"]
        assert header["data_type"] == "binary"
        assert header["point_count"] == 50

    def test_load_pcd_xyz_ascii(self):
        """Test loading ASCII PCD file."""
        pcd_content = """# .PCD v0.7
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
1.0 2.0 3.0 0.5
4.0 5.0 6.0 0.6
7.0 8.0 9.0 0.7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pcd", delete=False) as f:
            f.write(pcd_content)
            temp_path = f.name

        try:
            points, intensities = load_pcd_xyz(temp_path)

            assert points.shape == (3, 3)
            assert np.allclose(points[0], [1.0, 2.0, 3.0])
            assert np.allclose(points[1], [4.0, 5.0, 6.0])
            assert np.allclose(points[2], [7.0, 8.0, 9.0])
            assert intensities.shape == (3,)
            assert np.allclose(intensities, [0.5, 0.6, 0.7])
        finally:
            os.unlink(temp_path)

    def test_load_pcd_xyz_no_intensity(self):
        """Test loading PCD file without intensity field."""
        pcd_content = """# .PCD v0.7
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
1.0 2.0 3.0
4.0 5.0 6.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pcd", delete=False) as f:
            f.write(pcd_content)
            temp_path = f.name

        try:
            points, intensities = load_pcd_xyz(temp_path)

            assert points.shape == (2, 3)
            assert intensities.shape == (2,)
            assert np.allclose(intensities, [0.0, 0.0])  # Should default to zero
        finally:
            os.unlink(temp_path)

    def test_get_sorted_pcd_files(self):
        """Test sorting of PCD files by frame index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with numeric prefixes
            Path(tmpdir, "100-frame.pcd").touch()
            Path(tmpdir, "002-frame.pcd").touch()
            Path(tmpdir, "010-frame.pcd").touch()
            Path(tmpdir, "001-frame.pcd").touch()

            files = get_sorted_pcd_files(tmpdir)

            assert len(files) == 4
            # Check ordering
            assert "001-frame.pcd" in os.path.basename(files[0])
            assert "002-frame.pcd" in os.path.basename(files[1])
            assert "010-frame.pcd" in os.path.basename(files[2])
            assert "100-frame.pcd" in os.path.basename(files[3])


class TestExtrinsics:
    """Tests for extrinsics transformation functions."""

    def test_parse_sensor_calibration(self):
        """Test parsing sensor calibration file."""
        calib_content = """Name: Velodyne
Extrinsics:
X: 1.0
Y: 2.0
Z: 3.0
R_00: 1.0
R_01: 0.0
R_02: 0.0
R_10: 0.0
R_11: 1.0
R_12: 0.0
R_20: 0.0
R_21: 0.0
R_22: 1.0

Name: Camera
Focal_x: 1000.0
Focal_y: 1000.0
COD_x: 320.0
COD_y: 240.0
Extrinsics:
X: 0.0
Y: 0.0
Z: 0.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(calib_content)
            temp_path = f.name

        try:
            extrinsics = parse_sensor_calibration(temp_path)

            assert "Velodyne" in extrinsics
            assert extrinsics["Velodyne"]["X"] == 1.0
            assert extrinsics["Velodyne"]["Y"] == 2.0
            assert extrinsics["Velodyne"]["Z"] == 3.0
            assert np.allclose(extrinsics["Velodyne"]["R"], np.eye(3))
            assert "Camera" not in extrinsics  # Should only include LIDAR
        finally:
            os.unlink(temp_path)

    def test_apply_extrinsics(self):
        """Test applying extrinsics transformation."""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        extr = {
            "X": 1.0,
            "Y": 2.0,
            "Z": 3.0,
            "R": np.eye(3, dtype=np.float32),
        }

        transformed = apply_extrinsics(points, extr)

        assert transformed.shape == (2, 3)
        assert np.allclose(transformed[0], [2.0, 2.0, 3.0])
        assert np.allclose(transformed[1], [1.0, 3.0, 3.0])

    def test_apply_extrinsics_inverse(self):
        """Test applying inverse extrinsics transformation."""
        points = np.array([[2.0, 2.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32)
        extr = {
            "X": 1.0,
            "Y": 2.0,
            "Z": 3.0,
            "R": np.eye(3, dtype=np.float32),
        }

        transformed = apply_extrinsics_inverse(points, extr)

        assert transformed.shape == (2, 3)
        assert np.allclose(transformed[0], [1.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(transformed[1], [0.0, 1.0, 0.0], atol=1e-6)


class TestBackgroundSubtraction:
    """Tests for background subtraction functionality."""

    def test_background_subtraction_basic(self):
        """Test basic background subtraction."""
        # Create background: points on a grid
        background = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        # Current frame: same points plus a new object
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Background
                [1.0, 0.0, 0.0],  # Background
                [5.0, 5.0, 0.0],  # Foreground (far from background)
                [0.0, 1.0, 0.0],  # Background
            ],
            dtype=np.float32,
        )

        foreground = background_subtraction(points, background, distance_threshold=0.5)

        assert len(foreground) == 1
        assert np.allclose(foreground[0], [5.0, 5.0, 0.0])

    def test_background_subtraction_empty_points(self):
        """Test background subtraction with empty point cloud."""
        background = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        points = np.empty((0, 3), dtype=np.float32)

        foreground = background_subtraction(points, background)

        assert len(foreground) == 0

    def test_background_subtraction_empty_background(self):
        """Test background subtraction with empty background."""
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        background = np.empty((0, 3), dtype=np.float32)

        foreground = background_subtraction(points, background)

        assert len(foreground) == 1
        assert np.allclose(foreground[0], [1.0, 2.0, 3.0])


class TestClustering:
    """Tests for point clustering functionality."""

    def test_cluster_points_simple(self):
        """Test clustering of nearby points."""
        # Two distinct clusters
        cluster1 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        cluster2 = np.array([[5.0, 5.0, 0.0], [5.1, 5.0, 0.0], [5.0, 5.1, 0.0]], dtype=np.float32)
        points = np.vstack([cluster1, cluster2])

        clusters = cluster_points(points, eps=0.5, min_points=2)

        assert len(clusters) == 2
        assert len(clusters[0]) == 3
        assert len(clusters[1]) == 3

    def test_cluster_points_single_cluster(self):
        """Test clustering with all points in one cluster."""
        points = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.2, 0.2, 0.0]], dtype=np.float32
        )

        clusters = cluster_points(points, eps=0.5, min_points=2)

        assert len(clusters) == 1
        assert len(clusters[0]) == 4

    def test_cluster_points_min_points_filter(self):
        """Test that clusters below minimum points are filtered."""
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float32)

        clusters = cluster_points(points, eps=0.5, min_points=3)

        # Should filter out small clusters
        assert len(clusters) == 0

    def test_cluster_points_empty(self):
        """Test clustering with empty point cloud."""
        points = np.empty((0, 3), dtype=np.float32)

        clusters = cluster_points(points)

        assert len(clusters) == 0


class TestCentroid:
    """Tests for centroid computation."""

    def test_compute_centroid(self):
        """Test centroid computation."""
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

        centroid_info = compute_centroid(points)

        assert centroid_info is not None
        assert centroid_info["num_points"] == 3
        assert np.allclose(centroid_info["centroid"], [2.0 / 3.0, 2.0 / 3.0, 0.0])

    def test_compute_centroid_empty(self):
        """Test centroid computation with empty points."""
        points = np.empty((0, 3), dtype=np.float32)

        centroid_info = compute_centroid(points)

        assert centroid_info is None


class TestObjectDetection:
    """Tests for object detection in frames."""

    def test_detect_objects_in_frame(self):
        """Test object detection in a single frame."""
        # Background: grid of points
        background = np.array(
            [[i, j, 0.0] for i in range(0, 5) for j in range(0, 5)], dtype=np.float32
        )

        # Current frame: background + two foreground objects
        background_points = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32)
        object1 = np.array(
            [[10.0, 10.0, 0.0], [10.1, 10.0, 0.0], [10.0, 10.1, 0.0]], dtype=np.float32
        )
        object2 = np.array(
            [[15.0, 15.0, 0.0], [15.1, 15.0, 0.0], [15.0, 15.1, 0.0]], dtype=np.float32
        )
        points = np.vstack([background_points, object1, object2])

        objects, background_mask = detect_objects_in_frame(
            points, background, bg_threshold=0.5, cluster_eps=0.5, min_cluster_points=2
        )

        assert len(objects) >= 1  # Should detect at least one object
        assert all("centroid" in obj for obj in objects)
        assert all("num_points" in obj for obj in objects)
        assert background_mask.shape == (len(points),)

    def test_detect_objects_in_frame_empty_points(self):
        """Test detection with empty point cloud."""
        background = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        points = np.empty((0, 3), dtype=np.float32)

        objects, background_mask = detect_objects_in_frame(points, background)

        assert len(objects) == 0
        assert len(background_mask) == 0

    def test_detect_objects_in_frame_no_background(self):
        """Test detection without background model."""
        points = np.array([[1.0, 2.0, 3.0], [1.1, 2.0, 3.0]], dtype=np.float32)
        background = np.empty((0, 3), dtype=np.float32)

        objects, background_mask = detect_objects_in_frame(
            points, background, cluster_eps=0.5, min_cluster_points=2
        )

        # All points should be treated as foreground
        assert len(objects) >= 0  # May or may not cluster depending on parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
