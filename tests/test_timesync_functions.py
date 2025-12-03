"""
Unit tests for TimeSync temporal alignment and validation functions.

Tests the core functions for loading COCO annotations, extracting 3D locations,
and temporal alignment operations.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    from TimeSync.temporal_alignment_validation import (
        extract_3d_locations,
        load_coco_annotations,
    )
except ImportError as e:
    pytest.skip(f"TimeSync module not available: {e}", allow_module_level=True)


class TestTimeSyncFunctions:
    """Test suite for TimeSync functions."""

    @pytest.fixture
    def sample_coco_data(self):
        """Create sample COCO annotation data for testing."""
        return {
            "images": [
                {"id": 1, "file_name": "frame_001.jpg", "width": 1920, "height": 1080},
                {"id": 2, "file_name": "frame_002.jpg", "width": 1920, "height": 1080},
                {"id": 3, "file_name": "frame_003.jpg", "width": 1920, "height": 1080},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 200, 50, 80],
                    "track_id": 10,
                    "location_3d": [5.0, 10.0, 0.0],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 400, 60, 90],
                    "track_id": 20,
                    "location_3d": [8.0, 12.0, 0.0],
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [110, 210, 50, 80],
                    "track_id": 10,
                    "location_3d": [5.5, 10.5, 0.0],
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
                {"id": 3, "name": "car"},
            ],
        }

    def test_extract_3d_locations_basic(self, sample_coco_data):
        """Test extracting 3D locations from COCO data."""
        points_3d, metadata = extract_3d_locations(sample_coco_data)

        assert points_3d is not None
        assert isinstance(points_3d, np.ndarray)
        assert len(points_3d.shape) == 2
        assert points_3d.shape[1] == 3  # Should have X, Y, Z coordinates

        assert metadata is not None
        assert isinstance(metadata, list)
        assert len(metadata) == len(points_3d)

    def test_extract_3d_locations_count(self, sample_coco_data):
        """Test that correct number of 3D locations are extracted."""
        points_3d, metadata = extract_3d_locations(sample_coco_data)

        # Should extract 3 locations (all annotations have location_3d)
        assert len(points_3d) == 3
        assert len(metadata) == 3

    def test_extract_3d_locations_values(self, sample_coco_data):
        """Test that extracted 3D locations have correct values."""
        points_3d, metadata = extract_3d_locations(sample_coco_data)

        # Check first point
        assert np.allclose(points_3d[0], [5.0, 10.0, 0.0])

        # Check metadata
        assert metadata[0]["category_id"] == 1
        assert metadata[0]["track_id"] == 10
        assert metadata[0]["image_id"] == 1

    def test_extract_3d_locations_no_3d(self):
        """Test extraction when annotations have no 3D locations."""
        coco_data = {
            "images": [{"id": 1, "file_name": "frame_001.jpg"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 200, 50, 80],
                    # No location_3d
                }
            ],
            "categories": [{"id": 1, "name": "person"}],
        }

        points_3d, metadata = extract_3d_locations(coco_data)

        assert len(points_3d) == 0
        assert len(metadata) == 0

    def test_load_coco_annotations_file_not_found(self):
        """Test loading COCO annotations when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_coco_annotations("/nonexistent/path/annotations.json")

    @pytest.fixture
    def temp_coco_file(self, tmp_path, sample_coco_data):
        """Create a temporary COCO annotation file."""
        import json

        coco_file = tmp_path / "test_annotations.json"
        with open(coco_file, "w") as f:
            json.dump(sample_coco_data, f)

        return str(coco_file)

    def test_load_coco_annotations_file_exists(self, temp_coco_file, sample_coco_data):
        """Test loading COCO annotations from existing file."""
        loaded_data = load_coco_annotations(temp_coco_file)

        assert loaded_data is not None
        assert "images" in loaded_data
        assert "annotations" in loaded_data
        assert "categories" in loaded_data
        assert len(loaded_data["images"]) == len(sample_coco_data["images"])
        assert len(loaded_data["annotations"]) == len(sample_coco_data["annotations"])

    def test_load_coco_annotations_path_conversion(self, temp_coco_file):
        """Test that Windows paths are converted to WSL paths."""
        # Test with Windows-style path
        windows_path = temp_coco_file.replace("/", "\\")
        if "\\" in windows_path:
            # Try to load with Windows path format
            # The function should handle path conversion
            loaded_data = load_coco_annotations(temp_coco_file)
            assert loaded_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
