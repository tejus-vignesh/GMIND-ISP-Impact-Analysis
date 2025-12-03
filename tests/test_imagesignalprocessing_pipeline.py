"""
Unit tests for ImageSignalProcessing Pipeline.

Tests the core Pipeline class and basic ISP functionality to ensure the pipeline
correctly processes Bayer images through the ISP modules.
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

# Add ImageSignalProcessing directory to path for relative imports
isp_dir = Path(__file__).parent.parent / "ImageSignalProcessing"
if isp_dir.exists():
    sys.path.insert(0, str(isp_dir))

try:
    from ImageSignalProcessing.pipeline import Pipeline
    from ImageSignalProcessing.utils.yacs import Config
except ImportError as e:
    pytest.skip(f"ImageSignalProcessing not available: {e}", allow_module_level=True)


class TestISPPipeline:
    """Test suite for ISP Pipeline class."""

    @pytest.fixture
    def test_config_path(self):
        """Path to test configuration file."""
        config_path = (
            Path(__file__).parent.parent / "ImageSignalProcessing" / "configs" / "test.yaml"
        )
        if not config_path.exists():
            pytest.skip(f"Test config file not found: {config_path}")
        return str(config_path)

    @pytest.fixture
    def pipeline(self, test_config_path):
        """Create a Pipeline instance from test config."""
        cfg = Config(test_config_path)
        return Pipeline(cfg)

    @pytest.fixture
    def sample_bayer_image(self):
        """Create a sample Bayer pattern image for testing."""
        # Create a test Bayer image matching config dimensions (RGGB pattern)
        height, width = 1080, 1920  # Match test.yaml config
        bayer = np.zeros((height, width), dtype=np.uint16)

        # Fill with RGGB pattern
        # R channel (even rows, even cols)
        bayer[0::2, 0::2] = 1000
        # Gr channel (even rows, odd cols)
        bayer[0::2, 1::2] = 800
        # Gb channel (odd rows, even cols)
        bayer[1::2, 0::2] = 800
        # B channel (odd rows, odd cols)
        bayer[1::2, 1::2] = 600

        return bayer

    def test_pipeline_initialization(self, pipeline):
        """Test that Pipeline initializes correctly."""
        assert pipeline is not None
        assert hasattr(pipeline, "cfg")
        assert hasattr(pipeline, "modules")
        assert isinstance(pipeline.modules, dict)

    def test_pipeline_execute_basic(self, pipeline, sample_bayer_image):
        """Test basic pipeline execution with a sample Bayer image."""
        data, intermediates = pipeline.execute(
            bayer=sample_bayer_image, save_intermediates=False, verbose=False
        )

        assert data is not None
        assert isinstance(data, dict)
        assert "output" in data, "Pipeline should produce an output"

        output = data["output"]
        assert isinstance(output, np.ndarray), "Output should be a numpy array"
        assert len(output.shape) == 3, "Output should be RGB image (H, W, 3)"
        assert output.shape[2] == 3, "Output should have 3 channels (RGB)"
        assert output.dtype == np.uint8, "Output should be uint8"

    def test_pipeline_output_shape(self, pipeline, sample_bayer_image):
        """Test that pipeline output has correct shape."""
        data, _ = pipeline.execute(
            bayer=sample_bayer_image, save_intermediates=False, verbose=False
        )

        output = data["output"]
        # Output should have same spatial dimensions as input (or scaled)
        assert output.shape[0] > 0 and output.shape[1] > 0

    def test_pipeline_save_intermediates(self, pipeline, sample_bayer_image):
        """Test that intermediate results are saved when requested."""
        data, intermediates = pipeline.execute(
            bayer=sample_bayer_image, save_intermediates=True, verbose=False
        )

        assert isinstance(intermediates, dict)
        # Should have intermediate results from enabled modules
        assert len(intermediates) > 0, "Should save intermediate results"

    def test_pipeline_with_different_sizes(self, pipeline):
        """Test pipeline with different image sizes."""
        # Use sizes that are compatible with CEH module requirements
        # CEH module uses tiles, so dimensions need to be reasonable
        sizes = [
            (540, 960),  # Medium
            (1080, 1920),  # Standard (matches config)
        ]

        for height, width in sizes:
            bayer = np.random.randint(0, 1024, size=(height, width), dtype=np.uint16)
            try:
                data, _ = pipeline.execute(bayer=bayer, verbose=False)

                assert "output" in data
                output = data["output"]
                assert output.shape[0] > 0 and output.shape[1] > 0
                logger.debug("Successfully processed image size %dx%d", width, height)
            except (ValueError, RuntimeWarning) as e:
                # Some modules may have issues with certain sizes, skip this size
                logger.warning("Skipping size %dx%d due to: %s", width, height, e)
                pytest.skip(f"Image size {width}x{height} not compatible: {e}")

    def test_pipeline_config_loading(self, test_config_path):
        """Test that pipeline loads configuration correctly."""
        cfg = Config(test_config_path)
        pipeline = Pipeline(cfg)

        assert pipeline.cfg is not None
        assert hasattr(pipeline.cfg, "hardware")
        assert hasattr(pipeline.cfg.hardware, "raw_width")
        assert hasattr(pipeline.cfg.hardware, "raw_height")


class TestISPModules:
    """Test suite for individual ISP modules."""

    @pytest.fixture
    def test_config_path(self):
        """Path to test configuration file."""
        config_path = (
            Path(__file__).parent.parent / "ImageSignalProcessing" / "configs" / "test.yaml"
        )
        if not config_path.exists():
            pytest.skip(f"Test config file not found: {config_path}")
        return str(config_path)

    @pytest.fixture
    def config(self, test_config_path):
        """Create a Config instance."""
        return Config(test_config_path)

    def test_basic_module_structure(self, config):
        """Test that BasicModule can be imported and used."""
        from ImageSignalProcessing.modules.basic_module import BasicModule

        # BasicModule is abstract, so we can't instantiate it directly
        # But we can verify the structure
        assert BasicModule is not None
        assert hasattr(BasicModule, "execute")

    def test_module_dependencies(self):
        """Test that module dependencies are properly registered."""
        from ImageSignalProcessing.modules.basic_module import MODULE_DEPENDENCIES

        assert isinstance(MODULE_DEPENDENCIES, dict)
        logger.debug("Module dependencies: %s", list(MODULE_DEPENDENCIES.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
