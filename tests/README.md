# Tests

This folder contains test suites and validation scripts for the GMIND SDK.

## Test Structure

### Unit Tests
- `test_*.py`: Individual unit tests for specific components
- `test_calibration_camera_model.py`: Tests for camera calibration utilities (CameraModel class)
- `test_3d_projection_*.py`: Tests for 3D projection functions
- `test_geometric_3d_projection*.py`: Tests for geometric projection methods
- `test_imagesignalprocessing_pipeline.py`: Tests for ISP Pipeline class and modules
- `test_timesync_functions.py`: Tests for temporal alignment and validation functions
- `test_validation_functions.py`: Tests for validation utilities (sensor calibration parsing, ray transformations)

### Integration Tests
- `test_annotation_generation_integration.py`: End-to-end annotation generation tests
- `test_train_infer_all_models.py`: Model training and inference tests
- `test_dataloader_visualization.py`: DataLoader functionality tests
- `test_3d_projection_robust_pipeline.py`: Comprehensive robust projection pipeline tests

### Smoke Tests
- `test_train_models_smoke.py`: Quick smoke tests for model training
- `smoke_train.py`: Minimal training validation
- `test_training_loss_decreases.py`: Training loss verification tests

### Model Tests
- `test_models.py`: Tests for model loading and initialization
- `test_supported_models.py`: Tests for supported model backends
- `test_load_model_weights.py`: Tests for weight loading

### Evaluation Tests
- `test_evaluate_no_pycocotools.py`: Evaluation tests without pycocotools dependency

### Visualization Scripts
- `visualise_dataloader.py`: Visualize DataLoader output (utility script)
- `visualise_all_flir89.py`: Visualization for FLIR camera data (utility script)

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test
```bash
pytest tests/test_calibration_camera_model.py
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Smoke Tests Only
```bash
pytest tests/test_train_models_smoke.py
```

## Test Coverage

### Module Coverage
- **Annotation**: Comprehensive coverage (geometric projection, integration tests)
- **DataLoader**: Good coverage (visualization, batch loading)
- **DeepLearning**: Good coverage (models, training, inference)
- **ImageSignalProcessing**: Basic coverage (pipeline tests)
- **Calibration**: Basic coverage (CameraModel tests)
- **TimeSync**: Basic coverage (function tests)
- **Validation**: Basic coverage (utility tests)

## Test Data

### COCO Sample Dataset
The `coco_sample/` directory contains a minimal COCO dataset for testing:
- Download automatically on first run (if configured)
- Used for DataLoader and training tests
- Can be manually populated with COCO format data

### Checkpoints
The `checkpoints/` directory stores model checkpoints for testing:
- Created during training tests
- Used for inference validation
- Excluded from git (see `.gitignore`)

## Dependencies

Tests require:
- `pytest` for test framework
- All SDK dependencies (see `requirements.txt`)
- Optional: `pycocotools` for evaluation tests

Install test dependencies:
```bash
pip install pytest pytest-cov
pip install -e ".[all]"  # Install all optional dependencies
```

## Writing New Tests

### Test Template
```python
import pytest
from your_module import YourClass

def test_your_function():
    # Arrange
    test_input = "test_data"
    
    # Act
    result = YourClass().your_function(test_input)
    
    # Assert
    assert result is not None
    assert result == expected_output
```

### Test Categories
- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **Smoke tests**: Quick validation that code runs without errors
- **Visualization tests**: Generate visual outputs for manual inspection

## Continuous Integration

Tests should pass before merging PRs. Key requirements:
- All unit tests pass
- Smoke tests complete successfully
- No critical errors in integration tests

## Troubleshooting

**Import errors:**
- Ensure SDK is installed: `pip install -e .`
- Check that test paths are correct
- Verify all dependencies are installed

**Model loading failures:**
- Ensure model weights are downloaded:
  - **Dome-DETR**: Download `Dome-L-VisDrone-best.pth` and `Dome-L-VisDrone.yml` (see root README for setup)
  - **YOLO models**: Automatically downloaded on first use
- Check that model paths are correct
- Verify CUDA/CPU compatibility

**Test data missing:**
- COCO sample dataset downloads automatically
- Check network connection for automatic downloads
- Manually download if needed

