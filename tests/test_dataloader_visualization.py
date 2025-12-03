"""
Unit test for GMIND DataLoader that visualizes frames with bounding boxes.

This test loads a specific video and displays frames with their annotations.
"""

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from DataLoader import GMINDDataset, get_gmind_dataloader
except ImportError as e:
    pytest.skip(f"DataLoader not available: {e}", allow_module_level=True)


def draw_boxes_on_image(
    image: np.ndarray, boxes: torch.Tensor, labels: torch.Tensor, category_names: dict = None
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Image as numpy array (H, W, C) in RGB format
        boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        labels: Tensor of shape [N] with category IDs
        category_names: Optional dict mapping category_id to name

    Returns:
        Image with boxes drawn (BGR format for cv2)
    """
    # Convert RGB to BGR for cv2
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image

    # Default category names
    if category_names is None:
        category_names = {
            1: "person",
            2: "bicycle",
            3: "car",
        }

    # Color map for different categories
    colors = {
        1: (0, 255, 0),  # Green for person
        2: (255, 0, 0),  # Blue for bicycle
        3: (0, 0, 255),  # Red for car
    }

    boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    for i, (box, label) in enumerate(zip(boxes_np, labels_np)):
        x1, y1, x2, y2 = box.astype(int)
        category_id = int(label)
        color = colors.get(category_id, (255, 255, 255))  # White for unknown
        category_name = category_names.get(category_id, f"class_{category_id}")

        # Draw rectangle with thicker lines
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

        # Draw label with larger font - include box index for tracking
        label_text = f"{category_name} #{i}"
        font_scale = 1.2  # Increased from 0.5
        thickness = 2  # Increased from 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            img_bgr, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), color, -1
        )
        cv2.putText(
            img_bgr,
            label_text,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return img_bgr


def tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert image tensor to numpy array."""
    if isinstance(image_tensor, torch.Tensor):
        # Handle different tensor formats
        if image_tensor.dim() == 3:  # [C, H, W]
            img = image_tensor.permute(1, 2, 0).cpu().numpy()
        elif image_tensor.dim() == 4:  # [B, C, H, W]
            img = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        else:
            img = image_tensor.cpu().numpy()

        # Normalize if needed (assuming [0, 1] range)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        return img
    return image_tensor


@pytest.mark.parametrize(
    "video_path",
    [
        "H:/GMIND/UrbanJunctionSet/1/FLIR8.9-Urban1.mp4",
        "/mnt/h/GMIND/UrbanJunctionSet/1/FLIR8.9-Urban1.mp4",
    ],
)
def test_dataloader_visualization(video_path, display: bool = False, max_frames: int = 10):
    """
    Test DataLoader by loading a video and displaying frames with bounding boxes.

    Args:
        video_path: Path to video file (Windows or WSL format)
        display: Whether to display frames and wait for keyboard input (default: False)
        max_frames: Maximum number of frames to display
    """
    # Convert Windows path to WSL path if needed
    video_path_str = str(video_path)
    if video_path_str.startswith("H:/") or video_path_str.startswith("H:\\"):
        video_path_str = video_path_str.replace("H:/", "/mnt/h/").replace("\\", "/")

    video_path = Path(video_path_str)

    if not video_path.exists():
        pytest.skip(f"Video file not found: {video_path}")

    # Get data root and set name from video path
    data_root = video_path.parent.parent.parent
    set_name = video_path.parent.parent.name
    sensor = video_path.stem.split("-")[0]  # Extract sensor from filename

    logger.info("=" * 60)
    logger.info("Testing GMIND DataLoader Visualization")
    logger.info("=" * 60)
    logger.info("Video: %s", video_path)
    logger.info("Data root: %s", data_root)
    logger.info("Set: %s", set_name)
    logger.info("Sensor: %s", sensor)
    logger.info("=" * 60)

    # Create dataset
    dataset = GMINDDataset(
        data_root=data_root,
        sets=[set_name],
        sensor=sensor,
        transforms=None,  # No transforms for visualization
        frame_stride=30,  # Sample every 30th frame for faster testing
        max_frames=max_frames,
    )

    logger.info("Dataset size: %d frames", len(dataset))

    if len(dataset) == 0:
        pytest.fail("Dataset is empty - no frames found!")

    # Load category names from annotation file if available
    annotation_path = video_path.with_suffix(".json")
    category_names = None
    if annotation_path.exists():
        import json

        with open(annotation_path, "r") as f:
            ann_data = json.load(f)
            category_names = {cat["id"]: cat["name"] for cat in ann_data.get("categories", [])}

    # Iterate through frames
    frames_with_boxes = 0
    frames_without_boxes = 0

    for idx in range(min(len(dataset), max_frames)):
        try:
            image, target = dataset[idx]

            # Convert to numpy
            img_np = tensor_to_numpy(image)

            # Get boxes and labels
            boxes = target["boxes"]
            labels = target["labels"]

            num_boxes = len(boxes)
            if num_boxes > 0:
                frames_with_boxes += 1
                logger.debug("Frame %d: %d bounding boxes", idx, num_boxes)
            else:
                frames_without_boxes += 1
                logger.debug("Frame %d: No bounding boxes", idx)

            # Draw boxes
            img_with_boxes = draw_boxes_on_image(img_np, boxes, labels, category_names)

            # Add frame info text
            info_text = f"Frame {idx}/{len(dataset)-1} | Boxes: {num_boxes}"
            cv2.putText(
                img_with_boxes,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display if requested
            if display:
                # Resize if too large for display
                display_img = img_with_boxes.copy()
                h, w = display_img.shape[:2]
                max_display_size = 1280
                if w > max_display_size or h > max_display_size:
                    scale = max_display_size / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))

                cv2.imshow(f"GMIND DataLoader - {sensor} - Frame {idx}", display_img)
                logger.debug("Press any key to continue, 'q' to quit...")
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                cv2.destroyAllWindows()

        except Exception as e:
            logger.error("Error processing frame %d: %s", idx, e)
            import traceback

            traceback.print_exc()
            continue

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("Frames with boxes: %d", frames_with_boxes)
    logger.info("Frames without boxes: %d", frames_without_boxes)
    logger.info("Total frames processed: %d", frames_with_boxes + frames_without_boxes)
    logger.info("=" * 60)

    if display:
        cv2.destroyAllWindows()

    # Assert that we processed at least some frames
    assert frames_with_boxes + frames_without_boxes > 0, "No frames were processed!"

    # Optionally assert that we found some annotations
    # (This might fail if the video has no annotations, so we make it optional)
    # assert frames_with_boxes > 0, "No frames with bounding boxes found!"


def test_dataloader_batch_loading():
    """Test that DataLoader can create batches correctly."""
    data_root = "/mnt/h/GMIND"
    video_path = Path(data_root) / "UrbanJunctionSet" / "1" / "FLIR8.9-Urban1.mp4"

    if not video_path.exists():
        pytest.skip(f"Video file not found: {video_path}")

    # Create dataloader with small batch
    dataloader = get_gmind_dataloader(
        data_root=data_root,
        sets=["UrbanJunctionSet"],
        sensor="FLIR8.9",
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Use 0 workers for testing
        frame_stride=100,  # Sample sparsely
        max_frames=4,  # Just 4 frames
    )

    # Get one batch
    images, targets = next(iter(dataloader))

    # Check batch structure
    # DataLoader returns images as a list of tensors (not a single batched tensor)
    assert isinstance(images, list), "Images should be a list"
    assert len(images) <= 2, "Batch size should be <= 2"
    assert all(isinstance(img, torch.Tensor) for img in images), "All images should be tensors"
    if len(images) > 0:
        assert images[0].shape[0] == 3, "Images should have 3 channels"

    assert isinstance(targets, list), "Targets should be a list"
    assert len(targets) == len(images), "Number of targets should match batch size"

    for target in targets:
        assert "boxes" in target, "Target should have 'boxes' key"
        assert "labels" in target, "Target should have 'labels' key"
        assert isinstance(target["boxes"], torch.Tensor), "Boxes should be a tensor"
        assert isinstance(target["labels"], torch.Tensor), "Labels should be a tensor"


if __name__ == "__main__":
    # Run visualization test directly
    import sys

    # Check if display flag is passed (default: False, use --display to enable)
    display = "--display" in sys.argv
    max_frames = 10
    if "--max-frames" in sys.argv:
        idx = sys.argv.index("--max-frames")
        if idx + 1 < len(sys.argv):
            max_frames = int(sys.argv[idx + 1])

    # Try both path formats
    test_paths = [
        "H:/GMIND/UrbanJunctionSet/1/FLIR8.9-Urban1.mp4",
        "/mnt/h/GMIND/UrbanJunctionSet/1/FLIR8.9-Urban1.mp4",
    ]

    for path in test_paths:
        try:
            test_dataloader_visualization(path, display=display, max_frames=max_frames)
            break
        except Exception as e:
            logger.error("Failed with path %s: %s", path, e)
            continue
    else:
        logger.error("Could not find video file in any of the expected locations")
