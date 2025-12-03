#!/usr/bin/env python3
"""
Inference script for running object detection models on video files.

This script provides a lightweight utility for running inference on a single video
file using Ultralytics YOLO models. It is designed for quick testing, debugging,
and visual inspection of model predictions without the overhead of the full
evaluation pipeline.

Features:
    - Support for pretrained models and custom trained weights
    - Real-time video display with predictions overlaid
    - Configurable confidence threshold
    - Optional saving of annotated video and text predictions
    - Frame-by-frame detection summaries

Use Cases:
    - Quick model testing and debugging
    - Visual inspection of predictions
    - Testing different confidence thresholds
    - Validating model performance on specific videos

For full evaluation with metrics on the test dataset, use:
    python -m Evaluation.core.baseline_detector_and_tracker

Usage Examples:
    # Use pretrained model
    python -m DeepLearning.run_inference \\
        --weights yolo11m \\
        --video /path/to/video.mp4 \\
        --save-vid \\
        --conf 0.25

    # Use custom trained weights
    python -m DeepLearning.run_inference \\
        --weights checkpoints/yolo11m/best.pt \\
        --video /path/to/video.mp4 \\
        --save-vid \\
        --conf 0.3 \\
        --device cuda

    # Run without display (headless mode)
    python -m DeepLearning.run_inference \\
        --weights yolo11m \\
        --video /path/to/video.mp4 \\
        --save-vid \\
        --no-show
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "Error: Ultralytics YOLO is not installed.\n" "Install with: pip install ultralytics",
        file=sys.stderr,
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_pretrained_model_name(weights: str) -> bool:
    """
    Determine if the weights argument is a pretrained model name or a file path.

    Pretrained model names are identifiers like 'yolo11m', 'yolov8m', etc.
    that don't have file extensions and aren't existing file paths.

    Args:
        weights: The weights argument from command line

    Returns:
        True if weights appears to be a pretrained model name, False if it's a file path
    """
    # Check for file extensions that indicate a file path
    if weights.endswith(".pt") or weights.endswith(".yaml"):
        return False

    # Check for path separators
    if "/" in weights or "\\" in weights:
        return False

    # Check if it's an existing file path
    if Path(weights).exists():
        return False

    # Likely a pretrained model name
    return True


def validate_inputs(
    weights: str, video_path: Path, is_pretrained: bool
) -> Tuple[bool, Optional[str]]:
    """
    Validate input arguments before running inference.

    Args:
        weights: Model weights path or name
        video_path: Path to video file
        is_pretrained: Whether weights is a pretrained model name

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if inputs are valid, False otherwise
        - error_message: Error message if validation fails, None otherwise
    """
    # Validate weights
    if not is_pretrained:
        weights_path = Path(weights)
        if not weights_path.exists():
            return (
                False,
                f"Weights file not found: {weights_path}\n"
                f"Please provide a valid path to model weights (.pt file)\n"
                f"Or use a pretrained model name like 'yolo11m', 'yolov8m', etc.",
            )

    # Validate video file
    if not video_path.exists():
        return (
            False,
            f"Video file not found: {video_path}\n"
            f"Please provide a valid path to the test video.",
        )

    # Check video file extension
    if video_path.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv", ".m4v"]:
        logger.warning(
            f"Video file has unusual extension: {video_path.suffix}\n"
            f"Supported formats: .mp4, .avi, .mov, .mkv, .m4v"
        )

    return True, None


def load_model(weights: str, is_pretrained: bool) -> Optional[YOLO]:
    """
    Load YOLO model from weights file or pretrained model name.

    Args:
        weights: Model weights path or pretrained model name
        is_pretrained: Whether weights is a pretrained model name

    Returns:
        Loaded YOLO model object, or None if loading failed
    """
    model_source = weights if is_pretrained else str(Path(weights).resolve())

    logger.info(f"Loading model: {model_source}")
    if is_pretrained:
        logger.info("Using pretrained Ultralytics weights (will download if needed)")

    try:
        model = YOLO(model_source)
        logger.info(f"Model loaded successfully (task: {model.task})")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(
            "Tip: Use --weights to specify a local .pt file or a different pretrained model name"
        )
        return None


def print_detection_summary(results, model: YOLO) -> None:
    """
    Print a summary of detections across all frames.

    Args:
        results: List of YOLO result objects from inference
        model: YOLO model object (for class name mapping)
    """
    if not results:
        logger.warning("No results to summarise")
        return

    logger.info("Detection Summary:")
    total_detections = 0
    all_classes = set()

    for i, result in enumerate(results):
        if hasattr(result, "boxes") and result.boxes is not None:
            num_detections = len(result.boxes)
            total_detections += num_detections

            if num_detections > 0:
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_names = [model.names[int(c)] for c in classes]
                all_classes.update(class_names)

                avg_conf = confs.mean()
                logger.info(
                    f"  Frame {i+1}: {num_detections} detections "
                    f"(classes: {', '.join(set(class_names))}, "
                    f"avg confidence: {avg_conf:.3f})"
                )

    logger.info(f"\nTotal: {total_detections} detections across {len(results)} frames")
    if all_classes:
        logger.info(f"Classes detected: {', '.join(sorted(all_classes))}")


def print_output_files(output_dir: Path) -> None:
    """
    Print information about generated output files.

    Args:
        output_dir: Directory containing inference output files
    """
    output_path = output_dir / "inference"
    if not output_path.exists():
        logger.warning(f"Output directory not found: {output_path}")
        return

    files = [f for f in output_path.iterdir() if f.is_file()]
    if not files:
        logger.warning("No output files generated")
        return

    logger.info("Output files:")
    total_size = 0
    for file in sorted(files):
        size_bytes = file.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        total_size += size_bytes
        logger.info(f"  - {file.name} ({size_mb:.2f} MB)")

    total_mb = total_size / (1024 * 1024)
    logger.info(f"Total output size: {total_mb:.2f} MB")


def run_inference(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    conf_threshold: float,
    device: str,
    save_video: bool,
    save_txt: bool,
    show_video: bool,
) -> Optional[list]:
    """
    Run inference on a video file.

    Args:
        model: Loaded YOLO model
        video_path: Path to input video file
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        device: Device to use ('cuda' or 'cpu')
        save_video: Whether to save annotated video
        save_txt: Whether to save predictions as text files
        show_video: Whether to display video in real-time

    Returns:
        List of YOLO result objects, or None if inference failed
    """
    logger.info(f"Running inference on video: {video_path.name}")
    if show_video:
        logger.info("Video will be displayed with predictions in real-time (press 'q' to quit)")
    else:
        logger.info("Processing video in headless mode (no display)")

    try:
        results = model.predict(
            source=str(video_path),
            conf=conf_threshold,
            device=device,
            save=save_video,
            save_txt=save_txt,
            show=show_video,
            project=str(output_dir),
            name="inference",
            exist_ok=True,
            verbose=True,
        )

        logger.info(f"Inference completed successfully")
        logger.info(f"Results saved to: {output_dir / 'inference'}")
        return results

    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user")
        return None
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return None


def main() -> int:
    """
    Main entry point for the inference script.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Run object detection inference on a video file using YOLO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pretrained model
  python -m DeepLearning.run_inference --weights yolo11m --video video.mp4 --save-vid

  # Use custom trained weights
  python -m DeepLearning.run_inference \\
      --weights checkpoints/yolo11m/best.pt \\
      --video video.mp4 \\
      --save-vid \\
      --conf 0.3

  # Headless mode (no display)
  python -m DeepLearning.run_inference \\
      --weights yolo11m \\
      --video video.mp4 \\
      --save-vid \\
      --no-show
        """,
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11m",
        help="Pretrained model name (e.g., 'yolo11m', 'yolov8m') or path to trained weights (.pt file). "
        "Default: yolo11m",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_results",
        help="Directory to save inference results (default: inference_results)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0.0-1.0, default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--save-vid",
        action="store_true",
        help="Save annotated video output",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save predictions as text files (YOLO format)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display video in real-time (useful for headless environments)",
    )

    args = parser.parse_args()

    # Validate confidence threshold
    if not 0.0 <= args.conf <= 1.0:
        logger.error(f"Confidence threshold must be between 0.0 and 1.0, got: {args.conf}")
        return 1

    # Convert to Path objects
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Determine if weights is a pretrained model name or file path
    is_pretrained = is_pretrained_model_name(args.weights)

    # Validate inputs
    is_valid, error_msg = validate_inputs(args.weights, video_path, is_pretrained)
    if not is_valid:
        logger.error(error_msg)
        return 1

    # Print configuration
    logger.info("=" * 70)
    logger.info("YOLO Inference Configuration")
    logger.info("=" * 70)
    model_display = args.weights if is_pretrained else str(Path(args.weights).resolve())
    logger.info(f"Model: {model_display}")
    logger.info(f"Video: {video_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Display video: {not args.no_show}")
    logger.info(f"Save video: {args.save_vid}")
    logger.info(f"Save text: {args.save_txt}")
    logger.info("=" * 70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.weights, is_pretrained)
    if model is None:
        return 1

    # Run inference
    results = run_inference(
        model=model,
        video_path=video_path,
        output_dir=output_dir,
        conf_threshold=args.conf,
        device=args.device,
        save_video=args.save_vid,
        save_txt=args.save_txt,
        show_video=not args.no_show,
    )

    if results is None:
        return 1

    # Print summaries
    print_detection_summary(results, model)
    print_output_files(output_dir)

    logger.info("=" * 70)
    logger.info("Inference complete!")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
