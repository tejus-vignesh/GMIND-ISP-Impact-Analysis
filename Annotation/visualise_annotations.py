#!/usr/bin/env python3
"""
Visualise bounding box annotations on video.

Displays a video with bounding box annotations overlaid. The annotation file
should be in COCO format and have the same name as the video file with a
.json extension.

Features:
    - Real-time video playback with annotations
    - Adjustable playback delay
    - Colour-coded bounding boxes by category or track ID
    - Category labels on each bounding box

Controls:
    - Space: Pause/Resume playback
    - Q or ESC: Quit
    - Right Arrow / D: Step forward one frame (when paused)
    - Left Arrow / A: Step backward one frame (when paused)

Usage:
    python -m Annotation.visualise_annotations \\
        /path/to/video.mp4 \\
        --delay 10

Note:
    The annotation file should be located at the same path as the video
    with a .json extension (e.g., video.mp4 -> video.json).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_annotations(json_path: Path) -> Tuple[Dict[int, List[Dict]], Dict[int, str]]:
    """Load annotations from JSON file and organise by frame/image_id.

    Args:
        json_path: Path to the JSON annotation file

    Returns:
        Tuple of (annotations_by_frame, category_names)
        - annotations_by_frame: Dict mapping image_id to list of annotation dicts
        - category_names: Dict mapping category_id to category name
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build category name mapping
    category_names = {}
    for cat in data.get("categories", []):
        category_names[cat["id"]] = cat["name"]

    # Organise annotations by image_id (frame number)
    annotations_by_frame = {}
    for ann in data.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in annotations_by_frame:
            annotations_by_frame[image_id] = []
        annotations_by_frame[image_id].append(ann)

    return annotations_by_frame, category_names


def draw_bboxes(
    frame: np.ndarray, annotations: List[Dict], category_names: Dict[int, str]
) -> np.ndarray:
    """Draw bounding boxes on a frame.

    Args:
        frame: The video frame (numpy array)
        annotations: List of annotation dictionaries
        category_names: Dict mapping category_id to category name

    Returns:
        Frame with bounding boxes drawn
    """
    vis_frame = frame.copy()

    for ann in annotations:
        # COCO format: bbox is [x, y, width, height]
        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Get category info
        category_id = ann.get("category_id", 1)
        category_name = category_names.get(category_id, "object")

        # Generate color based on track_id or category_id
        track_id = ann.get("track_id", category_id)
        color = tuple(int((track_id * hash_val) % 200) + 55 for hash_val in [50, 100, 150])

        # Draw rectangle
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        score = ann.get("score", 1.0)
        if "track_id" in ann:
            label = f"ID:{track_id} {category_name} ({score:.2f})"
        else:
            label = f"{category_name} ({score:.2f})"

        # Draw label background and text
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(
            vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    return vis_frame


def visualise_video(video_path: str, delay_ms: int = 1):
    """Visualise video with bounding box annotations.

    Args:
        video_path: Path to the video file
        delay_ms: Delay between frames in milliseconds (default: 1ms)
    """
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    # Find corresponding JSON file
    json_path = video_path.parent / f"{video_path.stem}.json"

    if not json_path.exists():
        print(f"Error: Annotation file not found: {json_path}")
        print(f"Expected JSON file at: {json_path}")
        return

    print(f"Loading annotations from: {json_path}")
    annotations_by_frame, category_names = load_annotations(json_path)
    print(f"Loaded {len(annotations_by_frame)} frames with annotations")
    print(f"Categories: {list(category_names.values())}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
    print(f"Press 'q' or ESC to quit")
    print("-" * 70)

    # Create window
    cv2.namedWindow("Video Annotations", cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Get annotations for this frame (image_id is 1-indexed in COCO format)
            annotations = annotations_by_frame.get(frame_idx, [])

            # Draw bounding boxes
            vis_frame = draw_bboxes(frame, annotations, category_names)

            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Annotations: {len(annotations)}"
            (tw, th), baseline = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(vis_frame, (10, 10), (10 + tw, 10 + th + baseline), (0, 0, 0), -1)
            cv2.putText(
                vis_frame,
                info_text,
                (10, 10 + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            # Display frame
            cv2.imshow("Video Annotations", vis_frame)

            # Check for quit
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_idx} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise bounding box annotations on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualise_annotations.py /path/to/video.mp4
  python visualise_annotations.py /path/to/video.mp4 --delay 10
        """,
    )
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument(
        "--delay", type=int, default=1, help="Delay between frames in milliseconds (default: 1)"
    )

    args = parser.parse_args()
    visualise_video(args.video_path, args.delay)


if __name__ == "__main__":
    main()
