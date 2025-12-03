#!/usr/bin/env python3
"""
Visualise all FLIR 8.9MP videos in the GMIND dataset to check annotation quality.

Usage:
    python tests/visualise_all_flir89.py
    python tests/visualise_all_flir89.py --data-root "/mnt/h/GMIND"
    python tests/visualise_all_flir89.py --skip-videos 2  # Skip first 2 videos
    python tests/visualise_all_flir89.py --max-frames-per-video 50  # Limit frames per video
"""

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import cv2
    import numpy as np

    from DataLoader import GMINDDataset
    from tests.test_dataloader_visualization import draw_boxes_on_image, tensor_to_numpy

    parser = argparse.ArgumentParser(description="Visualize all FLIR 8.9MP videos")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/h/GMIND",
        help="Root directory of GMIND dataset",
    )
    parser.add_argument(
        "--skip-videos",
        type=int,
        default=0,
        help="Skip first N videos",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Maximum frames to show per video (None for all)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Load every Nth frame (default: 1, no skipping)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS (default: 10)",
    )
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Automatically advance to next video after showing all frames",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)

    # Discover all FLIR 8.9 videos
    print(f"Discovering all FLIR 8.9MP videos in {data_root}...")
    videos = []

    for set_dir in data_root.iterdir():
        if not set_dir.is_dir() or set_dir.name.startswith("."):
            continue

        try:
            for subdir in set_dir.iterdir():
                if not subdir.is_dir() or not subdir.name.isdigit():
                    continue

                # Look for FLIR8.9 videos
                for video_file in subdir.glob("FLIR8.9-*.mp4"):
                    videos.append(
                        {
                            "path": video_file,
                            "set": set_dir.name,
                            "subdir": subdir.name,
                        }
                    )
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot access {set_dir}: {e}")
            continue

    videos.sort(key=lambda x: (x["set"], x["subdir"], x["path"].name))

    print(f"\nFound {len(videos)} FLIR 8.9MP videos:")
    for i, vid in enumerate(videos):
        print(f"  {i+1}. {vid['set']}/{vid['subdir']}/{vid['path'].name}")

    if len(videos) == 0:
        print("Error: No FLIR 8.9MP videos found!")
        sys.exit(1)

    # Skip videos if requested
    if args.skip_videos > 0:
        print(f"\nSkipping first {args.skip_videos} videos...")
        videos = videos[args.skip_videos :]

    print(f"\n{'='*70}")
    print(f"Starting visualization of {len(videos)} videos")
    print(f"{'='*70}\n")

    window_name = "GMIND FLIR 8.9MP - All Videos"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_frames_with_boxes = 0
    total_frames_without_boxes = 0
    frame_delay_ms = int(1000.0 / args.fps)

    for video_idx, video_info in enumerate(videos):
        video_path = video_info["path"]
        set_name = video_info["set"]
        sensor = "FLIR8.9"

        print(f"\n{'='*70}")
        print(
            f"Video {video_idx + 1}/{len(videos)}: {set_name}/{video_info['subdir']}/{video_path.name}"
        )
        print(f"{'='*70}")

        # Find annotation file
        annotation_path = video_path.with_suffix(".json")
        if not annotation_path.exists():
            annotation_path = video_path.parent / f"{sensor}-{video_path.parent.name}.json"

        print(f"Video: {video_path}")
        print(f"Annotation: {annotation_path}")
        if annotation_path.exists():
            print(f"  Annotation file found")
        else:
            print(f"  Annotation file NOT found")

        # Load category names
        category_names = None
        if annotation_path.exists():
            import json

            with open(annotation_path, "r") as f:
                ann_data = json.load(f)
                category_names = {cat["id"]: cat["name"] for cat in ann_data.get("categories", [])}
                num_images = len(ann_data.get("images", []))
                num_annotations = len(ann_data.get("annotations", []))
                print(f"  Images in JSON: {num_images}")
                print(f"  Annotations in JSON: {num_annotations}")
                print(f"  Categories: {list(category_names.values())}")

        # Create dataset for this video
        try:
            # Convert subdir string to int
            subdir_int = int(video_info["subdir"])
            dataset = GMINDDataset(
                data_root=data_root,
                sets=[set_name],
                sensor=sensor,
                transforms=None,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames_per_video,
                set_subdirs={set_name: [subdir_int]},
            )

            print(f"Dataset size: {len(dataset)} frames")

            if len(dataset) == 0:
                print("  Warning: No frames found, skipping...")
                continue

            # Get display size from first frame
            try:
                first_image, _ = dataset[0]
                first_img_np = tensor_to_numpy(first_image)
                h, w = first_img_np.shape[:2]
                max_display_size = 1280
                if w > max_display_size or h > max_display_size:
                    scale = max_display_size / max(w, h)
                    display_w, display_h = int(w * scale), int(h * scale)
                else:
                    display_w, display_h = w, h
            except:
                display_w, display_h = 1280, 720

            # Playback state
            paused = False
            current_frame = 0
            frames_with_boxes = 0
            frames_without_boxes = 0

            print(f"\nControls:")
            print(f"  Space: Pause/Resume")
            print(f"  Q: Quit")
            print(f"  N: Next video")
            print(f"  Right Arrow / D: Step forward (when paused)")
            print(f"  Left Arrow / A: Step backward (when paused)")
            print()

            while current_frame < len(dataset):
                try:
                    image, target = dataset[current_frame]
                    img_np = tensor_to_numpy(image)
                    boxes = target["boxes"]
                    labels = target["labels"]
                    num_boxes = len(boxes)

                    # Get actual video frame index
                    frame_info = (
                        dataset.frame_index[current_frame]
                        if hasattr(dataset, "frame_index")
                        else None
                    )
                    actual_video_frame = frame_info["frame_idx"] if frame_info else current_frame

                    # Print frame info every 100 frames or first 10 frames
                    if current_frame < 10 or current_frame % 100 == 0:
                        print(
                            f"  Frame {current_frame}/{len(dataset)-1} (video frame {actual_video_frame}): {num_boxes} boxes"
                        )

                    if num_boxes > 0:
                        frames_with_boxes += 1
                    else:
                        frames_without_boxes += 1
                        # Warn about frames without boxes
                        if current_frame < 20 or (current_frame % 100 == 0):
                            print(
                                f"  Warning: Frame {current_frame} (video frame {actual_video_frame}): NO BOXES"
                            )

                    # Draw boxes
                    img_with_boxes = draw_boxes_on_image(img_np, boxes, labels, category_names)

                    # Add video and frame info
                    video_info_text = f"Video {video_idx + 1}/{len(videos)}: {video_path.name}"
                    cv2.putText(
                        img_with_boxes,
                        video_info_text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    status = "PAUSED" if paused else "PLAYING"
                    frame_info_text = (
                        f"Frame {current_frame}/{len(dataset)-1} | Boxes: {num_boxes} | {status}"
                    )
                    cv2.putText(
                        img_with_boxes,
                        frame_info_text,
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Resize if needed
                    display_img = img_with_boxes.copy()
                    if display_img.shape[1] != display_w or display_img.shape[0] != display_h:
                        display_img = cv2.resize(display_img, (display_w, display_h))

                    cv2.imshow(window_name, display_img)

                    # Handle keyboard input
                    if paused:
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        key = cv2.waitKey(frame_delay_ms) & 0xFF

                    # Handle controls
                    if key == ord("q") or key == 27:  # Q or ESC
                        print("\nQuitting...")
                        cv2.destroyAllWindows()
                        sys.exit(0)
                    elif key == ord("n"):  # Next video
                        print("\nSkipping to next video...")
                        break
                    elif key == ord(" "):  # Spacebar
                        paused = not paused
                    elif paused:
                        if key == 83 or key == 2 or key == ord("d"):  # Right arrow or 'd'
                            current_frame = min(current_frame + 1, len(dataset) - 1)
                        elif key == 81 or key == 0 or key == ord("a"):  # Left arrow or 'a'
                            current_frame = max(current_frame - 1, 0)

                    # Advance frame if not paused
                    if not paused:
                        current_frame += 1
                    elif args.auto_advance and current_frame >= len(dataset) - 1:
                        # Auto-advance to next video if at end
                        break

                except Exception as e:
                    print(f"Error processing frame {current_frame}: {e}")
                    import traceback

                    traceback.print_exc()
                    current_frame += 1
                    continue

            # Video summary
            print(f"\nVideo Summary:")
            print(f"  Frames with boxes: {frames_with_boxes}")
            print(f"  Frames without boxes: {frames_without_boxes}")
            print(f"  Total: {frames_with_boxes + frames_without_boxes}")

            total_frames_with_boxes += frames_with_boxes
            total_frames_without_boxes += frames_without_boxes

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'='*70}")
    print(f"Final Summary:")
    print(f"  Videos processed: {len(videos)}")
    print(f"  Total frames with boxes: {total_frames_with_boxes}")
    print(f"  Total frames without boxes: {total_frames_without_boxes}")
    print(f"  Total frames: {total_frames_with_boxes + total_frames_without_boxes}")
    print(f"{'='*70}\n")

    cv2.destroyAllWindows()
    print("Visualization completed!")
