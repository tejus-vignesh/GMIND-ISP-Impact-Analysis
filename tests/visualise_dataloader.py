#!/usr/bin/env python3
"""
Simple script to visualise GMIND DataLoader output with video playback.

Usage:
    # Search for video by filename (default: FLIR8.9-Urban1.mp4)
    python tests/visualise_dataloader.py
    python tests/visualise_dataloader.py --video-filename "FLIR3.2-Urban1.mp4"

    # Specify custom data root
    python tests/visualise_dataloader.py --video-filename "FLIR8.9-Urban1.mp4" --data-root "/mnt/h/GMIND"

    # Use full path (overrides filename search)
    python tests/visualise_dataloader.py --video-path "H:/GMIND/UrbanJunctionSet/1/FLIR8.9-Urban1.mp4"

    # Other options
    python tests/visualise_dataloader.py --max-frames 100
    python tests/visualise_dataloader.py --fps 30  # Playback speed
    python tests/visualise_dataloader.py --no-display  # Just test loading, don't show

Controls:
    Space: Pause/Resume
    Q: Quit
    Right Arrow / D: Step forward (when paused)
    Left Arrow / A: Step backward (when paused)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise GMIND DataLoader with video playback")
    parser.add_argument(
        "--video-filename",
        type=str,
        default="FLIR8.9-Urban1.mp4",
        help="Video filename to search for in GMIND dataset (e.g., 'FLIR8.9-Urban1.mp4')",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of GMIND dataset (default: auto-detect from common locations)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Full path to video file (overrides --video-filename if provided)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to display (None for all frames)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display frames, just test loading",
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
        default=30.0,
        help="Playback FPS (default: 30)",
    )

    args = parser.parse_args()

    # Import here to avoid issues if DataLoader not available
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from DataLoader import GMINDDataset

    # Find video file
    video_path = None
    data_root = None

    if args.video_path:
        # Use provided full path
        video_path_str = args.video_path
        if video_path_str.startswith("H:/") or video_path_str.startswith("H:\\"):
            video_path_str = video_path_str.replace("H:/", "/mnt/h/").replace("\\", "/")
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        data_root = video_path.parent.parent.parent
    else:
        # Search for video by filename
        # Try common data root locations
        possible_roots = []
        if args.data_root:
            possible_roots.append(Path(args.data_root))
        else:
            # Auto-detect common locations
            possible_roots = [
                Path("/mnt/h/GMIND"),
                Path("H:/GMIND"),
                Path("/mnt/c/GMIND"),
                Path("C:/GMIND"),
            ]

        # Convert Windows paths to WSL if needed and remove duplicates
        converted_roots = []
        seen_paths = set()
        for root in possible_roots:
            root_str = str(root)
            if root_str.startswith("H:/") or root_str.startswith("H:\\"):
                converted = Path(root_str.replace("H:/", "/mnt/h/").replace("\\", "/"))
            elif root_str.startswith("C:/") or root_str.startswith("C:\\"):
                converted = Path(root_str.replace("C:/", "/mnt/c/").replace("\\", "/"))
            else:
                converted = root

            # Only add if we haven't seen this path before
            path_str = str(converted.resolve() if converted.exists() else converted)
            if path_str not in seen_paths:
                converted_roots.append(converted)
                seen_paths.add(path_str)

        possible_roots = converted_roots

        print(f"Searching for video: {args.video_filename}")
        print(f"Searching in data roots: {[str(r) for r in possible_roots]}\n")

        # Search through dataset structure
        for root in possible_roots:
            if not root.exists():
                continue

            # Search in all sets (UrbanJunctionSet, DistanceTestSet, etc.)
            try:
                set_dirs = list(root.iterdir())
            except (PermissionError, OSError) as e:
                print(f"Warning: Cannot access {root}: {e}")
                continue

            for set_dir in set_dirs:
                try:
                    if not set_dir.is_dir() or set_dir.name.startswith("."):
                        continue

                    # Search in numbered subdirectories (1, 2, etc.)
                    try:
                        subdirs = list(set_dir.iterdir())
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Cannot access {set_dir}: {e}")
                        continue

                    for subdir in subdirs:
                        try:
                            if not subdir.is_dir() or not subdir.name.isdigit():
                                continue

                            # Check if video file exists here
                            video_file = subdir / args.video_filename
                            if video_file.exists():
                                video_path = video_file
                                data_root = root
                                print(f"Found video: {video_path}")
                                break
                        except (PermissionError, OSError) as e:
                            # Skip directories we can't access
                            continue

                    if video_path:
                        break
                except (PermissionError, OSError) as e:
                    # Skip directories we can't access
                    continue

            if video_path:
                break

        if not video_path:
            print(f"Error: Could not find video '{args.video_filename}' in GMIND dataset")
            print(f"Searched in:")
            for root in possible_roots:
                if root.exists():
                    print(f"  - {root}")
                else:
                    print(f"  - {root} (does not exist)")
            sys.exit(1)

    # Get set name and sensor from video path
    set_name = video_path.parent.parent.name
    sensor = video_path.stem.split("-")[0]

    # Find annotation file
    annotation_path = video_path.with_suffix(".json")
    if not annotation_path.exists():
        # Try alternative naming
        annotation_path = video_path.parent / f"{sensor}-{video_path.parent.name}.json"

    print(f"\n{'='*60}")
    print(f"GMIND DataLoader Visualization")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Annotation JSON: {annotation_path}")
    if annotation_path.exists():
        print(f"  Annotation file found")
    else:
        print(f"  Annotation file NOT found - will use empty annotations")
    print(f"Data root: {data_root}")
    print(f"Set: {set_name}")
    print(f"Sensor: {sensor}")
    print(f"Max frames: {args.max_frames if args.max_frames else 'All'}")
    print(
        f"Frame stride: {args.frame_stride} (no skipping)"
        if args.frame_stride == 1
        else f"Frame stride: {args.frame_stride}"
    )
    print(f"Playback FPS: {args.fps}")
    print(f"Display: {not args.no_display}")
    print(f"{'='*60}\n")
    if not args.no_display:
        print("Controls:")
        print("  Space: Pause/Resume")
        print("  Q: Quit")
        print("  Right Arrow / D: Step forward (when paused)")
        print("  Left Arrow / A: Step backward (when paused)")
        print()

    # Create dataset
    try:
        dataset = GMINDDataset(
            data_root=data_root,
            sets=[set_name],
            sensor=sensor,
            transforms=None,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )

        print(f"Dataset size: {len(dataset)} frames\n")

        if len(dataset) == 0:
            print("Error: Dataset is empty - no frames found!")
            sys.exit(1)

        # Load category names (annotation_path already found above)
        category_names = None
        if annotation_path.exists():
            import json

            with open(annotation_path, "r") as f:
                ann_data = json.load(f)
                category_names = {cat["id"]: cat["name"] for cat in ann_data.get("categories", [])}
                num_images = len(ann_data.get("images", []))
                num_annotations = len(ann_data.get("annotations", []))
                print(f"Annotation file loaded: {annotation_path}")
                print(f"  Images in JSON: {num_images}")
                print(f"  Annotations in JSON: {num_annotations}")
                print(f"  Categories: {category_names}\n")
        else:
            print(f"Warning: Annotation file not found at {annotation_path}\n")

        # Import visualization functions
        from tests.test_dataloader_visualization import (
            draw_boxes_on_image,
            tensor_to_numpy,
        )

        # Process frames with video playback
        frames_with_boxes = 0
        frames_without_boxes = 0

        # Calculate frame delay for video playback
        frame_delay_ms = int(1000.0 / args.fps)  # Convert FPS to milliseconds

        # Video playback state
        paused = False
        current_frame = 0
        window_name = f"GMIND DataLoader - {sensor}"

        # Get first frame to determine display size
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

        print(f"Starting video playback... (Press Space to pause, Q to quit)\n")

        while current_frame < len(dataset):
            try:
                image, target = dataset[current_frame]

                # Convert to numpy
                img_np = tensor_to_numpy(image)

                # Get boxes and labels
                boxes = target["boxes"]
                labels = target["labels"]

                num_boxes = len(boxes)

                # Debug: Track box counts to detect pattern
                if current_frame < 20 or (current_frame % 10 == 0):  # Print first 20 and every 10th
                    # Get actual video frame index from dataset
                    frame_info = (
                        dataset.frame_index[current_frame]
                        if hasattr(dataset, "frame_index")
                        else None
                    )
                    actual_video_frame = frame_info["frame_idx"] if frame_info else current_frame
                    print(
                        f"Frame {current_frame}: Video frame {actual_video_frame}, Boxes: {num_boxes}"
                    )

                if num_boxes > 0:
                    frames_with_boxes += 1
                else:
                    frames_without_boxes += 1
                    # Debug: print when boxes disappear
                    frame_info = (
                        dataset.frame_index[current_frame]
                        if hasattr(dataset, "frame_index")
                        else None
                    )
                    actual_video_frame = frame_info["frame_idx"] if frame_info else current_frame
                    image_id = (
                        frame_info.get("image_info", {}).get("id")
                        if frame_info and frame_info.get("image_info")
                        else None
                    )
                    print(
                        f"Debug: Dataset index {current_frame} -> Video frame {actual_video_frame}, Image ID: {image_id}, Boxes: {num_boxes}"
                    )

                if not args.no_display:
                    # Draw boxes
                    img_with_boxes = draw_boxes_on_image(img_np, boxes, labels, category_names)

                    # Add frame info with larger font
                    status = "PAUSED" if paused else "PLAYING"
                    info_text = (
                        f"Frame {current_frame}/{len(dataset)-1} | Boxes: {num_boxes} | {status}"
                    )
                    cv2.putText(
                        img_with_boxes,
                        info_text,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

                    # Add confirmation of sequential reading
                    seq_text = f"Video Frame: {current_frame} | JSON Frame: {current_frame}"
                    cv2.putText(
                        img_with_boxes,
                        seq_text,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Resize if too large
                    display_img = img_with_boxes.copy()
                    if display_img.shape[1] != display_w or display_img.shape[0] != display_h:
                        display_img = cv2.resize(display_img, (display_w, display_h))

                    cv2.imshow(window_name, display_img)

                    # Handle keyboard input
                    if paused:
                        # When paused, wait indefinitely for keypress
                        key = cv2.waitKey(0) & 0xFF
                    else:
                        # When playing, wait for frame delay or keypress
                        key = cv2.waitKey(frame_delay_ms) & 0xFF

                    # Handle controls
                    if key == ord("q") or key == 27:  # Q or ESC
                        break
                    elif key == ord(" "):  # Spacebar
                        paused = not paused
                    elif paused:
                        # Step controls when paused
                        # Arrow keys: 81=left, 82=up, 83=right, 84=down (on some systems)
                        # Alternative: 0=left, 1=up, 2=right, 3=down (on other systems)
                        if key == 83 or key == 2 or key == ord("d"):  # Right arrow or 'd'
                            current_frame = min(current_frame + 1, len(dataset) - 1)
                        elif key == 81 or key == 0 or key == ord("a"):  # Left arrow or 'a'
                            current_frame = max(current_frame - 1, 0)

                    # Advance frame if not paused
                    if not paused:
                        current_frame += 1
                else:
                    # No display mode - just process
                    if current_frame % 100 == 0:
                        print(f"Processed {current_frame}/{len(dataset)} frames...")
                    current_frame += 1

            except Exception as e:
                print(f"Error processing frame {current_frame}: {e}")
                import traceback

                traceback.print_exc()
                current_frame += 1
                continue

        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Frames with boxes: {frames_with_boxes}")
        print(f"  Frames without boxes: {frames_without_boxes}")
        print(f"  Total frames processed: {frames_with_boxes + frames_without_boxes}")
        print(f"  Total frames in dataset: {len(dataset)}")
        print(f"{'='*60}\n")

        if not args.no_display:
            cv2.destroyAllWindows()
            print("Video playback completed!")
        else:
            print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
