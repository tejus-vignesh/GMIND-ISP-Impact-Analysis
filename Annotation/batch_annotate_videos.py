#!/usr/bin/env python3
"""
Batch annotation generation for multiple videos.

Processes multiple video files in a directory, generating COCO-format annotations
for each. Supports filtering by filename patterns and recursive directory search.

Usage:
    # Process all videos in a directory
    python -m Annotation.batch_annotate_videos --directory /path/to/videos

    # Process only videos matching a pattern
    python -m Annotation.batch_annotate_videos --directory /path/to/videos --include "FLIR"

    # Exclude videos matching a pattern
    python -m Annotation.batch_annotate_videos --directory /path/to/videos --exclude "drone"

    # Recursive search
    python -m Annotation.batch_annotate_videos --directory /path/to/videos --recursive

    # Process specific videos
    python -m Annotation.batch_annotate_videos --videos video1.mp4 video2.mp4
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import List, Optional

import torch

from Annotation.annotation_generation import Config, process_video


def find_videos(
    base_path: Path,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    Find video files matching the specified criteria.

    Args:
        base_path: Directory to search for videos
        include_pattern: If provided, only include videos whose filename contains this pattern (case-insensitive)
        exclude_pattern: If provided, exclude videos whose filename contains this pattern (case-insensitive)
        recursive: If True, search recursively in subdirectories

    Returns:
        List of video file paths
    """
    if not base_path.exists():
        print(f"Warning: Base path does not exist: {base_path}")
        return []

    if not base_path.is_dir():
        print(f"Warning: Path is not a directory: {base_path}")
        return []

    # Find video files
    if recursive:
        videos = list(base_path.rglob("*.mp4"))
    else:
        videos = list(base_path.glob("*.mp4"))

    # Apply filters
    filtered_videos = []
    for video_path in videos:
        name_lower = video_path.name.lower()

        # Include filter
        if include_pattern and include_pattern.lower() not in name_lower:
            continue

        # Exclude filter
        if exclude_pattern and exclude_pattern.lower() in name_lower:
            continue

        filtered_videos.append(video_path)

    return sorted(filtered_videos)


def batch_annotate_videos(
    videos: Optional[List[Path]] = None,
    directory: Optional[Path] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    recursive: bool = False,
    force: bool = False,
):
    """
    Run annotation generation on multiple videos.

    Args:
        videos: List of specific video paths to process (if provided, other options are ignored)
        directory: Directory containing videos to process
        include_pattern: Only process videos whose filename contains this pattern
        exclude_pattern: Exclude videos whose filename contains this pattern
        recursive: Search recursively in subdirectories
        force: If True, reprocess videos even if annotations already exist
    """
    # Determine which videos to process
    if videos:
        # Use provided video list
        video_list = [Path(v) if isinstance(v, str) else v for v in videos]
        # Verify all videos exist
        video_list = [v for v in video_list if v.exists()]
        if not video_list:
            print("Error: None of the specified videos exist!")
            return
    elif directory:
        # Find videos in directory
        video_list = find_videos(Path(directory), include_pattern, exclude_pattern, recursive)
    else:
        print("Error: Must provide either --videos or --directory")
        return

    print(f"Found {len(video_list)} video(s) to process:")
    for v in video_list:
        print(f"  - {v}")
    print()

    if len(video_list) == 0:
        print("No matching videos found!")
        return

    config = Config()

    # Check which videos already have annotations (unless force is enabled)
    videos_to_process = []
    videos_skipped = []

    if not force:
        for video_path in video_list:
            annotation_file = video_path.parent / f"{video_path.stem}_anno.json"
            if annotation_file.exists():
                videos_skipped.append(video_path)
            else:
                videos_to_process.append(video_path)

        print(f"\nVideos already processed (skipping): {len(videos_skipped)}")
        for v in videos_skipped:
            print(f"  - {v.name}")

        print(f"\nVideos to process: {len(videos_to_process)}")
        for v in videos_to_process:
            print(f"  - {v.name}")
        print()
    else:
        videos_to_process = video_list
        print(f"\nForce mode enabled: will reprocess all {len(videos_to_process)} videos\n")

    if len(videos_to_process) == 0:
        print("All videos already processed! Use --force to reprocess.")
        return

    # Process videos
    total_videos = len(videos_to_process)
    print(f"Starting to process {total_videos} video(s)...")

    for i, video_path in enumerate(videos_to_process, 1):
        print(f"\n{'='*80}")
        print(f"Processing video {i}/{total_videos}: {video_path.name}")
        print(f"Full path: {video_path}")
        print(f"{'='*80}")

        try:
            process_video(str(video_path), config)
            print(f"Successfully processed: {video_path.name}")
        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Processed {i-1}/{total_videos} videos.")
            raise
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback

            traceback.print_exc()
            # Continue to next video even if this one fails
            continue
        finally:
            # Clean up GPU memory and Python garbage between videos
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n{'='*80}")
    print(f"Finished processing:")
    print(f"  - Processed: {len(videos_to_process)} video(s)")
    if not force:
        print(f"  - Skipped (already done): {len(videos_skipped)} video(s)")
    print(f"  - Total: {len(video_list)} video(s)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotation generation for multiple videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a directory
  python -m Annotation.batch_annotate_videos --directory /path/to/videos

  # Process only FLIR videos
  python -m Annotation.batch_annotate_videos --directory /path/to/videos --include FLIR

  # Exclude drone videos
  python -m Annotation.batch_annotate_videos --directory /path/to/videos --exclude drone

  # Process specific videos
  python -m Annotation.batch_annotate_videos --videos video1.mp4 video2.mp4

  # Recursive search with filtering
  python -m Annotation.batch_annotate_videos --directory /path/to/videos --recursive --include FLIR --exclude drone
        """,
    )

    # Video selection options
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument(
        "--directory",
        type=str,
        help="Directory containing videos to process",
    )
    video_group.add_argument(
        "--videos",
        nargs="+",
        help="Specific video files to process",
    )

    # Filtering options
    parser.add_argument(
        "--include",
        type=str,
        help="Only process videos whose filename contains this pattern (case-insensitive)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclude videos whose filename contains this pattern (case-insensitive)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively in subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess videos even if annotations already exist",
    )

    args = parser.parse_args()

    # Convert Windows paths to WSL paths if needed
    def convert_path(path_str: str) -> Path:
        path = Path(path_str)
        # Convert Windows path to WSL path if needed
        if path_str.startswith("H:\\") or path_str.startswith("H:/"):
            path_str = (
                path_str.replace("H:\\", "/mnt/h/").replace("H:/", "/mnt/h/").replace("\\", "/")
            )
            path = Path(path_str)
        return path

    if args.directory:
        directory = convert_path(args.directory)
        batch_annotate_videos(
            directory=directory,
            include_pattern=args.include,
            exclude_pattern=args.exclude,
            recursive=args.recursive,
            force=args.force,
        )
    elif args.videos:
        videos = [convert_path(v) for v in args.videos]
        batch_annotate_videos(videos=videos, force=args.force)


if __name__ == "__main__":
    main()
