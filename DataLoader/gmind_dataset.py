"""
GMIND Dataset DataLoader for videos with 2D bounding box annotations.

Supports loading videos and their corresponding COCO-format annotations.
Designed for per-sensor loading (will be extended for synchronized multi-sensor in future).
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configure logger for this module
logger = logging.getLogger(__name__)


class GMINDDataset(Dataset):
    """
    PyTorch Dataset for GMIND videos with 2D bounding box annotations.

    Loads video files and their corresponding COCO-format JSON annotations.
    Extracts frames on-demand and returns them with bounding box targets.

    Example:
        >>> from DataLoader import GMINDDataset
        >>> dataset = GMINDDataset(
        ...     data_root="/mnt/h/GMIND",
        ...     sets=["UrbanJunctionSet"],
        ...     sensor="FLIR3.2",
        ...     frame_stride=1,
        ... )
        >>> image, target = dataset[0]
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        sets: Optional[List[str]] = None,
        sensor: str = "FLIR3.2",
        annotation_format: str = "coco",
        transforms: Optional[Callable] = None,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        subdirs: Optional[List[int]] = None,
        set_subdirs: Optional[Dict[str, List[int]]] = None,
        percentage_split: Optional[Dict[str, float]] = None,
        percentage_split_start: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize GMIND Dataset.

        Args:
            data_root: Root directory of GMIND dataset (e.g., "/mnt/h/GMIND" or "H:/GMIND")
            sets: List of dataset sets to include (e.g., ["UrbanJunctionSet"]).
                  If None, searches all sets.
            sensor: Sensor type to load (e.g., "FLIR3.2", "FLIR8.9", "CCTV", "Thermal", "EB", "Drone")
            annotation_format: Format of annotations ("coco" supported)
            transforms: Optional transforms to apply to images
            frame_stride: Load every Nth frame (default: 1, load all frames)
            max_frames: Maximum number of frames to load per video (None for all)
            subdirs: Optional list of subdirectory numbers to include (e.g., [1, 2])
            set_subdirs: Optional dict mapping set names to subdir lists (e.g., {"UrbanJunctionSet": [1]})
            percentage_split: Optional dict for percentage splits (e.g., {"PedestrianInteractionSet/2": 0.6})
                             If specified, only uses the first N% of frames from that set/subdir
            percentage_split_start: Optional dict for starting point of percentage splits (e.g., {"PedestrianInteractionSet/2": 0.6})
                                   If specified with percentage_split, uses frames from start% to (start+split)%
                                   If specified alone, uses frames from start% to end

        Raises:
            ValueError: If no videos are found for the specified sensor and sets
        """
        self.data_root = Path(data_root)
        self.sensor = sensor
        self.annotation_format = annotation_format
        self.transforms = transforms
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.subdirs = subdirs
        self.set_subdirs = set_subdirs or {}
        self.percentage_split = percentage_split or {}
        self.percentage_split_start = percentage_split_start or {}

        # Default sets if not specified
        if sets is None:
            sets = ["UrbanJunctionSet", "DistanceTestSet", "OutlierSet", "PedestrianInteractionSet"]

        # Find all video files and their annotations
        self.video_items = self._discover_videos(sets)

        if len(self.video_items) == 0:
            raise ValueError(
                f"No videos found for sensor '{sensor}' in sets {sets} " f"at {data_root}"
            )

        # Build frame index: (video_idx, frame_idx) -> annotation
        self.frame_index = self._build_frame_index()

        # Build COCO ground truth object for pycocotools evaluation
        self.coco = self._build_coco_ground_truth()

        logger.info(f"GMINDDataset initialized:")
        logger.info(f"  - Videos found: {len(self.video_items)}")
        logger.info(f"  - Total frames: {len(self.frame_index)}")
        logger.info(f"  - Sensor: {sensor}")
        logger.info(f"  - Sets: {sets}")

    def _discover_videos(self, sets: List[str]) -> List[Dict]:
        """Discover all video files and their corresponding annotation files."""
        video_items = []

        logger.info(f"Discovering videos for sensor '{self.sensor}'...")
        logger.debug(f"  Data root: {self.data_root}")
        logger.debug(f"  Sets: {sets}")
        logger.debug(f"  Set subdirs filter: {self.set_subdirs}")

        for set_name in sets:
            set_dir = self.data_root / set_name
            if not set_dir.exists():
                logger.warning(f"Set '{set_name}' directory not found: {set_dir}")
                continue

            logger.debug(f"Set '{set_name}' found at: {set_dir}")

            # Get allowed subdirs for this set
            allowed_subdirs = self.set_subdirs.get(set_name, self.subdirs)
            if allowed_subdirs is not None:
                logger.debug(f"  Filtering subdirs to: {allowed_subdirs}")

            # Look for numbered subdirectories (e.g., "1", "2", etc.)
            found_videos_in_set = 0
            skipped_subdirs = []
            for subdir in sorted(set_dir.iterdir()):
                if not subdir.is_dir() or not subdir.name.isdigit():
                    continue

                # Filter by subdir if specified
                subdir_num = int(subdir.name)
                if allowed_subdirs is not None and subdir_num not in allowed_subdirs:
                    skipped_subdirs.append(subdir_num)
                    continue

                logger.debug(f"  Checking subdir {subdir.name}...")

                # Look for video file matching sensor pattern
                video_pattern = f"{self.sensor}-*.mp4"
                video_files = list(subdir.glob(video_pattern))

                if not video_files:
                    # Try alternative naming: {SetName}-{Number}.mp4
                    alt_pattern = f"{set_name}-{subdir.name}.mp4"
                    alt_videos = list(subdir.glob(alt_pattern))
                    if alt_videos:
                        video_files = alt_videos
                        logger.debug(
                            f"    Found {len(alt_videos)} video(s) with alternative naming"
                        )
                else:
                    logger.debug(
                        f"    Found {len(video_files)} video(s) matching pattern '{video_pattern}'"
                    )

                if not video_files:
                    logger.debug(f"    No video files found in {subdir}")
                    continue

                for video_path in video_files:
                    # Find corresponding annotation file
                    # Try: {Sensor}-{Name}.json
                    json_path = video_path.with_suffix(".json")
                    if not json_path.exists():
                        # Try alternative naming
                        json_path = subdir / f"{self.sensor}-{subdir.name}.json"
                        if not json_path.exists():
                            logger.warning(f"No annotation file found for {video_path.name}")
                            logger.debug(f"  Tried: {video_path.with_suffix('.json').name}")
                            logger.debug(f"  Tried: {self.sensor}-{subdir.name}.json")
                            continue

                    video_items.append(
                        {
                            "video_path": video_path,
                            "annotation_path": json_path,
                            "set_name": set_name,
                            "subdir": subdir.name,
                        }
                    )
                    found_videos_in_set += 1
                    logger.debug(f"    Added video: {video_path.name}")

            if skipped_subdirs:
                logger.debug(f"  Skipped subdirs (filtered out): {skipped_subdirs}")
            logger.info(f"  Total videos found in '{set_name}': {found_videos_in_set}")

        logger.info(f"Total videos discovered: {len(video_items)}")
        return video_items

    def _build_frame_index(self) -> List[Dict]:
        """Build index mapping (video_idx, frame_idx) to annotations.

        Also remaps image/annotation IDs to be globally unique across videos
        and caches a merged COCO dict (``self._merged_coco_dict``) for later
        use by ``_build_coco_ground_truth``.
        """
        frame_index = []

        # Accumulators for the merged COCO dict
        merged_images = []
        merged_annotations = []
        merged_categories = None  # take from first video

        # Cumulative offsets to make IDs globally unique
        next_image_id_offset = 0
        next_ann_id_offset = 0

        for video_idx, item in enumerate(self.video_items):
            # Load annotations
            with open(item["annotation_path"], "r") as f:
                ann_data = json.load(f)

            # Determine per-video max IDs for offset calculation
            raw_images = ann_data.get("images", [])
            raw_annotations = ann_data.get("annotations", [])

            img_offset = next_image_id_offset
            ann_offset = next_ann_id_offset

            # Build image_id lookup (original -> remapped image dict)
            image_id_to_info = {}
            frame_to_image_id = {}
            for img in raw_images:
                remapped = dict(img)
                remapped["id"] = img["id"] + img_offset
                image_id_to_info[img["id"]] = remapped

                # Extract frame number from filename
                filename = img["file_name"]
                try:
                    frame_num = int(filename.split("_frame_")[-1].split(".")[0])
                    frame_to_image_id[frame_num] = img["id"]
                except Exception as e:
                    if len(frame_to_image_id) < 5:
                        logger.debug(f"Could not parse frame number from '{filename}': {e}")
                    continue

            # Remap annotation IDs and image_id references
            image_id_to_anns = {}
            for ann in raw_annotations:
                remapped_ann = dict(ann)
                remapped_ann["id"] = ann["id"] + ann_offset
                remapped_ann["image_id"] = ann["image_id"] + img_offset

                orig_image_id = ann["image_id"]
                if orig_image_id not in image_id_to_anns:
                    image_id_to_anns[orig_image_id] = []
                image_id_to_anns[orig_image_id].append(remapped_ann)

                merged_annotations.append(remapped_ann)

            # Accumulate remapped images into merged list
            merged_images.extend(image_id_to_info.values())

            # Take categories from the first video
            if merged_categories is None:
                merged_categories = ann_data.get("categories", [])

            # Advance offsets past all IDs used by this video
            if raw_images:
                next_image_id_offset += max(img["id"] for img in raw_images)
            if raw_annotations:
                next_ann_id_offset += max(ann["id"] for ann in raw_annotations)

            # Get video info to determine frame count
            cap = cv2.VideoCapture(str(item["video_path"]))
            if not cap.isOpened():
                logger.warning(f"Could not open video {item['video_path']}")
                cap.release()
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Check if this video needs percentage split
            set_name = item["set_name"]
            subdir = item["subdir"]
            split_key = f"{set_name}/{subdir}"
            split_percentage = self.percentage_split.get(split_key)
            split_start = self.percentage_split_start.get(split_key)

            # Collect all frame indices first
            all_frame_indices = list(range(0, total_frames, self.frame_stride))

            if split_percentage is not None or split_start is not None:
                if split_start is not None:
                    # Start from a specific percentage
                    start_idx = int(len(all_frame_indices) * split_start)
                    if split_percentage is not None:
                        # Use range: from start% to (start+split)%
                        end_idx = int(len(all_frame_indices) * (split_start + split_percentage))
                        all_frame_indices = all_frame_indices[start_idx:end_idx]
                    else:
                        # Use from start% to end
                        all_frame_indices = all_frame_indices[start_idx:]
                else:
                    # Use first N% of frames
                    split_point = int(len(all_frame_indices) * split_percentage)
                    all_frame_indices = all_frame_indices[:split_point]

            # Add frames to index
            frames_added = 0
            missing_annotations_count = 0
            for frame_idx in all_frame_indices:
                if self.max_frames and frames_added >= self.max_frames:
                    break

                orig_image_id = frame_to_image_id.get(frame_idx)
                annotations = image_id_to_anns.get(orig_image_id, [])
                image_info = image_id_to_info.get(orig_image_id) if orig_image_id is not None else None

                # Track missing annotations
                if orig_image_id is None:
                    missing_annotations_count += 1
                    if missing_annotations_count <= 5:  # Only log first few
                        logger.debug(
                            f"Frame {frame_idx} not found in JSON frame_to_image_id mapping"
                        )
                        logger.debug(
                            f"  Available frames in mapping: {sorted(list(frame_to_image_id.keys()))[:10]}..."
                        )

                frame_index.append(
                    {
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                        "annotations": annotations,
                        "image_info": image_info,
                    }
                )
                frames_added += 1

            if missing_annotations_count > 0:
                logger.warning(f"{missing_annotations_count} frames had no matching JSON entry")

        # Cache merged COCO dict for _build_coco_ground_truth
        self._merged_coco_dict = {
            "images": merged_images,
            "annotations": merged_annotations,
            "categories": merged_categories or [],
        }

        return frame_index

    def _build_coco_ground_truth(self):
        """Create a pycocotools COCO object from the merged annotation dict.

        Returns:
            A ``pycocotools.coco.COCO`` instance, or ``None`` if pycocotools
            is not installed.
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            return None

        coco = COCO()
        coco.dataset = self._merged_coco_dict
        coco.createIndex()
        return coco

    def _load_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Load a specific frame from a video file.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to load (0-based)

        Returns:
            Frame as numpy array in RGB format

        Raises:
            IOError: If video cannot be opened
            ValueError: If frame cannot be read
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _annotations_to_target(self, annotations: List[Dict], image_info: Optional[Dict]) -> Dict:
        """
        Convert COCO annotations to PyTorch target format.

        Args:
            annotations: List of COCO annotation dictionaries
            image_info: Optional image metadata dictionary

        Returns:
            Dictionary with keys:
            - 'boxes': Tensor of shape [N, 4] (x1, y1, x2, y2 format)
            - 'labels': Tensor of shape [N] (category IDs, 1-based)
            - 'area': Tensor of shape [N]
            - 'iscrowd': Tensor of shape [N]
            - 'image_id': Tensor of shape [1] (required for COCO evaluation)
        """
        # Get image_id from image_info if available
        image_id = image_info.get("id", 0) if image_info else 0

        if not annotations:
            # Return empty target
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
            }

        boxes = []
        labels = []
        areas = []
        iscrowds = []

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann["bbox"]
            x, y, w, h = bbox

            # Convert to [x1, y1, x2, y2] format
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])  # Already 1-based in COCO
            areas.append(ann.get("area", w * h))
            iscrowds.append(ann.get("iscrowd", 0))

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowds, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }

    def __len__(self) -> int:
        """Return number of frames in dataset."""
        return len(self.frame_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a frame and its annotations.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (image_tensor, target_dict)
            - image_tensor: Tensor of shape [C, H, W]
            - target_dict: Dictionary with 'boxes', 'labels', 'area', 'iscrowd', 'image_id'
        """
        frame_info = self.frame_index[idx]
        video_item = self.video_items[frame_info["video_idx"]]

        # Load frame - reading sequentially: video frame_idx matches JSON frame number
        # frame_info['frame_idx'] is the video frame index (0, 1, 2, ...)
        # This matches the frame number extracted from JSON filename (e.g., "frame_000000" -> 0)
        frame = self._load_frame(video_item["video_path"], frame_info["frame_idx"])

        # Convert to PIL Image for transforms compatibility
        image = Image.fromarray(frame)

        # Get annotations
        annotations = frame_info["annotations"]
        image_info = frame_info["image_info"]

        # Convert annotations to target format
        target = self._annotations_to_target(annotations, image_info)

        # Apply transforms if provided
        if self.transforms:
            image, target = self.transforms(image, target)

        # Convert PIL to tensor if not already done by transforms
        if isinstance(image, Image.Image):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target


def get_gmind_dataloader(
    data_root: Union[str, Path],
    sets: Optional[List[str]] = None,
    sensor: str = "FLIR3.2",
    annotation_format: str = "coco",
    transforms: Optional[Callable] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    subdirs: Optional[List[int]] = None,
    set_subdirs: Optional[Dict[str, List[int]]] = None,
    percentage_split: Optional[Dict[str, float]] = None,
    percentage_split_start: Optional[Dict[str, float]] = None,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for GMIND videos.

    Args:
        data_root: Root directory of GMIND dataset
        sets: List of dataset sets to include
        sensor: Sensor type to load
        annotation_format: Format of annotations ("coco" supported)
        transforms: Optional transforms to apply
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        frame_stride: Load every Nth frame
        max_frames: Maximum frames per video
        subdirs: Optional list of subdirectory numbers to include
        set_subdirs: Optional dict mapping set names to subdir lists
        percentage_split: Optional dict for percentage splits
        percentage_split_start: Optional dict for starting point of percentage splits
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader instance

    Example:
        >>> from DataLoader import get_gmind_dataloader
        >>> loader = get_gmind_dataloader(
        ...     data_root="/mnt/h/GMIND",
        ...     sets=["UrbanJunctionSet"],
        ...     sensor="FLIR3.2",
        ...     batch_size=4,
        ... )
        >>> for images, targets in loader:
        ...     # Training loop
        ...     pass
    """
    dataset = GMINDDataset(
        data_root=data_root,
        sets=sets,
        sensor=sensor,
        annotation_format=annotation_format,
        transforms=transforms,
        frame_stride=frame_stride,
        max_frames=max_frames,
        subdirs=subdirs,
        set_subdirs=set_subdirs,
        percentage_split=percentage_split,
        percentage_split_start=percentage_split_start,
    )

    # Use standard collate function compatible with training module
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )
