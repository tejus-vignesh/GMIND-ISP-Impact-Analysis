"""ISP Variant Dataset — extends GMINDDataset for ISP sensitivity analysis.

Discovers video files produced by different ISP parameter variants without
modifying the original DataLoader code.
"""

import glob as glob_module
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from torch.utils.data import DataLoader

from DataLoader.gmind_dataset import GMINDDataset

logger = logging.getLogger(__name__)


class ISPVariantDataset(GMINDDataset):
    """GMINDDataset subclass that discovers ISP variant video files.

    When *isp_variant* is ``None`` the behaviour is identical to the parent
    class.  When set, ``_discover_videos`` looks under each subdir's
    ``Processed_Images/{sensor}/{variant}/`` tree for the variant video.

    Path conventions
    ----------------
    * **Default_ISP**: ``{subdir}/Processed_Images/{sensor}/Default_ISP/{sensor}.mp4``
    * **Bayer_GC**:    ``{subdir}/Processed_Images/Bayer_GC/{sensor}_Bayer_GC.mp4``
    * **Other**:       ``{subdir}/Processed_Images/{sensor}/{variant}/{variant}.mp4``
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        isp_variant: Optional[str] = None,
        sets: Optional[List[str]] = None,
        sensor: str = "FLIR8.9",
        annotation_format: str = "coco",
        transforms: Optional[Callable] = None,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        subdirs: Optional[List[int]] = None,
        set_subdirs: Optional[Dict[str, List[int]]] = None,
        percentage_split: Optional[Dict[str, float]] = None,
        percentage_split_start: Optional[Dict[str, float]] = None,
    ):
        # Store variant *before* super().__init__ so the overridden
        # _discover_videos can use it during construction.
        self.isp_variant = isp_variant

        super().__init__(
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

    # ------------------------------------------------------------------
    # Video discovery
    # ------------------------------------------------------------------

    def _discover_videos(self, sets: List[str]) -> List[Dict]:
        if self.isp_variant is None:
            return super()._discover_videos(sets)

        video_items: List[Dict] = []
        variant = self.isp_variant

        logger.info(
            f"Discovering ISP variant videos: variant='{variant}', sensor='{self.sensor}'"
        )

        for set_name in sets:
            set_dir = self.data_root / set_name
            if not set_dir.exists():
                logger.warning(f"Set directory not found: {set_dir}")
                continue

            allowed_subdirs = self.set_subdirs.get(set_name, self.subdirs)

            for subdir in sorted(set_dir.iterdir()):
                if not subdir.is_dir() or not subdir.name.isdigit():
                    continue

                subdir_num = int(subdir.name)
                if allowed_subdirs is not None and subdir_num not in allowed_subdirs:
                    continue

                # --- Locate variant video ---
                video_path = self._find_variant_video(subdir, variant)
                if video_path is None:
                    logger.warning(
                        f"No video for variant '{variant}' in {subdir}"
                    )
                    continue

                # --- Locate annotation file (flexible) ---
                ann_path = self._find_annotation(subdir, subdir_num)
                if ann_path is None:
                    logger.warning(
                        f"No annotation file found in {subdir}"
                    )
                    continue

                video_items.append(
                    {
                        "video_path": video_path,
                        "annotation_path": ann_path,
                        "set_name": set_name,
                        "subdir": subdir.name,
                    }
                )
                logger.debug(
                    f"  Added: {video_path.name}  ann={ann_path.name}"
                )

        logger.info(f"Total ISP variant videos discovered: {len(video_items)}")
        return video_items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_variant_video(self, subdir: Path, variant: str) -> Optional[Path]:
        """Return the video path for a given ISP variant under *subdir*."""
        proc = subdir / "Processed_Images"

        if variant == "Default_ISP":
            p = proc / self.sensor / "Default_ISP" / f"{self.sensor}.mp4"
            if p.exists():
                return p
        elif variant == "Bayer_GC":
            p = proc / "Bayer_GC" / f"{self.sensor}_Bayer_GC.mp4"
            if p.exists():
                return p
        else:
            p = proc / self.sensor / variant / f"{variant}.mp4"
            if p.exists():
                return p

        # Fallback: glob for any .mp4 that contains the variant name
        candidates = list(proc.rglob(f"*{variant}*.mp4"))
        if candidates:
            return candidates[0]

        return None

    def _find_annotation(self, subdir: Path, subdir_num: int) -> Optional[Path]:
        """Flexible annotation discovery for a numbered subdir."""
        # 1. {sensor}-{subdir_num}.json  (standard GMIND naming)
        p = subdir / f"{self.sensor}-{subdir_num}.json"
        if p.exists():
            return p

        # 2. *_sam3_annotations.json  (storage naming)
        matches = list(subdir.glob("*_sam3_annotations.json"))
        if matches:
            return matches[0]

        # 3. *_annotations.json  (generic annotation naming)
        matches = list(subdir.glob("*_annotations.json"))
        if matches:
            return matches[0]

        # 4. Any .json in the subdir
        matches = list(subdir.glob("*.json"))
        if matches:
            return matches[0]

        return None


def get_isp_dataloader(
    data_root: Union[str, Path],
    isp_variant: Optional[str] = None,
    sets: Optional[List[str]] = None,
    sensor: str = "FLIR8.9",
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
    """Create a DataLoader wrapping :class:`ISPVariantDataset`.

    Mirrors the signature of :func:`DataLoader.gmind_dataset.get_gmind_dataloader`
    with the addition of *isp_variant*.
    """
    dataset = ISPVariantDataset(
        data_root=data_root,
        isp_variant=isp_variant,
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
