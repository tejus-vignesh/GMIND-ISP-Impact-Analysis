#
# This script performs parameter sweeps across ISP configurations.
# It supports one-at-a-time sweeps where each parameter is changed independently
# while all others remain at their YAML default values.
#
# Features:
#   - Support for multiple source directories paired 1:1 with output directories
#   - One-at-a-time parameter sweeps (NOT combinations)
#   - Generates both PNG images and MP4 videos for each run
#   - Output directories named after configuration settings
#   - Run logs documenting each sweep configuration
#
# Usage:
#   - Edit the configuration section in __main__ to specify:
#     - SOURCE_DIRS: list of source directories
#     - OUTPUT_DIRS: list of output parent directories (must match SOURCE_DIRS count)
#     - BASE_CONFIG: base config file name from configs/ directory
#     - SWEEP_PARAMS: dict of blocks and their parameter value ranges
#   - Run: python run_isp_sweep.py


import glob
import logging
import os
import shutil
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import ffmpeg
import numpy as np
import rawpy
from pipeline import Pipeline
from tqdm import tqdm
from utils.yacs import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set the default config directory globally
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# Global toggles for output
SAVE_VIDEO = True
SAVE_PNG = True

# Global setting for number of processes in batch pipeline
NUM_PROCESSES = 12

# Global setting for video framerate
VIDEO_FRAMERATE = 30


def getWhiteBalanceGains(rawFile, gainMult):
    """
    Extract white balance gains from a raw file and return as integers.
    Args:
        rawFile: rawpy object
        gainMult: gain factor correction (e.g. 256=1 or 1024=1)
    Returns:
        [redGain, greenGain, blueGain]
    """
    (redGain, greenGain, blueGain, offset) = rawFile.camera_whitebalance
    redGain = np.multiply(redGain, gainMult).astype("u2")
    greenGain = np.multiply(greenGain, gainMult).astype("u2")
    blueGain = np.multiply(blueGain, gainMult).astype("u2")
    return [redGain, greenGain, blueGain]


def load_bayer(raw_path):
    """
    Load a Bayer image from PNG or raw file.
    Args:
        raw_path: path to image file
    Returns:
        bayer: numpy array
    """
    data = OrderedDict()
    if ".png" in raw_path:
        # For PNG, just load as grayscale
        bayer = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    else:
        # For raw, extract visible image and white balance
        redGain, greenGain, blueGain = 1024, 1024, 1024
        raw = rawpy.imread(raw_path)
        bayer = np.asarray(raw.raw_image_visible)
        [redGain, greenGain, blueGain] = getWhiteBalanceGains(raw, gainMult=4)
        data["RGB"] = [redGain, greenGain, blueGain]
    data["bayer"] = bayer
    return bayer


def generate_sweep_runs(sweep_params):
    """
    Generate individual parameter runs (NOT combinations).
    Each run changes ONE parameter while others stay at defaults.

    Args:
        sweep_params: dict of {block_name: {param_name: [values]}}

    Returns:
        list of tuples: [(block_name, param_name, value), ...]

    Example:
        Input: {"gac": {"gamma": [0.3, 0.4]}, "bnf": {"sigma": [0.5, 0.8]}}
        Output: [
            ("gac", "gamma", 0.3),
            ("gac", "gamma", 0.4),
            ("bnf", "sigma", 0.5),
            ("bnf", "sigma", 0.8),
        ]
    """
    runs = []
    for block_name, params in sweep_params.items():
        for param_name, values in params.items():
            for val in values:
                runs.append((block_name, param_name, val))
    return runs


def generate_dir_name(block_name, param_name, value):
    """
    Create descriptive directory name from single parameter.

    Args:
        block_name: ISP block name (e.g., "gac")
        param_name: parameter name (e.g., "gamma")
        value: parameter value (e.g., 0.45)

    Returns:
        str: directory name (e.g., "gac_gamma-0.45")
    """
    return f"{block_name}_{param_name}-{value}"


def create_config_with_single_param(base_config_path, block_name, param_name, value):
    """
    Load base config and override ONE parameter.
    All other params stay at their YAML default values.

    Args:
        base_config_path: path to base YAML config
        block_name: ISP block name to modify
        param_name: parameter name to modify
        value: new value for the parameter

    Returns:
        Config: modified config object
    """
    cfg = Config(base_config_path)
    with cfg.unfreeze():
        cfg[block_name][param_name] = value
    return cfg


def save_run_log(out_dir, dir_name, block_name, param_name, value, base_config, source_dir, image_count, elapsed_time):
    """
    Save a log file documenting the run parameters.

    Args:
        out_dir: output directory
        dir_name: configuration directory name
        block_name: ISP block name
        param_name: parameter name
        value: parameter value
        base_config: base config filename
        source_dir: source directory path
        image_count: number of images processed
        elapsed_time: processing time in seconds
    """
    log_path = os.path.join(out_dir, f"{dir_name}_run.log")
    with open(log_path, "w") as f:
        f.write("ISP Parameter Sweep Run Log\n")
        f.write("===========================\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Base Config: {base_config}\n")
        f.write(f"Source Directory: {source_dir}\n")
        f.write(f"Images Processed: {image_count}\n")
        f.write(f"\nParameter Override:\n")
        f.write(f"  Block: {block_name}\n")
        f.write(f"  Parameter: {param_name}\n")
        f.write(f"  Value: {value}\n")
        f.write(f"\nProcessing Time: {elapsed_time:.2f}s\n")
    logger.debug(f"Run log saved: {log_path}")


def create_video_from_pngs(save_dir, out_parent, dir_name, video_framerate):
    """
    Create MP4 video from processed PNGs using ffmpeg streaming.

    Args:
        save_dir: directory containing processed PNGs
        out_parent: parent output directory for video
        dir_name: base name for video file
        video_framerate: framerate for output video
    """
    images_out = glob.glob(f"{save_dir}/*.png")
    if not images_out:
        logger.warning(f"No images found in {save_dir} for video creation.")
        return

    # Sort images by timestamp (extracted from filename)
    img_info = [
        [os.path.basename(img), float(os.path.basename(img).split("-")[0])]
        for img in images_out
    ]
    img_info.sort(key=lambda x: x[1])

    # Get frame size from first image
    first_img = cv2.imread(os.path.join(save_dir, img_info[0][0]))
    h, w = first_img.shape[:2]

    out_video_path = os.path.join(out_parent, f"{dir_name}.mp4")
    logger.info(f"Creating video: {out_video_path} ({w}x{h} @ {video_framerate}fps)")

    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{w}x{h}",
            framerate=video_framerate,
        )
        .output(out_video_path, vcodec="libx264", crf=12, pix_fmt="yuv420p")
        .overwrite_output()
        .global_args("-loglevel", "error")
        .run_async(pipe_stdin=True)
    )

    for img in tqdm(img_info, desc=f"Writing video frames", unit="frame", leave=False):
        frame_path = os.path.join(save_dir, img[0])
        frame = cv2.imread(frame_path)
        if frame.shape[:2] != (h, w):
            logger.debug(f"Resizing frame {img[0]}: {frame.shape[:2]} -> ({h}, {w})")
            frame = cv2.resize(frame, (w, h))
        process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()
    logger.info(f"Video saved: {out_video_path}")


def process_sweep(source_dirs, output_dirs, base_config, sweep_params):
    """
    Main processing function for parameter sweeps.

    Args:
        source_dirs: list of source directories
        output_dirs: list of output parent directories (must match source_dirs count)
        base_config: base config filename (e.g., "FLIR8.9.yaml")
        sweep_params: dict of {block_name: {param_name: [values]}}
    """
    # Validate input/output directory pairing
    assert len(source_dirs) == len(output_dirs), \
        f"Source and output dirs must be 1:1. Got {len(source_dirs)} sources and {len(output_dirs)} outputs."

    # Generate sweep runs
    runs = generate_sweep_runs(sweep_params)
    total_param_configs = len(runs)
    total_runs = total_param_configs * len(source_dirs)

    logger.info(f"Parameter sweep configuration:")
    logger.info(f"  Base config: {base_config}")
    logger.info(f"  Source directories: {len(source_dirs)}")
    logger.info(f"  Parameter configurations: {total_param_configs}")
    logger.info(f"  Total processing runs: {total_runs}")

    base_config_path = os.path.join(CONFIG_DIR, base_config)
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    # Process each parameter configuration
    for run_idx, (block_name, param_name, value) in enumerate(tqdm(runs, desc="Parameter sweeps", unit="config")):
        dir_name = generate_dir_name(block_name, param_name, value)
        cfg = create_config_with_single_param(base_config_path, block_name, param_name, value)

        logger.info(f"[{run_idx + 1}/{total_param_configs}] Processing: {block_name}.{param_name} = {value}")

        # Process each source directory with this configuration
        for src_idx, (src_dir, out_parent) in enumerate(zip(source_dirs, output_dirs)):
            logger.info(f"  Source [{src_idx + 1}/{len(source_dirs)}]: {src_dir}")

            # Get images from source directory
            image_list = sorted(glob.glob(os.path.join(src_dir, "*.png")))
            if not image_list:
                logger.warning(f"  No PNG images found in {src_dir}. Skipping.")
                continue

            logger.info(f"  Found {len(image_list)} images")

            # Create output directories (config_dir contains PNGs, video, and log)
            config_dir = os.path.join(out_parent, dir_name)
            os.makedirs(config_dir, exist_ok=True)
            save_dir = os.path.join(config_dir, f"{dir_name}-HV")
            os.makedirs(save_dir, exist_ok=True)

            # Save the config used for this run
            cfg.dump(os.path.join(config_dir, f"{dir_name}_config.yaml"))

            # Process images with pipeline
            start_time = time.time()
            pipeline = Pipeline(cfg)
            pipeline.batch_run(
                image_list, save_dir, load_bayer, suffixes="", num_processes=NUM_PROCESSES
            )
            elapsed_time = time.time() - start_time

            # Verify processing
            processed_images = glob.glob(f"{save_dir}/*.png")
            if not processed_images:
                logger.warning(f"  No output images generated. Skipping video and cleanup.")
                shutil.rmtree(save_dir)
                continue

            logger.info(f"  Processed {len(processed_images)} images in {elapsed_time:.2f}s")

            # Save run log
            save_run_log(
                config_dir, dir_name, block_name, param_name, value,
                base_config, src_dir, len(processed_images), elapsed_time
            )

            # Generate video
            if SAVE_VIDEO:
                create_video_from_pngs(save_dir, config_dir, dir_name, VIDEO_FRAMERATE)

            # Clean up PNGs if not saving
            if not SAVE_PNG:
                logger.debug(f"  Removing intermediate PNG directory: {save_dir}")
                shutil.rmtree(save_dir)

    logger.info("Parameter sweep completed.")


if __name__ == "__main__":

    # Source directories - must match count of OUTPUT_DIRS
    SOURCE_DIRS = [
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/1/RAW_Images/FLIR8.9/",
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/2/RAW_Images/FLIR8.9/",  
    ]

    # Output parent directories 
    OUTPUT_DIRS = [
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/1/Processed_Images/FLIR8.9/",
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/2/Processed_Images/FLIR8.9/",
    ]

    # Base config file to modify (from configs/ directory)
    BASE_CONFIG = "FLIR8.9.yaml"

    # Parameter sweep configuration
    # Each parameter is swept INDEPENDENTLY while others stay at default
    SWEEP_PARAMS = {
        "gac": {
            "gamma": [0.3, 0.4, 0.45, 0.5],  # 4 runs with only gamma changed
            "gain": [8, 32, 512, 1024]
        },
        # "bnf": {
        #     "intensity_sigma": [8, 32, 512, 1024],
        # }
    }

    process_sweep(SOURCE_DIRS, OUTPUT_DIRS, BASE_CONFIG, SWEEP_PARAMS)
