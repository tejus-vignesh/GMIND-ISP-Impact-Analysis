#
# VIDEO-ONLY version of run_isp_sweep.py (Generated from already-processed PNGs)
#
# This script regenerates videos from already-processed PNGs without
# re-running the ISP pipeline. Use this when ISP processing completed
# but video generation failed (e.g., ffmpeg issues on cluster).
#
# Run with the same --sweep and --config args as the original run so that
# directory names are reconstructed correctly.
#
# Usage:
#   python run_isp_sweep_video.py --sweep "gac.gamma:0.3,0.4,0.45,0.5"
#   python run_isp_sweep_video.py --config FLIR8.9.yaml --sweep "gac.gamma:0.3,0.4,0.5"

import argparse
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
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

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


def parse_sweep_arg(sweep_str):
    """
    Parse a single sweep string into (block, param, values).

    Args:
        sweep_str: string like "gac.gamma:0.3,0.4,0.5"

    Returns:
        tuple: (block_name, param_name, [values])

    Raises:
        ValueError: if format is invalid
    """
    if ":" not in sweep_str or "." not in sweep_str.split(":")[0]:
        raise ValueError(
            f"Invalid sweep format: '{sweep_str}'. "
            f"Expected format: 'block.param:val1,val2,...' (e.g., 'gac.gamma:0.3,0.4,0.5')"
        )

    param_part, values_part = sweep_str.split(":", 1)
    block_name, param_name = param_part.split(".", 1)

    # Parse values with auto type detection
    values = []
    for val_str in values_part.split(","):
        val_str = val_str.strip()
        # Try int first, then float, then string
        try:
            values.append(int(val_str))
        except ValueError:
            try:
                values.append(float(val_str))
            except ValueError:
                values.append(val_str)

    return block_name, param_name, values


def build_sweep_params(sweep_args):
    """
    Build SWEEP_PARAMS dict from list of sweep argument strings.

    Args:
        sweep_args: list of strings like ["gac.gamma:0.3,0.4", "gac.gain:8,32"]

    Returns:
        dict: {block_name: {param_name: [values]}}
    """
    sweep_params = {}
    for sweep_str in sweep_args:
        block_name, param_name, values = parse_sweep_arg(sweep_str)
        if block_name not in sweep_params:
            sweep_params[block_name] = {}
        sweep_params[block_name][param_name] = values
    return sweep_params


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with 'sweep' (list) and 'config' (str)
    """
    parser = argparse.ArgumentParser(
        description="ISP Parameter Sweep VIDEO-ONLY Tool - Regenerate videos from existing PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single parameter sweep
  python run_isp_sweep_video.py --sweep "gac.gamma:0.3,0.4,0.45,0.5"

  # Multiple parameter sweeps
  python run_isp_sweep_video.py --sweep "gac.gamma:0.3,0.4,0.5" --sweep "gac.gain:8,32,512"

  # With custom base config
  python run_isp_sweep_video.py --config FLIR8.9.yaml --sweep "gac.gamma:0.3,0.4,0.5"
        """
    )
    parser.add_argument(
        "-s", "--sweep",
        action="append",
        required=True,
        metavar="BLOCK.PARAM:VAL1,VAL2,...",
        help="Sweep specification. Format: block.param:val1,val2,... (can be repeated)"
    )
    parser.add_argument(
        "-c", "--config",
        default="FLIR8.9.yaml",
        metavar="FILE",
        help="Base config file from configs/ directory (default: FLIR8.9.yaml)"
    )
    return parser.parse_args()


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
        .run_async(pipe_stdin=True, pipe_stderr=True)
    )

    try:
        for img in tqdm(img_info, desc=f"Writing video frames", unit="frame", leave=False):
            frame_path = os.path.join(save_dir, img[0])
            frame = cv2.imread(frame_path)
            if frame.shape[:2] != (h, w):
                logger.debug(f"Resizing frame {img[0]}: {frame.shape[:2]} -> ({h}, {w})")
                frame = cv2.resize(frame, (w, h))
            process.stdin.write(frame.astype(np.uint8).tobytes())
    except BrokenPipeError:
        stderr_output = process.stderr.read().decode() if process.stderr else "no stderr captured"
        logger.error(f"ffmpeg process died. stderr: {stderr_output}")
        process.wait()
        raise RuntimeError(f"ffmpeg failed for {out_video_path}: {stderr_output}")

    process.stdin.close()
    process.wait()
    stderr_output = process.stderr.read().decode() if process.stderr else ""
    if process.returncode != 0:
        logger.error(f"ffmpeg exited with code {process.returncode}: {stderr_output}")
        raise RuntimeError(f"ffmpeg failed for {out_video_path}: {stderr_output}")
    logger.info(f"Video saved: {out_video_path}")


def process_sweep(source_dirs, output_dirs, base_config, sweep_params):
    """
    Video-only processing function for parameter sweeps.
    Finds existing processed PNGs and generates videos from them.

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

    logger.info(f"Parameter sweep configuration (VIDEO-ONLY):")
    logger.info(f"  Base config: {base_config}")
    logger.info(f"  Source directories: {len(source_dirs)}")
    logger.info(f"  Parameter configurations: {total_param_configs}")
    logger.info(f"  Total video generation runs: {total_runs}")

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

            # --- SKIPPED: Getting image list from source dir (not needed for video-only) ---
            # image_list = sorted(glob.glob(os.path.join(src_dir, "*.png")))
            # if not image_list:
            #     logger.warning(f"  No PNG images found in {src_dir}. Skipping.")
            #     continue
            # logger.info(f"  Found {len(image_list)} images")

            # Output directories (config_dir contains PNGs, video, and log)
            config_dir = os.path.join(out_parent, dir_name)
            save_dir = os.path.join(config_dir, f"{dir_name}-HV")

            # --- SKIPPED: Creating directories (already exist from ISP run) ---
            # os.makedirs(config_dir, exist_ok=True)
            # os.makedirs(save_dir, exist_ok=True)

            # --- SKIPPED: Config dump (already done in ISP run) ---
            # cfg.dump(os.path.join(config_dir, f"{dir_name}_config.yaml"))

            # --- SKIPPED: ISP pipeline processing ---
            # start_time = time.time()
            # pipeline = Pipeline(cfg)
            # pipeline.batch_run(
            #     image_list, save_dir, load_bayer, suffixes="", num_processes=NUM_PROCESSES
            # )
            # elapsed_time = time.time() - start_time

            # --- SKIPPED: Processed image verification and timing ---
            # processed_images = glob.glob(f"{save_dir}/*.png")
            # if not processed_images:
            #     logger.warning(f"  No output images generated. Skipping video and cleanup.")
            #     shutil.rmtree(save_dir)
            #     continue
            # logger.info(f"  Processed {len(processed_images)} images in {elapsed_time:.2f}s")

            # Check that processed PNGs exist in save_dir
            if not os.path.isdir(save_dir):
                logger.warning(f"  Output directory does not exist: {save_dir}. Skipping.")
                continue
            existing_pngs = glob.glob(f"{save_dir}/*.png")
            if not existing_pngs:
                logger.warning(f"  No PNGs found in {save_dir}. Skipping.")
                continue
            logger.info(f"  Found {len(existing_pngs)} existing processed PNGs")

            # --- SKIPPED: Save run log (already done in ISP run) ---
            # save_run_log(
            #     config_dir, dir_name, block_name, param_name, value,
            #     base_config, src_dir, len(processed_images), elapsed_time
            # )

            # Generate video
            if SAVE_VIDEO:
                create_video_from_pngs(save_dir, config_dir, dir_name, VIDEO_FRAMERATE)

            # --- SKIPPED: PNG cleanup (don't delete the source images!) ---
            # if not SAVE_PNG:
            #     logger.debug(f"  Removing intermediate PNG directory: {save_dir}")
            #     shutil.rmtree(save_dir)

    logger.info("Video generation sweep completed.")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Build sweep parameters from CLI arguments
    SWEEP_PARAMS = build_sweep_params(args.sweep)

    # Log parsed sweep parameters
    logger.info("Parsed sweep parameters:")
    for block, params in SWEEP_PARAMS.items():
        for param, values in params.items():
            logger.info(f"  {block}.{param}: {values}")

    # Source directories - must match count of OUTPUT_DIRS
    SOURCE_DIRS = [
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/1/RAW_Images/FLIR8.9/",
        # "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/2/RAW_Images/FLIR8.9/",
    ]

    # Output parent directories
    OUTPUT_DIRS = [
        "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/1/Processed_Images/FLIR8.9/",
        # "/home/tejus/Workspace/GMIND-ISP-Impact-Analysis/SampleData/NightUrbanJunction/2/Processed_Images/FLIR8.9/",
    ]

    process_sweep(SOURCE_DIRS, OUTPUT_DIRS, args.config, SWEEP_PARAMS)
