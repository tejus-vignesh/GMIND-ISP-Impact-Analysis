## D.Molloy
## run_isp.py - Flexible RAW Image/Video Batch Processing Script
#
# This script provides three main modes:
# 1. Batch process multiple sets of images in parallel
# 2. Process, display, and optionally save a single image for quick testing
#
# Features:
#   - Robust config path handling (relative to script)
#   - Flexible batch processing with per-set config
#   - Direct ffmpeg streaming for video output
#   - User-friendly toggles for output (video, PNG)
#   - Auto-tuning for optimal multiprocessing
#   - User-settable video framerate
#
# Usage:
#   - Edit the __main__ section at the bottom to select your mode and provide paths.
#   - Requires: OpenCV, numpy, rawpy, tqdm, ffmpeg-python, and your custom pipeline/config modules.
#
# Example:
#   python run_isp.py
#
# Author: D. Molloy
# Date: 2025-06-26
#

import glob
import multiprocessing
import os
import shutil
import time
from collections import OrderedDict

import cv2
import ffmpeg
import numpy as np
import rawpy
from pipeline import Pipeline
from tqdm import tqdm
from utils.yacs import Config

# Set the default config directory globally
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# Global toggles for output
SAVE_VIDEO = True
SAVE_PNG = True

# Global setting for number of processes in batch pipeline
NUM_PROCESSES = 12  # Set to "Auto" to enable auto-tuning, or an integer for manual

# Global setting for video framerate (user must set this)
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


def process_image_dict(image_dict, out_dir):
    """
    Batch process multiple sets of images, each with its own config, and save output videos for each using ffmpeg-python (libx264, crf 12), streaming frames directly.
    Args:
        image_dict: dict where key = config name, value = list of image paths or a single directory
        out_dir: directory to save output videos, or 'Parent' to use parent of input directory
    """
    for cfg_name, image_list in tqdm(image_dict.items(), desc="Configs", unit="config"):
        cfg_filename = cfg_name + ".yaml"
        cfg_path = os.path.join(CONFIG_DIR, cfg_filename)
        set_name = cfg_name  # e.g. 'your_configA'
        # If image_list is a single directory, expand to all PNGs in that directory
        if isinstance(image_list, str) or (
            isinstance(image_list, list) and len(image_list) == 1 and os.path.isdir(image_list[0])
        ):
            input_dir = image_list if isinstance(image_list, str) else image_list[0]
            print(
                f"\n[DEBUG] Expanding directory for set '{set_name}':\n    Input directory: {input_dir}"
            )
            image_list = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        print(
            f"\n[DEBUG] Image list for set '{set_name}':\n    {len(image_list)} images\n    First 5: {image_list[:5] if image_list else '[]'}\n    ...\n"
        )
        # Determine output directory
        if out_dir == "Parent" and image_list:
            # Use parent of first image's directory
            parent_dir = os.path.dirname(os.path.dirname(image_list[0]))
            this_out_dir = parent_dir
        else:
            this_out_dir = out_dir
        os.makedirs(this_out_dir, exist_ok=True)
        print(
            f"[DEBUG] Output directory for set '{set_name}':\n    Config: {cfg_path}\n    Output PNGs: {os.path.abspath(os.path.join(this_out_dir, f'{set_name}-HV'))}\n    Output video: {os.path.abspath(os.path.join(this_out_dir, f'{set_name}.mp4')) if SAVE_VIDEO else '[SKIPPED]'}\n"
        )
        pipeline = Pipeline(Config(cfg_path))
        save_dir = os.path.join(this_out_dir, f"{set_name}-HV")
        os.makedirs(save_dir, exist_ok=True)
        pipeline.batch_run(
            image_list, save_dir, load_bayer, suffixes="", num_processes=NUM_PROCESSES
        )
        images_out = glob.glob(f"{save_dir}/*.png")
        print(
            f"[DEBUG] Processed PNGs for set '{set_name}':\n    {len(images_out)} images\n    First 5: {images_out[:5] if images_out else '[]'}\n    ...\n"
        )
        img_info = [
            [os.path.basename(img), float(os.path.basename(img).split("-")[0])]
            for img in images_out
        ]
        img_info.sort(key=lambda x: x[1])
        if not img_info:
            print(f"[DEBUG] No images found for set '{set_name}'. Skipping.\n")
            shutil.rmtree(save_dir)
            continue
        # Get frame size from first image
        first_img = cv2.imread(os.path.join(save_dir, img_info[0][0]))
        h, w = first_img.shape[:2]
        if SAVE_VIDEO:
            out_video_path = os.path.join(this_out_dir, f"{set_name}.mp4")
            print(
                f"[DEBUG] Starting video writing for set '{set_name}':\n    Output: {out_video_path}\n    Frame size: {w}x{h}\n    Framerate: {VIDEO_FRAMERATE}\n"
            )
            process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    s=f"{w}x{h}",
                    framerate=VIDEO_FRAMERATE,
                )
                .output(out_video_path, vcodec="libx264", crf=12, pix_fmt="yuv420p")
                .overwrite_output()
                .global_args("-loglevel", "error")  # Suppress ffmpeg output
                .run_async(pipe_stdin=True)
            )
        for idx, img in enumerate(tqdm(img_info, desc=f"Frames ({set_name})", unit="frame")):
            frame_path = os.path.join(save_dir, img[0])
            frame = cv2.imread(frame_path)
            if frame.shape[:2] != (h, w):
                print(
                    f"[DEBUG]    Resizing frame {os.path.basename(frame_path)}: {frame.shape[:2]} -> ({h}, {w})"
                )
                frame = cv2.resize(frame, (w, h))
            if SAVE_VIDEO:
                process.stdin.write(frame.astype(np.uint8).tobytes())
        if SAVE_VIDEO:
            process.stdin.close()
            process.wait()
            print(f"[DEBUG] Video saved for set '{set_name}': {out_video_path}\n")

        if not SAVE_PNG:
            print(f"[DEBUG] Removing intermediate PNG directory for set '{set_name}': {save_dir}\n")
            shutil.rmtree(save_dir)
    print(f"\n[DEBUG] All videos saved to: {out_dir if out_dir != 'Parent' else this_out_dir}\n")


def process_and_show_single_image(image_path, config_path, out_path=None):
    """
    Process a single image using the pipeline, display it, and optionally save the result.
    Args:
        image_path: Path to the input image (raw or png)
        config_path: Path to the config yaml
        out_path: Optional path to save the processed image (PNG)
    """
    pipeline = Pipeline(Config(config_path))
    temp_dir = "./_single_image_temp/"
    os.makedirs(temp_dir, exist_ok=True)
    # Use the same loader as batch, but for a single image
    pipeline.batch_run([image_path], temp_dir, load_bayer, suffixes="", num_processes=1)
    # Find the output image
    out_imgs = glob.glob(f"{temp_dir}/*.png")
    if not out_imgs:
        print("No output image generated.")
        shutil.rmtree(temp_dir)
        return
    img = cv2.imread(out_imgs[0])
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if out_path:
        cv2.imwrite(out_path, img)
        print(f"Saved processed image to {out_path}")
    shutil.rmtree(temp_dir)


def find_optimal_num_processes(
    raw_paths, save_dir, load_raw_fn, pipeline_class, config, max_procs=None, test_count=18
):
    """
    Benchmark different process counts and return the optimal number for batch_run.
    Args:
        raw_paths: list of image paths (use a small subset for testing)
        save_dir: directory to save outputs (can be a temp dir)
        load_raw_fn: function to load bayer data
        pipeline_class: the Pipeline class
        config: config object or path
        max_procs: max processes to try (default: cpu_count)
        test_count: number of images to use for benchmarking
    Returns:
        int: optimal number of processes
    """
    if max_procs is None:
        max_procs = multiprocessing.cpu_count()
    test_paths = raw_paths[:test_count]
    best_time = float("inf")
    best_proc = 1
    print("[BENCH] Benchmarking process counts...")
    for n_proc in [4, 6, 8, max_procs]:
        n_proc = min(n_proc, max_procs)
        if n_proc < 1:
            continue
        pipeline = pipeline_class(config)
        start = time.time()
        pipeline.batch_run(
            test_paths, save_dir, load_raw_fn, num_processes=n_proc, show_progress=True
        )
        elapsed = time.time() - start
        print(f"[BENCH] {n_proc} processes: {elapsed:.2f}s")
        if elapsed < best_time:
            best_time = elapsed
            best_proc = n_proc
    print(f"[BENCH] Optimal number of processes: {best_proc}")
    return best_proc


def auto_set_num_processes(image_dict, config_dir, load_raw_fn, pipeline_class):
    """
    If NUM_PROCESSES is set to 'Auto', benchmark and set the optimal number of processes.
    Modifies the global NUM_PROCESSES.
    """
    global NUM_PROCESSES
    first_key = next(iter(image_dict))
    first_list = image_dict[first_key]
    if isinstance(first_list, str) or (
        isinstance(first_list, list) and len(first_list) == 1 and os.path.isdir(first_list[0])
    ):
        input_dir = first_list if isinstance(first_list, str) else first_list[0]
        sample_images = sorted(glob.glob(os.path.join(input_dir, "*.png")))[:18]
    else:
        sample_images = first_list[:20]
    temp_bench_dir = "./_bench_temp/"
    os.makedirs(temp_bench_dir, exist_ok=True)
    NUM_PROCESSES = find_optimal_num_processes(
        sample_images,
        temp_bench_dir,
        load_raw_fn,
        pipeline_class,
        Config(os.path.join(config_dir, first_key + ".yaml")),
        test_count=20,
    )
    shutil.rmtree(temp_bench_dir)
    print(f"[INFO] Using NUM_PROCESSES = {NUM_PROCESSES} for batch processing.\n")


if __name__ == "__main__":
    # ----------------------
    # Select your mode below
    # ----------------------
    # 1. Single image test mode:
    # process_and_show_single_image('path/to/image.png', './configs/your_config.yaml', 'output_test.png')

    # 2. Batch process multiple sets of images using a dict:
    image_dict = {
        "FLIR8.9": ["H:/DanganDataset-Formatted2/Day/DistanceTest/1/FLIR-8.9/PNG"],
    }
    out_dir = "Parent"

    if NUM_PROCESSES == "Auto":
        auto_set_num_processes(image_dict, CONFIG_DIR, load_bayer, Pipeline)
    # ---
    process_image_dict(image_dict, out_dir)
