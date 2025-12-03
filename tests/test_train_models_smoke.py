import logging
import os
import subprocess
import sys

import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

COCO_DIR = "coco_sample"
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

SUPPORTED_MODELS = [
    # All torchvision supported detection models
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "keypointrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    # Ultralytics YOLO models
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    # MMDetection models (example names, update as needed)
    "mmdet_faster_rcnn_r50_fpn",
    "mmdet_mask_rcnn_r50_fpn",
    "mmdet_retinanet_r50_fpn",
    "mmdet_ssd300",
    "mmdet_yolox_s",
    # DETR models
    "detr_resnet50",
    "detr_resnet101",
]


@pytest.fixture(scope="session", autouse=True)
def setup_coco_sample():
    os.makedirs(COCO_DIR, exist_ok=True)
    val_zip = os.path.join(COCO_DIR, "val2017.zip")
    ann_zip = os.path.join(COCO_DIR, "annotations_trainval2017.zip")
    val_dir = os.path.join(COCO_DIR, "val2017")
    ann_dir = os.path.join(COCO_DIR, "annotations")
    # Download if missing
    if not os.path.exists(val_zip):
        subprocess.run(["wget", COCO_VAL_URL, "-O", val_zip], check=True)
    if not os.path.exists(ann_zip):
        subprocess.run(["wget", COCO_ANN_URL, "-O", ann_zip], check=True)
    # Extract if missing
    if not os.path.exists(val_dir):
        subprocess.run(["unzip", val_zip, "-d", COCO_DIR], check=True)
    if not os.path.exists(ann_dir):
        subprocess.run(["unzip", ann_zip, "-d", COCO_DIR], check=True)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_train_model_smoke(model_name):
    logger.info("=== Testing model: %s ===", model_name)
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "DeepLearning.train_models",
        "--data",
        COCO_DIR,
        "--model",
        model_name,
        "--epochs",
        "1",
        "--eval-subset",
        "10",
        "--device",
        "cpu",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logger.debug(line.rstrip())
    process.wait()
    assert process.returncode == 0, f"Model {model_name} failed."
