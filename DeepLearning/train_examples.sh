#!/bin/bash
# Example training commands for GMIND dataset
# These use the gmind_config.yaml configuration file

# ============================================
# YOLOv12m Training (Recommended YOLO model - Latest)
# ============================================
# Latest YOLO version with Area Attention Module (A²) and R-ELAN architecture
# Best balanced speed/accuracy with improved feature aggregation
python DeepLearning/train_models.py \
    --use-gmind \
    --gmind-config DeepLearning/gmind_config.yaml \
    --model yolov12m \
    --backend ultralytics \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.01 \
    --checkpoint-dir ./checkpoints/yolov12m \
    --device cuda

# ============================================
# RT-DETR-l Training (Recommended Transformer)
# ============================================
# Real-time transformer, good speed/accuracy balance
python DeepLearning/train_models.py \
    --use-gmind \
    --gmind-config DeepLearning/gmind_config.yaml \
    --model rtdetr-l \
    --backend ultralytics \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.01 \
    --checkpoint-dir ./checkpoints/rtdetr-l \
    --device cuda

# ============================================
# Alternative: YOLOv12 Variants
# ============================================
# YOLOv12n (nano - fastest, smallest, ~40.6 mAP)
# python DeepLearning/train_models.py \
#     --use-gmind \
#     --gmind-config DeepLearning/gmind_config.yaml \
#     --model yolov12n \
#     --backend ultralytics \
#     --epochs 50 \
#     --batch-size 16 \
#     --lr 0.01 \
#     --checkpoint-dir ./checkpoints/yolov12n \
#     --device cuda

# YOLOv12x (extra large - highest accuracy)
# python DeepLearning/train_models.py \
#     --use-gmind \
#     --gmind-config DeepLearning/gmind_config.yaml \
#     --model yolov12x \
#     --backend ultralytics \
#     --epochs 50 \
#     --batch-size 4 \
#     --lr 0.01 \
#     --checkpoint-dir ./checkpoints/yolov12x \
#     --device cuda

# ============================================
# Alternative: DINO Transformer (via MMDetection)
# ============================================
# Note: Requires MMDetection config file
# Higher accuracy but more complex setup
# python DeepLearning/train_models.py \
#     --use-gmind \
#     --gmind-config DeepLearning/gmind_config.yaml \
#     --model dino \
#     --backend mmdet \
#     --backend-config /path/to/dino_config.py \
#     --epochs 50 \
#     --batch-size 4 \
#     --lr 0.0001 \
#     --checkpoint-dir ./checkpoints/dino \
#     --device cuda

