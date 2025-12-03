"""Test suite for GMIND SDK detection models and training pipeline.

Tests cover:
- Model loading from all backends (TorchVision, Ultralytics, MMDetection, Detectron2)
- Training loops with mixed precision
- Inference and evaluation
- Checkpoint save/load
- Dataset handling
"""
