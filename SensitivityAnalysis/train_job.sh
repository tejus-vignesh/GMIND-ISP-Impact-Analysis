#!/bin/bash
#SBATCH --job-name=isp_sensitivity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#
# Standalone SLURM job template for ISP sensitivity training.
#
# Usage:
#   sbatch SensitivityAnalysis/train_job.sh <isp_variant> <data_root> <results_dir> [model] [epochs]
#
# Example:
#   sbatch SensitivityAnalysis/train_job.sh gac_gain-1024 /storage/data /results yolov8m 50
#
# The script passes --checkpoint-dir as the base results directory.
# train_models.py auto-creates {model}/{variant}/ underneath.

ISP_VARIANT="${1:?Usage: sbatch train_job.sh <isp_variant> <data_root> <results_dir> [model] [epochs]}"
DATA_ROOT="${2:?Missing data_root}"
OUTPUT_DIR="${3:?Missing results_dir}"
MODEL="${4:-yolov8m}"
EPOCHS="${5:-50}"

echo "============================================"
echo "ISP Sensitivity Training Job"
echo "  Variant : ${ISP_VARIANT}"
echo "  Data    : ${DATA_ROOT}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Model   : ${MODEL}"
echo "  Epochs  : ${EPOCHS}"
echo "  Host    : $(hostname)"
echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

module load miniconda/2405

conda run -n gmind python -m DeepLearning.train_models \
    --use-gmind \
    --gmind-config SensitivityAnalysis/sensitivity_config.yaml \
    --isp-variant "${ISP_VARIANT}" \
    --model "${MODEL}" \
    --backend auto \
    --epochs "${EPOCHS}" \
    --checkpoint-dir "${OUTPUT_DIR}" \
    --device cuda
