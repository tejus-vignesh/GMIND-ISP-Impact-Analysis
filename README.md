# Same Scene, Different Pipeline: ISP Impact on Automotive Detection at Range

This repository contains the code accompanying the paper:

> **Same Scene, Different Pipeline: ISP Impact on Automotive Detection at Range**
> Tejus Vijayakumar, Tim Brophy, Brian Deegan, Ciarán Eising, and Patrick Denny
> *Submitted to Electronic Imaging 2026 (EI2026)*


We systematically evaluate how Image Signal Processor (ISP) parameter variation affects nighttime object detection across four detector architectures and five distance bins (0–75 m), using raw Bayer data from the [G-MIND dataset](https://ieee-dataport.org/documents/galway-multimodal-infrastructure-node-dataset). 23 ISP configurations are produced by sweeping gain, gamma correction, saturation, bilateral noise filtering, and edge enhancement (plus raw Bayer with and without gamma correction); each configuration is used to independently fine-tune YOLOv8m, YOLO26m, Faster R-CNN (ResNet-50 FPN), and RT-DETR-L, yielding 92 trained models that are evaluated per-class and per-distance.

## Citation

Will be updated once published.

## Reproducing the paper

The end-to-end reproduction has three stages. Each stage links to a sub-module README with full command reference and a list of hyperparameters / config keys to change.

1. **Generate the 23 ISP variants** — see [`ImageSignalProcessing/`](ImageSignalProcessing/README.md).
   Run the default ISP, then sweep individual blocks (`gac.gamma`, `gac.gain`, `hsc.saturation_gain`) or batch multi-parameter variants (BNF, EEH) onto every raw frame in the nighttime subset.

2. **Fine-tune the four detector architectures** — see [`DeepLearning/`](DeepLearning/README.md).
   `python -m DeepLearning.train_models` with `--use-gmind`, `--gmind-config SensitivityAnalysis/sensitivity_config.yaml`, and `--isp-variant <name>` fine-tunes a model on a single ISP variant. Repeat for all 23 variants × 4 architectures.

3. **Distance-binned evaluation, plotting, and analysis** — see [`SensitivityAnalysis/`](SensitivityAnalysis/README.md) and [`Evaluation/`](Evaluation/README.md).
   Use `--eval-only --bin-distance` to compute per-class AP across the 0–15, 15–30, 30–45, 45–60, 60–75 m bins, then aggregate with `SensitivityAnalysis/plot_sensitivity_results.py` and render qualitative frames with `Evaluation/visualisation/save_gt_and_pred_frames.py`.

### Paper section → code artefact

| Paper section | Artefact | File / command |
|---|---|---|
| Tables 2–4 (gain / gamma / saturation) | Single-parameter sweep | `ImageSignalProcessing/run_isp_sweep.py --sweep "<block.param>:<vals>"` |
| Tables 5–6 (BNF / EEH multi-param) | Multi-parameter batch | `ImageSignalProcessing/run_isp_sweep.py --batch "<block.param>=<val>,..."` |
| Table 7 (detector architectures) | Training entry-point | `python -m DeepLearning.train_models --model <name> --backend <ultralytics\|torchvision> ...` |
| Figure 4 (ΔmAP heatmap) | Sensitivity plot | `SensitivityAnalysis/plot_sensitivity_results.py` |
| Figures 6–7 (per-class AP vs distance) | Distance-binned eval + plot | `train_models.py --eval-only --bin-distance` → `plot_sensitivity_results.py` |
| Figures 8–9 (qualitative) | Frame visualiser | `Evaluation/visualisation/save_gt_and_pred_frames.py` |

## Module Documentation

Modules used directly to produce the paper are listed first:

- **[ImageSignalProcessing/](ImageSignalProcessing/README.md)** — ISP pipeline (vendored, modified [fast-openISP](https://github.com/QiuJueqin/fast-openISP)) and the variant sweep tool used to produce the 23 ISP configurations.
- **[DeepLearning/](DeepLearning/README.md)** — Multi-backend training framework (TorchVision, Ultralytics, MMDetection) and the `--eval-only --bin-distance` evaluation entry-point.
- **[SensitivityAnalysis/](SensitivityAnalysis/README.md)** — Config (`sensitivity_config.yaml`), ISP variant naming convention, and aggregation/plotting scripts.
- **[Evaluation/](Evaluation/README.md)** — COCO-format evaluation tools, GT generation, and visualisation utilities for the qualitative figures.

Supporting modules:

- **[Annotation/](Annotation/README.md)** — Video annotation generation (object detection + tracking → COCO).
- **[Calibration/](Calibration/README.md)** — Camera and sensor calibration tools.
- **[DataLoader/](DataLoader/README.md)** — PyTorch DataLoader for the G-MIND format.
- **[TimeSync/](TimeSync/README.md)** — Temporal synchronisation utilities.
- **[Validation/](Validation/README.md)** — Sensor-fusion validation and visualisation.
- **[tests/](tests/README.md)** — Test suite.

## Getting Started

```sh
# Clone with submodules
git clone --recurse-submodules https://github.com/tejus-vignesh/GMIND-ISP-Impact-Analysis.git
cd GMIND-ISP-Impact-Analysis

# Or, if already cloned, initialise submodules
git submodule update --init --recursive

# Install
pip install -e ".[ultralytics,eval]"   # ISP, training (Ultralytics), and COCO eval
# pip install -e ".[all]"              # everything (incl. MMDetection)
```

Submodules used:
- `Calibration/` — Sensor Calibration Toolbox
- `OC_SORT/` — OC-SORT tracker (used by the Annotation and Evaluation modules)

For the paper reproduction you do not need to download any extra model weights — Ultralytics and TorchVision fetch the COCO-pretrained backbones automatically on first use.

## Annotation generation (supporting module)

The repository also includes an end-to-end video annotation pipeline (`Annotation/annotation_generation.py`) that runs Dome-DETR or YOLOv12x as a detector, tracks objects with OC-SORT, optionally lifts boxes to 3D via ground-plane intersection, and writes COCO-format JSON. This is not exercised by the EI2026 paper but is part of the underlying G-MIND SDK. See [`Annotation/README.md`](Annotation/README.md) for the full configuration and usage walkthrough.

## Contributing

Contributions are welcome. Please open issues or pull requests for bug fixes, new features, or improvements.

## Third-Party Code and Attributions

- **Submodules**
  - `OC_SORT/` — OC-SORT tracker (see `OC_SORT/LICENSE`)
  - `Calibration/` — Sensor Calibration Toolbox
- **Heavily modified third-party code**
  - `ImageSignalProcessing/` — based on [fast-openISP](https://github.com/QiuJueqin/fast-openISP) by Qiu Jueqin (MIT)
- **License**: MIT. Third-party components retain their original licenses.

## Forked From

This repository is forked from [daramolloy/GMIND-sdk](https://github.com/daramolloy/GMIND-sdk), the official toolkit for the G-MIND dataset.
