# Sensitivity Analysis

This directory orchestrates the per-variant fine-tuning and result aggregation used in the EI2026 paper *"Same Scene, Different Pipeline: ISP Impact on Automotive Detection at Range"*. It holds the dataset / split / distance-binning configuration consumed by `DeepLearning/train_models.py`, the dataset adapter that loads pre-rendered ISP-variant videos, and the plotting script that produces the ΔmAP heatmap (Fig. 4) and per-class distance-binned line plots (Figs. 6–7).

## Contents

- `sensitivity_config.yaml` — dataset paths, train/val/test splits, sensor selection, frame stride, and distance-evaluation parameters.
- `isp_dataset.py` — `ISPVariantDataset` loader: given a variant name, locates the corresponding pre-processed video file in the dataset tree and yields frames in COCO-style training format.
- `plot_sensitivity_results.py` — reads `eval_results_last_checkpoint.json` from each `(architecture, variant)` and emits overall, per-class, and per-distance-bin comparison plots.

## Workflow

1. **Generate ISP variants** for every cell in paper Tables 2–6 — see `[../ImageSignalProcessing/README.md](../ImageSignalProcessing/README.md)`.
2. **Fine-tune each detector** (YOLOv8m, YOLO26m, Faster R-CNN, RT-DETR-L) on each variant by passing `--gmind-config SensitivityAnalysis/sensitivity_config.yaml --isp-variant <name>` to `python -m DeepLearning.train_models` — see `[../DeepLearning/README.md](../DeepLearning/README.md)`.
3. **Run distance-binned evaluation** on each trained checkpoint with `--eval-only --bin-distance`, writing one `eval_results_last_checkpoint.json` per `(architecture, variant)` under `sensitivity_results/<arch>/<variant>/`.
4. **Plot**:
  ```bash
   python -m SensitivityAnalysis.plot_sensitivity_results
  ```
   No CLI args — edit the configuration block at the top of `plot_sensitivity_results.py` (`RESULTS_ROOT`, `OUTPUT_ROOT`, `MODELS`, `ISP_VARIANTS`, metric selection, exclusions) before running. Output PNGs land in `sensitivity_plots/eval_results_last_checkpoint/` by default.

## Configuration: `sensitivity_config.yaml`

```yaml
data:
  root: <path to G-MIND root>     # parent directory containing the dataset sets
  sensor: "FLIR8.9"               # sensor name; selects the per-camera ISP config
  frame_stride: 30                # sample every Nth frame for train/val/test
  max_frames_per_video: null      # cap per video, or null for no cap

train / validation / test:
  sets: [...]                     # G-MIND set names (e.g. NightUrbanJunction)
  set_subdirs:                    # which subdirectories of each set to include
    "<set>": [...]
  percentage_split:               # fraction of frames per subdir to use
    "<set>/<subdir>": <0..1>
  percentage_split_start:         # (val/test only) starting offset into the subdir

distance_eval:
  camera_height: 4.0              # metres above ground
  camera_pitch_deg: 20.0          # degrees, positive = downward
  bins: [0, 15, 30,...]           # bin edges in metres
```

The paper uses five distance bins (0–15, 15–30, 30–45, 45–60, 60–75 m) derived from the bin edges plus the camera geometry. Bounding-box foot points are back-projected to the ground plane using `sensor_calibration.txt` (camera intrinsics) at the repository root, together with the height and pitch above.

## ISP variant naming

Variant directory names are produced by `run_isp_sweep.py` and re-used both as `--isp-variant` arguments to `train_models.py` and as keys when aggregating results. The naming patterns are:


| Paper table            | Pattern                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| Table 2 — gamma        | `gac_gamma-{value}`                                                                        |
| Table 3 — digital gain | `gac_gain-{value}`                                                                         |
| Table 4 — saturation   | `hsc_saturation_gain-{value}`                                                              |
| Table 5 — BNF          | `bnf_intensity_sigma-{is}_bnf_spatial_sigma-{ss}_bnf_kernel_size-{ks}`                     |
| Table 6 — EEH          | `eeh_edge_gain-{eg}_eeh_flat_threshold-{ft}_eeh_delta_threshold-{dt}_eeh_kernel_size-{ks}` |


Plus the fixed names `Default_ISP`, `Bayer`, and `Bayer_GC`. See paper Tables 2–6 for the actual values used at each step.

`ISPVariantDataset` locates the corresponding pre-rendered video under:

- `Default_ISP`: `<set>/<subdir>/Processed_Images/<sensor>/Default_ISP/<sensor>.mp4`
- `Bayer_GC`: `<set>/<subdir>/Processed_Images/Bayer_GC/<sensor>_Bayer_GC.mp4`
- All others: `<set>/<subdir>/Processed_Images/<sensor>/<variant>/<variant>.mp4`

## Result directory layout

Once training and `--eval-only --bin-distance` have been run for every cell, results are expected under:

```
sensitivity_results/
├── yolov8m/
│   ├── gac_gamma-0.1/
│   │   └── eval_results_last_checkpoint.json
│   ├── gac_gamma-0.25/
│   │   └── eval_results_last_checkpoint.json
│   ├── ...
├── yolo26m/
├── fasterrcnn_resnet50_fpn/
└── rtdetr-l/
```

`plot_sensitivity_results.py` reads exactly this layout. Each JSON contains overall mAP, per-class AP, and the `distance_binned_metrics` block produced by `--bin-distance`.

## Where to change hyperparameters


| What                                                                  | Where                                                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Dataset root, sensor, frame stride, max-frames-per-video              | `data:` block of `sensitivity_config.yaml`                                                         |
| Train / validation / test splits                                      | `train:` / `validation:` / `test:` of `sensitivity_config.yaml`                                    |
| Distance bin edges (paper Figs 4, 6, 7)                               | `distance_eval.bins` of `sensitivity_config.yaml`                                                  |
| Camera height / pitch                                                 | `distance_eval.camera_height`, `distance_eval.camera_pitch_deg`                                    |
| Camera intrinsics                                                     | `sensor_calibration.txt` (repository root)                                                         |
| Architectures plotted                                                 | `MODELS` constant in `plot_sensitivity_results.py`                                                 |
| Variants plotted                                                      | `ISP_VARIANTS` constant in `plot_sensitivity_results.py`                                           |
| Metric (mAP50 vs mAP50-95)                                            | `OVERALL_METRIC` / `BINNED_METRIC` / `PER_CLASS_METRIC` constants in `plot_sensitivity_results.py` |
| Per-class bin exclusions (e.g. dropping the 60–75 m bin for `person`) | `EXCLUDE_BINS_PER_CLASS` in `plot_sensitivity_results.py`                                          |
| Plot DPI / file format                                                | `FIGURE_DPI` / `FIGURE_FORMAT` in `plot_sensitivity_results.py`                                    |


