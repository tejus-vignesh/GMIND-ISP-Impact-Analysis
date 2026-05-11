# Image Signal Processing (ISP)

This directory contains the ISP pipeline used to generate the 23 ISP variants evaluated in the EI2026 paper *"Same Scene, Different Pipeline: ISP Impact on Automotive Detection at Range"*. The pipeline is based on [fast-openISP](https://github.com/QiuJueqin/fast-openISP) (a NumPy-based, ~300× faster re-implementation of [openISP](https://github.com/cruxopen/openISP)) and has been extended for the G-MIND SDK with a Bayer-domain gamma-correction module (`gac_bayer`), batch / parallel raw-to-MP4 processing, and a parameter-sweep driver.

## Contents

- `pipeline.py` — core ISP pipeline (sequence of `modules/`).
- `run_isp.py` — process one or more directories of raw frames with a chosen YAML config; write per-config PNGs and an MP4.
- `run_isp_sweep.py` — produce ISP variants by overriding one or more block parameters relative to a base YAML; each variant is written to its own folder.
- `run_isp_sweep_video.py` — re-encode the MP4 for an already-processed variant directory without re-running the pipeline.
- `modules/` — individual ISP blocks (`gac`, `gac_bayer`, `bnf`, `eeh`, `ceh`, `hsc`, `ccm`, …).
- `configs/*.yaml` — per-camera baseline configurations.

## Usage

### Default ISP

```bash
python ImageSignalProcessing/run_isp.py
```

Edit the `__main__` block of `run_isp.py` to set the input image directories, their per-set YAML config, and the output directory. The active baseline used for the paper is `configs/FLIR8.9.yaml`; `configs/FLIR8.9_Bayer_GC.yaml` is used for the `Bayer + GC` condition (gamma correction applied directly in the Bayer domain via `gac_bayer`, with no other processing).

### Generating ISP variants

`run_isp_sweep.py` has two modes:

**Single-parameter sweep** — vary one block parameter at a time (used for the paper's gain, gamma, and saturation sweeps in Tables 2–4):

```bash
# Paper Table 2 — gamma correction
python ImageSignalProcessing/run_isp_sweep.py --sweep "gac.gamma:0.1,0.25,1,1.5"

# Paper Table 3 — digital gain (256 = 1.0×)
python ImageSignalProcessing/run_isp_sweep.py --sweep "gac.gain:8,32,512,1024"

# Paper Table 4 — saturation (256 = 1.0×, 0 = grayscale)
python ImageSignalProcessing/run_isp_sweep.py --sweep "hsc.saturation_gain:0,512,1024,2048"
```

Each sweep produces one output directory per value (e.g. `gac_gamma-0.1/`, `gac_gamma-0.25/`, ...), each containing the processed PNGs and an MP4.

**Multi-parameter batch** — set several parameters together per variant (used for the paper's BNF and EEH variants in Tables 5–6, where intensity sigma, spatial sigma, and kernel size must move together):

```bash
# Paper Table 5 — bilateral noise filter, Step -1 / +1 / +2 / +3
python ImageSignalProcessing/run_isp_sweep.py \
    --batch "bnf.intensity_sigma=0.35,bnf.spatial_sigma=0.3,bnf.kernel_size=5" \
    --batch "bnf.intensity_sigma=6,bnf.spatial_sigma=6,bnf.kernel_size=7" \
    --batch "bnf.intensity_sigma=16,bnf.spatial_sigma=16,bnf.kernel_size=13" \
    --batch "bnf.intensity_sigma=72,bnf.spatial_sigma=72,bnf.kernel_size=25"
```

`--sweep` and `--batch` are mutually exclusive within a single invocation. Either flag can be passed multiple times to enumerate several variants in one run.

`--config <name.yaml>` selects the base config (default `FLIR8.9.yaml`). All other block parameters retain their values from the base YAML.

Set `SOURCE_DIRS` and `OUTPUT_DIRS` in the `__main__` block of `run_isp_sweep.py` before running — these tell the sweep which raw input folders to process and where to write the per-variant subfolders.

### Re-encoding MP4 only

If the PNGs were produced successfully but the MP4 encoding failed (e.g. `ffmpeg` interrupted), re-run with the same `--sweep` / `--batch` / `--config` arguments against `run_isp_sweep_video.py` to regenerate only the MP4s from the existing PNG sequences.

## Where to change ISP hyperparameters

| What | Where |
|---|---|
| Baseline per-camera config (sensor format, default block params) | `ImageSignalProcessing/configs/FLIR8.9.yaml` (and other `*.yaml`) |
| Which modules are run, and in what order | `module_enable_status:` block at the top of each YAML |
| Sensor resolution, bit depth, Bayer pattern | `hardware:` block of each YAML |
| Per-block parameters (`gac`, `bnf`, `eeh`, `hsc`, `ccm`, `ceh`, …) | Block-named sections at the bottom of each YAML |
| Values swept per ISP variant | CLI args (`--sweep` / `--batch`) of `run_isp_sweep.py` |
| Input / output directories for sweeps | `SOURCE_DIRS` / `OUTPUT_DIRS` in `__main__` of `run_isp_sweep.py` |
| Multiprocessing count, output framerate, save flags | `NUM_PROCESSES`, `VIDEO_FRAMERATE`, `SAVE_PNG`, `SAVE_VIDEO` at the top of `run_isp.py` / `run_isp_sweep.py` |

## ISP variant naming convention

Variant directories are named from the parameters they override (used as the `--isp-variant` argument when fine-tuning):

| Paper table | Pattern |
|---|---|
| Table 2 — gamma | `gac_gamma-{value}` |
| Table 3 — digital gain | `gac_gain-{value}` |
| Table 4 — saturation | `hsc_saturation_gain-{value}` |
| Table 5 — BNF | `bnf_intensity_sigma-{is}_bnf_spatial_sigma-{ss}_bnf_kernel_size-{ks}` |
| Table 6 — EEH | `eeh_edge_gain-{eg}_eeh_flat_threshold-{ft}_eeh_delta_threshold-{dt}_eeh_kernel_size-{ks}` |

The baseline and the two raw conditions use the fixed names `Default_ISP`, `Bayer`, and `Bayer_GC`.

## Algorithm notes

All modules implement the [openISP algorithms](https://github.com/cruxopen/openISP/blob/master/docs/Image%20Signal%20Processor.pdf), with two replacements and one addition:

- **EEH (edge enhancement)** — uses the original-minus-Gaussian Y-channel as the edge estimate (replaces openISP's asymmetric kernel; reduces artefacts at large gains).
- **BCC (brightness & contrast control)** — uses the median of the frame instead of a hard-coded value of 128.
- **CEH (contrast enhancement)** — new module implementing [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE); not present in openISP.
- **gac_bayer** — gamma correction applied directly to the single-channel Bayer mosaic (used by `FLIR8.9_Bayer_GC.yaml` for the paper's `Bayer + GC` condition).

## License and attribution

Original fast-openISP code © 2021 Qiu Jueqin, MIT licensed. GMIND-SDK additions and modifications are also MIT licensed. A fork of the upstream repository is maintained at [daramolloy/fast-openISP](https://github.com/daramolloy/fast-openISP) for attribution and code-lineage visibility.
