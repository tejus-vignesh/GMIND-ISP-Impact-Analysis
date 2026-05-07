"""
Sensitivity Analysis Plotting Script
=====================================
Reads evaluation JSON files from sensitivity_results/ and produces
comparison plots showing how ISP configuration changes affect each
model's detection performance.

No CLI arguments - edit the configuration section below.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "sensitivity_results"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "sensitivity_plots"

# ── Choose which JSON results file to plot ────────────────────────────
# Available: "eval_results.json", "eval_results_3_bins.json", "eval_results_6_bins.json"
# RESULTS_FILENAME = "eval_results.json"
RESULTS_FILENAME = "eval_results_last_checkpoint.json"

MODELS = ["fasterrcnn_resnet50_fpn", "rtdetr-l", "yolo26m", "yolov8m"]

ISP_VARIANTS = [
    "Default_ISP",
    "Bayer",
    "Bayer_GC",
    "gac_gain-8",
    "gac_gain-32",
    "gac_gain-512",
    "gac_gain-1024",
    "gac_gamma-0.1",
    "gac_gamma-0.25",
    "gac_gamma-1",
    "gac_gamma-1.5",
    "hsc_saturation_gain-0",
    "hsc_saturation_gain-512",
    "hsc_saturation_gain-1024",
    "hsc_saturation_gain-2048",
    "bnf_intensity_sigma-0.35_bnf_spatial_sigma-0.3_bnf_kernel_size-5",
    "bnf_intensity_sigma-6_bnf_spatial_sigma-6_bnf_kernel_size-7",
    "bnf_intensity_sigma-16_bnf_spatial_sigma-16_bnf_kernel_size-13",
    "bnf_intensity_sigma-72_bnf_spatial_sigma-72_bnf_kernel_size-25",
    "eeh_edge_gain-0_eeh_flat_threshold-16_eeh_delta_threshold-32_eeh_kernel_size-5",
    "eeh_edge_gain-768_eeh_flat_threshold-8_eeh_delta_threshold-64_eeh_kernel_size-7",
    "eeh_edge_gain-1408_eeh_flat_threshold-6_eeh_delta_threshold-128_eeh_kernel_size-13",
    "eeh_edge_gain-2048_eeh_flat_threshold-2_eeh_delta_threshold-128_eeh_kernel_size-21",
]

CLASSES = ["person", "bicycle", "car"]

# ── Per-class bin exclusions ──────────────────────────────────────────
# Bins listed here are dropped from per-class binned plots (Plot 3).
# Leave a class out or set to an empty list to include all bins.
EXCLUDE_BINS_PER_CLASS: dict[str, list[str]] = {
    "person": ["60-75m"],
}

# OVERALL_METRIC = "map50"          # top-level key (lowercase)
# BINNED_METRIC = "AP50"            # distance_binned_metrics key (uppercase)
# PER_CLASS_METRIC = "ap50"         # per_class key (lowercase)
# PER_CLASS_BINNED_METRIC = "AP50"  # binned per_class key (uppercase)

OVERALL_METRIC = "map50-95"          # top-level key (lowercase)
BINNED_METRIC = "AP50-95"            # distance_binned_metrics key (uppercase)
PER_CLASS_METRIC = "ap50-95"         # per_class key (lowercase)
PER_CLASS_BINNED_METRIC = "AP50-95"  # binned per_class key (uppercase)

GT_CLASS_COUNTS_FILE: Optional[Path] = RESULTS_ROOT / "gt_class_bin_counts.json"
# How to show per-class GT instance counts on Plot 3.
#   "bar" — semi-transparent bars on a secondary y-axis (uniform scale)
#   "num" — text annotations showing the count below each distance bin
#   None  — disabled
GT_COUNTS_DISPLAY: Optional[str] = "num" # "num", "bar", None

FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Derived output directory: strip .json -> subdirectory name
OUTPUT_DIR = OUTPUT_ROOT / Path(RESULTS_FILENAME).stem

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Display-name helpers
# ═══════════════════════════════════════════════════════════════════════

_MODEL_DISPLAY = {
    "fasterrcnn_resnet50_fpn": "Faster R-CNN",
    "rtdetr-l": "RT-DETR-L",
    "yolo26m": "YOLO26m",
    "yolov8m": "YOLOv8m",
}

_VARIANT_DISPLAY = {
    "Default_ISP": "Default ISP",
    "Bayer": "Bayer",
    "Bayer_GC": "Bayer + GC",
    "gac_gain-8": "Gain 8",
    "gac_gain-32": "Gain 32",
    "gac_gain-512": "Gain 512",
    "gac_gain-1024": "Gain 1024",
    "gac_gamma-0.1": "Gamma 0.1",
    "gac_gamma-0.25": "Gamma 0.25",
    "gac_gamma-1": "Gamma 1",
    "gac_gamma-1.5": "Gamma 1.5",
    "hsc_saturation_gain-0": "Saturation 0",
    "hsc_saturation_gain-512": "Saturation 512",
    "hsc_saturation_gain-1024": "Saturation 1024",
    "hsc_saturation_gain-2048": "Saturation 2048",
    "bnf_intensity_sigma-0.35_bnf_spatial_sigma-0.3_bnf_kernel_size-5": "BNF Step -1",
    "bnf_intensity_sigma-6_bnf_spatial_sigma-6_bnf_kernel_size-7": "BNF Step 1",
    "bnf_intensity_sigma-16_bnf_spatial_sigma-16_bnf_kernel_size-13": "BNF Step 2",
    "bnf_intensity_sigma-72_bnf_spatial_sigma-72_bnf_kernel_size-25": "BNF Step 3",
    "eeh_edge_gain-0_eeh_flat_threshold-16_eeh_delta_threshold-32_eeh_kernel_size-5": "EEH Step -1",
    "eeh_edge_gain-768_eeh_flat_threshold-8_eeh_delta_threshold-64_eeh_kernel_size-7": "EEH Step 1",
    "eeh_edge_gain-1408_eeh_flat_threshold-6_eeh_delta_threshold-128_eeh_kernel_size-13": "EEH Step 2",
    "eeh_edge_gain-2048_eeh_flat_threshold-2_eeh_delta_threshold-128_eeh_kernel_size-21": "EEH Step 3",
}


def _format_model_name(model: str) -> str:
    return _MODEL_DISPLAY.get(model, model)


def _format_variant_name(variant: str) -> str:
    return _VARIANT_DISPLAY.get(variant, variant)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


def load_all_results() -> dict[str, dict[str, dict]]:
    """Load all JSON files into results[model][variant] = parsed dict."""
    results: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        results[model] = {}
        for variant in ISP_VARIANTS:
            path = RESULTS_ROOT / model / variant / RESULTS_FILENAME
            if not path.exists():
                log.warning("Missing: %s", path)
                continue
            with open(path) as f:
                results[model][variant] = json.load(f)
    loaded = sum(len(v) for v in results.values())
    log.info("Loaded %d / %d JSON files", loaded, len(MODELS) * len(ISP_VARIANTS))
    return results


def load_gt_class_counts(path: Optional[Path]) -> Optional[dict]:
    """Load per-class GT instance counts from JSON.  Returns None if missing."""
    if path is None or not path.exists():
        log.info("GT class counts file not found (%s) — bars disabled", path)
        return None
    with open(path) as f:
        data = json.load(f)
    log.info("Loaded GT class counts from %s", path)
    return data


def _bin_sort_key(bin_name: str) -> float:
    """Sort key for bin names like '0-25m', '60-80m', '100m+'."""
    m = re.match(r"(\d+)", bin_name)
    return float(m.group(1)) if m else float("inf")


def discover_bins(results: dict[str, dict[str, dict]]) -> list[str]:
    """Union of all bin names across loaded data, sorted numerically."""
    bins: set[str] = set()
    for model_variants in results.values():
        for data in model_variants.values():
            bins.update(data.get("distance_binned_metrics", {}).keys())
    return sorted(bins, key=_bin_sort_key)


def discover_per_class_bins(results: dict[str, dict[str, dict]]) -> list[str]:
    """Union of bin names that have valid per-class data for any class, sorted numerically.

    Bins that exist in distance_binned_metrics but contain no per_class data
    with valid (non -1.0) values for any class are excluded.
    """
    bins: set[str] = set()
    for model_variants in results.values():
        for data in model_variants.values():
            for bin_name, bin_data in data.get("distance_binned_metrics", {}).items():
                per_class = bin_data.get("per_class", {})
                for cls in CLASSES:
                    cls_data = per_class.get(cls, {})
                    val = cls_data.get(PER_CLASS_BINNED_METRIC)
                    if val is not None and val != -1.0:
                        bins.add(bin_name)
                        break
    return sorted(bins, key=_bin_sort_key)


# ═══════════════════════════════════════════════════════════════════════
# Metric extraction helpers (centralise -1.0 / missing-key handling)
# ═══════════════════════════════════════════════════════════════════════


def _get_overall_metric(data: dict, metric: str) -> Optional[float]:
    val = data.get(metric)
    if val is None or val == -1.0:
        return None
    return float(val)


def _get_binned_metric(data: dict, bin_name: str, metric: str) -> Optional[float]:
    bin_data = data.get("distance_binned_metrics", {}).get(bin_name)
    if bin_data is None:
        return None
    val = bin_data.get(metric)
    if val is None or val == -1.0:
        return None
    return float(val)


def _get_per_class_binned_metric(
    data: dict, bin_name: str, cls: str, metric: str
) -> Optional[float]:
    bin_data = data.get("distance_binned_metrics", {}).get(bin_name)
    if bin_data is None:
        return None
    per_class = bin_data.get("per_class", {}).get(cls)
    if per_class is None:
        return None
    val = per_class.get(metric)
    if val is None or val == -1.0:
        return None
    return float(val)


# ═══════════════════════════════════════════════════════════════════════
# Styling helpers
# ═══════════════════════════════════════════════════════════════════════

_TAB20 = plt.cm.tab20  # type: ignore[attr-defined]


def _get_variant_style(variant: str, idx: int) -> dict:
    """Return colour / marker / linewidth for a variant."""
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "d"]
    if variant == "Default_ISP":
        return {
            "color": "black",
            "marker": "o",
            "linewidth": 2.5,
            "zorder": 10,
        }
    return {
        "color": _TAB20(idx % 20),
        "marker": markers[idx % len(markers)],
        "linewidth": 1.3,
        "zorder": 5,
    }


# ═══════════════════════════════════════════════════════════════════════
# Save helper
# ═══════════════════════════════════════════════════════════════════════


def _save_figure(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _nice_ylim(values: list[float], pad_frac: float = 0.10) -> tuple[float, float]:
    """Compute padded y-axis limits from actual data values.

    Adds *pad_frac* of the data range on each side, then rounds outward
    to the nearest 0.05 for clean tick marks.  Always clamps to [0, 1].
    """
    if not values:
        return (0.0, 1.0)
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 0.05  # fallback for constant data
    lo = lo - span * pad_frac
    hi = hi + span * pad_frac
    # Round to nearest 0.05 outward
    lo = max(0.0, math.floor(lo / 0.05) * 0.05)
    hi = min(1.0, math.ceil(hi / 0.05) * 0.05)
    # Guarantee a minimum visible range
    if hi - lo < 0.05:
        hi = min(1.0, lo + 0.05)
    return (lo, hi)


# ═══════════════════════════════════════════════════════════════════════
# Plot 1 – Overall mAP per model (horizontal bar chart)
# ═══════════════════════════════════════════════════════════════════════


def _plot_overall_single(
    ax: plt.Axes,
    model: str,
    model_results: dict[str, dict],
) -> None:
    """Draw a horizontal bar chart of overall metric for one model."""
    variants = []
    values = []
    for v in ISP_VARIANTS:
        if v not in model_results:
            continue
        val = _get_overall_metric(model_results[v], OVERALL_METRIC)
        if val is None:
            continue
        variants.append(v)
        values.append(val)

    y_pos = np.arange(len(variants))
    default_val = None
    bar_colors = []
    for v, val in zip(variants, values):
        if v == "Default_ISP":
            bar_colors.append("black")
            default_val = val
        else:
            bar_colors.append("#4C72B0")

    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="white", height=0.7)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=14,
        )

    # Reference line at Default_ISP value
    if default_val is not None:
        ax.axvline(default_val, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_variant_name(v) for v in variants], fontsize=15)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"$\mathrm{mAP}_{50\text{-}95}$", fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_title(_format_model_name(model), fontsize=18, fontweight="bold")
    ax.invert_yaxis()


def plot_overall(results: dict[str, dict[str, dict]]) -> None:
    """Plot 1: overall metric – combined 2x2 + per-model standalone."""
    # Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, model in zip(axes.flat, MODELS):
        _plot_overall_single(ax, model, results.get(model, {}))
    fig.tight_layout()
    _save_figure(fig, f"overall_{OVERALL_METRIC}")

    # Per-model standalone
    for model in MODELS:
        fig_s, ax_s = plt.subplots(figsize=(10, 7))
        _plot_overall_single(ax_s, model, results.get(model, {}))
        fig_s.tight_layout()
        _save_figure(fig_s, f"overall_{OVERALL_METRIC}_{model}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2 – Binned performance per model (line plot)
# ═══════════════════════════════════════════════════════════════════════


def _plot_binned_single(
    ax: plt.Axes,
    model: str,
    model_results: dict[str, dict],
    bins: list[str],
) -> list[tuple[str, dict]]:
    """Draw binned line plot for one model. Returns legend handles info."""
    legend_entries: list[tuple[str, dict]] = []
    all_y: list[float] = []
    color_idx = 0
    for v in ISP_VARIANTS:
        if v not in model_results:
            continue
        data = model_results[v]
        style = _get_variant_style(v, color_idx)
        color_idx += 1

        y_vals = []
        for b in bins:
            val = _get_binned_metric(data, b, BINNED_METRIC)
            y_vals.append(val if val is not None else float("nan"))

        ax.plot(
            bins,
            y_vals,
            label=_format_variant_name(v),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=5,
            zorder=style["zorder"],
        )
        legend_entries.append((_format_variant_name(v), style))
        all_y.extend(v for v in y_vals if not np.isnan(v))

    ax.set_ylabel(r"$\mathrm{AP}_{50\text{-}95}$", fontsize=9)
    ax.set_xlabel("Distance bin", fontsize=9)
    ax.set_title(_format_model_name(model), fontsize=11, fontweight="bold")
    ax.set_ylim(_nice_ylim(all_y))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    return legend_entries


def plot_binned(results: dict[str, dict[str, dict]], bins: list[str]) -> None:
    """Plot 2: binned metric – combined 2x2 + per-model standalone."""
    # Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Distance-Binned {BINNED_METRIC} by ISP Variant",
        fontsize=14,
        fontweight="bold",
    )
    legend_entries = []
    for ax, model in zip(axes.flat, MODELS):
        legend_entries = _plot_binned_single(ax, model, results.get(model, {}), bins)

    # Shared legend below subplots
    if legend_entries:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    _save_figure(fig, f"binned_{BINNED_METRIC}")

    # Per-model standalone
    for model in MODELS:
        fig_s, ax_s = plt.subplots(figsize=(8, 5))
        _plot_binned_single(ax_s, model, results.get(model, {}), bins)
        ax_s.legend(fontsize=7, loc="best")
        fig_s.tight_layout()
        _save_figure(fig_s, f"binned_{BINNED_METRIC}_{model}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3 – Per-class binned performance (line plot)
# ═══════════════════════════════════════════════════════════════════════


def _gt_counts_global_max(gt_counts: Optional[dict], classes: list[str]) -> int:
    """Return the maximum per-class bin count across all classes and bins."""
    if gt_counts is None:
        return 0
    max_val = 0
    for cls in classes:
        cls_counts = gt_counts.get("counts", {}).get(cls, {})
        for v in cls_counts.values():
            if v > max_val:
                max_val = v
    return max_val


def _plot_per_class_binned_single(
    ax: plt.Axes,
    model: str,
    cls: str,
    model_results: dict[str, dict],
    bins: list[str],
    gt_counts: Optional[dict] = None,
    gt_max_count: int = 0,
) -> None:
    """Draw per-class binned line plot for one model + class."""
    excluded = set(EXCLUDE_BINS_PER_CLASS.get(cls, []))

    # Use numeric x positions so every subplot has the same uniform axis,
    # even when a class has no data for some bins.
    x_pos = np.arange(len(bins))

    color_idx = 0
    for v in ISP_VARIANTS:
        if v not in model_results:
            continue
        data = model_results[v]
        style = _get_variant_style(v, color_idx)
        color_idx += 1

        y_vals = []
        for b in bins:
            if b in excluded:
                y_vals.append(float("nan"))
            else:
                val = _get_per_class_binned_metric(data, b, cls, PER_CLASS_BINNED_METRIC)
                y_vals.append(val if val is not None else float("nan"))

        ax.plot(
            x_pos,
            y_vals,
            label=_format_variant_name(v),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=4,
            zorder=style["zorder"],
        )

    # ── GT instance counts ──
    if gt_counts is not None and cls in gt_counts.get("counts", {}):
        cls_counts = gt_counts["counts"][cls]
        bar_values = [cls_counts.get(b, 0) for b in bins]

        if GT_COUNTS_DISPLAY == "bar" and any(v > 0 for v in bar_values):
            ax2 = ax.twinx()
            ax2.bar(
                x_pos, bar_values,
                width=0.5, alpha=0.20, color="grey", zorder=1,
            )
            ax2.set_ylabel("GT instances", fontsize=10, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=9)
            # Uniform y-axis across all subplots
            if gt_max_count > 0:
                ax2.set_ylim(0, gt_max_count * 1.05)
            # Keep line plots on top of bars
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        elif GT_COUNTS_DISPLAY == "num":
            _num_tick_labels = [f"{b}\n(n={cls_counts.get(b, 0)})" for b in bins]

    ax.set_xticks(x_pos)
    # When "num" mode has counts, use composite tick labels; skip "Distance bin" xlabel
    if GT_COUNTS_DISPLAY == "num" and gt_counts is not None and cls in gt_counts.get("counts", {}):
        ax.set_xticklabels(_num_tick_labels)
        ax.set_xlabel("")
    else:
        ax.set_xticklabels(bins)
        ax.set_xlabel("Distance bin", fontsize=14)
    ax.set_xlim(-0.3, len(bins) - 0.7)
    ax.set_ylabel(r"$\mathrm{AP}_{50\text{-}95}$", fontsize=16)
    ax.set_title(f"{_format_model_name(model)} – {cls}", fontsize=17, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.3)


def _gt_counts_annotation_text() -> Optional[str]:
    """Return the annotation string for the current GT_COUNTS_DISPLAY mode."""
    if GT_COUNTS_DISPLAY == "bar":
        return "Grey bars = Ground Truth instance count"
    elif GT_COUNTS_DISPLAY == "num":
        return "n = Ground Truth instance count"
    return None


def plot_per_class_binned(
    results: dict[str, dict[str, dict]],
    bins: list[str],
    gt_counts: Optional[dict] = None,
) -> None:
    """Plot 3: per-class binned – combined grid + per-model standalone."""
    n_models = len(MODELS)
    n_classes = len(CLASSES)
    gt_max = _gt_counts_global_max(gt_counts, CLASSES) if GT_COUNTS_DISPLAY == "bar" else 0
    annotation = _gt_counts_annotation_text() if gt_counts is not None else None

    # Combined: models (rows) x classes (cols)
    fig, axes = plt.subplots(
        n_models, n_classes, figsize=(5 * n_classes, 8 * n_models) # Vertical Long
    )
    fig.suptitle(
        f"Per-Class Distance-Binned {PER_CLASS_BINNED_METRIC} by ISP Variant",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    for r, model in enumerate(MODELS):
        for c, cls in enumerate(CLASSES):
            ax = axes[r, c]
            _plot_per_class_binned_single(
                ax, model, cls, results.get(model, {}), bins,
                gt_counts=gt_counts, gt_max_count=gt_max,
            )

    # Shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(8, len(labels)),
            fontsize=10,
            frameon=True,
        )
    if annotation is not None:
        fig.text(
            0.99, 0.98, annotation,
            ha="right", va="top", fontsize=9, color="grey", style="italic",
        )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save_figure(fig, f"per_class_binned_{PER_CLASS_BINNED_METRIC}")

    # Per-model standalone: 1 row x n_classes cols
    for model in MODELS:
        fig_s, axes_s = plt.subplots(1, n_classes, figsize=(6 * n_classes, 9))
        for c, cls in enumerate(CLASSES):
            _plot_per_class_binned_single(
                axes_s[c], model, cls, results.get(model, {}), bins,
                gt_counts=gt_counts, gt_max_count=gt_max,
            )
        handles_s, labels_s = axes_s[0].get_legend_handles_labels()
        if handles_s:
            fig_s.legend(
                handles_s,
                labels_s,
                loc="lower center",
                ncol=min(8, len(labels_s)),
                fontsize=13,
                frameon=True,
            )
        fig_s.tight_layout(rect=[0, 0.10, 1, 1.0])
        _save_figure(fig_s, f"per_class_binned_{PER_CLASS_BINNED_METRIC}_{model}")


# ═══════════════════════════════════════════════════════════════════════
# Delta helpers
# ═══════════════════════════════════════════════════════════════════════


def _compute_deltas(
    results: dict[str, dict[str, dict]],
) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    """Compute per-variant, per-model delta from Default_ISP.

    Returns (variant_keys, variant_labels, deltas) where
    deltas[model][variant] = value - default_value.
    Variants without data or Default_ISP itself are skipped.
    """
    non_default = [v for v in ISP_VARIANTS if v != "Default_ISP"]
    labels = [_format_variant_name(v) for v in non_default]

    deltas: dict[str, dict[str, float]] = {}
    for model in MODELS:
        model_results = results.get(model, {})
        default_data = model_results.get("Default_ISP")
        if default_data is None:
            continue
        default_val = _get_overall_metric(default_data, OVERALL_METRIC)
        if default_val is None:
            continue
        deltas[model] = {}
        for v in non_default:
            if v not in model_results:
                continue
            val = _get_overall_metric(model_results[v], OVERALL_METRIC)
            if val is not None:
                deltas[model][v] = val - default_val

    return non_default, labels, deltas


# ═══════════════════════════════════════════════════════════════════════
# Plot 4 – Heatmap with annotated cells
# ═══════════════════════════════════════════════════════════════════════


def plot_delta_heatmap(results: dict[str, dict[str, dict]]) -> None:
    """Option B: heatmap of Δ mAP from Default ISP."""
    variants, labels, deltas = _compute_deltas(results)
    models_with_data = [m for m in MODELS if m in deltas]
    n_variants = len(variants)
    n_models = len(models_with_data)

    # Build matrix (variants × models)
    matrix = np.full((n_variants, n_models), np.nan)
    for j, model in enumerate(models_with_data):
        for i, v in enumerate(variants):
            matrix[i, j] = deltas[model].get(v, np.nan)

    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(
        matrix, cmap="RdBu", aspect="auto",
        vmin=-vmax, vmax=vmax,
    )

    # Annotate cells
    for i in range(n_variants):
        for j in range(n_models):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            sign = "+" if val > 0 else ""
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                j, i, f"{sign}{val:.3f}",
                ha="center", va="center", fontsize=12, color=color,
            )

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([_format_model_name(m) for m in models_with_data], fontsize=14)
    ax.set_yticks(np.arange(n_variants))
    ax.set_yticklabels(labels, fontsize=14)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(r"$\Delta\;\mathrm{mAP}_{50\text{-}95}$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    _save_figure(fig, f"delta_heatmap_{OVERALL_METRIC}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    log.info("Results file: %s", RESULTS_FILENAME)
    log.info("Output dir:   %s", OUTPUT_DIR)

    results = load_all_results()
    gt_counts = load_gt_class_counts(GT_CLASS_COUNTS_FILE) if GT_COUNTS_DISPLAY else None
    bins = discover_bins(results)
    per_class_bins = discover_per_class_bins(results)
    log.info("Discovered bins: %s", bins)
    log.info("Discovered per-class bins: %s", per_class_bins)

    plot_overall(results)
    plot_binned(results, bins)
    plot_per_class_binned(results, per_class_bins, gt_counts=gt_counts)
    plot_delta_heatmap(results)

    log.info("Done – all plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
