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
RESULTS_FILENAME = "eval_results_6_bins.json"

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
            fontsize=7,
        )

    # Reference line at Default_ISP value
    if default_val is not None:
        ax.axvline(default_val, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_variant_name(v) for v in variants], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(OVERALL_METRIC, fontsize=9)
    ax.set_title(_format_model_name(model), fontsize=11, fontweight="bold")
    ax.invert_yaxis()


def plot_overall(results: dict[str, dict[str, dict]]) -> None:
    """Plot 1: overall metric – combined 2x2 + per-model standalone."""
    # Combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Overall {OVERALL_METRIC} by ISP Variant", fontsize=14, fontweight="bold")
    for ax, model in zip(axes.flat, MODELS):
        _plot_overall_single(ax, model, results.get(model, {}))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, f"overall_{OVERALL_METRIC}")

    # Per-model standalone
    for model in MODELS:
        fig_s, ax_s = plt.subplots(figsize=(8, 5))
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

    ax.set_ylabel(BINNED_METRIC, fontsize=9)
    ax.set_xlabel("Distance bin", fontsize=9)
    ax.set_title(_format_model_name(model), fontsize=11, fontweight="bold")
    # ax.set_ylim(_nice_ylim(all_y))
    ax.set_ylim(0.0, 1.0)
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


def _plot_per_class_binned_single(
    ax: plt.Axes,
    model: str,
    cls: str,
    model_results: dict[str, dict],
    bins: list[str],
) -> None:
    """Draw per-class binned line plot for one model + class."""
    excluded = set(EXCLUDE_BINS_PER_CLASS.get(cls, []))

    # Always use full bin list for consistent x-axis; mark excluded bins
    x_labels = [f"{b}\n(skipped)" if b in excluded else b for b in bins]

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
            x_labels,
            y_vals,
            label=_format_variant_name(v),
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=4,
            zorder=style["zorder"],
        )

    ax.set_ylabel(PER_CLASS_BINNED_METRIC, fontsize=8)
    ax.set_xlabel("Distance bin", fontsize=8)
    ax.set_title(f"{_format_model_name(model)} – {cls}", fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)


def plot_per_class_binned(results: dict[str, dict[str, dict]], bins: list[str]) -> None:
    """Plot 3: per-class binned – combined grid + per-model standalone."""
    n_models = len(MODELS)
    n_classes = len(CLASSES)

    # Combined: models (rows) x classes (cols)
    fig, axes = plt.subplots(
        n_models, n_classes, figsize=(5 * n_classes, 4 * n_models)
    )
    fig.suptitle(
        f"Per-Class Distance-Binned {PER_CLASS_BINNED_METRIC} by ISP Variant",
        fontsize=14,
        fontweight="bold",
    )
    for r, model in enumerate(MODELS):
        for c, cls in enumerate(CLASSES):
            ax = axes[r, c]
            _plot_per_class_binned_single(
                ax, model, cls, results.get(model, {}), bins
            )

    # Shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            fontsize=8,
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save_figure(fig, f"per_class_binned_{PER_CLASS_BINNED_METRIC}")

    # Per-model standalone: 1 row x n_classes cols
    for model in MODELS:
        fig_s, axes_s = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
        fig_s.suptitle(
            f"{_format_model_name(model)} – Per-Class {PER_CLASS_BINNED_METRIC}",
            fontsize=12,
            fontweight="bold",
        )
        for c, cls in enumerate(CLASSES):
            _plot_per_class_binned_single(
                axes_s[c], model, cls, results.get(model, {}), bins
            )
        handles_s, labels_s = axes_s[0].get_legend_handles_labels()
        if handles_s:
            fig_s.legend(
                handles_s,
                labels_s,
                loc="lower center",
                ncol=min(6, len(labels_s)),
                fontsize=7,
                frameon=True,
            )
        fig_s.tight_layout(rect=[0, 0.06, 1, 0.93])
        _save_figure(fig_s, f"per_class_binned_{PER_CLASS_BINNED_METRIC}_{model}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    log.info("Results file: %s", RESULTS_FILENAME)
    log.info("Output dir:   %s", OUTPUT_DIR)

    results = load_all_results()
    bins = discover_bins(results)
    log.info("Discovered bins: %s", bins)

    plot_overall(results)
    plot_binned(results, bins)
    plot_per_class_binned(results, bins)

    log.info("Done – all plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
