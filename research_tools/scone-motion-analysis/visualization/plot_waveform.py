"""Raw waveform plotting for selected motion variables."""

from collections import OrderedDict
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.constants import SD_RANGE
from utils.file_utils import format_legend_label, get_clean_variable_name
from utils.unit_utils import convert_values_for_display, get_axis_label


PLOT_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#6A4C93", "#4D4D4D",
]


def _extract_waveform_xy(df: pd.DataFrame, variable: str, prefer_time_axis: bool = True) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Return finite x/y arrays and the x-axis label for a variable."""
    y_raw = convert_values_for_display(variable, pd.to_numeric(df[variable], errors="coerce").to_numpy(dtype=float))
    x_label = "Frame / Index"
    x_raw = np.arange(len(y_raw), dtype=float)

    if prefer_time_axis and "time" in df.columns:
        time_raw = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
        if len(time_raw) == len(y_raw):
            finite_time = time_raw[np.isfinite(time_raw)]
            if len(finite_time) > 0 and (len(finite_time) == 1 or np.all(np.diff(finite_time) >= 0)):
                x_raw = time_raw
                x_label = "Time (s)"

    finite_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    return x_raw[finite_mask], y_raw[finite_mask], x_label, get_axis_label(variable)


def _align_series_to_grid(entries: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate same-condition waveforms to the first valid x grid."""
    reference_x = entries[0]["x"]
    aligned = []

    for entry in entries:
        x = entry["x"]
        y = entry["y"]
        if len(x) < 2 or len(y) < 2:
            continue

        unique_x, unique_indices = np.unique(x, return_index=True)
        unique_y = y[unique_indices]
        if len(unique_x) < 2:
            continue

        if len(unique_x) == len(reference_x) and np.allclose(unique_x, reference_x, equal_nan=False):
            aligned.append(unique_y)
        else:
            aligned.append(np.interp(reference_x, unique_x, unique_y, left=np.nan, right=np.nan))

    if not aligned:
        return reference_x, np.empty((0, len(reference_x)))

    return reference_x, np.vstack(aligned)


def draw_raw_waveform(
    ax,
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    variable: str,
    show_mean_sd: bool = True,
    prefer_time_axis: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Draw a raw waveform plot for one selected variable.

    Multiple files are overlaid. If several files share the same legend label,
    their condition mean curve and SD band are drawn together.
    """
    warnings = []
    grouped_series: Dict[str, List[Dict]] = OrderedDict()
    x_labels = []
    y_labels = []

    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(file_paths[file_index]) if file_index < len(file_paths) else f"file_{file_index + 1}"
        legend_label = format_legend_label(filename)

        if variable not in df.columns:
            warnings.append(f"{filename}: '{variable}' variable not found; skipped.")
            continue

        x, y, x_label, y_label = _extract_waveform_xy(df, variable, prefer_time_axis=prefer_time_axis)
        if len(y) == 0:
            warnings.append(f"{filename}: '{variable}' has no finite numeric data; skipped.")
            continue

        grouped_series.setdefault(legend_label, []).append({
            "filename": filename,
            "label": legend_label,
            "x": x,
            "y": y,
            "x_label": x_label,
        })
        x_labels.append(x_label)
        y_labels.append(y_label)

    has_data = bool(grouped_series)
    if not has_data:
        ax.text(
            0.5,
            0.5,
            f"No valid data for {variable}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        _style_raw_waveform_axes(ax, variable, "Frame / Index", get_axis_label(variable), has_data=False)
        return False, warnings

    color_index = 0
    for condition_label, entries in grouped_series.items():
        color = PLOT_COLORS[color_index % len(PLOT_COLORS)]
        color_index += 1

        if show_mean_sd and len(entries) > 1:
            for entry in entries:
                ax.plot(entry["x"], entry["y"], color=color, alpha=0.22, linewidth=0.9)

            x_grid, aligned_y = _align_series_to_grid(entries)
            if aligned_y.size > 0:
                mean_curve = np.nanmean(aligned_y, axis=0)
                sd_curve = np.nanstd(aligned_y, axis=0)
                ax.fill_between(
                    x_grid,
                    mean_curve - SD_RANGE * sd_curve,
                    mean_curve + SD_RANGE * sd_curve,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
                ax.plot(
                    x_grid,
                    mean_curve,
                    label=f"{condition_label} mean n={len(entries)}",
                    color=color,
                    alpha=0.98,
                    linewidth=2.6,
                )
            continue

        for entry_idx, entry in enumerate(entries):
            label = condition_label if len(entries) == 1 else f"{condition_label} #{entry_idx + 1}"
            ax.plot(
                entry["x"],
                entry["y"],
                label=label,
                color=color,
                alpha=0.92,
                linewidth=2.0,
            )

            if show_mean_sd and len(entry["y"]) > 1:
                sd_value = float(np.nanstd(entry["y"]))
                if np.isfinite(sd_value) and sd_value > 0:
                    ax.fill_between(
                        entry["x"],
                        entry["y"] - SD_RANGE * sd_value,
                        entry["y"] + SD_RANGE * sd_value,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )

    x_label = "Time (s)" if x_labels and all(label == "Time (s)" for label in x_labels) else "Frame / Index"
    y_label = y_labels[0] if y_labels and all(label == y_labels[0] for label in y_labels) else "Value"
    _style_raw_waveform_axes(ax, variable, x_label, y_label, has_data=True)
    return True, warnings


def _style_raw_waveform_axes(ax, variable: str, x_label: str, y_label: str, has_data: bool) -> None:
    clean_var_name = get_clean_variable_name(variable)
    ax.set_title(clean_var_name, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    if has_data:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
