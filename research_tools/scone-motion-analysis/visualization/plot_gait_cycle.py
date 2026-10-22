"""Percent gait-cycle waveform plotting with SCONE state phase shading."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    RESAMPLE_POINTS,
    SCONE_PHASE_STATES,
    SCONE_STATE_PHASE_LABELS,
    SD_RANGE,
    STATE_COLS,
    VALIDATION_CONFIG,
)
from utils.file_utils import format_legend_label, get_clean_variable_name
from utils.unit_utils import convert_values_for_display, get_axis_label
from .plot_waveform import PLOT_COLORS


SCONE_PHASE_LABELS = SCONE_STATE_PHASE_LABELS

PHASE_COLORS = {
    0: "#DCEBFF",
    1: "#DFF4E6",
    2: "#FFF1CC",
    3: "#F3E8FF",
    4: "#FFE0E0",
}


def detect_state_column_for_variable(df: pd.DataFrame, variable: str) -> Optional[str]:
    """Choose the most likely SCONE state column for a variable."""
    var_lower = variable.lower()
    side_order = []

    if "_r" in var_lower or ".r" in var_lower or "right" in var_lower or "leg1_r" in var_lower:
        side_order.append("right")
    if "_l" in var_lower or ".l" in var_lower or "left" in var_lower or "leg0_l" in var_lower:
        side_order.append("left")

    for side in ("right", "left"):
        if side not in side_order:
            side_order.append(side)

    for side in side_order:
        state_col = STATE_COLS.get(side)
        if state_col in df.columns:
            return state_col
    return None


def detect_gait_cycle_boundaries(state_arr: np.ndarray) -> List[int]:
    """Return cycle boundary indices from entries into state 0."""
    if len(state_arr) == 0:
        return []

    try:
        raw_state = np.asarray(state_arr, dtype=float)
    except (TypeError, ValueError):
        return []

    state_values = _valid_state_values(raw_state)
    valid_indices = np.where(np.isin(state_values, [0, 1, 2, 3, 4]))[0]
    if len(valid_indices) == 0:
        return []

    boundaries = []
    first_idx = int(valid_indices[0])
    if state_values[first_idx] == 0:
        boundaries.append(first_idx)

    previous_state = int(state_values[first_idx])

    for idx in valid_indices[1:]:
        current_state = int(state_values[idx])
        if previous_state != 0 and current_state == 0:
            if not boundaries:
                boundaries.append(int(idx))
                previous_state = current_state
                continue

            cycle_length = idx - boundaries[-1]
            if (
                VALIDATION_CONFIG.get("min_cycle_length", 10)
                <= cycle_length
                <= VALIDATION_CONFIG.get("max_cycle_length", 1000)
            ):
                boundaries.append(int(idx))
        previous_state = current_state

    return boundaries


def _valid_state_values(state_arr: np.ndarray) -> np.ndarray:
    raw_state = np.asarray(state_arr, dtype=float)
    state_values = np.full(raw_state.shape, -1, dtype=int)
    finite_indices = np.where(np.isfinite(raw_state))[0]
    if len(finite_indices) == 0:
        return state_values

    rounded = np.rint(raw_state[finite_indices]).astype(int)
    valid_mask = np.isin(rounded, [0, 1, 2, 3, 4])
    state_values[finite_indices[valid_mask]] = rounded[valid_mask]
    return state_values


def _normalized_source_axis(segment_length: int, x_segment: Optional[np.ndarray] = None) -> np.ndarray:
    if x_segment is None:
        return np.linspace(0.0, 1.0, segment_length)

    try:
        x_values = np.asarray(x_segment, dtype=float)
    except (TypeError, ValueError):
        return np.linspace(0.0, 1.0, segment_length)

    if len(x_values) != segment_length or len(x_values) < 2:
        return np.linspace(0.0, 1.0, segment_length)
    if not np.isfinite(x_values).all():
        return np.linspace(0.0, 1.0, segment_length)
    if np.any(np.diff(x_values) <= 0):
        return np.linspace(0.0, 1.0, segment_length)

    duration = x_values[-1] - x_values[0]
    if not np.isfinite(duration) or duration <= 0:
        return np.linspace(0.0, 1.0, segment_length)
    return (x_values - x_values[0]) / duration


def _resample_numeric_segment(
    segment: np.ndarray,
    n_points: int,
    x_segment: Optional[np.ndarray] = None,
) -> np.ndarray:
    segment = np.asarray(segment, dtype=float)
    if len(segment) < 2:
        return np.full(n_points, np.nan)

    source_x = _normalized_source_axis(len(segment), x_segment)
    target_x = np.linspace(0.0, 1.0, n_points)
    finite_mask = np.isfinite(segment)

    if finite_mask.sum() < 2:
        return np.full(n_points, np.nan)

    unique_x, unique_indices = np.unique(source_x[finite_mask], return_index=True)
    unique_y = segment[finite_mask][unique_indices]
    if len(unique_x) < 2:
        return np.full(n_points, np.nan)

    return np.interp(target_x, unique_x, unique_y)


def _resample_state_segment(
    segment: np.ndarray,
    n_points: int,
    x_segment: Optional[np.ndarray] = None,
) -> np.ndarray:
    segment = _valid_state_values(np.asarray(segment, dtype=float))
    if len(segment) < 2:
        return np.full(n_points, -1, dtype=int)

    source_x = _normalized_source_axis(len(segment), x_segment)
    target_x = np.linspace(0.0, 1.0, n_points)
    valid_indices = np.where(np.isin(segment, [0, 1, 2, 3, 4]))[0]
    if len(valid_indices) == 0:
        return np.full(n_points, -1, dtype=int)

    nearest = np.array([
        valid_indices[int(np.argmin(np.abs(source_x[valid_indices] - target)))]
        for target in target_x
    ])
    return segment[nearest].astype(int)


def normalize_to_gait_cycle(
    df: pd.DataFrame,
    variable: str,
    n_points: int = RESAMPLE_POINTS,
) -> Tuple[Optional[str], np.ndarray, np.ndarray]:
    """Return variable cycles and matching state cycles resampled to 0-100%."""
    if variable not in df.columns:
        return None, np.empty((0, n_points)), np.empty((0, n_points), dtype=int)

    state_col = detect_state_column_for_variable(df, variable)
    if state_col is None:
        return None, np.empty((0, n_points)), np.empty((0, n_points), dtype=int)

    data_arr = convert_values_for_display(
        variable,
        pd.to_numeric(df[variable], errors="coerce").to_numpy(dtype=float),
    )
    state_arr = pd.to_numeric(df[state_col], errors="coerce").to_numpy(dtype=float)
    time_arr = None
    if "time" in df.columns:
        time_arr = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    boundaries = detect_gait_cycle_boundaries(state_arr)
    if len(boundaries) < 2:
        return state_col, np.empty((0, n_points)), np.empty((0, n_points), dtype=int)

    normalized_cycles = []
    normalized_states = []
    for start_idx, end_idx in zip(boundaries[:-1], boundaries[1:]):
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        if start_idx >= end_idx:
            continue

        x_segment = time_arr[start_idx:end_idx + 1] if time_arr is not None and len(time_arr) == len(data_arr) else None
        cycle = _resample_numeric_segment(data_arr[start_idx:end_idx + 1], n_points, x_segment)
        state_cycle = _resample_state_segment(state_arr[start_idx:end_idx + 1], n_points, x_segment)
        if np.isfinite(cycle).any():
            normalized_cycles.append(cycle)
            normalized_states.append(state_cycle)

    if not normalized_cycles:
        return state_col, np.empty((0, n_points)), np.empty((0, n_points), dtype=int)

    return state_col, np.vstack(normalized_cycles), np.vstack(normalized_states)


def create_normalized_phase_masks(state_cycle: np.ndarray) -> Dict[str, np.ndarray]:
    """Create state and composite phase masks from a normalized state cycle."""
    state_values = np.asarray(state_cycle, dtype=int)
    valid_mask = np.isin(state_values, [0, 1, 2, 3, 4])
    return {
        phase_name: valid_mask & np.isin(state_values, phase_state_values)
        for phase_name, phase_state_values in SCONE_PHASE_STATES.items()
    }


def _dominant_state_cycle(state_cycles: List[np.ndarray], n_points: int) -> Optional[np.ndarray]:
    valid_cycles = [np.asarray(cycle, dtype=int) for cycle in state_cycles if len(cycle) == n_points]
    if not valid_cycles:
        return None

    stacked = np.vstack(valid_cycles)
    dominant = np.full(n_points, -1, dtype=int)
    for col_idx in range(n_points):
        values = stacked[:, col_idx]
        values = values[np.isin(values, [0, 1, 2, 3, 4])]
        if len(values) == 0:
            continue
        counts = np.bincount(values, minlength=5)
        dominant[col_idx] = int(np.argmax(counts))
    return dominant


def _state_runs(state_cycle: np.ndarray) -> List[Tuple[int, int, int]]:
    runs = []
    current_state = None
    start_idx = 0
    for idx, state_value in enumerate(state_cycle):
        state_value = int(state_value)
        if state_value not in SCONE_PHASE_LABELS:
            if current_state is not None:
                runs.append((current_state, start_idx, idx - 1))
                current_state = None
            continue

        if current_state is None:
            current_state = state_value
            start_idx = idx
        elif state_value != current_state:
            runs.append((current_state, start_idx, idx - 1))
            current_state = state_value
            start_idx = idx

    if current_state is not None:
        runs.append((current_state, start_idx, len(state_cycle) - 1))
    return runs


def plot_with_phase_shading(ax, state_cycle: Optional[np.ndarray]) -> None:
    """Draw SCONE state phase shading on a 0-100% gait-cycle axis."""
    if state_cycle is None:
        return

    for state_value, start_idx, end_idx in _state_runs(state_cycle):
        start_pct = start_idx / (len(state_cycle) - 1) * 100.0
        end_pct = end_idx / (len(state_cycle) - 1) * 100.0
        ax.axvspan(
            start_pct,
            end_pct,
            color=PHASE_COLORS[state_value],
            alpha=0.28,
            linewidth=0,
            zorder=0,
        )
        if end_pct - start_pct >= 6:
            ax.text(
                (start_pct + end_pct) / 2,
                0.99,
                SCONE_PHASE_LABELS[state_value],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#475569",
            )


def add_peak_markers(ax, x_percent: np.ndarray, y_values: np.ndarray) -> None:
    """Add max/min markers to a mean gait-cycle waveform."""
    valid_mask = np.isfinite(x_percent) & np.isfinite(y_values)
    if valid_mask.sum() == 0:
        return

    valid_indices = np.where(valid_mask)[0]
    max_idx = int(valid_indices[np.argmax(y_values[valid_mask])])
    min_idx = int(valid_indices[np.argmin(y_values[valid_mask])])

    for marker, color, idx in (
        ("^", "#B91C1C", max_idx),
        ("v", "#1D4ED8", min_idx),
    ):
        x_val = float(x_percent[idx])
        y_val = float(y_values[idx])
        ax.plot(
            [x_val],
            [y_val],
            marker=marker,
            markersize=8,
            color=color,
            linestyle="None",
            label="_nolegend_",
            zorder=5,
        )


def draw_gait_cycle_waveform(
    ax,
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    variable: str,
    show_mean_sd: bool = True,
    show_phase_shading: bool = True,
    show_peak_markers: bool = True,
    n_points: int = RESAMPLE_POINTS,
) -> Tuple[bool, List[str]]:
    """Draw selected variable waveforms normalized to 0-100% gait cycle."""
    warnings = []
    x_percent = np.linspace(0, 100, n_points)
    all_state_cycles = []
    plotted_means = []
    has_data = False

    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(file_paths[file_index]) if file_index < len(file_paths) else f"file_{file_index + 1}"
        legend_label = format_legend_label(filename)

        if variable not in df.columns:
            warnings.append(f"{filename}: '{variable}' variable not found; skipped.")
            continue

        state_col, cycles, state_cycles = normalize_to_gait_cycle(df, variable, n_points=n_points)
        if cycles.size == 0:
            if state_col is None:
                warnings.append(f"{filename}: state column not found; percent gait cycle plot skipped.")
            else:
                warnings.append(f"{filename}: no valid gait cycles found from {state_col}; skipped.")
            continue

        color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
        for cycle in cycles:
            ax.plot(x_percent, cycle, color=color, alpha=0.12, linewidth=0.8)

        mean_cycle = np.nanmean(cycles, axis=0)
        std_cycle = np.nanstd(cycles, axis=0)
        if show_mean_sd and len(cycles) > 1:
            ax.fill_between(
                x_percent,
                mean_cycle - SD_RANGE * std_cycle,
                mean_cycle + SD_RANGE * std_cycle,
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        ax.plot(
            x_percent,
            mean_cycle,
            label=f"{legend_label}  n={len(cycles)}",
            color=color,
            alpha=0.95,
            linewidth=3,
        )
        plotted_means.append(mean_cycle)
        all_state_cycles.extend(list(state_cycles))
        has_data = True

    if not has_data:
        from .plot_waveform import draw_raw_waveform

        raw_has_data, raw_warnings = draw_raw_waveform(
            ax,
            dataframes,
            file_paths,
            variable,
            show_mean_sd=show_mean_sd,
            prefer_time_axis=True,
        )
        if raw_has_data:
            warnings.append("Percent gait cycle data unavailable; fell back to time/index waveform without phase shading.")
        return raw_has_data, warnings + raw_warnings

    if show_phase_shading:
        dominant_state = _dominant_state_cycle(all_state_cycles, n_points)
        plot_with_phase_shading(ax, dominant_state)

    if show_peak_markers:
        for mean_cycle in plotted_means:
            add_peak_markers(ax, x_percent, mean_cycle)

    _style_gait_cycle_axes(ax, variable, has_data)
    return has_data, warnings


def _style_gait_cycle_axes(ax, variable: str, has_data: bool) -> None:
    clean_var_name = get_clean_variable_name(variable)
    ax.set_title(clean_var_name, fontsize=17, fontweight="bold", loc="center", pad=14)
    ax.set_xlabel("Gait Cycle (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel(get_axis_label(variable), fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B2C1")
    ax.spines["bottom"].set_color("#A8B2C1")
    ax.tick_params(colors="#374151", labelsize=10)

    for x_ref in (0, 50, 100):
        ax.axvline(x_ref, color="#94A3B8", linestyle="--", linewidth=1, alpha=0.65)

    if has_data:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid gait cycles found.\nCheck state columns: leg1_r.state / leg0_l.state",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
