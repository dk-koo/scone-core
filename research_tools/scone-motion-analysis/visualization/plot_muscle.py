"""Muscle activation and co-contraction visualization helpers."""

from dataclasses import dataclass
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import RESAMPLE_POINTS
from utils.file_utils import format_legend_label, sanitize_dataframe_for_spreadsheet_export
from .plot_gait_cycle import create_normalized_phase_masks, normalize_to_gait_cycle
from .plot_waveform import PLOT_COLORS


SIGNAL_TYPES = ("activation", "excitation", "input")
SIDE_LABELS = {"r": "Right", "l": "Left", "": "Unknown"}
SIDE_VALUES = ("r", "l")

CCI_PAIR_DEFINITIONS = {
    "hip": (("glut_max", "hamstrings"), ("iliopsoas",)),
    "knee": (("vasti", "rect_fem"), ("hamstrings", "bifemsh")),
    "ankle": (("gastroc", "soleus"), ("tib_ant",)),
}

SIGNAL_STYLES = {
    "activation": {"linestyle": "-", "linewidth": 2.4, "alpha": 0.95},
    "excitation": {"linestyle": "--", "linewidth": 1.7, "alpha": 0.78},
    "input": {"linestyle": ":", "linewidth": 1.8, "alpha": 0.78},
}


@dataclass(frozen=True)
class MuscleSignal:
    variable: str
    muscle: str
    side: str
    signal: str


def detect_muscle_signal_variables(columns: Iterable[str]) -> List[MuscleSignal]:
    """Detect activation/excitation/input variables from column names."""
    detected = []
    for column in columns:
        parsed = parse_muscle_signal_name(str(column))
        if parsed is not None:
            detected.append(parsed)
    return sorted(detected, key=lambda item: (item.muscle, item.side, item.signal, item.variable))


def parse_muscle_signal_name(variable: str) -> Optional[MuscleSignal]:
    """Parse a SCONE/OpenSim-style muscle signal variable name."""
    lower = variable.lower()
    signal = None
    base = variable

    for candidate in SIGNAL_TYPES:
        for suffix in (f".{candidate}", f"_{candidate}"):
            if lower.endswith(suffix):
                signal = candidate
                base = variable[: -len(suffix)]
                break
        if signal is not None:
            break

    if signal is None:
        return None

    side, muscle = _extract_side_and_muscle(base)
    if not muscle:
        return None

    return MuscleSignal(variable=variable, muscle=muscle, side=side, signal=signal)


def _extract_side_and_muscle(base_name: str) -> Tuple[str, str]:
    base = base_name.strip().lower()
    side = ""

    if "leg0_l" in base or re.search(r"(^|[._-])left([._-]|$)", base):
        side = "l"
    elif "leg1_r" in base or re.search(r"(^|[._-])right([._-]|$)", base):
        side = "r"

    cleaned = re.sub(r"^(leg0_l|leg1_r)[._-]?", "", base)
    cleaned = re.sub(r"(^|[._-])(left|right)([._-]|$)", r"\1", cleaned).strip("._-")

    suffix_match = re.search(r"([._-])([rl])$", cleaned)
    if suffix_match:
        side = suffix_match.group(2)
        cleaned = cleaned[: suffix_match.start()].strip("._-")

    prefix_match = re.search(r"^([rl])([._-])", cleaned)
    if prefix_match and not side:
        side = prefix_match.group(1)
        cleaned = cleaned[prefix_match.end():].strip("._-")

    return side, cleaned


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _muscle_matches(muscle_name: str, requested_token: str) -> bool:
    muscle = _normalize_token(muscle_name)
    token = _normalize_token(requested_token)
    return bool(token) and (token in muscle or muscle in token)


def _filter_side(records: Sequence[MuscleSignal], side_filter: str) -> List[MuscleSignal]:
    normalized_filter = _normalize_side_filter(side_filter)
    if normalized_filter == "both":
        return list(records)
    side = "r" if normalized_filter == "right" else "l"
    return [record for record in records if record.side == side]


def _normalize_side_filter(side_filter: str) -> str:
    value = str(side_filter or "both").strip().lower()
    if value in ("r", "right"):
        return "right"
    if value in ("l", "left"):
        return "left"
    return "both"


def _sides_for_filter(side_filter: str) -> Tuple[str, ...]:
    normalized_filter = _normalize_side_filter(side_filter)
    if normalized_filter == "right":
        return ("r",)
    if normalized_filter == "left":
        return ("l",)
    return SIDE_VALUES


def _condition_label(file_paths: Sequence[str], file_index: int) -> str:
    if file_index < len(file_paths):
        filename = os.path.basename(str(file_paths[file_index]))
    else:
        filename = f"file_{file_index + 1}"
    return format_legend_label(filename)


def _records_by_key(records: Sequence[MuscleSignal]) -> Dict[Tuple[str, str, str], MuscleSignal]:
    return {(record.muscle, record.side, record.signal): record for record in records}


def _find_signal_record(
    records: Sequence[MuscleSignal],
    muscle: str,
    side: str,
    signal: str,
) -> Optional[MuscleSignal]:
    exact = _records_by_key(records).get((muscle, side, signal))
    if exact is not None:
        return exact

    matches = [
        record
        for record in records
        if record.side == side and record.signal == signal and _muscle_matches(record.muscle, muscle)
    ]
    return sorted(matches, key=lambda record: record.variable)[0] if matches else None


def _normalized_mean_curve(
    df: pd.DataFrame,
    variable: str,
    filename: str,
    warnings: List[str],
) -> Tuple[Optional[np.ndarray], int]:
    mean_curve, n_cycles, _ = _normalized_mean_curve_with_state(df, variable, filename, warnings)
    return mean_curve, n_cycles


def _normalized_mean_curve_with_state(
    df: pd.DataFrame,
    variable: str,
    filename: str,
    warnings: List[str],
) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
    state_col, cycles, state_cycles = normalize_to_gait_cycle(df, variable, n_points=RESAMPLE_POINTS)
    if cycles.size == 0:
        if state_col is None:
            warnings.append(f"{filename}: state column not found for '{variable}'; skipped.")
        else:
            warnings.append(f"{filename}: no valid gait cycles for '{variable}' from {state_col}; skipped.")
        return None, 0, None

    return np.nanmean(cycles, axis=0), int(cycles.shape[0]), _dominant_state_cycle(state_cycles)


def _dominant_state_cycle(state_cycles: np.ndarray) -> Optional[np.ndarray]:
    if state_cycles.size == 0:
        return None
    states = np.asarray(state_cycles, dtype=int)
    if states.ndim != 2:
        return None

    dominant = np.full(states.shape[1], -1, dtype=int)
    for col_idx in range(states.shape[1]):
        values = states[:, col_idx]
        values = values[np.isin(values, [0, 1, 2, 3, 4])]
        if len(values) == 0:
            continue
        dominant[col_idx] = int(np.argmax(np.bincount(values, minlength=5)))
    return dominant


def _activation_events(curve: np.ndarray, threshold: float) -> Dict[str, float]:
    values = np.asarray(curve, dtype=float)
    x_percent = np.linspace(0.0, 100.0, len(values))
    finite_mask = np.isfinite(values)
    active_mask = finite_mask & (values >= threshold)

    if not np.any(active_mask):
        return {
            "onset_pct": np.nan,
            "offset_pct": np.nan,
            "active_duration_pct": 0.0,
            "duty_factor": 0.0,
        }

    active_indices = np.where(active_mask)[0]
    onset_pct = float(x_percent[active_indices[0]])
    offset_pct = float(x_percent[active_indices[-1]])
    duty_factor = float(np.mean(active_mask))
    return {
        "onset_pct": onset_pct,
        "offset_pct": offset_pct,
        "active_duration_pct": duty_factor * 100.0,
        "duty_factor": duty_factor,
    }


def _integrate_percent_curve(curve: np.ndarray) -> float:
    values = np.asarray(curve, dtype=float)
    x_percent = np.linspace(0.0, 100.0, len(values))
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.nan
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(values[finite_mask], x_percent[finite_mask]))


def _integrate_percent_window(curve: np.ndarray, start_pct: float, end_pct: float) -> float:
    values = np.asarray(curve, dtype=float)
    x_percent = np.linspace(0.0, 100.0, len(values))
    window_mask = (x_percent >= start_pct) & (x_percent <= end_pct)
    finite_mask = window_mask & np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.nan
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(values[finite_mask], x_percent[finite_mask]))


def _integrate_percent_mask(curve: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(curve, dtype=float)
    phase_mask = np.asarray(mask, dtype=bool)
    if len(values) != len(phase_mask):
        return np.nan

    x_percent = np.linspace(0.0, 100.0, len(values))
    valid_mask = phase_mask & np.isfinite(values)
    integrator = getattr(np, "trapezoid", np.trapz)
    total = 0.0
    has_segment = False
    start_idx = None

    for idx, is_valid in enumerate(valid_mask):
        if is_valid and start_idx is None:
            start_idx = idx
        elif not is_valid and start_idx is not None:
            end_idx = idx - 1
            if end_idx > start_idx:
                total += float(integrator(values[start_idx:end_idx + 1], x_percent[start_idx:end_idx + 1]))
                has_segment = True
            start_idx = None

    if start_idx is not None and len(values) - 1 > start_idx:
        total += float(integrator(values[start_idx:], x_percent[start_idx:]))
        has_segment = True

    return total if has_segment else np.nan


def _peak_activation_metrics(curve: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(curve, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.nan, np.nan
    finite_indices = np.where(finite_mask)[0]
    peak_local_idx = int(np.nanargmax(values[finite_mask]))
    peak_idx = int(finite_indices[peak_local_idx])
    x_percent = np.linspace(0.0, 100.0, len(values))
    return float(values[peak_idx]), float(x_percent[peak_idx])


def _style_axis(ax, title: str, y_label: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Gait cycle (%)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=8)


def _scope_records(records: Sequence[MuscleSignal], scope_variables: Optional[Sequence[str]]) -> List[MuscleSignal]:
    if scope_variables is None:
        return list(records)
    scope_set = {str(variable) for variable in scope_variables}
    return [record for record in records if record.variable in scope_set]


def _finalize_grid(fig, axes, used_count: int, total_count: int, right: float = 1.0) -> None:
    for unused_idx in range(used_count, total_count):
        row_idx, col_idx = divmod(unused_idx, axes.shape[1])
        axes[row_idx][col_idx].set_visible(False)
    fig.tight_layout(rect=(0, 0, right, 0.96))


def _add_shared_legend(fig, legend_entries: Dict[str, object], x_anchor: float = 0.82) -> None:
    if not legend_entries:
        return
    labels = list(legend_entries)
    handles = [legend_entries[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(x_anchor, 0.5),
        frameon=False,
        fontsize=6 if len(labels) > 18 else 7,
        title="Condition / signal",
        title_fontsize=8,
    )


def create_muscle_activation_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    side_filter: str = "both",
    threshold: float = 0.05,
    scope_variables: Optional[Sequence[str]] = None,
) -> Tuple[object, bool, List[str]]:
    """Create a subplot grid for activation, excitation, and input waveforms."""
    import matplotlib.pyplot as plt

    all_records = _scope_records(
        detect_muscle_signal_variables(column for df in dataframes for column in df.columns),
        scope_variables,
    )
    filtered_records = _filter_side(all_records, side_filter)
    panel_records = (
        [record for record in filtered_records if record.signal == "activation"]
        if scope_variables is None
        else filtered_records
    )
    panel_keys = sorted({(record.muscle, record.side) for record in panel_records})
    warnings: List[str] = []

    if not panel_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No muscle activation variables found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No muscle activation variables found."]

    n_panels = len(panel_keys)
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.1 * n_rows), squeeze=False)
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    used_panels = 0
    legend_entries: Dict[str, object] = {}

    for panel_idx, (muscle, side) in enumerate(panel_keys):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx][col_idx]
        panel_has_data = False

        for file_index, df in enumerate(dataframes):
            filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
            condition = _condition_label(file_paths, file_index)
            records = _scope_records(detect_muscle_signal_variables(df.columns), scope_variables)
            color = PLOT_COLORS[file_index % len(PLOT_COLORS)]

            for signal in SIGNAL_TYPES:
                record = _find_signal_record(records, muscle, side, signal)
                if record is None:
                    continue
                mean_curve, n_cycles = _normalized_mean_curve(df, record.variable, filename, warnings)
                if mean_curve is None:
                    continue

                style = SIGNAL_STYLES[signal]
                label = f"{condition} {signal}"
                line, = ax.plot(
                    x_percent,
                    mean_curve,
                    label=label,
                    color=color,
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                )
                if label not in legend_entries:
                    legend_entries[label] = line
                panel_has_data = True

                if signal == "activation":
                    events = _activation_events(mean_curve, threshold)
                    onset = events["onset_pct"]
                    offset = events["offset_pct"]
                    if np.isfinite(onset) and np.isfinite(offset):
                        ax.axvspan(onset, offset, color=color, alpha=0.06, linewidth=0)
                        ax.axvline(onset, color=color, linestyle=":", linewidth=0.9, alpha=0.7)
                        ax.axvline(offset, color=color, linestyle=":", linewidth=0.9, alpha=0.7)

        side_label = SIDE_LABELS.get(side, "Unknown")
        _style_axis(ax, f"{muscle} ({side_label})", "Activation / signal")
        if panel_has_data:
            used_panels += 1
        else:
            ax.text(0.5, 0.5, "No valid gait cycles", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Muscle Activation Waveforms", fontsize=14, fontweight="bold")
    legend_right = 0.78 if legend_entries else 1.0
    _finalize_grid(fig, axes, len(panel_keys), n_rows * n_cols, right=legend_right)
    _add_shared_legend(fig, legend_entries, x_anchor=0.79)
    return fig, used_panels > 0, warnings


def compute_activation_summary_rows(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    side_filter: str = "both",
    threshold: float = 0.05,
    scope_variables: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict], List[str]]:
    """Return threshold-independent activation magnitude and timing summary rows."""
    warnings: List[str] = []
    rows: List[Dict] = []

    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        records = [
            record
            for record in _filter_side(
                _scope_records(detect_muscle_signal_variables(df.columns), scope_variables),
                side_filter,
            )
            if record.signal == "activation"
        ]

        for record in records:
            mean_curve, n_cycles, state_cycle = _normalized_mean_curve_with_state(
                df,
                record.variable,
                filename,
                warnings,
            )
            if mean_curve is None:
                continue
            peak_activation, peak_timing_pct = _peak_activation_metrics(mean_curve)
            phase_masks = create_normalized_phase_masks(state_cycle) if state_cycle is not None else {}
            stance_integral = (
                _integrate_percent_mask(mean_curve, phase_masks["stance"])
                if "stance" in phase_masks
                else _integrate_percent_window(mean_curve, 0.0, 60.0)
            )
            swing_integral = (
                _integrate_percent_mask(mean_curve, phase_masks["swing"])
                if "swing" in phase_masks
                else _integrate_percent_window(mean_curve, 60.0, 100.0)
            )
            rows.append({
                "file": filename,
                "condition": condition,
                "muscle": record.muscle,
                "side": SIDE_LABELS.get(record.side, "Unknown"),
                "variable": record.variable,
                "integrated_activation": _integrate_percent_curve(mean_curve),
                "peak_activation": peak_activation,
                "peak_timing_pct": peak_timing_pct,
                "stance_integrated_activation": stance_integral,
                "swing_integrated_activation": swing_integral,
                "n_cycles": n_cycles,
            })

    if not rows:
        warnings.append("No valid activation summary rows were created.")
    return rows, warnings


def create_activation_summary_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    side_filter: str = "both",
    threshold: float = 0.05,
    scope_variables: Optional[Sequence[str]] = None,
) -> Tuple[object, bool, List[str], List[Dict]]:
    """Create activation magnitude, peak timing, and phase-summary plots."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, warnings = compute_activation_summary_rows(
        dataframes,
        file_paths,
        side_filter,
        threshold,
        scope_variables=scope_variables,
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False, sharey=True)
    axes_flat = axes.ravel()

    if not rows:
        axes_flat[0].text(0.5, 0.5, "No activation summary data found.", ha="center", va="center")
        for ax in axes_flat:
            ax.set_axis_off()
        fig.tight_layout()
        return fig, False, warnings, rows

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["muscle", "side", "condition"]).reset_index(drop=True)
    grouped = (
        summary
        .groupby(["muscle", "side", "condition"], as_index=False)
        .agg({
            "integrated_activation": "mean",
            "peak_activation": "mean",
            "peak_timing_pct": "mean",
            "stance_integrated_activation": "mean",
            "swing_integrated_activation": "mean",
        })
    )
    group_keys = list(grouped[["muscle", "side"]].drop_duplicates().itertuples(index=False, name=None))
    conditions = list(grouped["condition"].drop_duplicates())
    y = np.arange(len(group_keys))
    group_labels = [f"{muscle} ({str(side)[0]})" for muscle, side in group_keys]
    bar_height = min(0.72 / max(len(conditions), 1), 0.28)
    offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * bar_height
    condition_colors = {
        condition: PLOT_COLORS[index % len(PLOT_COLORS)]
        for index, condition in enumerate(conditions)
    }
    fig_height = min(18.0, max(9.0, 4.2 + 0.45 * len(group_keys)))
    fig.set_size_inches(15, fig_height, forward=True)

    plot_specs = (
        ("integrated_activation", "Integrated activation", "Activation x % gait"),
        ("peak_activation", "Peak activation", "Activation"),
        ("peak_timing_pct", "Peak timing", "% gait cycle"),
        ("stance_integrated_activation", "Stance activation", "Activation x % gait"),
        ("swing_integrated_activation", "Swing activation", "Activation x % gait"),
    )
    for ax, (column, title, y_label) in zip(axes_flat, plot_specs):
        value_lookup = {
            (row.muscle, row.side, row.condition): float(getattr(row, column))
            for row in grouped.itertuples(index=False)
        }
        for condition_index, condition in enumerate(conditions):
            values = np.array(
                [value_lookup.get((muscle, side, condition), np.nan) for muscle, side in group_keys],
                dtype=float,
            )
            valid_mask = np.isfinite(values)
            if not np.any(valid_mask):
                continue
            ax.barh(
                y[valid_mask] + offsets[condition_index],
                values[valid_mask],
                height=bar_height,
                color=condition_colors[condition],
                alpha=0.86,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(y_label)
        ax.set_yticks(y)
        ax.set_yticklabels(group_labels, fontsize=8)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")

    axes_flat[0].invert_yaxis()

    for idx, ax in enumerate(axes_flat):
        if idx % 3 != 0:
            ax.tick_params(labelleft=False)
    for ax in axes_flat[len(plot_specs):]:
        ax.set_visible(False)

    fig.suptitle("Muscle Activation Summary", fontsize=14, fontweight="bold")
    legend_handles = [
        Patch(facecolor=condition_colors[condition], alpha=0.86, label=condition)
        for condition in conditions
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        fontsize=8,
        title="Condition",
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 0.95))
    return fig, True, warnings, rows


def _group_activation_curve(
    df: pd.DataFrame,
    records: Sequence[MuscleSignal],
    group_tokens: Sequence[str],
    side: str,
    filename: str,
    warnings: List[str],
) -> Tuple[Optional[np.ndarray], List[str], int]:
    curves = []
    missing_tokens = []
    max_cycles = 0

    for token in group_tokens:
        record = _find_signal_record(records, token, side, "activation")
        if record is None:
            missing_tokens.append(token)
            continue
        mean_curve, n_cycles = _normalized_mean_curve(df, record.variable, filename, warnings)
        if mean_curve is None:
            missing_tokens.append(token)
            continue
        curves.append(mean_curve)
        max_cycles = max(max_cycles, n_cycles)

    if missing_tokens or not curves:
        return None, missing_tokens, max_cycles

    return np.nanmean(np.vstack(curves), axis=0), [], max_cycles


def _compute_cci(agonist_curve: np.ndarray, antagonist_curve: np.ndarray) -> np.ndarray:
    agonist = np.asarray(agonist_curve, dtype=float)
    antagonist = np.asarray(antagonist_curve, dtype=float)
    act_low = np.minimum(agonist, antagonist)
    act_high = np.maximum(agonist, antagonist)
    cci = np.zeros_like(act_high, dtype=float)
    valid_mask = np.isfinite(act_low) & np.isfinite(act_high) & (act_high > 0)
    cci[valid_mask] = (act_low[valid_mask] / act_high[valid_mask]) * (
        act_low[valid_mask] + act_high[valid_mask]
    )
    cci[~np.isfinite(cci)] = np.nan
    return cci


def compute_cci_results(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    side_filter: str = "both",
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Compute CCI waveforms and integrated CCI rows."""
    warnings: List[str] = []
    series: List[Dict] = []
    rows: List[Dict] = []
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)

    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        records = detect_muscle_signal_variables(df.columns)

        for side in _sides_for_filter(side_filter):
            for pair_name, (agonist_tokens, antagonist_tokens) in CCI_PAIR_DEFINITIONS.items():
                agonist_curve, agonist_missing, agonist_cycles = _group_activation_curve(
                    df, records, agonist_tokens, side, filename, warnings
                )
                antagonist_curve, antagonist_missing, antagonist_cycles = _group_activation_curve(
                    df, records, antagonist_tokens, side, filename, warnings
                )
                missing = list(agonist_missing) + list(antagonist_missing)
                if missing or agonist_curve is None or antagonist_curve is None:
                    side_label = SIDE_LABELS.get(side, side)
                    warnings.append(
                        f"{filename}: {pair_name} {side_label} CCI skipped; missing activation variables: "
                        f"{', '.join(sorted(set(missing)))}"
                    )
                    continue

                cci = _compute_cci(agonist_curve, antagonist_curve)
                finite_mask = np.isfinite(cci)
                if finite_mask.sum() == 0:
                    warnings.append(f"{filename}: {pair_name} {SIDE_LABELS.get(side, side)} CCI has no finite values.")
                    continue

                peak_idx = int(np.nanargmax(cci))
                row = {
                    "file": filename,
                    "condition": condition,
                    "muscle_pair": pair_name,
                    "side": SIDE_LABELS.get(side, side),
                    "integrated_cci": _integrate_percent_curve(cci),
                    "mean_cci": float(np.nanmean(cci)),
                    "peak_cci": float(np.nanmax(cci)),
                    "peak_cci_pct": float(x_percent[peak_idx]),
                    "n_cycles": max(agonist_cycles, antagonist_cycles),
                }
                rows.append(row)
                series.append({
                    **row,
                    "file_index": file_index,
                    "cci": cci,
                })

    if not series:
        warnings.append("No CCI waveforms were created.")
    return series, rows, warnings


def create_cci_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    side_filter: str = "both",
) -> Tuple[object, bool, List[str], List[Dict]]:
    """Create CCI waveform, integrated CCI bar plot, and heatmap."""
    import matplotlib.pyplot as plt

    series, rows, warnings = compute_cci_results(dataframes, file_paths, side_filter)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), squeeze=False)
    ax_wave, ax_bar, ax_heatmap = axes.ravel()
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)

    if not series:
        ax_wave.text(0.5, 0.5, "No CCI data found.", ha="center", va="center")
        for ax in (ax_wave, ax_bar, ax_heatmap):
            ax.set_axis_off()
        fig.tight_layout()
        return fig, False, warnings, rows

    pair_styles = {"hip": "-", "knee": "--", "ankle": ":"}
    for item in series:
        color = PLOT_COLORS[item["file_index"] % len(PLOT_COLORS)]
        side_short = str(item["side"])[0]
        label = f"{item['condition']} {item['muscle_pair']} {side_short}"
        ax_wave.plot(
            x_percent,
            item["cci"],
            color=color,
            linestyle=pair_styles.get(item["muscle_pair"], "-"),
            linewidth=1.9,
            alpha=0.88,
            label=label,
        )

    _style_axis(ax_wave, "CCI waveform", "CCI")
    ax_wave.legend(fontsize=7, loc="best", frameon=False)

    summary = pd.DataFrame(rows)
    summary["label"] = (
        summary["condition"] + " " + summary["muscle_pair"] + " " + summary["side"].str[0]
    )
    summary = summary.sort_values(["muscle_pair", "side", "condition"]).reset_index(drop=True)
    x = np.arange(len(summary))
    ax_bar.bar(x, summary["integrated_cci"].to_numpy(dtype=float), color="#D55E00", alpha=0.85)
    ax_bar.set_title("Integrated CCI", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("CCI x % gait")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(summary["label"], rotation=60, ha="right", fontsize=7)
    ax_bar.grid(True, axis="y", alpha=0.25, linestyle="--")

    heatmap_rows = summary.copy()
    heatmap_rows["pair_side"] = heatmap_rows["muscle_pair"] + " " + heatmap_rows["side"].str[0]
    matrix = heatmap_rows.pivot_table(
        index="condition",
        columns="pair_side",
        values="integrated_cci",
        aggfunc="mean",
    )
    values = matrix.to_numpy(dtype=float)
    image = ax_heatmap.imshow(values, aspect="auto", cmap="magma")
    ax_heatmap.set_title("Condition x muscle-pair integrated CCI", fontsize=11, fontweight="bold")
    ax_heatmap.set_xlabel("Muscle pair / side")
    ax_heatmap.set_ylabel("Condition")
    ax_heatmap.set_xticks(np.arange(len(matrix.columns)))
    ax_heatmap.set_xticklabels(matrix.columns.astype(str), rotation=35, ha="right")
    ax_heatmap.set_yticks(np.arange(len(matrix.index)))
    ax_heatmap.set_yticklabels(matrix.index.astype(str))
    fig.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02, label="Integrated CCI")

    fig.suptitle("Co-contraction Index", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, True, warnings, rows


def save_integrated_cci_csv(rows: Sequence[Dict], output_path: str) -> str:
    """Save integrated CCI rows to CSV and return the saved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_df = sanitize_dataframe_for_spreadsheet_export(pd.DataFrame(list(rows)))
    export_df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
