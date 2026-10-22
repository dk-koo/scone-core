"""GRF, COP, COM, momentum, and balance-strategy visualization helpers."""

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import RESAMPLE_POINTS
from utils.file_utils import format_legend_label
from .plot_gait_cycle import normalize_to_gait_cycle
from .plot_waveform import PLOT_COLORS


SIDE_PREFIXES = {
    "leg0_l": "Left",
    "leg1_r": "Right",
}
AXES = ("x", "y", "z")
SIDE_LINESTYLES = {"Left": "-", "Right": "--"}


def detect_grf_variables(columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Detect leg0_l/leg1_r GRF variables by side and axis."""
    return _detect_side_axis_variables(columns, token="grf")


def detect_cop_variables(columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Detect leg0_l/leg1_r COP variables by side and axis."""
    return _detect_side_axis_variables(columns, token="cop")


def detect_com_variables(columns: Iterable[str]) -> Dict[str, str]:
    """Detect com_x/com_y/com_z variables."""
    detected = {}
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for axis in AXES:
        for candidate in (f"com_{axis}", f"com.{axis}"):
            if candidate in lower_to_original:
                detected[axis] = lower_to_original[candidate]
                break
    return detected


def detect_momentum_variables(columns: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Detect linear and angular momentum variables by axis."""
    result = {"linear": {}, "angular": {}}
    for column in columns:
        name = str(column)
        lower = name.lower()
        axis = _axis_suffix(lower)
        if axis is None:
            continue
        if lower.startswith(("lin_mom_", "lin_mom.")):
            result["linear"][axis] = name
        elif lower.startswith(("ang_mom_", "ang_mom.")):
            result["angular"][axis] = name
    return result


def _detect_side_axis_variables(columns: Iterable[str], token: str) -> Dict[str, Dict[str, str]]:
    detected = {side_label: {} for side_label in SIDE_PREFIXES.values()}
    for column in columns:
        name = str(column)
        lower = name.lower()
        axis = _axis_suffix(lower)
        if axis is None:
            continue

        for prefix, side_label in SIDE_PREFIXES.items():
            if not lower.startswith(prefix):
                continue
            remainder = lower[len(prefix):]
            if token == "grf" and "grf" in remainder and "grm" not in remainder:
                detected[side_label][axis] = name
            elif token == "cop" and "cop" in remainder:
                detected[side_label][axis] = name
    return {side: axes for side, axes in detected.items() if axes}


def _axis_suffix(name: str) -> Optional[str]:
    for axis in AXES:
        if name.endswith(f"_{axis}") or name.endswith(f".{axis}"):
            return axis
    return None


def _condition_label(file_paths: Sequence[str], file_index: int) -> str:
    if file_index < len(file_paths):
        filename = os.path.basename(str(file_paths[file_index]))
    else:
        filename = f"file_{file_index + 1}"
    return format_legend_label(filename)


def _side_linestyle(side: str) -> str:
    return SIDE_LINESTYLES.get(side, "-")


def _add_shared_legend(fig, axes: Sequence[object]) -> bool:
    legend_entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or str(label).startswith("_") or label in legend_entries:
                continue
            legend_entries[label] = handle

    if not legend_entries:
        return False

    fig.legend(
        list(legend_entries.values()),
        list(legend_entries.keys()),
        loc="center left",
        bbox_to_anchor=(0.81, 0.5),
        frameon=False,
        fontsize=6 if len(legend_entries) > 18 else 7,
        title="Legend",
        title_fontsize=8,
    )
    return True


def _resample_whole_series(df: pd.DataFrame, variable: str, n_points: int = RESAMPLE_POINTS) -> np.ndarray:
    values = pd.to_numeric(df[variable], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.full(n_points, np.nan)
    source_x = np.arange(len(values), dtype=float)
    target_x = np.linspace(0.0, len(values) - 1, n_points)
    return np.interp(target_x, source_x[finite_mask], values[finite_mask])


def _mean_percent_curve(
    df: pd.DataFrame,
    variable: str,
    filename: str,
    warnings: List[str],
) -> Tuple[Optional[np.ndarray], int]:
    if variable not in df.columns:
        warnings.append(f"{filename}: '{variable}' not found; skipped.")
        return None, 0

    state_col, cycles, _ = normalize_to_gait_cycle(df, variable, n_points=RESAMPLE_POINTS)
    if cycles.size > 0:
        return np.nanmean(cycles, axis=0), int(cycles.shape[0])

    fallback_curve = _resample_whole_series(df, variable, n_points=RESAMPLE_POINTS)
    if np.isfinite(fallback_curve).sum() >= 2:
        if state_col is None:
            warnings.append(f"{filename}: state column not found for '{variable}'; whole signal resampled.")
        else:
            warnings.append(f"{filename}: no gait cycles for '{variable}' from {state_col}; whole signal resampled.")
        return fallback_curve, 1

    warnings.append(f"{filename}: '{variable}' has no finite numeric data; skipped.")
    return None, 0


def _integrate_curve(curve: np.ndarray, positive: bool) -> float:
    values = np.asarray(curve, dtype=float)
    x_percent = np.linspace(0.0, 100.0, len(values))
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.nan
    if positive:
        target = np.maximum(values, 0.0)
    else:
        target = np.minimum(values, 0.0)
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(target[finite_mask], x_percent[finite_mask]))


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    return float(np.sqrt(np.mean(finite ** 2)))


def _style_wave_axis(ax, title: str, y_label: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Gait cycle (%)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=8)


def _mark_vertical_grf_events(ax, x_percent: np.ndarray, curve: np.ndarray, color: str) -> None:
    values = np.asarray(curve, dtype=float)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 6:
        return

    first_region = np.where(finite_mask & (x_percent <= 50.0))[0]
    second_region = np.where(finite_mask & (x_percent >= 50.0))[0]
    if len(first_region) == 0 or len(second_region) == 0:
        return

    first_idx = int(first_region[np.argmax(values[first_region])])
    second_idx = int(second_region[np.argmax(values[second_region])])
    if first_idx > second_idx:
        first_idx, second_idx = second_idx, first_idx

    between = np.arange(first_idx, second_idx + 1)
    between = between[np.isfinite(values[between])]
    valley_idx = int(between[np.argmin(values[between])]) if len(between) else None

    markers = [("1st", "^", first_idx), ("2nd", "^", second_idx)]
    if valley_idx is not None:
        markers.append(("valley", "v", valley_idx))

    for label, marker, idx in markers:
        ax.plot(
            [x_percent[idx]],
            [values[idx]],
            marker=marker,
            markersize=6,
            linestyle="None",
            color=color,
            label="_nolegend_",
            zorder=5,
        )
        ax.text(
            x_percent[idx],
            values[idx],
            f"{label}\n{x_percent[idx]:.0f}%",
            fontsize=7,
            color=color,
            ha="center",
            va="bottom" if marker == "^" else "top",
        )


def _all_numeric_columns(dataframes: Sequence[pd.DataFrame]) -> List[str]:
    return sorted({
        str(column)
        for df in dataframes
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    })


def _draw_vertical_grf(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        grf_vars = detect_grf_variables(df.columns)
        for side, axes in grf_vars.items():
            variable = axes.get("y")
            if not variable:
                continue
            curve, n_cycles = _mean_percent_curve(df, variable, filename, warnings)
            if curve is None:
                continue
            color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
            ax.plot(
                x_percent,
                curve,
                color=color,
                linestyle=_side_linestyle(side),
                linewidth=2.0,
                alpha=0.9,
                label=f"{condition} {side}",
            )
            _mark_vertical_grf_events(ax, x_percent, curve, color)
            has_data = True
    if has_data:
        _style_wave_axis(ax, "Vertical GRF", "GRF y")
    return has_data


def _draw_ap_grf(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    has_data = False
    text_lines = []
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        grf_vars = detect_grf_variables(df.columns)
        for side, axes in grf_vars.items():
            variable = axes.get("x")
            if not variable:
                continue
            curve, n_cycles = _mean_percent_curve(df, variable, filename, warnings)
            if curve is None:
                continue
            color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
            braking_impulse = abs(_integrate_curve(curve, positive=False))
            propulsive_impulse = _integrate_curve(curve, positive=True)
            label = f"{condition} {side}"
            ax.plot(
                x_percent,
                curve,
                color=color,
                linestyle=_side_linestyle(side),
                linewidth=2.0,
                alpha=0.9,
                label=label,
            )
            ax.fill_between(x_percent, 0, curve, where=np.asarray(curve) < 0, color="#1D4ED8", alpha=0.10)
            ax.fill_between(x_percent, 0, curve, where=np.asarray(curve) > 0, color="#D97706", alpha=0.10)
            text_lines.append(f"{condition} {side}: brake {braking_impulse:.3g}, prop {propulsive_impulse:.3g}")
            has_data = True
    if has_data:
        _style_wave_axis(ax, "AP GRF braking / propulsion", "GRF x")
        ax.axhline(0, color="#64748B", linewidth=0.8, linestyle="--")
        ax.text(0.01, 0.98, "\n".join(text_lines[:5]), transform=ax.transAxes, ha="left", va="top", fontsize=7)
    return has_data


def _draw_cop_trajectory(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        cop_vars = detect_cop_variables(df.columns)
        for side, axes in cop_vars.items():
            x_var = axes.get("x")
            z_var = axes.get("z") or axes.get("y")
            if not x_var or not z_var:
                warnings.append(f"{filename}: {side} COP trajectory needs x and z/y variables; skipped.")
                continue
            x_curve, _ = _mean_percent_curve(df, x_var, filename, warnings)
            z_curve, _ = _mean_percent_curve(df, z_var, filename, warnings)
            if x_curve is None or z_curve is None:
                continue
            color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
            ax.plot(
                x_curve,
                z_curve,
                color=color,
                linestyle=_side_linestyle(side),
                linewidth=1.9,
                alpha=0.9,
                label=f"{condition} {side}",
            )
            ax.scatter([x_curve[0]], [z_curve[0]], color=color, marker="o", s=18, label="_nolegend_")
            ax.scatter([x_curve[-1]], [z_curve[-1]], color=color, marker="x", s=28, label="_nolegend_")
            has_data = True
    if has_data:
        ax.set_title("COP trajectory", fontsize=10, fontweight="bold")
        ax.set_xlabel("COP x")
        ax.set_ylabel("COP z/y")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    return has_data


def _draw_com_trajectory(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        com_vars = detect_com_variables(df.columns)
        x_var = com_vars.get("x")
        z_var = com_vars.get("z") or com_vars.get("y")
        if not x_var or not z_var:
            continue
        x_curve, _ = _mean_percent_curve(df, x_var, filename, warnings)
        z_curve, _ = _mean_percent_curve(df, z_var, filename, warnings)
        if x_curve is None or z_curve is None:
            continue
        color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
        ax.plot(x_curve, z_curve, color=color, linewidth=2.1, alpha=0.9, label=condition)
        ax.scatter([x_curve[0]], [z_curve[0]], color=color, marker="o", s=18, label="_nolegend_")
        ax.scatter([x_curve[-1]], [z_curve[-1]], color=color, marker="x", s=28, label="_nolegend_")
        has_data = True
    if has_data:
        ax.set_title("COM trajectory", fontsize=10, fontweight="bold")
        ax.set_xlabel("COM x")
        ax.set_ylabel("COM z/y")
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    return has_data


def _draw_com_cop_distance(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        com_vars = detect_com_variables(df.columns)
        cop_vars = detect_cop_variables(df.columns)
        vertical_axis = "z" if "z" in com_vars else "y" if "y" in com_vars else None
        if "x" not in com_vars or vertical_axis is None:
            continue
        com_x, _ = _mean_percent_curve(df, com_vars["x"], filename, warnings)
        com_vertical, _ = _mean_percent_curve(df, com_vars[vertical_axis], filename, warnings)
        if com_x is None or com_vertical is None:
            continue
        for side, axes in cop_vars.items():
            cop_vertical_axis = "z" if "z" in axes else vertical_axis if vertical_axis in axes else None
            if "x" not in axes or cop_vertical_axis is None:
                continue
            cop_x, _ = _mean_percent_curve(df, axes["x"], filename, warnings)
            cop_vertical, _ = _mean_percent_curve(df, axes[cop_vertical_axis], filename, warnings)
            if cop_x is None or cop_vertical is None:
                continue
            distance = np.sqrt((com_x - cop_x) ** 2 + (com_vertical - cop_vertical) ** 2)
            color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
            ax.plot(
                x_percent,
                distance,
                color=color,
                linestyle=_side_linestyle(side),
                linewidth=1.9,
                alpha=0.9,
                label=f"{condition} {side}",
            )
            has_data = True
    if has_data:
        _style_wave_axis(ax, "COM-COP distance", "Distance")
    return has_data


def _draw_com_velocity(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        com_vars = detect_com_variables(df.columns)
        curves = []
        for axis in AXES:
            variable = com_vars.get(axis)
            if variable:
                curve, _ = _mean_percent_curve(df, variable, filename, warnings)
                if curve is not None:
                    curves.append(curve)
        if len(curves) < 2:
            continue
        velocity_components = [np.gradient(curve, x_percent) for curve in curves]
        speed = np.sqrt(np.sum(np.vstack(velocity_components) ** 2, axis=0))
        color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
        ax.plot(x_percent, speed, color=color, linewidth=2.0, alpha=0.9, label=f"{condition} RMS={_rms(speed):.3g}")
        has_data = True
    if has_data:
        _style_wave_axis(ax, "COM position change rate", "Position change per % gait")
    return has_data


def _draw_momentum(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    x_percent = np.linspace(0.0, 100.0, RESAMPLE_POINTS)
    has_data = False
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        momentum_vars = detect_momentum_variables(df.columns)
        for kind, axes in momentum_vars.items():
            for axis, variable in sorted(axes.items()):
                curve, _ = _mean_percent_curve(df, variable, filename, warnings)
                if curve is None:
                    continue
                color = PLOT_COLORS[file_index % len(PLOT_COLORS)]
                linestyle = "-" if kind == "linear" else "--"
                label = f"{condition} {kind} {axis} RMS={_rms(curve):.3g}"
                ax.plot(x_percent, curve, color=color, linestyle=linestyle, linewidth=1.7, alpha=0.82, label=label)
                has_data = True
    if has_data:
        _style_wave_axis(ax, "Linear / angular momentum", "Momentum")
    return has_data


def _balance_joint_variable(joint: str, column: str) -> bool:
    lower = column.lower()
    return joint in lower and any(token in lower for token in ("moment", "torque", "load"))


def _classify_strategy(contributions: Dict[str, float]) -> str:
    finite = {joint: value for joint, value in contributions.items() if np.isfinite(value) and value > 0}
    total = sum(finite.values())
    if total <= 0:
        return "mixed strategy"
    shares = {joint: value / total for joint, value in finite.items()}
    top_joint, top_share = max(shares.items(), key=lambda item: item[1])
    sorted_shares = sorted(shares.values(), reverse=True)
    if top_share >= 0.50 and (len(sorted_shares) == 1 or top_share - sorted_shares[1] >= 0.15):
        return f"{top_joint} strategy"
    return "mixed strategy"


def _draw_balance_strategy(ax, dataframes: List[pd.DataFrame], file_paths: List[str], warnings: List[str]) -> bool:
    rows = []
    for file_index, df in enumerate(dataframes):
        filename = os.path.basename(str(file_paths[file_index])) if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = _condition_label(file_paths, file_index)
        row = {"condition": condition, "ankle": 0.0, "knee": 0.0, "hip": 0.0}
        found = False
        for joint in ("ankle", "knee", "hip"):
            joint_values = []
            for column in df.columns:
                if not _balance_joint_variable(joint, str(column)):
                    continue
                curve, _ = _mean_percent_curve(df, str(column), filename, warnings)
                if curve is not None:
                    joint_values.append(_rms(curve))
            if joint_values:
                row[joint] = float(np.nanmean(joint_values))
                found = True
        if found:
            row["strategy"] = _classify_strategy({joint: row[joint] for joint in ("ankle", "knee", "hip")})
            rows.append(row)

    if not rows:
        warnings.append("Balance strategy plot skipped; no ankle/knee/hip moment, torque, or load variables found.")
        return False

    table = pd.DataFrame(rows)
    x = np.arange(len(table))
    bottom = np.zeros(len(table), dtype=float)
    colors = {"ankle": "#0072B2", "knee": "#009E73", "hip": "#D55E00"}
    for joint in ("ankle", "knee", "hip"):
        values = table[joint].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors[joint], alpha=0.85, label=joint)
        bottom += values
    for idx, strategy in enumerate(table["strategy"]):
        ax.text(idx, bottom[idx], strategy, ha="center", va="bottom", fontsize=8, rotation=20)
    ax.set_title("Balance strategy proxy", fontsize=10, fontweight="bold")
    ax.set_ylabel("Joint load/moment RMS")
    ax.set_xticks(x)
    ax.set_xticklabels(table["condition"], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    return True


def create_grf_cop_com_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
) -> Tuple[object, bool, List[str]]:
    """Create a multi-panel GRF/COP/COM overview figure."""
    import matplotlib.pyplot as plt

    warnings: List[str] = []
    if not dataframes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No loaded data.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No loaded data."]

    columns = _all_numeric_columns(dataframes)
    if not detect_grf_variables(columns):
        warnings.append("No leg0_l/leg1_r GRF variables found.")
    if not detect_cop_variables(columns):
        warnings.append("No leg0_l/leg1_r COP variables found.")
    if not detect_com_variables(columns):
        warnings.append("No com_x/com_y/com_z variables found.")

    draw_steps = (
        _draw_vertical_grf,
        _draw_ap_grf,
        _draw_cop_trajectory,
        _draw_com_trajectory,
        _draw_com_cop_distance,
        _draw_com_velocity,
        _draw_momentum,
        _draw_balance_strategy,
    )

    n_cols = 2
    n_rows = int(math.ceil(len(draw_steps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    used_count = 0

    for draw_step in draw_steps:
        ax = axes_flat[used_count]
        has_panel_data = draw_step(ax, dataframes, file_paths, warnings)
        if has_panel_data:
            used_count += 1

    for idx in range(used_count, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if used_count == 0:
        axes_flat[0].set_visible(True)
        axes_flat[0].text(0.5, 0.5, "No GRF/COP/COM data could be plotted.", ha="center", va="center")
        axes_flat[0].set_axis_off()
        return fig, False, warnings

    fig.suptitle("GRF / COP / COM Overview", fontsize=14, fontweight="bold")
    has_legend = _add_shared_legend(fig, axes_flat[:used_count])
    fig.tight_layout(rect=(0, 0, 0.79 if has_legend else 1, 0.96))
    return fig, True, warnings
