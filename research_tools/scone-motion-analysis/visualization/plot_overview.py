"""Multi-panel gait overview figures."""

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.file_utils import format_legend_label
from utils.unit_utils import get_display_unit
from .plot_gait_cycle import normalize_to_gait_cycle
from .plot_waveform import PLOT_COLORS


PANEL_PRESETS = ("Gait overview",)


@dataclass(frozen=True)
class OverviewPanelSpec:
    title: str
    unit: str
    matcher: Callable[[str], bool]


def _name_has_any(name: str, tokens: Tuple[str, ...]) -> bool:
    return any(token in name for token in tokens)


def _is_load_force_torque(name: str, joint: str) -> bool:
    return joint in name and _name_has_any(
        name,
        (".load", "_load", ".force", "_force", ".torque", "_torque"),
    )


def _is_power(name: str, joint: str) -> bool:
    return joint in name and "power" in name


def _is_angle_panel_variable(name: str, prefix: str, joint: str) -> bool:
    if get_display_unit(name) != "deg":
        return False
    return name.startswith(prefix) or (joint in name and "angle" in name)


OVERVIEW_PANEL_SPECS = [
    OverviewPanelSpec("Hip angle", "deg", lambda n: _is_angle_panel_variable(n, "hip_flexion_", "hip")),
    OverviewPanelSpec("Knee angle", "deg", lambda n: _is_angle_panel_variable(n, "knee_angle_", "knee")),
    OverviewPanelSpec("Ankle angle", "deg", lambda n: _is_angle_panel_variable(n, "ankle_angle_", "ankle")),
    OverviewPanelSpec("Hip load / force / torque", "N / Nm / raw", lambda n: _is_load_force_torque(n, "hip")),
    OverviewPanelSpec("Knee load / force / torque", "N / Nm / raw", lambda n: _is_load_force_torque(n, "knee")),
    OverviewPanelSpec("Ankle load / force / torque", "N / Nm / raw", lambda n: _is_load_force_torque(n, "ankle")),
    OverviewPanelSpec("AP GRF", "BW / raw", lambda n: "grf" in n and (n.endswith("_x") or n.endswith(".x") or "grf_norm_x" in n)),
    OverviewPanelSpec("Vertical GRF", "BW / raw", lambda n: "grf" in n and (n.endswith("_y") or n.endswith(".y") or "grf_norm_y" in n)),
    OverviewPanelSpec("Hip power", "W / raw", lambda n: _is_power(n, "hip")),
    OverviewPanelSpec("Knee power", "W / raw", lambda n: _is_power(n, "knee")),
    OverviewPanelSpec("Ankle power", "W / raw", lambda n: _is_power(n, "ankle")),
]


def get_condition_labels(file_paths: List[str]) -> List[str]:
    """Return unique condition labels in file order."""
    labels = []
    for path in file_paths:
        label = format_legend_label(str(path).split("\\")[-1].split("/")[-1])
        if label not in labels:
            labels.append(label)
    return labels


def _select_panel_variables(spec: OverviewPanelSpec, all_variables: List[str]) -> List[str]:
    matched = [var for var in all_variables if spec.matcher(var.lower())]
    return sorted(matched, key=lambda value: (value.lower().replace("leg0_l", "0").replace("leg1_r", "1"), value))


def collect_overview_panels(
    dataframes: List[pd.DataFrame],
    scope_variables: Optional[List[str]] = None,
) -> Tuple[List[Tuple[OverviewPanelSpec, List[str]]], List[str]]:
    """Find preset panel variables from loaded dataframes."""
    scope_set = set(scope_variables or [])
    all_variables = sorted({
        str(column)
        for df in dataframes
        for column in df.columns
        if str(column).lower() != "time" and pd.api.types.is_numeric_dtype(df[column])
        and (not scope_set or str(column) in scope_set)
    })

    panels = []
    warnings = []
    for spec in OVERVIEW_PANEL_SPECS:
        variables = _select_panel_variables(spec, all_variables)
        if variables:
            panels.append((spec, variables))
        else:
            warnings.append(f"{spec.title}: matching variable not found; skipped.")

    return panels, warnings


def _line_style_for_variable(variable: str) -> str:
    lower = variable.lower()
    if "_l" in lower or ".l" in lower or "leg0_l" in lower or "left" in lower:
        return "--"
    return "-"


def _short_variable_label(variable: str) -> str:
    lower = variable.lower()
    if "_r" in lower or ".r" in lower or "leg1_r" in lower or "right" in lower:
        return "R"
    if "_l" in lower or ".l" in lower or "leg0_l" in lower or "left" in lower:
        return "L"
    return variable


def _collect_panel_series(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    variables: List[str],
    sd_multiplier: float,
) -> Tuple[List[Dict], List[str]]:
    series = []
    warnings = []

    for file_index, df in enumerate(dataframes):
        filename = str(file_paths[file_index]).split("\\")[-1].split("/")[-1] if file_index < len(file_paths) else f"file_{file_index + 1}"
        condition = format_legend_label(filename)

        for variable in variables:
            if variable not in df.columns:
                warnings.append(f"{filename}: '{variable}' not found; skipped.")
                continue

            state_col, cycles, _ = normalize_to_gait_cycle(df, variable)
            if cycles.size == 0:
                if state_col is None:
                    warnings.append(f"{filename}: state column not found for '{variable}'; skipped.")
                else:
                    warnings.append(f"{filename}: no valid gait cycles for '{variable}' from {state_col}; skipped.")
                continue

            mean_curve = np.nanmean(cycles, axis=0)
            sd_curve = np.nanstd(cycles, axis=0) * sd_multiplier
            series.append({
                "condition": condition,
                "variable": variable,
                "label": f"{condition} {_short_variable_label(variable)}",
                "mean": mean_curve,
                "sd": sd_curve,
                "cycles": len(cycles),
                "file_index": file_index,
                "line_style": _line_style_for_variable(variable),
            })

    return series, warnings


def _style_overview_axis(ax, spec: OverviewPanelSpec, reference_condition: Optional[str]) -> None:
    ax.set_title(spec.title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Gait Cycle (%)", fontsize=9, fontweight="bold")
    ax.set_ylabel(spec.unit, fontsize=9, fontweight="bold")
    ax.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A8B2C1")
    ax.spines["bottom"].set_color("#A8B2C1")
    ax.tick_params(colors="#374151", labelsize=8)
    for x_ref in (0, 50, 100):
        ax.axvline(x_ref, color="#94A3B8", linestyle="--", linewidth=0.8, alpha=0.6)
    if reference_condition:
        ax.text(
            0.01,
            0.98,
            f"ref: {reference_condition}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#475569",
        )


def _add_shared_legend(fig, legend_entries: Dict[str, object]) -> None:
    if not legend_entries:
        return
    labels = list(legend_entries)
    handles = [legend_entries[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.985, 0.5),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )


def create_overview_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    preset_name: str = "Gait overview",
    reference_condition: Optional[str] = None,
    sd_multiplier: float = 1.0,
    show_sd_band: bool = True,
    scope_variables: Optional[List[str]] = None,
) -> Tuple[object, bool, List[str]]:
    """Create a matplotlib multi-panel overview figure."""
    import matplotlib.pyplot as plt

    panels, warnings = collect_overview_panels(dataframes, scope_variables=scope_variables)
    if preset_name not in PANEL_PRESETS:
        warnings.append(f"Unknown overview preset '{preset_name}', using Gait overview.")

    drawable_panels = []
    panel_series_cache = []
    for spec, variables in panels:
        series, panel_warnings = _collect_panel_series(dataframes, file_paths, variables, sd_multiplier)
        warnings.extend(panel_warnings)
        if series:
            drawable_panels.append((spec, variables))
            panel_series_cache.append(series)

    if not drawable_panels:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No overview variables with valid gait cycles found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, warnings

    n_panels = len(drawable_panels)
    n_cols = 3 if n_panels >= 3 else n_panels
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols + 1.2, 3.3 * n_rows), squeeze=False)
    x_percent = np.linspace(0, 100, 101)
    legend_entries: Dict[str, object] = {}

    for panel_idx, ((spec, _), series_list) in enumerate(zip(drawable_panels, panel_series_cache)):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx][col_idx]

        for series in series_list:
            color = PLOT_COLORS[series["file_index"] % len(PLOT_COLORS)]
            is_reference = reference_condition and series["condition"] == reference_condition
            line_width = 2.7 if is_reference else 1.8
            alpha = 1.0 if is_reference else 0.82

            if show_sd_band and np.isfinite(series["sd"]).any():
                ax.fill_between(
                    x_percent,
                    series["mean"] - series["sd"],
                    series["mean"] + series["sd"],
                    color=color,
                    alpha=0.12 if is_reference else 0.08,
                    linewidth=0,
                )

            line = ax.plot(
                x_percent,
                series["mean"],
                label=series["label"],
                color=color,
                linestyle=series["line_style"],
                linewidth=line_width,
                alpha=alpha,
            )[0]
            if series["label"] not in legend_entries:
                legend_entries[series["label"]] = line

        _style_overview_axis(ax, spec, reference_condition)

    for unused_idx in range(n_panels, n_rows * n_cols):
        row_idx, col_idx = divmod(unused_idx, n_cols)
        axes[row_idx][col_idx].set_visible(False)

    _add_shared_legend(fig, legend_entries)
    fig.suptitle("Gait Overview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    return fig, True, warnings


def create_overview_plotly_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    preset_name: str = "Gait overview",
    reference_condition: Optional[str] = None,
    sd_multiplier: float = 1.0,
    show_sd_band: bool = True,
):
    """Create a Plotly HTML-ready multi-panel overview figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels, warnings = collect_overview_panels(dataframes)
    if preset_name not in PANEL_PRESETS:
        warnings.append(f"Unknown overview preset '{preset_name}', using Gait overview.")

    drawable_panels = []
    panel_series_cache = []
    for spec, variables in panels:
        series, panel_warnings = _collect_panel_series(dataframes, file_paths, variables, sd_multiplier)
        warnings.extend(panel_warnings)
        if series:
            drawable_panels.append((spec, variables))
            panel_series_cache.append(series)

    if not drawable_panels:
        fig = go.Figure()
        fig.add_annotation(text="No overview variables with valid gait cycles found.", x=0.5, y=0.5, showarrow=False)
        return fig, False, warnings

    n_panels = len(drawable_panels)
    n_cols = 3 if n_panels >= 3 else n_panels
    n_rows = int(math.ceil(n_panels / n_cols))
    subplot_titles = [spec.title for spec, _ in drawable_panels]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    x_percent = np.linspace(0, 100, 101)

    for panel_idx, ((spec, _), series_list) in enumerate(zip(drawable_panels, panel_series_cache)):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        row = row_idx + 1
        col = col_idx + 1

        for series in series_list:
            color = PLOT_COLORS[series["file_index"] % len(PLOT_COLORS)]
            is_reference = reference_condition and series["condition"] == reference_condition
            line_width = 3 if is_reference else 2
            dash = "dash" if series["line_style"] == "--" else "solid"

            if show_sd_band and np.isfinite(series["sd"]).any():
                upper = series["mean"] + series["sd"]
                lower = series["mean"] - series["sd"]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_percent, x_percent[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.10 if is_reference else 0.07),
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    mode="lines",
                ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=x_percent,
                y=series["mean"],
                mode="lines",
                name=series["label"],
                legendgroup=series["label"],
                line=dict(color=color, width=line_width, dash=dash),
                hovertemplate=(
                    f"<b>{series['label']}</b><br>"
                    "Gait cycle: %{x:.1f}%<br>"
                    "Mean: %{y:.4f}<extra></extra>"
                ),
            ), row=row, col=col)

        fig.update_xaxes(title_text="Gait cycle (%)", range=[0, 100], row=row, col=col)
        fig.update_yaxes(title_text=spec.unit, row=row, col=col)

    title = "Gait Overview"
    if reference_condition:
        title += f" | reference: {reference_condition}"

    fig.update_layout(
        title=title,
        template="simple_white",
        hovermode="x unified",
        height=max(420, 310 * n_rows),
        width=max(900, 390 * n_cols),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, True, warnings


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"
