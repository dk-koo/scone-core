"""HTML dashboard generation for motion-analysis visualization results."""

import base64
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.file_utils import format_legend_label
from .plot_features import create_feature_plot
from .plot_gait_cycle import draw_gait_cycle_waveform
from .plot_grf_cop_com import (
    create_grf_cop_com_figure,
    detect_com_variables,
    detect_cop_variables,
    detect_grf_variables,
    detect_momentum_variables,
)
from .plot_muscle import (
    compute_activation_summary_rows,
    create_cci_figure,
    create_muscle_activation_figure,
)
from .plot_overview import create_overview_figure
from .plot_symmetry import build_symmetry_table


DASHBOARD_SECTIONS = (
    "Data quality summary",
    "Selected variable waveform",
    "Gait phase summary",
    "Multi-panel gait overview",
    "Feature summary",
    "Muscle activation",
    "CCI",
    "GRF/COP/COM",
    "Symmetry",
    "Condition comparison",
)


def create_visualization_dashboard(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    selected_variables: Sequence[str],
    graph_variables: Sequence[str],
    analysis_results: Dict,
    feature_csv_paths: Optional[Iterable[str]] = None,
    output_root: str = "output",
    section_scope: str = "all",
) -> Tuple[str, str, List[str]]:
    """Create dashboard.html and visualization_summary.md."""
    import matplotlib.pyplot as plt

    warnings: List[str] = []
    feature_csv_paths = list(feature_csv_paths or [])
    output_path = Path(output_root)
    html_dir = output_path / "html"
    reports_dir = output_path / "reports"
    html_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    selected = _unique_nonempty(selected_variables)
    graph_vars = _unique_nonempty(graph_variables) or selected[:4]
    scoped_dataframes = _scope_dataframes_for_variables(dataframes, selected) if section_scope == "selected" else dataframes
    summary = _collect_dashboard_tables(scoped_dataframes, file_paths, selected, analysis_results)

    sections = []
    fig, has_data, section_warnings = _create_quality_figure(summary["quality"])
    warnings.extend(section_warnings)
    sections.append(_section("Data quality summary", fig, _html_table(summary["quality"]), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = _create_waveform_figure(dataframes, file_paths, graph_vars)
    warnings.extend(section_warnings)
    sections.append(_section("Selected variable waveform", fig, _html_table(summary["features"].head(12)), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = _create_phase_summary_figure(summary["phase"])
    warnings.extend(section_warnings)
    sections.append(_section("Gait phase summary", fig, _html_table(summary["phase"].head(20)), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = create_overview_figure(scoped_dataframes, file_paths, scope_variables=selected if section_scope == "selected" else None)
    warnings.extend(section_warnings)
    sections.append(_section("Multi-panel gait overview", fig, _html_table(summary["detected_variables"]), section_warnings))
    plt.close(fig)

    fig, feature_has_data, section_warnings = _create_feature_summary_figure(
        feature_csv_paths,
        summary["features"],
    )
    warnings.extend(section_warnings)
    sections.append(_section("Feature summary", fig, _html_table(summary["features"].head(20)), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = create_muscle_activation_figure(scoped_dataframes, file_paths)
    warnings.extend(section_warnings)
    activation_rows, activation_warnings = compute_activation_summary_rows(scoped_dataframes, file_paths)
    warnings.extend(activation_warnings)
    sections.append(
        _section(
            "Muscle activation",
            fig,
            _html_table(pd.DataFrame(activation_rows).head(20)),
            section_warnings + activation_warnings,
        )
    )
    plt.close(fig)

    fig, has_data, section_warnings, cci_rows = create_cci_figure(scoped_dataframes, file_paths)
    warnings.extend(section_warnings)
    sections.append(_section("CCI", fig, _html_table(pd.DataFrame(cci_rows).head(20)), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = create_grf_cop_com_figure(scoped_dataframes, file_paths)
    warnings.extend(section_warnings)
    sections.append(_section("GRF/COP/COM", fig, _html_table(summary["grf_cop_com"]), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = _create_symmetry_figure(summary["symmetry"])
    warnings.extend(section_warnings)
    sections.append(_section("Symmetry", fig, _html_table(summary["symmetry"].head(20)), section_warnings))
    plt.close(fig)

    fig, has_data, section_warnings = _create_condition_comparison_figure(summary["comparison"])
    warnings.extend(section_warnings)
    sections.append(_section("Condition comparison", fig, _html_table(summary["comparison"].head(20)), section_warnings))
    plt.close(fig)

    html = _render_dashboard_html(sections, warnings, file_paths, selected, section_scope)
    dashboard_path = html_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")

    summary_md = _render_summary_markdown(sections, warnings, file_paths, selected, section_scope)
    summary_path = reports_dir / "visualization_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    return str(dashboard_path), str(summary_path), warnings


def _unique_nonempty(values: Sequence[str]) -> List[str]:
    unique = []
    for value in values:
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _scope_dataframes_for_variables(dataframes: List[pd.DataFrame], selected_variables: Sequence[str]) -> List[pd.DataFrame]:
    selected = set(selected_variables)
    if not selected:
        return dataframes

    scoped = []
    for df in dataframes:
        keep_columns = []
        for column in df.columns:
            name = str(column)
            lower = name.lower()
            if (
                name in selected
                or lower == "time"
                or lower.endswith(".state")
                or lower.endswith("_state")
            ):
                keep_columns.append(column)
        scoped.append(df.loc[:, keep_columns].copy())
    return scoped


def _collect_dashboard_tables(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    selected_variables: Sequence[str],
    analysis_results: Dict,
) -> Dict[str, pd.DataFrame]:
    return {
        "quality": _quality_table(dataframes, file_paths),
        "features": _feature_table(analysis_results),
        "phase": _phase_table(analysis_results),
        "symmetry": _symmetry_table(analysis_results),
        "comparison": _comparison_table(analysis_results),
        "detected_variables": _detected_variable_table(dataframes),
        "grf_cop_com": _grf_cop_com_table(dataframes),
    }


def _quality_table(dataframes: List[pd.DataFrame], file_paths: List[str]) -> pd.DataFrame:
    rows = []
    for idx, df in enumerate(dataframes):
        numeric_cols = [
            column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        total_cells = int(df.shape[0] * max(len(df.columns), 1))
        missing_cells = int(df.isna().sum().sum())
        time_duration = np.nan
        if "time" in df.columns:
            time_values = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
            finite_time = time_values[np.isfinite(time_values)]
            if len(finite_time) >= 2:
                time_duration = float(np.nanmax(finite_time) - np.nanmin(finite_time))
        rows.append({
            "file": _filename(file_paths, idx),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": len(numeric_cols),
            "missing_percent": missing_cells / total_cells * 100.0 if total_cells else 0.0,
            "time_duration": time_duration,
        })
    return pd.DataFrame(rows)


def _feature_table(analysis_results: Dict) -> pd.DataFrame:
    rows = []
    for key, result in analysis_results.items():
        common = result.get("common_features", {}) or {}
        stats = result.get("overall_statistics", {}) or {}
        if not common and not stats:
            continue
        variable = result.get("variable_name", "")
        rows.append({
            "file": _file_from_result_key(key, variable),
            "variable": variable,
            "unit": result.get("display_unit", ""),
            "mean": stats.get("mean", np.nan),
            "std": stats.get("std", np.nan),
            "max": common.get("max", stats.get("max", np.nan)),
            "min": common.get("min", stats.get("min", np.nan)),
            "rom": common.get("rom", np.nan),
            "rms": common.get("rms", np.nan),
            "auc": common.get("auc", np.nan),
        })
    return pd.DataFrame(rows)


def _phase_table(analysis_results: Dict) -> pd.DataFrame:
    rows = []
    for key, result in analysis_results.items():
        variable = result.get("variable_name", "")
        for phase, stats in (result.get("scone_phase_statistics", {}) or {}).items():
            rows.append({
                "file": _file_from_result_key(key, variable),
                "variable": variable,
                "phase": phase,
                "mean": stats.get("mean", np.nan),
                "std": stats.get("std", np.nan),
                "max": stats.get("max", np.nan),
                "min": stats.get("min", np.nan),
                "rms": stats.get("rms", np.nan),
                "auc": stats.get("auc", np.nan),
            })
    return pd.DataFrame(rows)


def _symmetry_table(analysis_results: Dict) -> pd.DataFrame:
    return build_symmetry_table(analysis_results)


def _comparison_table(analysis_results: Dict) -> pd.DataFrame:
    features = _feature_table(analysis_results)
    if features.empty or "file" not in features.columns:
        return pd.DataFrame()

    rows = []
    for variable, group in features.groupby("variable", dropna=False):
        if len(group) < 2:
            continue
        reference = group.iloc[0]
        for _, row in group.iloc[1:].iterrows():
            for metric in ("mean", "max", "rms", "auc"):
                ref_value = reference.get(metric, np.nan)
                value = row.get(metric, np.nan)
                if not np.isfinite(ref_value) or not np.isfinite(value):
                    continue
                reference_magnitude = abs(ref_value)
                rows.append({
                    "reference": reference["file"],
                    "condition": row["file"],
                    "variable": variable,
                    "metric": metric,
                    "delta": value - ref_value,
                    "delta_percent": (value - ref_value) / reference_magnitude * 100.0 if reference_magnitude != 0 else np.nan,
                })
    return pd.DataFrame(rows)


def _detected_variable_table(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    all_columns = sorted({str(column) for df in dataframes for column in df.columns})
    groups = [
        ("GRF", sum(len(axes) for axes in detect_grf_variables(all_columns).values())),
        ("COP", sum(len(axes) for axes in detect_cop_variables(all_columns).values())),
        ("COM", len(detect_com_variables(all_columns))),
        (
            "Momentum",
            sum(len(axes) for axes in detect_momentum_variables(all_columns).values()),
        ),
    ]
    return pd.DataFrame([{"group": group, "detected_variables": count} for group, count in groups])


def _grf_cop_com_table(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    all_columns = sorted({str(column) for df in dataframes for column in df.columns})
    grf = detect_grf_variables(all_columns)
    cop = detect_cop_variables(all_columns)
    com = detect_com_variables(all_columns)
    momentum = detect_momentum_variables(all_columns)
    rows = []
    for side, axes in grf.items():
        rows.append({"kind": "GRF", "side": side, "axes": ", ".join(sorted(axes)), "variables": ", ".join(axes.values())})
    for side, axes in cop.items():
        rows.append({"kind": "COP", "side": side, "axes": ", ".join(sorted(axes)), "variables": ", ".join(axes.values())})
    if com:
        rows.append({"kind": "COM", "side": "", "axes": ", ".join(sorted(com)), "variables": ", ".join(com.values())})
    for kind, axes in momentum.items():
        if axes:
            rows.append({"kind": kind.title() + " momentum", "side": "", "axes": ", ".join(sorted(axes)), "variables": ", ".join(axes.values())})
    return pd.DataFrame(rows)


def _create_waveform_figure(
    dataframes: List[pd.DataFrame],
    file_paths: List[str],
    variables: Sequence[str],
):
    import matplotlib.pyplot as plt

    warnings: List[str] = []
    if not variables:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No selected waveform variables.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No selected waveform variables."]

    plot_vars = list(variables[:6])
    n_cols = 2 if len(plot_vars) > 1 else 1
    n_rows = int(np.ceil(len(plot_vars) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 3.8 * n_rows), squeeze=False)
    has_any = False
    for idx, variable in enumerate(plot_vars):
        ax = axes.ravel()[idx]
        has_data, section_warnings = draw_gait_cycle_waveform(
            ax,
            dataframes,
            file_paths,
            variable,
            show_mean_sd=True,
            show_phase_shading=True,
            show_peak_markers=True,
        )
        warnings.extend(section_warnings)
        has_any = has_any or has_data
    for idx in range(len(plot_vars), len(axes.ravel())):
        axes.ravel()[idx].set_visible(False)
    fig.tight_layout()
    return fig, has_any, warnings


def _create_quality_figure(quality_table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if quality_table.empty:
        ax.text(0.5, 0.5, "No data quality rows found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No data quality rows found."]
    rows = quality_table.copy()
    rows["complete_percent"] = 100.0 - pd.to_numeric(rows["missing_percent"], errors="coerce").fillna(0.0)
    labels = rows["file"].astype(str)
    ax.bar(np.arange(len(rows)), rows["complete_percent"], color="#0072B2", alpha=0.85)
    ax.set_ylim(0, 105)
    ax.set_title("Data completeness by file")
    ax.set_ylabel("Completeness (%)")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig, True, []


def _create_phase_summary_figure(phase_table: pd.DataFrame):
    import matplotlib.pyplot as plt

    warnings = []
    fig, ax = plt.subplots(figsize=(10, 5))
    if phase_table.empty:
        ax.text(0.5, 0.5, "No gait phase results found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No gait phase results found."]
    rows = phase_table.copy()
    rows["mean"] = pd.to_numeric(rows["mean"], errors="coerce")
    matrix = rows.pivot_table(index="phase", columns="variable", values="mean", aggfunc="mean")
    if matrix.empty:
        ax.text(0.5, 0.5, "No finite phase mean values found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No finite phase mean values found."]
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_title("Phase mean by variable")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Phase")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, True, warnings


def _create_feature_summary_figure(feature_csv_paths: Sequence[str], feature_table: pd.DataFrame):
    import matplotlib.pyplot as plt

    if feature_csv_paths:
        fig, has_data, warnings = create_feature_plot(feature_csv_paths, "Feature bar plot")
        if has_data:
            return fig, has_data, warnings
        plt.close(fig)

    warnings = []
    fig, ax = plt.subplots(figsize=(10, 5))
    if feature_table.empty:
        ax.text(0.5, 0.5, "No feature summary data found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No feature summary data found."]
    rows = feature_table.copy()
    rows["label"] = rows["file"].astype(str) + " | " + rows["variable"].astype(str)
    rows = rows.head(12)
    ax.bar(np.arange(len(rows)), pd.to_numeric(rows["rms"], errors="coerce"), color="#0072B2", alpha=0.85)
    ax.set_title("RMS feature summary")
    ax.set_ylabel("RMS")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows["label"], rotation=55, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig, True, warnings


def _create_symmetry_figure(symmetry_table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if symmetry_table.empty:
        ax.text(0.5, 0.5, "No symmetry results found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No symmetry results found."]
    rows = symmetry_table.head(16).copy()
    rows["label"] = rows["variable"].astype(str) + " | " + rows["feature"].astype(str)
    values = pd.to_numeric(rows["symmetry_index"], errors="coerce")
    ax.bar(np.arange(len(rows)), values, color="#009E73", alpha=0.85)
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.set_title("Symmetry index")
    ax.set_ylabel("SI (%)")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows["label"], rotation=55, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig, True, []


def _create_condition_comparison_figure(comparison_table: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if comparison_table.empty:
        ax.text(0.5, 0.5, "No condition comparison data found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No condition comparison data found."]
    rows = comparison_table.copy()
    rows["delta"] = pd.to_numeric(rows["delta"], errors="coerce")
    rows = rows[np.isfinite(rows["delta"])]
    if rows.empty:
        ax.text(0.5, 0.5, "No finite condition deltas found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No finite condition deltas found."]
    rows["metric_variable"] = rows["metric"].astype(str) + " | " + rows["variable"].astype(str)
    matrix = rows.pivot_table(index="metric_variable", columns="condition", values="delta", aggfunc="mean")
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_title("Condition delta vs first file")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Metric | variable")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, True, []


def _section(title: str, fig, table_html: str, warnings: Sequence[str]) -> Dict:
    return {
        "title": title,
        "figure": _fig_to_base64(fig) if fig is not None else "",
        "table": table_html,
        "warnings": list(warnings),
    }


def _fig_to_base64(fig) -> str:
    if fig is None:
        return ""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class=\"muted\">No table data available.</p>"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(_format_number)
    header = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    body_rows = []
    for _, row in display.iterrows():
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _format_number(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4g}"


def _render_dashboard_html(
    sections: Sequence[Dict],
    warnings: Sequence[str],
    file_paths: Sequence[str],
    selected_variables: Sequence[str],
    section_scope: str,
) -> str:
    nav = "".join(
        f"<a href=\"#{_slug(section['title'])}\">{escape(section['title'])}</a>"
        for section in sections
    )
    section_html = "\n".join(_render_section(section) for section in sections)
    warning_items = "".join(f"<li>{escape(str(warning))}</li>" for warning in _dedupe(warnings))
    if not warning_items:
        warning_items = "<li>No warnings.</li>"
    file_list = ", ".join(escape(_filename(file_paths, idx)) for idx in range(len(file_paths))) or "No files"
    variable_list = ", ".join(escape(str(variable)) for variable in selected_variables) or "No selected variables"
    scope_label = "Selected variables" if section_scope == "selected" else "All detected variables"
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Motion Analysis Visualization Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f8fafc; }}
    header {{ background: #ffffff; border-bottom: 1px solid #dbe3ef; padding: 20px 28px; position: sticky; top: 0; z-index: 10; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    nav a {{ color: #0f5e9c; text-decoration: none; border: 1px solid #bfd4e8; padding: 5px 8px; border-radius: 4px; background: #f4f9ff; font-size: 13px; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    section {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 6px; margin-bottom: 22px; padding: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; }}
    .meta {{ color: #526070; font-size: 13px; line-height: 1.45; }}
    .figure {{ text-align: center; margin: 12px 0 16px; }}
    .figure img {{ max-width: 100%; height: auto; border: 1px solid #edf1f6; }}
    .table-wrap {{ overflow-x: auto; margin-top: 10px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #dbe3ef; padding: 6px 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #eef4fb; }}
    .muted {{ color: #6b7280; }}
    .warning {{ background: #fff8e6; border: 1px solid #f2d38b; padding: 8px 10px; border-radius: 4px; color: #6b4e00; font-size: 13px; }}
    footer {{ max-width: 1220px; margin: 0 auto 28px; background: #fff; border: 1px solid #dbe3ef; border-radius: 6px; padding: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>Motion Analysis Visualization Dashboard</h1>
    <div class="meta">Created: {escape(created)}<br>Scope: {escape(scope_label)}<br>Files: {file_list}<br>Selected variables: {variable_list}</div>
    <nav>{nav}</nav>
  </header>
  <main>
    {section_html}
  </main>
  <footer id="warnings">
    <h2>Warnings and skipped items</h2>
    <ul>{warning_items}</ul>
  </footer>
</body>
</html>
"""


def _render_section(section: Dict) -> str:
    warning_html = ""
    if section["warnings"]:
        warning_html = "<div class=\"warning\">" + "<br>".join(escape(str(w)) for w in _dedupe(section["warnings"])[:8]) + "</div>"
    figure_html = ""
    if section["figure"]:
        figure_html = f"<div class=\"figure\"><img alt=\"{escape(section['title'])}\" src=\"data:image/png;base64,{section['figure']}\"></div>"
    return f"""<section id="{_slug(section['title'])}">
  <h2>{escape(section['title'])}</h2>
  {figure_html}
  {section['table']}
  {warning_html}
</section>"""


def _render_summary_markdown(
    sections: Sequence[Dict],
    warnings: Sequence[str],
    file_paths: Sequence[str],
    selected_variables: Sequence[str],
    section_scope: str,
) -> str:
    lines = [
        "# Visualization Summary",
        "",
        f"- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Files: {len(file_paths)}",
        f"- Selected variables: {len(selected_variables)}",
        f"- Scope: {'selected variables' if section_scope == 'selected' else 'all detected variables'}",
        "",
        "## Sections",
    ]
    for section in sections:
        lines.append(f"- {section['title']}: {'figure' if section['figure'] else 'table only'}")
    lines.extend(["", "## Warnings"])
    deduped = _dedupe(warnings)
    if deduped:
        lines.extend(f"- {warning}" for warning in deduped)
    else:
        lines.append("- No warnings.")
    lines.append("")
    return "\n".join(lines)


def _dedupe(values: Sequence[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value)
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _filename(file_paths: Sequence[str], idx: int) -> str:
    if idx < len(file_paths):
        return Path(str(file_paths[idx])).name
    return f"file_{idx + 1}"


def _file_from_result_key(key: str, variable: str = "") -> str:
    text = str(key)
    parts = text.split("_", 1)
    if len(parts) != 2:
        return text
    remainder = parts[1]
    suffix = f"_{variable}" if variable else ""
    if suffix and remainder.endswith(suffix):
        remainder = remainder[: -len(suffix)]
    return remainder or text


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
