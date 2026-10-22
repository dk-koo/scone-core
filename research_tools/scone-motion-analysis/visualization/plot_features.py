"""Feature CSV visualization helpers."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_FILE_NAMES = ("cycle_features.csv", "phase_features.csv", "comparison_features.csv")

FEATURE_PLOT_TYPES = (
    "Feature bar plot",
    "Phase feature bar plot",
    "Condition progression line",
    "Variable x condition heatmap",
    "Phase x variable heatmap",
    "Delta comparison",
    "RMSE/NCC heatmap",
)

def discover_feature_csvs(search_roots: Iterable[str]) -> List[str]:
    """Find known feature CSV files under shallow search roots."""
    found = []
    seen = set()
    for root in search_roots:
        if not root:
            continue
        root_path = Path(root)
        if root_path.is_file():
            candidates = [root_path]
        elif root_path.is_dir():
            candidates = []
            for file_name in FEATURE_FILE_NAMES:
                candidates.extend(root_path.glob(file_name))
                candidates.extend(root_path.glob(f"*/{file_name}"))
                candidates.extend(root_path.glob(f"*/*/{file_name}"))
        else:
            continue

        for candidate in candidates:
            if candidate.name not in FEATURE_FILE_NAMES:
                continue
            resolved = str(candidate.resolve())
            if resolved not in seen:
                found.append(resolved)
                seen.add(resolved)
    return found


def classify_feature_file(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    if name == "cycle_features.csv":
        return "cycle"
    if name == "phase_features.csv":
        return "phase"
    if name == "comparison_features.csv":
        return "comparison"
    return None


def load_feature_tables(paths: Iterable[str]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Load recognized feature CSV files into a table dict."""
    tables: Dict[str, pd.DataFrame] = {}
    warnings = []
    for path in paths:
        table_key = classify_feature_file(path)
        if table_key is None:
            warnings.append(f"{path}: unsupported feature CSV name; skipped.")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            warnings.append(f"{path}: failed to read CSV ({exc}); skipped.")
            continue
        tables[table_key] = _normalize_columns(df)
    return tables, warnings


def get_conditions_from_feature_tables(paths: Iterable[str]) -> List[str]:
    tables, _ = load_feature_tables(paths)
    conditions = []
    for df in tables.values():
        if "condition" not in df.columns:
            continue
        for value in df["condition"].dropna().astype(str):
            if value not in conditions:
                conditions.append(value)
    return conditions


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if "delta%" in normalized.columns and "delta_percent" not in normalized.columns:
        normalized = normalized.rename(columns={"delta%": "delta_percent"})
    return normalized


def _require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str, warnings: List[str]) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        warnings.append(f"{table_name}: missing columns {missing}; skipped.")
        return False
    return True


def _prepare_value_rows(df: pd.DataFrame, warnings: List[str], table_name: str) -> pd.DataFrame:
    if not _require_columns(df, ("variable", "feature", "value"), table_name, warnings):
        return pd.DataFrame()
    rows = df.copy()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows[np.isfinite(rows["value"])]
    if rows.empty:
        warnings.append(f"{table_name}: no finite feature values.")
    return rows


def _choose_feature(rows: pd.DataFrame, preferred: Tuple[str, ...]) -> str:
    available = [str(value) for value in rows["feature"].dropna().unique()]
    available_lower = {value.lower(): value for value in available}
    for feature_name in preferred:
        if feature_name.lower() in available_lower:
            return available_lower[feature_name.lower()]
    return available[0] if available else ""


def _pivot_mean(rows: pd.DataFrame, index: str, columns: str, values: str = "value") -> pd.DataFrame:
    return rows.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")


def create_feature_plot(
    feature_csv_paths: Iterable[str],
    plot_type: str,
    reference_condition: Optional[str] = None,
):
    """Create a matplotlib plot from feature CSV files."""
    import matplotlib.pyplot as plt

    tables, warnings = load_feature_tables(feature_csv_paths)
    fig, ax = plt.subplots(figsize=(11, 6))

    if plot_type == "Feature bar plot":
        has_data = _draw_feature_bar(ax, tables, warnings)
    elif plot_type == "Phase feature bar plot":
        has_data = _draw_phase_feature_bar(ax, tables, warnings)
    elif plot_type == "Condition progression line":
        has_data = _draw_condition_progression(ax, tables, warnings)
    elif plot_type == "Variable x condition heatmap":
        has_data = _draw_variable_condition_heatmap(ax, tables, warnings)
    elif plot_type == "Phase x variable heatmap":
        has_data = _draw_phase_variable_heatmap(ax, tables, warnings)
    elif plot_type == "Delta comparison":
        has_data = _draw_delta_comparison(ax, tables, reference_condition, warnings)
    elif plot_type == "RMSE/NCC heatmap":
        has_data = _draw_rmse_ncc_heatmap(ax, tables, warnings)
    else:
        warnings.append(f"Unknown feature plot type '{plot_type}'.")
        has_data = False

    if not has_data:
        ax.text(0.5, 0.5, "No plottable feature data found.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    return fig, has_data, warnings


def _feature_source_table(tables: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    if "cycle" in tables:
        return "cycle_features.csv", tables["cycle"]
    if "phase" in tables:
        return "phase_features.csv", tables["phase"]
    return "", pd.DataFrame()


def _draw_feature_bar(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    table_name, df = _feature_source_table(tables)
    rows = _prepare_value_rows(df, warnings, table_name)
    if rows.empty:
        return False

    feature_name = _choose_feature(rows, ("peak", "max", "rom", "auc", "impulse", "rms", "mean"))
    rows = rows[rows["feature"].astype(str) == feature_name]
    group_col = "condition" if "condition" in rows.columns else "variable"
    summary = rows.groupby(group_col, dropna=False)["value"].mean().sort_index()
    summary.plot(kind="bar", ax=ax, color="#0072B2")
    ax.set_title(f"{feature_name} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    return True


def _draw_phase_feature_bar(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    df = tables.get("phase", pd.DataFrame())
    rows = _prepare_value_rows(df, warnings, "phase_features.csv")
    if rows.empty or "phase" not in rows.columns:
        warnings.append("phase_features.csv: phase column is required for phase bar plot.")
        return False

    feature_name = _choose_feature(rows, ("max", "mean", "rms", "auc", "rom", "range"))
    rows = rows[rows["feature"].astype(str) == feature_name]
    summary = rows.groupby("phase", dropna=False)["value"].mean().sort_index()
    summary.plot(kind="bar", ax=ax, color="#009E73")
    ax.set_title(f"{feature_name} by phase")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    return True


def _draw_condition_progression(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    table_name, df = _feature_source_table(tables)
    rows = _prepare_value_rows(df, warnings, table_name)
    if rows.empty or "condition" not in rows.columns:
        warnings.append(f"{table_name}: condition column is required for progression plot.")
        return False

    feature_name = _choose_feature(rows, ("peak", "max", "rom", "auc", "impulse", "rms", "mean"))
    rows = rows[rows["feature"].astype(str) == feature_name]
    pivot = _pivot_mean(rows, "condition", "variable")
    if pivot.empty:
        return False

    for variable in pivot.columns[:8]:
        ax.plot(pivot.index.astype(str), pivot[variable], marker="o", label=str(variable))
    ax.set_title(f"Condition progression: {feature_name}")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8, frameon=False)
    return True


def _draw_heatmap(ax, matrix: pd.DataFrame, title: str, x_label: str, y_label: str) -> bool:
    if matrix.empty:
        return False
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(col) for col in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(idx) for idx in matrix.index])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return True


def _draw_variable_condition_heatmap(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    table_name, df = _feature_source_table(tables)
    rows = _prepare_value_rows(df, warnings, table_name)
    if rows.empty or "condition" not in rows.columns:
        warnings.append(f"{table_name}: condition column is required for variable x condition heatmap.")
        return False

    feature_name = _choose_feature(rows, ("max", "peak", "rom", "auc", "impulse", "rms", "mean"))
    matrix = _pivot_mean(rows[rows["feature"].astype(str) == feature_name], "variable", "condition")
    return _draw_heatmap(ax, matrix, f"Variable x condition: {feature_name}", "Condition", "Variable")


def _draw_phase_variable_heatmap(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    df = tables.get("phase", pd.DataFrame())
    rows = _prepare_value_rows(df, warnings, "phase_features.csv")
    if rows.empty or "phase" not in rows.columns:
        warnings.append("phase_features.csv: phase column is required for phase x variable heatmap.")
        return False

    feature_name = _choose_feature(rows, ("max", "mean", "rms", "auc", "rom", "range"))
    matrix = _pivot_mean(rows[rows["feature"].astype(str) == feature_name], "phase", "variable")
    return _draw_heatmap(ax, matrix, f"Phase x variable: {feature_name}", "Variable", "Phase")


def _draw_delta_comparison(
    ax,
    tables: Dict[str, pd.DataFrame],
    reference_condition: Optional[str],
    warnings: List[str],
) -> bool:
    df = tables.get("comparison", pd.DataFrame()).copy()
    required = ("condition", "variable")
    if not _require_columns(df, required, "comparison_features.csv", warnings):
        return False

    value_column = _choose_comparison_value_column(df, ("delta", "delta_percent", "delta_timing"))
    if not value_column:
        warnings.append("comparison_features.csv: delta, delta_percent, or delta_timing column is required.")
        return False

    if reference_condition and "reference_condition" in df.columns:
        filtered = df[df["reference_condition"].astype(str) == str(reference_condition)]
        if not filtered.empty:
            df = filtered
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df[np.isfinite(df[value_column])]
    if df.empty:
        return False

    matrix = df.pivot_table(index="variable", columns="condition", values=value_column, aggfunc="mean")
    return _draw_heatmap(ax, matrix, f"Comparison {value_column}", "Condition", "Variable")


def _choose_comparison_value_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    return ""


def _draw_rmse_ncc_heatmap(ax, tables: Dict[str, pd.DataFrame], warnings: List[str]) -> bool:
    df = tables.get("comparison", pd.DataFrame()).copy()
    if not _require_columns(df, ("condition", "variable", "feature"), "comparison_features.csv", warnings):
        return False

    value_column = "value" if "value" in df.columns else ""
    if not value_column:
        for candidate in ("condition_value", "delta", "delta_percent"):
            if candidate in df.columns:
                value_column = candidate
                break
    if not value_column:
        warnings.append("comparison_features.csv: value-like column is required for RMSE/NCC heatmap.")
        return False

    metric_mask = df["feature"].astype(str).str.lower().isin(
        ["rmse", "normalized_rmse", "nrmse", "ncc", "area_between_curves"]
    )
    rows = df[metric_mask].copy()
    rows[value_column] = pd.to_numeric(rows[value_column], errors="coerce")
    rows = rows[np.isfinite(rows[value_column])]
    if rows.empty:
        warnings.append("comparison_features.csv: RMSE/NCC metrics were not found.")
        return False

    rows["metric_variable"] = rows["feature"].astype(str) + " | " + rows["variable"].astype(str)
    matrix = rows.pivot_table(index="metric_variable", columns="condition", values=value_column, aggfunc="mean")
    return _draw_heatmap(ax, matrix, "RMSE / normalized RMSE / NCC", "Condition", "Metric | Variable")
