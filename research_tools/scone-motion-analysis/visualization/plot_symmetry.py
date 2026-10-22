"""Symmetry visualization helpers."""

from typing import Dict, Sequence

import numpy as np
import pandas as pd


_SYMMETRY_COLUMNS = (
    "file",
    "variable",
    "pair_variable",
    "pair_label",
    "feature",
    "right_value",
    "left_value",
    "symmetry_index",
)


def build_symmetry_table(analysis_results: Dict, selected_variables: Sequence[str] = ()) -> pd.DataFrame:
    """Build one de-duplicated symmetry row per file, side pair, and feature."""
    selected_set = set(selected_variables or [])
    rows = []
    seen = set()

    for key, result in analysis_results.items():
        if not isinstance(result, dict):
            continue

        variable = str(result.get("variable_name", "")).strip()
        pair_variable = str(result.get("symmetry_pair_variable", "")).strip()
        if selected_set and variable not in selected_set and pair_variable not in selected_set:
            continue
        if not variable or not pair_variable:
            continue

        file_name = _file_from_result_key(str(key), variable)
        pair_key = tuple(sorted((variable, pair_variable)))
        pair_label = f"{pair_key[0]} / {pair_key[1]}"

        for feature, stats in (result.get("symmetry_features", {}) or {}).items():
            dedupe_key = (file_name, pair_key[0], pair_key[1], str(feature))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            rows.append({
                "file": file_name,
                "variable": variable,
                "pair_variable": pair_variable,
                "pair_label": pair_label,
                "feature": feature,
                "right_value": stats.get("right_value", np.nan),
                "left_value": stats.get("left_value", np.nan),
                "symmetry_index": stats.get("symmetry_index", np.nan),
            })

    return pd.DataFrame(rows, columns=_SYMMETRY_COLUMNS)


def create_symmetry_figure(symmetry_table: pd.DataFrame, max_rows: int = 20):
    """Create a bar plot of symmetry index values."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    warnings = []

    if symmetry_table.empty:
        ax.text(0.5, 0.5, "No symmetry results found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No symmetry results found."]

    rows = symmetry_table.copy()
    rows["symmetry_index"] = pd.to_numeric(rows["symmetry_index"], errors="coerce")
    rows = rows[np.isfinite(rows["symmetry_index"])]
    if rows.empty:
        ax.text(0.5, 0.5, "No finite symmetry index values found.", ha="center", va="center")
        ax.set_axis_off()
        return fig, False, ["No finite symmetry index values found."]

    rows = rows.reindex(rows["symmetry_index"].abs().sort_values(ascending=False).index)
    if len(rows) > max_rows:
        warnings.append(f"Showing top {max_rows} symmetry rows by absolute SI.")
        rows = rows.head(max_rows)

    labels = rows["file"].astype(str) + "\n" + rows["pair_label"].astype(str) + " | " + rows["feature"].astype(str)
    x = np.arange(len(rows))
    values = rows["symmetry_index"].to_numpy(dtype=float)
    colors = np.where(values >= 0, "#0072B2", "#D55E00")

    ax.bar(x, values, color=colors, alpha=0.85)
    ax.axhline(0, color="#64748B", linewidth=0.9)
    ax.axhline(10, color="#94A3B8", linewidth=0.8, linestyle="--")
    ax.axhline(-10, color="#94A3B8", linewidth=0.8, linestyle="--")
    ax.set_title("Symmetry index by side pair")
    ax.set_ylabel("Symmetry Index (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig, True, warnings


def _file_from_result_key(key: str, variable: str) -> str:
    suffix = f"_{variable}"
    if suffix and key.endswith(suffix):
        prefix = key[:-len(suffix)]
    else:
        prefix = key
    if "_" in prefix:
        return prefix.split("_", 1)[1]
    return prefix
