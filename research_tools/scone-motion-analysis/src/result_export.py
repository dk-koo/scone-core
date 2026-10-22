"""Result export helpers separated from the Tkinter UI."""

import os
from typing import Dict, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from utils.file_utils import format_legend_label, sanitize_dataframe_for_spreadsheet_export


_CYCLE_FEATURE_COLUMNS = ("file", "condition", "variable", "phase", "feature", "value", "unit")
_PHASE_FEATURE_COLUMNS = ("file", "condition", "variable", "phase", "feature", "value", "unit")
_COMPARISON_FEATURE_COLUMNS = (
    "condition",
    "reference_condition",
    "variable",
    "phase",
    "feature",
    "condition_value",
    "reference_value",
    "delta",
    "delta_percent",
    "delta_timing",
    "unit",
)


def build_results_dataframe(
    selected_vars: Sequence[str],
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Build the tabular result export from completed analysis results."""
    all_stats = []
    for file_idx, path in enumerate(file_paths):
        file_name = os.path.basename(path)
        exported_symmetry_pairs: Set[Tuple[str, str, str]] = set()
        for var in selected_vars:
            analysis_key = f"{file_idx}_{file_name}_{var}"
            if analysis_key not in analysis_results:
                continue

            analysis_result = analysis_results[analysis_key]
            variable_preset = analysis_result.get("variable_preset", {})

            if analysis_result.get("time_coverage_failed", False):
                basic_row = _build_time_coverage_failed_row(file_name, var)
            else:
                basic_row = _build_overall_row(file_name, var, analysis_result, variable_preset)
            all_stats.append(basic_row)

            if analysis_result.get("time_coverage_failed", False):
                continue

            all_stats.extend(_build_phase_rows(file_name, var, analysis_result, variable_preset))
            all_stats.extend(
                _build_symmetry_rows(
                    file_name,
                    var,
                    analysis_result,
                    variable_preset,
                    exported_symmetry_pairs,
                )
            )

    return sanitize_dataframe_for_spreadsheet_export(pd.DataFrame(all_stats))


def save_results_dataframe(df_results: pd.DataFrame, save_path: str):
    """Save the result dataframe to CSV or Excel based on extension."""
    if save_path.endswith(".xlsx"):
        df_results.to_excel(save_path, index=False)
    else:
        df_results.to_csv(save_path, index=False, encoding="utf-8-sig")


def save_feature_csv_bundle(
    selected_vars: Sequence[str],
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
    output_dir: str,
) -> Sequence[str]:
    """Save standard feature CSV files used by the visualization module."""
    os.makedirs(output_dir, exist_ok=True)
    tables = build_feature_csv_tables(selected_vars, file_paths, analysis_results)
    if all(dataframe.empty for dataframe in tables.values()):
        return []
    saved_paths = []
    for file_name, dataframe in tables.items():
        path = os.path.join(output_dir, file_name)
        sanitize_dataframe_for_spreadsheet_export(dataframe).to_csv(path, index=False, encoding="utf-8-sig")
        saved_paths.append(path)
    return saved_paths


def build_feature_csv_tables(
    selected_vars: Sequence[str],
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
) -> Dict[str, pd.DataFrame]:
    """Build cycle/phase/comparison feature tables for downstream plots."""
    cycle_rows = []
    phase_rows = []
    feature_rows_by_variable: Dict[str, list] = {}

    for file_idx, path in enumerate(file_paths):
        file_name = os.path.basename(path)
        condition = format_legend_label(file_name)
        for var in selected_vars:
            analysis_key = f"{file_idx}_{file_name}_{var}"
            analysis_result = analysis_results.get(analysis_key)
            if not analysis_result or analysis_result.get("time_coverage_failed", False):
                continue

            cycle_feature_rows = _build_cycle_feature_rows(file_name, condition, var, analysis_result)
            cycle_rows.extend(cycle_feature_rows)
            feature_rows_by_variable.setdefault(var, []).extend(cycle_feature_rows)
            phase_rows.extend(_build_phase_feature_rows(file_name, condition, var, analysis_result))

    comparison_rows = _build_comparison_feature_rows(feature_rows_by_variable)
    return {
        "cycle_features.csv": pd.DataFrame(cycle_rows, columns=_CYCLE_FEATURE_COLUMNS),
        "phase_features.csv": pd.DataFrame(phase_rows, columns=_PHASE_FEATURE_COLUMNS),
        "comparison_features.csv": pd.DataFrame(comparison_rows, columns=_COMPARISON_FEATURE_COLUMNS),
    }


def _build_cycle_feature_rows(file_name: str, condition: str, var: str, analysis_result: Dict):
    rows = []
    common = analysis_result.get("common_features", {}) or {}
    stats = analysis_result.get("overall_statistics", {}) or {}
    unit = analysis_result.get("display_unit", "")
    axis_basis = common.get("axis_basis", "")
    feature_values = {
        "mean": common.get("mean", stats.get("mean", np.nan)),
        "std": common.get("std", stats.get("std", np.nan)),
        "max": common.get("max", stats.get("max", np.nan)),
        "min": common.get("min", stats.get("min", np.nan)),
        "rom": common.get("rom", np.nan),
        "rms": common.get("rms", np.nan),
        "auc": common.get("auc", np.nan),
        "positive_auc": common.get("positive_auc", np.nan),
        "negative_auc": common.get("negative_auc", np.nan),
        "max_time": common.get("max_time", np.nan),
        "max_percent": common.get("max_percent", np.nan),
        "min_time": common.get("min_time", np.nan),
        "min_percent": common.get("min_percent", np.nan),
        "threshold_exceed_count": common.get("threshold_exceed_count", np.nan),
        "onset_time": common.get("onset_time", np.nan),
        "offset_time": common.get("offset_time", np.nan),
        "active_duration": common.get("active_duration", np.nan),
    }
    for feature, value in feature_values.items():
        if _is_finite_number(value):
            rows.append({
                "file": file_name,
                "condition": condition,
                "variable": var,
                "phase": "whole_cycle",
                "feature": feature,
                "value": float(value),
                "unit": _feature_unit(feature, unit, axis_basis),
            })
    return rows


def _build_phase_feature_rows(file_name: str, condition: str, var: str, analysis_result: Dict):
    rows = []
    phase_stats = analysis_result.get("scone_phase_statistics", {}) or {}
    unit = analysis_result.get("display_unit", "")
    for phase, stats in phase_stats.items():
        for feature in ("mean", "std", "min", "max", "range", "rms", "auc", "count"):
            value = stats.get(feature, np.nan)
            if _is_finite_number(value):
                rows.append({
                    "file": file_name,
                    "condition": condition,
                    "variable": var,
                    "phase": phase,
                    "feature": feature,
                    "value": float(value),
                    "unit": _feature_unit(feature, unit, stats.get("auc_time_basis", "")),
                })
    return rows


def _build_comparison_feature_rows(feature_rows_by_variable: Dict[str, list]):
    rows = []
    for variable, feature_rows in feature_rows_by_variable.items():
        by_feature: Dict[str, list] = {}
        for row in feature_rows:
            by_feature.setdefault(row["feature"], []).append(row)

        for feature, rows_for_feature in by_feature.items():
            if len(rows_for_feature) < 2:
                continue
            reference = rows_for_feature[0]
            reference_value = reference["value"]
            for row in rows_for_feature[1:]:
                value = row["value"]
                delta = value - reference_value
                reference_magnitude = abs(reference_value)
                rows.append({
                    "condition": row["condition"],
                    "reference_condition": reference["condition"],
                    "variable": variable,
                    "phase": row.get("phase", "whole_cycle"),
                    "feature": feature,
                    "condition_value": value,
                    "reference_value": reference_value,
                    "delta": delta,
                    "delta_percent": delta / reference_magnitude * 100.0 if not np.isclose(reference_magnitude, 0.0) else np.nan,
                    "delta_timing": delta if feature.endswith(("_time", "_percent")) else np.nan,
                    "unit": reference.get("unit", ""),
                })
    return rows


def _is_finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _feature_unit(feature: str, display_unit: str, axis_basis: str = "") -> str:
    axis = str(axis_basis or "").lower()
    unit = str(display_unit or "")

    if feature in {"count", "threshold_exceed_count", "threshold_exceed_points"}:
        return "count"
    if feature.endswith("_percent"):
        return "%"
    if feature.endswith("_time") or feature in {"active_duration"}:
        return "s" if axis == "time" else "index"
    if feature in {"auc", "positive_auc", "negative_auc"}:
        return _integrated_unit(unit, axis)
    return unit


def _integrated_unit(display_unit: str, axis_basis: str) -> str:
    unit = display_unit or "value"
    if axis_basis == "time":
        if unit.endswith("/s^2"):
            return unit[:-2]
        if unit.endswith("/s"):
            return unit[:-2]
        return f"{unit}*s"
    if axis_basis == "index":
        return f"{unit}*index"
    return unit


def _build_time_coverage_failed_row(file_name: str, var: str) -> Dict:
    return {
        "파일명": file_name,
        "변수명": var,
        "분석유형": "전체 데이터",
        "단위": "",
        "변수군": "",
        "추천 Preset": "",
        "매칭 패턴": "",
        "평균": "",
        "표준편차": "",
        "최댓값": "",
        "최솟값": "",
        "중앙값": "",
        "범위": "",
        "RMS": "",
        "AUC": "",
        "Positive AUC": "",
        "Negative AUC": "",
        "Max Time": "",
        "Max Percent": "",
        "Min Time": "",
        "Min Percent": "",
        "Threshold": "",
        "Threshold 초과구간": "",
        "Onset": "",
        "Offset": "",
        "Active Duration": "",
        "Pair 변수": "",
        "대칭 Feature": "",
        "오른쪽 값": "",
        "왼쪽 값": "",
        "Symmetry Index": "",
        "데이터수": "",
        "상태컬럼": "",
    }


def _build_overall_row(file_name: str, var: str, analysis_result: Dict, variable_preset: Dict) -> Dict:
    overall_stats = analysis_result.get("overall_statistics", {})
    common_features = analysis_result.get("common_features", {})
    overall_min = overall_stats.get("min", np.nan)
    overall_max = overall_stats.get("max", np.nan)
    overall_range = (
        overall_max - overall_min
        if np.isfinite(overall_min) and np.isfinite(overall_max)
        else np.nan
    )

    return {
        "파일명": file_name,
        "변수명": var,
        "분석유형": "전체 데이터",
        "단위": analysis_result.get("display_unit", ""),
        "변수군": variable_preset.get("variable_group", ""),
        "추천 Preset": variable_preset.get("recommended_preset", ""),
        "매칭 패턴": variable_preset.get("matched_pattern", ""),
        "평균": overall_stats.get("mean", np.nan),
        "표준편차": overall_stats.get("std", np.nan),
        "최댓값": overall_stats.get("max", np.nan),
        "최솟값": overall_stats.get("min", np.nan),
        "중앙값": overall_stats.get("median", np.nan),
        "범위": common_features.get("rom", overall_range),
        "RMS": common_features.get("rms", np.nan),
        "AUC": common_features.get("auc", np.nan),
        "Positive AUC": common_features.get("positive_auc", np.nan),
        "Negative AUC": common_features.get("negative_auc", np.nan),
        "Max Time": common_features.get("max_time", np.nan),
        "Max Percent": common_features.get("max_percent", np.nan),
        "Min Time": common_features.get("min_time", np.nan),
        "Min Percent": common_features.get("min_percent", np.nan),
        "Threshold": common_features.get("threshold", np.nan),
        "Threshold 초과구간": common_features.get("threshold_exceed_count", 0),
        "Onset": common_features.get("onset_time", np.nan),
        "Offset": common_features.get("offset_time", np.nan),
        "Active Duration": common_features.get("active_duration", np.nan),
        "Pair 변수": analysis_result.get("symmetry_pair_variable", ""),
        "대칭 Feature": "",
        "오른쪽 값": "",
        "왼쪽 값": "",
        "Symmetry Index": "",
        "데이터수": overall_stats.get("count", 0),
        "상태컬럼": analysis_result.get("scone_phase_state_column", ""),
    }


def _build_phase_rows(file_name: str, var: str, analysis_result: Dict, variable_preset: Dict):
    rows = []
    scone_phase_stats = analysis_result.get("scone_phase_statistics", {})
    for phase_name, stats in scone_phase_stats.items():
        if stats.get("count", 0) <= 0:
            continue
        rows.append({
            "파일명": file_name,
            "변수명": var,
            "분석유형": f"SCONE phase - {phase_name}",
            "단위": analysis_result.get("display_unit", ""),
            "변수군": variable_preset.get("variable_group", ""),
            "추천 Preset": variable_preset.get("recommended_preset", ""),
            "매칭 패턴": variable_preset.get("matched_pattern", ""),
            "평균": stats.get("mean", np.nan),
            "표준편차": stats.get("std", np.nan),
            "최댓값": stats.get("max", np.nan),
            "최솟값": stats.get("min", np.nan),
            "중앙값": "",
            "범위": stats.get("range", np.nan),
            "RMS": stats.get("rms", np.nan),
            "AUC": stats.get("auc", np.nan),
            "Positive AUC": "",
            "Negative AUC": "",
            "Max Time": "",
            "Max Percent": "",
            "Min Time": "",
            "Min Percent": "",
            "Threshold": "",
            "Threshold 초과구간": "",
            "Onset": "",
            "Offset": "",
            "Active Duration": "",
            "Pair 변수": "",
            "대칭 Feature": "",
            "오른쪽 값": "",
            "왼쪽 값": "",
            "Symmetry Index": "",
            "데이터수": stats.get("count", 0),
            "상태컬럼": analysis_result.get("scone_phase_state_column", ""),
        })
    return rows


def _build_symmetry_rows(
    file_name: str,
    var: str,
    analysis_result: Dict,
    variable_preset: Dict,
    exported_symmetry_pairs: Set[Tuple[str, str, str]],
):
    rows = []
    symmetry_features = analysis_result.get("symmetry_features", {})
    pair_var = analysis_result.get("symmetry_pair_variable", "")
    if not pair_var:
        return rows

    pair_key = tuple(sorted((str(var), str(pair_var))))
    for feature_name, stats in symmetry_features.items():
        export_key = (pair_key[0], pair_key[1], str(feature_name))
        if export_key in exported_symmetry_pairs:
            continue
        exported_symmetry_pairs.add(export_key)

        rows.append({
            "파일명": file_name,
            "변수명": var,
            "분석유형": "좌우 대칭",
            "단위": analysis_result.get("display_unit", ""),
            "변수군": variable_preset.get("variable_group", ""),
            "추천 Preset": variable_preset.get("recommended_preset", ""),
            "매칭 패턴": variable_preset.get("matched_pattern", ""),
            "평균": "",
            "표준편차": "",
            "최댓값": "",
            "최솟값": "",
            "중앙값": "",
            "범위": "",
            "RMS": "",
            "AUC": "",
            "Positive AUC": "",
            "Negative AUC": "",
            "Max Time": "",
            "Max Percent": "",
            "Min Time": "",
            "Min Percent": "",
            "Threshold": "",
            "Threshold 초과구간": "",
            "Onset": "",
            "Offset": "",
            "Active Duration": "",
            "Pair 변수": pair_var,
            "대칭 Feature": feature_name,
            "오른쪽 값": stats.get("right_value", np.nan),
            "왼쪽 값": stats.get("left_value", np.nan),
            "Symmetry Index": stats.get("symmetry_index", np.nan),
            "데이터수": "",
            "상태컬럼": "",
        })
    return rows
