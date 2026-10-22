"""Automatic summary helpers for motion-analysis results."""

import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from utils.file_utils import format_legend_label, sanitize_dataframe_for_spreadsheet_export


SUMMARY_METRICS = ("mean", "max", "rom", "rms", "auc")


def build_analysis_summary(
    selected_vars: Sequence[str],
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
    dataframes: Sequence[pd.DataFrame] = (),
    analysis_purpose: str = "",
) -> str:
    """Build a short Korean text summary that can sit above detailed results."""
    selected_vars = list(selected_vars or _variables_from_results(analysis_results))
    records = _result_records(file_paths, analysis_results)
    comparison = build_condition_comparison_table(file_paths, analysis_results)

    lines = [
        "=== 자동 요약 ===",
        f"- 분석 목적: {analysis_purpose or '선택 안 됨'}",
        f"- 파일/조건: {len(file_paths)}개 ({_condition_label_text(file_paths)})",
        f"- 선택 변수: {len(selected_vars)}개",
        f"- 완료된 분석 항목: {len(records)}개",
    ]

    quality_line = _quality_summary_line(records)
    if quality_line:
        lines.append(f"- 데이터 품질: {quality_line}")

    phase_line = _phase_summary_line(records)
    if phase_line:
        lines.append(f"- SCONE phase: {phase_line}")

    symmetry_line = _symmetry_summary_line(records)
    if symmetry_line:
        lines.append(f"- 좌우 대칭: {symmetry_line}")

    missing_line = _missing_variable_summary_line(selected_vars, dataframes)
    if missing_line:
        lines.append(f"- 누락 변수: {missing_line}")

    comparison_lines = _comparison_highlight_lines(comparison)
    if comparison_lines:
        lines.append("- 조건 비교 하이라이트:")
        lines.extend(f"  {line}" for line in comparison_lines)
    elif len(file_paths) >= 2:
        lines.append("- 조건 비교 하이라이트: 비교 가능한 유한한 feature 값이 없습니다.")

    lines.append("=" * 60)
    return "\n".join(lines)


def build_condition_comparison_table(
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Return a user-friendly comparison table against the first condition."""
    records = _result_records(file_paths, analysis_results)
    if not records:
        return pd.DataFrame(columns=_comparison_columns())

    rows = []
    by_variable: Dict[str, List[Dict]] = {}
    for record in records:
        by_variable.setdefault(record["variable"], []).append(record)

    for variable, variable_records in by_variable.items():
        variable_records = sorted(variable_records, key=lambda item: item["file_index"])
        reference = variable_records[0]
        for record in variable_records[1:]:
            for metric in SUMMARY_METRICS:
                ref_value = reference["features"].get(metric, np.nan)
                value = record["features"].get(metric, np.nan)
                if not np.isfinite(ref_value) or not np.isfinite(value):
                    continue
                delta = value - ref_value
                reference_magnitude = abs(ref_value)
                rows.append({
                    "reference_condition": reference["condition"],
                    "condition": record["condition"],
                    "variable": variable,
                    "metric": metric,
                    "reference_value": ref_value,
                    "condition_value": value,
                    "delta": delta,
                    "delta_percent": delta / reference_magnitude * 100.0 if not np.isclose(reference_magnitude, 0.0) else np.nan,
                })

    return sanitize_dataframe_for_spreadsheet_export(pd.DataFrame(rows, columns=_comparison_columns()))


def save_condition_comparison_table(
    file_paths: Sequence[str],
    analysis_results: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Save condition comparison table if comparison rows exist."""
    table = build_condition_comparison_table(file_paths, analysis_results)
    if table.empty:
        return ""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "condition_comparison_summary.csv")
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _comparison_columns() -> List[str]:
    return [
        "reference_condition",
        "condition",
        "variable",
        "metric",
        "reference_value",
        "condition_value",
        "delta",
        "delta_percent",
    ]


def _variables_from_results(analysis_results: Dict[str, Dict]) -> List[str]:
    variables = []
    for result in analysis_results.values():
        if not isinstance(result, dict):
            continue
        variable = str(result.get("variable_name", "")).strip()
        if variable and variable not in variables:
            variables.append(variable)
    return variables


def _result_records(file_paths: Sequence[str], analysis_results: Dict[str, Dict]) -> List[Dict]:
    records = []
    for key, result in analysis_results.items():
        if not isinstance(result, dict) or result.get("time_coverage_failed", False):
            continue
        variable = result.get("variable_name", "")
        if not variable:
            continue
        file_index = _file_index_from_key(key)
        file_name = _file_name(file_paths, file_index)
        stats = result.get("overall_statistics", {}) or {}
        common = result.get("common_features", {}) or {}
        features = {
            "mean": common.get("mean", stats.get("mean", np.nan)),
            "max": common.get("max", stats.get("max", np.nan)),
            "rom": common.get("rom", np.nan),
            "rms": common.get("rms", np.nan),
            "auc": common.get("auc", np.nan),
        }
        records.append({
            "file_index": file_index,
            "file": file_name,
            "condition": format_legend_label(file_name),
            "variable": variable,
            "features": features,
            "quality": result.get("data_quality", {}) or {},
            "phase": result.get("scone_phase_statistics", {}) or {},
            "state_column": result.get("scone_phase_state_column", ""),
            "symmetry": result.get("symmetry_features", {}) or {},
            "pair_variable": result.get("symmetry_pair_variable", ""),
        })
    return records


def _file_index_from_key(key: str) -> int:
    try:
        return int(str(key).split("_", 1)[0])
    except (TypeError, ValueError):
        return 0


def _file_name(file_paths: Sequence[str], file_index: int) -> str:
    if 0 <= file_index < len(file_paths):
        return os.path.basename(str(file_paths[file_index]))
    return f"file_{file_index + 1}"


def _condition_label_text(file_paths: Sequence[str]) -> str:
    labels = []
    for path in file_paths:
        label = format_legend_label(os.path.basename(str(path)))
        if label not in labels:
            labels.append(label)
    return ", ".join(labels) if labels else "조건 없음"


def _quality_summary_line(records: Sequence[Dict]) -> str:
    counts: Dict[str, int] = {}
    for record in records:
        grade = str(record.get("quality", {}).get("grade", "Unknown"))
        counts[grade] = counts.get(grade, 0) + 1
    return ", ".join(f"{grade} {count}개" for grade, count in sorted(counts.items()))


def _phase_summary_line(records: Sequence[Dict]) -> str:
    state_columns = []
    phase_count = 0
    for record in records:
        state_column = record.get("state_column", "")
        if state_column and state_column not in state_columns:
            state_columns.append(state_column)
        if record.get("phase"):
            phase_count += 1
    if not state_columns:
        return "state 컬럼이 없어 phase 분석은 생략됨"
    return f"{', '.join(state_columns)} 사용, phase 결과 {phase_count}개"


def _symmetry_summary_line(records: Sequence[Dict]) -> str:
    rows = []
    for record in records:
        for feature, stats in record.get("symmetry", {}).items():
            value = stats.get("symmetry_index", np.nan)
            if np.isfinite(value):
                rows.append((abs(float(value)), record["variable"], feature, float(value)))
    if not rows:
        return ""
    rows.sort(reverse=True)
    _, variable, feature, value = rows[0]
    return f"pair feature {len(rows)}개, 최대 |SI| {variable} {feature} = {value:.2f}%"


def _missing_variable_summary_line(selected_vars: Sequence[str], dataframes: Sequence[pd.DataFrame]) -> str:
    if not selected_vars or not dataframes:
        return ""
    missing = []
    for variable in selected_vars:
        missing_count = sum(1 for df in dataframes if variable not in df.columns)
        if missing_count:
            missing.append(f"{variable}({missing_count}개 파일)")
    if not missing:
        return ""
    shown = ", ".join(missing[:5])
    suffix = f" 외 {len(missing) - 5}개" if len(missing) > 5 else ""
    return shown + suffix


def _comparison_highlight_lines(table: pd.DataFrame) -> List[str]:
    if table.empty:
        return []
    rows = table.copy()
    rows["abs_delta_percent"] = pd.to_numeric(rows["delta_percent"], errors="coerce").abs()
    rows["abs_delta"] = pd.to_numeric(rows["delta"], errors="coerce").abs()
    rows = rows.sort_values(["abs_delta_percent", "abs_delta"], ascending=False)
    highlights = []
    for _, row in rows.head(5).iterrows():
        delta_percent = row.get("delta_percent", np.nan)
        percent_text = f"{delta_percent:+.1f}%" if np.isfinite(delta_percent) else "Δ% N/A"
        highlights.append(
            f"- {row['condition']} vs {row['reference_condition']}: "
            f"{row['variable']} {row['metric']} Δ={row['delta']:+.4g} ({percent_text})"
        )
    return highlights
