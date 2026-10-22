"""Analysis workflow helpers separated from the Tkinter UI."""

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis_engine import (
    analyze_variable_comprehensive,
    compute_common_features,
    compute_symmetry_features,
    find_side_pair_variable,
)
from utils.unit_utils import convert_values_for_display


ProgressCallback = Callable[[float, str], None]
TimeCoverageCallback = Callable[[pd.DataFrame], bool]
StateColumnCallback = Callable[[pd.DataFrame, str], Optional[str]]


def run_analysis_workflow(
    dataframes: Sequence[pd.DataFrame],
    file_paths: Sequence[str],
    selected_vars: Sequence[str],
    threshold_value: float,
    lang: Dict[str, str],
    data_validator,
    check_time_coverage: TimeCoverageCallback,
    get_state_column_for_variable: StateColumnCallback,
    progress_callback: Optional[ProgressCallback] = None,
    skip_missing_variables: bool = False,
):
    """Run variable analysis without depending on Tkinter widgets."""
    _notify(progress_callback, 0, lang["analysis_progress"])
    results_parts: List[str] = []
    analysis_results: Dict[str, Dict] = {}
    total_variables = max(len(selected_vars) * len(dataframes), 1)
    current_var = 0

    for file_idx, (path, df) in enumerate(zip(file_paths, dataframes)):
        file_name = os.path.basename(path)
        results_parts.append(f"{'=' * 60}\n")
        results_parts.append(f"파일: {file_name}\n")
        results_parts.append(f"{'=' * 60}\n\n")

        for var in selected_vars:
            current_var += 1
            progress = (current_var / total_variables) * 100
            _notify(progress_callback, progress, f"분석 중: {var} ({current_var}/{total_variables})")

            try:
                if var not in df.columns:
                    results_parts.append(f"변수: {var}\n")
                    if skip_missing_variables:
                        results_parts.append("  건너뜀: 해당 파일에 변수가 없습니다.\n\n")
                    else:
                        results_parts.append(f"  오류: {lang['var_not_exist']}\n\n")
                    continue

                if not pd.api.types.is_numeric_dtype(df[var]):
                    results_parts.append(f"변수: {var}\n")
                    label = "건너뜀" if skip_missing_variables else "오류"
                    results_parts.append(f"  {label}: {lang['not_numeric']}\n\n")
                    continue

                if not check_time_coverage(df):
                    warning_msg = _get_time_coverage_warning(data_validator, df)
                    results_parts.append(f"변수: {var}\n")
                    results_parts.append(f"  경고: {warning_msg} (분석 제외)\n\n")
                    analysis_key = f"{file_idx}_{file_name}_{var}"
                    analysis_results[analysis_key] = _build_time_coverage_skip_result(var, warning_msg)
                    continue

                data_arr = df[var].values
                time_arr = df["time"].values if "time" in df.columns else None
                state_col = get_state_column_for_variable(df, var)
                state_arr = df[state_col].values if state_col else None
                analysis_result = analyze_variable_comprehensive(
                    data_arr,
                    time_arr,
                    var,
                    state_arr=state_arr,
                    state_column_name=state_col or "",
                    threshold=threshold_value,
                )

                pair_var = find_side_pair_variable(var, list(df.columns))
                if pair_var and pair_var in df.columns and pd.api.types.is_numeric_dtype(df[pair_var]):
                    pair_common_features = compute_common_features(
                        convert_values_for_display(pair_var, df[pair_var].values),
                        time_arr,
                        threshold=threshold_value,
                    )
                    analysis_result["symmetry_pair_variable"] = pair_var
                    analysis_result["symmetry_features"] = compute_symmetry_features(
                        var,
                        pair_var,
                        analysis_result.get("common_features", {}),
                        pair_common_features,
                    )

                analysis_key = f"{file_idx}_{file_name}_{var}"
                analysis_results[analysis_key] = analysis_result
                results_parts.append(format_analysis_result_text(analysis_result))
            except Exception as exc:
                results_parts.append(f"변수 '{var}' 분석 중 오류: {str(exc)}\n\n")

        results_parts.append("\n")

    return "".join(results_parts), analysis_results


def format_analysis_result_text(analysis_result: Dict) -> str:
    """Format one analysis result for the text output panel."""
    result_text = ""
    var_name = analysis_result["variable_name"]
    result_text += f"변수: {var_name}\n"
    display_unit = analysis_result.get("display_unit", "")
    if display_unit:
        result_text += f"  - 표시/저장 단위: {display_unit}\n"

    variable_preset = analysis_result.get("variable_preset", {})
    if variable_preset:
        result_text += "  [추천 Preset]\n"
        result_text += f"  - 변수군: {variable_preset.get('variable_group', 'N/A')}\n"
        result_text += f"  - 추천 preset: {variable_preset.get('recommended_preset', 'N/A')}\n"
        result_text += f"  - 매칭 패턴: {variable_preset.get('matched_pattern', 'N/A')}\n"

    quality = analysis_result.get("data_quality", {})
    result_text += f"  [데이터 품질: {quality.get('grade', 'Unknown')}]\n"
    result_text += f"  - 전체 데이터 포인트: {quality.get('total_points', 0):,}개\n"
    result_text += f"  - 유효 데이터 포인트: {quality.get('valid_points', 0):,}개\n"
    result_text += f"  - 완성도: {quality.get('completeness', 0):.1f}%\n"

    overall_stats = analysis_result.get("overall_statistics", {})
    if overall_stats.get("count", 0) > 0:
        result_text += "  \n  [전체 데이터 통계]\n"
        result_text += f"  - 평균: {overall_stats.get('mean', np.nan):.6f}\n"
        result_text += f"  - 표준편차: {overall_stats.get('std', np.nan):.6f}\n"
        result_text += f"  - 최댓값: {overall_stats.get('max', np.nan):.6f}\n"
        result_text += f"  - 최솟값: {overall_stats.get('min', np.nan):.6f}\n"
        result_text += f"  - 중앙값: {overall_stats.get('median', np.nan):.6f}\n"

    common_features = analysis_result.get("common_features", {})
    if common_features:
        result_text += "  \n  [공통 Feature]\n"
        result_text += f"  - 최댓값 시점: {common_features.get('max_time', np.nan):.6f}, {common_features.get('max_percent', np.nan):.2f}%\n"
        result_text += f"  - 최솟값 시점: {common_features.get('min_time', np.nan):.6f}, {common_features.get('min_percent', np.nan):.2f}%\n"
        result_text += f"  - ROM/Range: {common_features.get('rom', np.nan):.6f}\n"
        result_text += f"  - RMS: {common_features.get('rms', np.nan):.6f}\n"
        result_text += f"  - AUC: {common_features.get('auc', np.nan):.6f}\n"
        result_text += f"  - Positive AUC: {common_features.get('positive_auc', np.nan):.6f}\n"
        result_text += f"  - Negative AUC: {common_features.get('negative_auc', np.nan):.6f}\n"
        result_text += f"  - Threshold: {common_features.get('threshold', np.nan):.6f}\n"
        result_text += f"  - Threshold 초과 구간: {common_features.get('threshold_exceed_count', 0)}개\n"
        result_text += f"  - Onset/Offset: {common_features.get('onset_time', np.nan):.6f} / {common_features.get('offset_time', np.nan):.6f}\n"
        result_text += f"  - Active duration: {common_features.get('active_duration', np.nan):.6f}\n"

    scone_phase_stats = analysis_result.get("scone_phase_statistics", {})
    if scone_phase_stats:
        state_column = analysis_result.get("scone_phase_state_column", "")
        result_text += "  \n  [SCONE state 기반 Phase 통계]\n"
        if state_column:
            result_text += f"  - 사용 state 컬럼: {state_column}\n"
        for phase_name, stats in scone_phase_stats.items():
            if stats.get("count", 0) > 0:
                result_text += f"  [{phase_name}] (state {stats.get('state_values', '')})\n"
                result_text += f"    - 평균: {stats.get('mean', np.nan):.6f}\n"
                result_text += f"    - 표준편차: {stats.get('std', np.nan):.6f}\n"
                result_text += f"    - 최댓값: {stats.get('max', np.nan):.6f}\n"
                result_text += f"    - 최솟값: {stats.get('min', np.nan):.6f}\n"
                result_text += f"    - 범위: {stats.get('range', np.nan):.6f}\n"
                result_text += f"    - RMS: {stats.get('rms', np.nan):.6f}\n"
                result_text += f"    - AUC: {stats.get('auc', np.nan):.6f}\n"
                result_text += f"    - 데이터 포인트: {stats.get('count', 0)}개\n\n"

    symmetry_features = analysis_result.get("symmetry_features", {})
    if symmetry_features:
        pair_var = analysis_result.get("symmetry_pair_variable", "")
        result_text += "  \n  [좌우 대칭성]\n"
        if pair_var:
            result_text += f"  - Pair 변수: {pair_var}\n"
        for feature_name, stats in symmetry_features.items():
            result_text += (
                f"  - {feature_name}: SI={stats.get('symmetry_index', np.nan):.6f}% "
                f"(R={stats.get('right_value', np.nan):.6f}, L={stats.get('left_value', np.nan):.6f})\n"
            )

    notes = analysis_result.get("analysis_notes", [])
    if notes:
        result_text += "  [분석 노트]\n"
        for note in notes:
            result_text += f"  - {note}\n"

    quality_score = analysis_result.get("quality_score", 0)
    result_text += f"  \n  [분석 품질 점수: {quality_score:.1f}/100]\n"
    result_text += "-" * 60 + "\n\n"
    return result_text


def _notify(callback: Optional[ProgressCallback], value: float, message: str):
    if callback is not None:
        callback(value, message)


def _get_time_coverage_warning(data_validator, df: pd.DataFrame) -> str:
    time_valid, time_msg, _ = data_validator.validate_time_column(df)
    return "5초 미만 시뮬레이션 데이터" if time_valid else time_msg


def _build_time_coverage_skip_result(var: str, warning_msg: str) -> Dict:
    return {
        "variable_name": var,
        "overall_statistics": {},
        "common_features": {},
        "scone_phase_statistics": {},
        "scone_phase_state_column": "",
        "symmetry_pair_variable": "",
        "symmetry_features": {},
        "data_quality": {},
        "analysis_notes": [warning_msg],
        "time_coverage_failed": True,
    }
