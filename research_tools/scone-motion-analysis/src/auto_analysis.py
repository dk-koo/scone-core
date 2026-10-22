"""Automatic analysis candidate detection for loaded motion data."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from config.constants import STATE_COLS
from src.analysis_engine import find_side_pair_variable, recommend_variable_preset
from visualization.plot_grf_cop_com import (
    detect_com_variables,
    detect_cop_variables,
    detect_grf_variables,
    detect_momentum_variables,
)
from visualization.plot_muscle import CCI_PAIR_DEFINITIONS, detect_muscle_signal_variables


FEATURE_ANALYSIS_OPTIONS = (
    "All analyzable",
    "Joint angle",
    "Joint load / torque / force",
    "Muscle activation",
    "Muscle mechanics / force",
    "GRF / COP / COM",
    "Symmetry pairs",
    "Velocity",
    "Spatiotemporal",
    "Penalty / quality",
    "General numeric",
)

ANALYSIS_PURPOSE_OPTIONS = (
    "전체 자동 리포트",
    "보행 각도 분석",
    "관절 부하/토크 분석",
    "역학/관절 토크 분석",
    "근활성 분석",
    "GRF/COP/COM 균형 분석",
    "좌우 대칭 분석",
    "품질/패널티 분석",
    "일반 숫자 분석",
)

ANALYSIS_PURPOSE_TO_FEATURE = {
    "전체 자동 리포트": "All analyzable",
    "보행 각도 분석": "Joint angle",
    "관절 부하 분석": "Joint load / torque / force",
    "관절 부하/토크 분석": "Joint load / torque / force",
    "역학/관절 토크 분석": "Joint load / torque / force",
    "근활성 분석": "Muscle activation",
    "GRF/균형 분석": "GRF / COP / COM",
    "GRF/COP/COM 균형 분석": "GRF / COP / COM",
    "좌우 대칭 분석": "Symmetry pairs",
    "품질/패널티 분석": "Penalty / quality",
    "일반 숫자 분석": "General numeric",
}


def feature_filter_for_purpose(purpose: str) -> str:
    """Return the internal feature filter for a user-facing analysis purpose."""
    return ANALYSIS_PURPOSE_TO_FEATURE.get(purpose, FEATURE_ANALYSIS_OPTIONS[0])


def is_known_analysis_purpose(purpose: str) -> bool:
    """Return True when a purpose is a current option or supported legacy alias."""
    return purpose in ANALYSIS_PURPOSE_TO_FEATURE

_FEATURE_FILTER_PRESET_KEYS = {
    "Joint angle": {"joint_angle"},
    "Joint load / torque / force": {"joint_load"},
    "Muscle activation": {"muscle_activation"},
    "Muscle mechanics / force": {"muscle_mechanics", "muscle_force"},
    "GRF / COP / COM": {"grf_cop_com"},
    "Velocity": {"velocity"},
    "Spatiotemporal": {"spatiotemporal"},
    "Penalty / quality": {"penalty_quality"},
    "General numeric": {"general"},
}

_JOINT_LOAD_SIGNAL_TOKENS = (
    ".load",
    "_load",
    ".force",
    "_force",
    ".torque",
    "_torque",
    ".moment",
    "_moment",
)
_JOINT_LOAD_JOINT_TOKENS = (
    "hip",
    "knee",
    "ankle",
    "pelvis",
    "subtalar",
    "mtp",
    "lumbar",
    "back",
)
_JOINT_LOAD_EXCLUDED_TOKENS = (
    "activation",
    "excitation",
    "fiber",
    "tendon",
    "mtu_force",
    "grf",
    "cop",
    "com_",
    "lin_mom",
    "ang_mom",
    "penalty",
)


@dataclass
class AutoAnalysisPlan:
    selected_variables: List[str]
    graph_variables: List[str]
    available_analyses: List[str]
    skipped_analyses: List[str]
    group_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    feature_filter: str = FEATURE_ANALYSIS_OPTIONS[0]


def build_auto_analysis_plan(
    dataframes: Sequence[pd.DataFrame],
    file_paths: Sequence[str] = (),
    max_graph_variables: int = 30,
    feature_filter: str = FEATURE_ANALYSIS_OPTIONS[0],
) -> AutoAnalysisPlan:
    """Detect variables and analysis types that can be run safely."""
    numeric_variables = _collect_numeric_analysis_variables(dataframes)
    all_columns = _collect_all_columns(dataframes)
    selected_variables, filter_warnings, selected_filter = _filter_variables_by_feature(
        numeric_variables,
        feature_filter,
    )
    group_counts = _count_variable_groups(selected_variables)

    recognized_variables = [
        variable
        for variable in selected_variables
        if recommend_variable_preset(variable).get("preset_key") != "general"
    ]
    graph_variables = (recognized_variables or selected_variables)[:max_graph_variables]

    available = []
    skipped = []

    if selected_variables:
        available.append(f"공통 feature 분석: 숫자 변수 {len(selected_variables)}개")
        available.append("Raw waveform 시각화")
    else:
        skipped.append(f"{selected_filter}: 분석할 숫자 변수가 없습니다.")
        skipped.append("Raw waveform 시각화: 숫자 변수가 없습니다.")

    state_columns = _detect_state_columns(all_columns)
    if selected_variables and state_columns:
        available.append(f"SCONE phase 분석: {', '.join(state_columns)}")
        available.append("Gait cycle waveform / phase shading")
    elif selected_variables:
        skipped.append("SCONE phase 분석: leg0_l.state 또는 leg1_r.state가 없습니다.")
        skipped.append("Gait cycle waveform: state 컬럼이 없어 time-axis 그래프로 대체됩니다.")

    pair_count = _count_left_right_pairs(selected_variables)
    if pair_count:
        available.append(f"좌우 symmetry 분석: pair {pair_count}개")
    elif selected_variables:
        skipped.append("좌우 symmetry 분석: 자동 감지된 좌우 pair가 없습니다.")

    selected_set = set(selected_variables)
    muscle_records = [
        record
        for record in detect_muscle_signal_variables(all_columns)
        if record.variable in selected_set
    ]
    activation_records = [record for record in muscle_records if record.signal == "activation"]
    if muscle_records:
        available.append(f"Muscle activation 시각화: muscle signal {len(muscle_records)}개")
    elif selected_variables:
        skipped.append("Muscle activation 시각화: activation/excitation/input 변수가 없습니다.")

    cci_pairs = _detect_available_cci_pairs(activation_records)
    if cci_pairs:
        available.append(f"Co-contraction CCI: {', '.join(cci_pairs)}")
    elif selected_variables:
        skipped.append("Co-contraction CCI: 필요한 muscle activation pair가 부족합니다.")

    selected_columns = list(selected_variables)
    grf = detect_grf_variables(selected_columns)
    cop = detect_cop_variables(selected_columns)
    com = detect_com_variables(selected_columns)
    momentum = detect_momentum_variables(selected_columns)
    if grf:
        available.append("GRF 시각화")
    elif selected_variables:
        skipped.append("GRF 시각화: leg0_l/leg1_r grf 변수가 없습니다.")
    if cop or com:
        available.append("COP/COM 시각화")
    elif selected_variables:
        skipped.append("COP/COM 시각화: COP 또는 COM 변수가 없습니다.")
    if momentum.get("linear") or momentum.get("angular"):
        available.append("Momentum 시각화")
    elif selected_variables:
        skipped.append("Momentum 시각화: lin_mom 또는 ang_mom 변수가 없습니다.")

    if len(file_paths) > 1 and selected_variables:
        available.append("조건 비교 시각화")
    elif selected_variables:
        skipped.append("조건 비교 시각화: 비교할 파일/조건이 2개 이상 필요합니다.")

    warnings = list(filter_warnings)
    if len(recognized_variables) > max_graph_variables:
        warnings.append(
            f"그래프 변수는 화면 복잡도를 줄이기 위해 후보 {len(recognized_variables)}개 중 "
            f"처음 {max_graph_variables}개만 자동 추가했습니다."
        )

    return AutoAnalysisPlan(
        selected_variables=selected_variables,
        graph_variables=graph_variables,
        available_analyses=available,
        skipped_analyses=skipped,
        group_counts=group_counts,
        warnings=warnings,
        feature_filter=selected_filter,
    )


def format_auto_analysis_plan(plan: AutoAnalysisPlan) -> str:
    """Format the automatic analysis plan for the results panel."""
    lines = [
        "=== 자동 분석 후보 ===",
        "",
        f"Feature 선택: {plan.feature_filter}",
        f"분석 대상 변수: {len(plan.selected_variables)}개",
        f"그래프 후보 변수: {len(plan.graph_variables)}개",
        "",
        "[변수군 요약]",
    ]

    if plan.group_counts:
        for group_name, count in sorted(plan.group_counts.items()):
            lines.append(f"- {group_name}: {count}개")
    else:
        lines.append("- 감지된 숫자 변수가 없습니다.")

    lines.extend(["", "[실행 가능한 분석]"])
    lines.extend(f"- {item}" for item in plan.available_analyses)

    lines.extend(["", "[건너뛸 분석]", "없는 변수군은 오류가 아니라 자동으로 건너뜁니다."])
    lines.extend(f"- {item}" for item in plan.skipped_analyses)

    if plan.warnings:
        lines.extend(["", "[주의]"])
        lines.extend(f"- {item}" for item in plan.warnings)

    if plan.graph_variables:
        lines.extend(["", "[그래프 변수 미리보기]"])
        preview = plan.graph_variables[:12]
        lines.extend(f"- {variable}" for variable in preview)
        if len(plan.graph_variables) > len(preview):
            lines.append(f"- ... 외 {len(plan.graph_variables) - len(preview)}개")

    return "\n".join(lines) + "\n"


def _filter_variables_by_feature(variables: Sequence[str], feature_filter: str):
    selected_filter = feature_filter if feature_filter in FEATURE_ANALYSIS_OPTIONS else FEATURE_ANALYSIS_OPTIONS[0]
    warnings = []
    if selected_filter != feature_filter:
        warnings.append(f"알 수 없는 Feature 선택 '{feature_filter}'을 All analyzable로 처리했습니다.")

    if selected_filter == "All analyzable":
        return list(variables), warnings, selected_filter

    if selected_filter == "Symmetry pairs":
        return _variables_with_left_right_pairs(variables), warnings, selected_filter

    if selected_filter == "Joint load / torque / force":
        filtered = [variable for variable in variables if _is_joint_load_candidate(variable)]
        warnings.append(
            "관절 부하 후보는 hip/knee/ankle/pelvis 등 관절 이름과 "
            "load/force/torque/moment 패턴이 함께 있는 변수만 선택했습니다."
        )
        return filtered, warnings, selected_filter

    if selected_filter == "GRF / COP / COM":
        filtered = [
            variable
            for variable in variables
            if (
                recommend_variable_preset(variable).get("preset_key") == "grf_cop_com"
                or _is_balance_strategy_candidate(variable)
            )
        ]
        return filtered, warnings, selected_filter

    preset_keys = _FEATURE_FILTER_PRESET_KEYS.get(selected_filter, set())
    filtered = [
        variable
        for variable in variables
        if recommend_variable_preset(variable).get("preset_key") in preset_keys
    ]
    return filtered, warnings, selected_filter


def _collect_all_columns(dataframes: Sequence[pd.DataFrame]) -> List[str]:
    columns = {str(column) for df in dataframes for column in df.columns}
    return sorted(columns)


def _collect_numeric_analysis_variables(dataframes: Sequence[pd.DataFrame]) -> List[str]:
    variables = []
    for column in _collect_all_columns(dataframes):
        if _is_excluded_column(column):
            continue
        if _is_numeric_with_data(column, dataframes):
            variables.append(column)
    return variables


def _is_excluded_column(column: str) -> bool:
    lower = str(column).lower()
    state_columns = {value.lower() for value in STATE_COLS.values()}
    return lower == "time" or lower in state_columns or lower.endswith(".state") or lower.endswith("_state")


def _is_numeric_with_data(column: str, dataframes: Sequence[pd.DataFrame]) -> bool:
    for df in dataframes:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            return True
    return False


def _count_variable_groups(variables: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for variable in variables:
        group = recommend_variable_preset(variable).get("variable_group", "Unknown")
        counts[group] = counts.get(group, 0) + 1
    return counts


def _detect_state_columns(columns: Iterable[str]) -> List[str]:
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    detected = []
    for state_column in STATE_COLS.values():
        original = lower_to_original.get(state_column.lower())
        if original:
            detected.append(original)
    return detected


def _count_left_right_pairs(variables: Sequence[str]) -> int:
    seen = set()
    pair_count = 0
    for variable in variables:
        if variable in seen:
            continue
        pair = find_side_pair_variable(variable, list(variables))
        if not pair:
            continue
        seen.add(variable)
        seen.add(pair)
        pair_count += 1
    return pair_count


def _variables_with_left_right_pairs(variables: Sequence[str]) -> List[str]:
    selected = []
    selected_set = set()
    for variable in variables:
        pair = find_side_pair_variable(variable, list(variables))
        if not pair:
            continue
        for candidate in (variable, pair):
            if candidate not in selected_set:
                selected.append(candidate)
                selected_set.add(candidate)
    return selected


def _is_joint_load_candidate(variable: str) -> bool:
    lower = str(variable).lower()
    if any(token in lower for token in _JOINT_LOAD_EXCLUDED_TOKENS):
        return False
    if not any(token in lower for token in _JOINT_LOAD_SIGNAL_TOKENS):
        return False
    if not any(token in lower for token in _JOINT_LOAD_JOINT_TOKENS):
        return False
    return recommend_variable_preset(variable).get("preset_key") == "joint_load"


def _is_balance_strategy_candidate(variable: str) -> bool:
    lower = str(variable).lower()
    if not any(joint in lower for joint in ("ankle", "knee", "hip")):
        return False
    if not any(token in lower for token in ("moment", "torque", "load")):
        return False
    return recommend_variable_preset(variable).get("preset_key") == "joint_load"


def _detect_available_cci_pairs(activation_records) -> List[str]:
    detected = []
    for pair_name, (agonists, antagonists) in CCI_PAIR_DEFINITIONS.items():
        if _has_muscle_group(activation_records, agonists) and _has_muscle_group(activation_records, antagonists):
            detected.append(pair_name)
    return detected


def _has_muscle_group(records, requested_muscles) -> bool:
    return any(
        any(_muscle_name_matches(record.muscle, requested) for record in records)
        for requested in requested_muscles
    )


def _muscle_name_matches(actual: str, requested: str) -> bool:
    actual_norm = _normalize_name(actual)
    requested_norm = _normalize_name(requested)
    return bool(actual_norm and requested_norm) and (
        actual_norm in requested_norm or requested_norm in actual_norm
    )


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())
