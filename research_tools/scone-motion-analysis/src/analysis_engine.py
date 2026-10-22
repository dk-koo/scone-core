"""
분석 엔진 모듈 v3.0
- 기본 통계, 공통 feature, SCONE state phase 분석 기능 제공
"""

import numpy as np
from typing import Dict, List, Optional

from config.constants import SCONE_PHASE_STATES, STATISTICS_CONFIG
from utils.math_utils import calculate_comprehensive_stats, validate_data
from utils.unit_utils import convert_values_for_display, get_display_unit


def recommend_variable_preset(var_name: str) -> Dict[str, str]:
    """Recommend a feature preset from the variable name pattern table."""
    name = str(var_name)
    lower_name = name.lower()

    def result(preset_key: str, variable_group: str, recommended_preset: str, matched_pattern: str) -> Dict[str, str]:
        return {
            "preset_key": preset_key,
            "variable_group": variable_group,
            "recommended_preset": recommended_preset,
            "matched_pattern": matched_pattern,
        }

    # Put penalty before load/force so names like knee_l.load_penalty are quality signals.
    if "penalty" in lower_name:
        return result(
            "penalty_quality",
            "Quality/objective",
            "AUC, max, active duration",
            "*.penalty / *penalty*",
        )

    if lower_name.endswith((".rl", ".rf", ".rdp", ".rdv")):
        return result(
            "penalty_quality",
            "Reflex/controller signal",
            "peak, timing, AUC",
            "*.RL / *.RF / *.RDP / *.RDV",
        )

    if _is_velocity_variable(lower_name):
        return result(
            "velocity",
            "Velocity",
            "RMS, peak velocity, slope",
            "*_u / *.rot_vel_* / *.ang_vel_* / *.lin_vel_*",
        )

    if _is_joint_angle_position_variable(lower_name):
        return result(
            "joint_angle",
            "Joint kinematics",
            "ROM, peak, timing, phase feature",
            "hip_flexion_* / knee_angle_* / ankle_angle_* / pelvis_tilt",
        )

    if any(token in lower_name for token in ("fiber_length", "fiber_velocity", "tendon_length")):
        return result(
            "muscle_mechanics",
            "Muscle mechanics",
            "min/max/ROM, velocity peak",
            "*.fiber_length* / *.fiber_velocity* / *.tendon_length*",
        )

    if "mtu_force_norm" in lower_name or lower_name.endswith(".f"):
        return result(
            "muscle_force",
            "Muscle force",
            "peak, impulse",
            "*.mtu_force_norm / *.F",
        )

    if lower_name.endswith(".activation") or lower_name.endswith("_activation") or ".activation_" in lower_name:
        return result(
            "muscle_activation",
            "Muscle activation",
            "peak, timing, iEMG, onset/offset, CCI",
            "*.activation",
        )

    if lower_name.endswith((".excitation", ".input")) or lower_name.endswith(("_excitation", "_input")):
        return result(
            "muscle_activation",
            "Neural/controller",
            "peak, timing, delay, phase activation",
            "*.excitation / *.input",
        )

    if "leg" in lower_name and ("_grf" in lower_name or ".grf" in lower_name):
        return result(
            "grf_cop_com",
            "GRF",
            "VGRF peaks, AP impulse, stance detection",
            "leg*_grf* / leg*.grf*",
        )

    if "leg" in lower_name and ("_cop" in lower_name or ".cop" in lower_name):
        return result(
            "grf_cop_com",
            "COP",
            "range, path length, velocity",
            "leg*_cop* / leg*.cop*",
        )

    if lower_name.startswith(("com_", "com.", "lin_mom_", "lin_mom.", "ang_mom_", "ang_mom.")):
        return result(
            "grf_cop_com",
            "COM/momentum",
            "excursion, RMS, peak, COM-COP",
            "com_* / com.* / lin_mom_* / lin_mom.* / ang_mom_* / ang_mom.*",
        )

    if lower_name in {"step_length", "step_velocity"}:
        return result(
            "spatiotemporal",
            "Spatiotemporal",
            "mean, SD, asymmetry, condition trend",
            "step_length / step_velocity",
        )

    if any(token in lower_name for token in (".torque", "_torque", ".moment", "_moment")):
        return result(
            "joint_load",
            "Joint torque/moment",
            "peak, impulse, positive/negative AUC",
            "*.torque / *_torque / *.moment / *_moment",
        )

    if any(token in lower_name for token in (".load", "_load", ".force", "_force")):
        return result(
            "joint_load",
            "Joint load/force",
            "peak, impulse, loading rate, threshold",
            "*.load / *_load / *.force / *_force",
        )

    return result(
        "general",
        "General numeric",
        "basic statistics, peak/timing, AUC, threshold",
        "default",
    )


def _is_velocity_variable(lower_name: str) -> bool:
    velocity_tokens = (
        ".rot_vel_",
        "_rot_vel_",
        ".ang_vel_",
        "_ang_vel_",
        ".lin_vel_",
        "_lin_vel_",
    )
    return lower_name.endswith("_u") or any(token in lower_name for token in velocity_tokens)


def _is_joint_angle_position_variable(lower_name: str) -> bool:
    return (
        lower_name.startswith(("hip_flexion_", "knee_angle_", "ankle_angle_"))
        or lower_name == "pelvis_tilt"
        or lower_name.startswith(("pelvis.ori_", "torso.ori_"))
    )


def classify_variable(var_name: str) -> str:
    """Return the variable group recommended for the variable name."""
    return recommend_variable_preset(var_name)["variable_group"]


def _get_integration_function():
    return (getattr(np, "trapezoid", None) or getattr(np, "trapz"))


def _get_analysis_axis(time_arr: Optional[np.ndarray], expected_length: int) -> tuple:
    prepared_time = _prepare_time_array(time_arr, expected_length)
    if prepared_time is not None:
        return prepared_time, "time"
    return np.arange(expected_length, dtype=float), "index"


def _percent_axis(x_arr: np.ndarray) -> np.ndarray:
    if len(x_arr) == 0:
        return np.array([])
    if len(x_arr) == 1:
        return np.array([0.0])

    duration = x_arr[-1] - x_arr[0]
    if np.isfinite(duration) and duration > 0:
        return (x_arr - x_arr[0]) / duration * 100.0
    return np.linspace(0.0, 100.0, len(x_arr))


def _integrate_values(y_arr: np.ndarray, x_arr: np.ndarray) -> float:
    valid_mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    integrate = _get_integration_function()
    auc = 0.0
    has_segment = False
    for start_idx, end_idx in _true_runs(valid_mask):
        if end_idx <= start_idx:
            continue
        auc += float(integrate(y_arr[start_idx:end_idx + 1], x_arr[start_idx:end_idx + 1]))
        has_segment = True
    return auc if has_segment else np.nan


def _safe_percent(time_value: float, x_arr: np.ndarray, pct_arr: np.ndarray) -> float:
    if len(x_arr) == 0 or not np.isfinite(time_value):
        return np.nan
    return float(np.interp(time_value, x_arr, pct_arr))


def _calculate_threshold_features(
    data_arr: np.ndarray,
    x_arr: np.ndarray,
    pct_arr: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    if not np.isfinite(threshold):
        return {
            "threshold": np.nan,
            "threshold_exceed_count": 0,
            "threshold_exceed_points": 0,
            "onset_time": np.nan,
            "onset_percent": np.nan,
            "offset_time": np.nan,
            "offset_percent": np.nan,
            "active_duration": np.nan,
            "active_duration_percent": np.nan,
        }

    active_mask = np.isfinite(data_arr) & np.isfinite(x_arr) & (data_arr > threshold)
    runs = _true_runs(active_mask)
    total_duration = x_arr[-1] - x_arr[0] if len(x_arr) > 1 else 0.0

    active_duration = 0.0
    offset_idx = None
    for start_idx, end_idx in runs:
        next_idx = end_idx + 1 if end_idx + 1 < len(x_arr) else end_idx
        offset_idx = next_idx
        if np.isfinite(x_arr[next_idx]) and np.isfinite(x_arr[start_idx]):
            active_duration += max(0.0, float(x_arr[next_idx] - x_arr[start_idx]))

    onset_idx = runs[0][0] if runs else None
    onset_time = float(x_arr[onset_idx]) if onset_idx is not None else np.nan
    offset_time = float(x_arr[offset_idx]) if offset_idx is not None else np.nan

    return {
        "threshold": float(threshold),
        "threshold_exceed_count": len(runs),
        "threshold_exceed_points": int(active_mask.sum()),
        "onset_time": onset_time,
        "onset_percent": _safe_percent(onset_time, x_arr, pct_arr),
        "offset_time": offset_time,
        "offset_percent": _safe_percent(offset_time, x_arr, pct_arr),
        "active_duration": active_duration if runs else 0.0,
        "active_duration_percent": (
            active_duration / total_duration * 100.0
            if runs and np.isfinite(total_duration) and total_duration > 0
            else 0.0
        ),
    }


def compute_common_features(
    data_arr: np.ndarray,
    time_arr: Optional[np.ndarray] = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Calculate common scalar features for one selected variable."""
    try:
        data_arr = np.asarray(data_arr, dtype=float)
    except (TypeError, ValueError):
        return {}

    if data_arr.size == 0:
        return {}

    x_arr, axis_basis = _get_analysis_axis(time_arr, len(data_arr))
    pct_arr = _percent_axis(x_arr)
    valid_mask = np.isfinite(data_arr) & np.isfinite(x_arr)
    valid_values = data_arr[valid_mask]

    if valid_values.size == 0:
        return {}

    valid_indices = np.where(valid_mask)[0]
    max_idx = int(valid_indices[np.argmax(data_arr[valid_mask])])
    min_idx = int(valid_indices[np.argmin(data_arr[valid_mask])])
    max_value = float(data_arr[max_idx])
    min_value = float(data_arr[min_idx])
    auc = _integrate_values(data_arr, x_arr)
    positive_values = np.where(np.isfinite(data_arr), np.maximum(data_arr, 0.0), np.nan)
    negative_values = np.where(np.isfinite(data_arr), np.minimum(data_arr, 0.0), np.nan)
    positive_auc = _integrate_values(positive_values, x_arr)
    negative_auc = _integrate_values(negative_values, x_arr)

    features = {
        "count": int(valid_values.size),
        "mean": float(np.mean(valid_values)),
        "std": (
            float(np.std(valid_values, ddof=STATISTICS_CONFIG.get('ddof', 0)))
            if valid_values.size > STATISTICS_CONFIG.get('ddof', 0)
            else np.nan
        ),
        "max": max_value,
        "max_time": float(x_arr[max_idx]),
        "max_percent": float(pct_arr[max_idx]),
        "min": min_value,
        "min_time": float(x_arr[min_idx]),
        "min_percent": float(pct_arr[min_idx]),
        "rom": max_value - min_value,
        "rms": float(np.sqrt(np.mean(np.square(valid_values)))),
        "auc": auc,
        "positive_auc": positive_auc,
        "negative_auc": negative_auc,
        "axis_basis": axis_basis,
    }
    features.update(_calculate_threshold_features(data_arr, x_arr, pct_arr, threshold))
    return features


def detect_variable_side(var_name: str) -> Optional[str]:
    var_lower = var_name.lower()
    if "leg1_r" in var_lower or "_r" in var_lower or ".r" in var_lower or "right" in var_lower:
        return "r"
    if "leg0_l" in var_lower or "_l" in var_lower or ".l" in var_lower or "left" in var_lower:
        return "l"
    return None


def find_side_pair_variable(var_name: str, available_columns: List[str]) -> Optional[str]:
    lower_lookup = {column.lower(): column for column in available_columns}
    candidates = []

    replacements = [
        ("leg0_l", "leg1_r"),
        ("leg1_r", "leg0_l"),
        ("_l", "_r"),
        ("_r", "_l"),
        (".l", ".r"),
        (".r", ".l"),
        ("left", "right"),
        ("right", "left"),
    ]
    for source, target in replacements:
        if source in var_name.lower():
            start_idx = var_name.lower().find(source)
            candidate = var_name[:start_idx] + target + var_name[start_idx + len(source):]
            candidates.append(candidate)

    for candidate in candidates:
        matched = lower_lookup.get(candidate.lower())
        if matched and matched != var_name:
            return matched
    return None


def _symmetry_index(right_value: float, left_value: float) -> float:
    if not np.isfinite(right_value) or not np.isfinite(left_value):
        return np.nan
    denominator = 0.5 * (right_value + left_value)
    if np.isclose(denominator, 0.0):
        return np.nan
    return float((right_value - left_value) / denominator * 100.0)


def compute_symmetry_features(
    var_name: str,
    pair_var_name: str,
    selected_features: Dict[str, float],
    pair_features: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    selected_side = detect_variable_side(var_name)
    pair_side = detect_variable_side(pair_var_name)
    if selected_side not in ("l", "r") or pair_side not in ("l", "r") or selected_side == pair_side:
        return {}

    scalar_features = ["mean", "max", "min", "rom", "rms", "auc", "positive_auc", "negative_auc"]
    result = {}
    for feature_name in scalar_features:
        selected_value = selected_features.get(feature_name, np.nan)
        pair_value = pair_features.get(feature_name, np.nan)
        if selected_side == "r":
            right_value, left_value = selected_value, pair_value
        else:
            right_value, left_value = pair_value, selected_value
        result[feature_name] = {
            "right_value": right_value,
            "left_value": left_value,
            "symmetry_index": _symmetry_index(right_value, left_value),
        }
    return result


def create_scone_phase_masks(state_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Return boolean masks for SCONE state phases and composite phases."""
    try:
        raw_state = np.asarray(state_arr, dtype=float)
    except (TypeError, ValueError):
        return {}

    if raw_state.size == 0:
        return {}

    rounded_state = np.full(raw_state.shape, -1, dtype=int)
    finite_mask = np.isfinite(raw_state)
    rounded_state[finite_mask] = np.rint(raw_state[finite_mask]).astype(int)
    valid_state_mask = finite_mask & np.isin(rounded_state, [0, 1, 2, 3, 4])

    return {
        phase_name: valid_state_mask & np.isin(rounded_state, state_values)
        for phase_name, state_values in SCONE_PHASE_STATES.items()
    }


def _true_runs(mask: np.ndarray) -> List[tuple]:
    """Return inclusive start/end index pairs for consecutive True values."""
    runs = []
    start_idx = None
    for idx, is_true in enumerate(mask):
        if is_true and start_idx is None:
            start_idx = idx
        elif not is_true and start_idx is not None:
            runs.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        runs.append((start_idx, len(mask) - 1))

    return runs


def _masked_auc(data_arr: np.ndarray, mask: np.ndarray, time_arr: Optional[np.ndarray]) -> float:
    """Integrate each continuous masked segment and sum the segment AUC values."""
    if time_arr is None:
        x_arr = np.arange(len(data_arr), dtype=float)
    else:
        x_arr = np.asarray(time_arr, dtype=float)

    valid_mask = mask & np.isfinite(data_arr) & np.isfinite(x_arr)
    auc = 0.0
    integrate = (getattr(np, "trapezoid", None) or getattr(np, "trapz"))
    for start_idx, end_idx in _true_runs(valid_mask):
        if end_idx <= start_idx:
            continue
        auc += float(integrate(data_arr[start_idx:end_idx + 1], x_arr[start_idx:end_idx + 1]))
    return auc


def _prepare_time_array(time_arr: Optional[np.ndarray], expected_length: int) -> Optional[np.ndarray]:
    if time_arr is None:
        return None

    try:
        prepared_time = np.asarray(time_arr, dtype=float)
    except (TypeError, ValueError):
        return None

    if len(prepared_time) != expected_length:
        return None
    if not np.isfinite(prepared_time).all():
        return None
    if len(prepared_time) > 1 and np.any(np.diff(prepared_time) <= 0):
        return None

    return prepared_time


def _calculate_phase_statistics(
    data_arr: np.ndarray,
    mask: np.ndarray,
    time_arr: Optional[np.ndarray],
    ddof: int,
) -> Dict[str, float]:
    phase_values = data_arr[mask]
    valid_values = phase_values[np.isfinite(phase_values)]

    if valid_values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "rms": np.nan,
            "auc": np.nan,
            "count": 0,
        }

    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))
    std_value = (
        float(np.std(valid_values, ddof=ddof))
        if valid_values.size > ddof
        else np.nan
    )

    return {
        "mean": float(np.mean(valid_values)),
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "rms": float(np.sqrt(np.mean(np.square(valid_values)))),
        "auc": _masked_auc(data_arr, mask, time_arr),
        "count": int(valid_values.size),
    }


def analyze_data_by_scone_phases(
    data_arr: np.ndarray,
    state_arr: np.ndarray,
    time_arr: Optional[np.ndarray] = None,
    ddof: int = None,
) -> Dict[str, Dict]:
    """Calculate selected-variable statistics for SCONE state-based phases."""
    if len(data_arr) != len(state_arr):
        return {}

    try:
        data_arr = np.asarray(data_arr, dtype=float)
    except (TypeError, ValueError):
        return {}

    if ddof is None:
        ddof = STATISTICS_CONFIG.get('ddof', 0)

    prepared_time = _prepare_time_array(time_arr, len(data_arr))
    phase_masks = create_scone_phase_masks(state_arr)
    if not phase_masks:
        return {}

    phase_stats = {}
    for phase_name, mask in phase_masks.items():
        stats = _calculate_phase_statistics(data_arr, mask, prepared_time, ddof)
        stats["phase_name"] = phase_name
        stats["state_values"] = ",".join(str(value) for value in SCONE_PHASE_STATES[phase_name])
        stats["auc_time_basis"] = "time" if prepared_time is not None else "index"
        phase_stats[phase_name] = stats

    return phase_stats


def calculate_overall_statistics(data_arr: np.ndarray, 
                               variable_name: str = "") -> Dict[str, any]:
    """
    전체 데이터에 대한 포괄적 통계 계산
    """
    try:
        ddof = STATISTICS_CONFIG.get('ddof', 0)
        remove_outliers = STATISTICS_CONFIG.get('remove_outliers', False)
        
        stats = calculate_comprehensive_stats(
            data_arr, 
            ddof=ddof,
            remove_outliers=remove_outliers,
            include_percentiles=True
        )
        
        stats.update({
            'variable_name': variable_name,
            'analysis_type': 'overall',
            'outliers_removed': remove_outliers,
            'ddof_used': ddof
        })
        
        return stats
        
    except Exception as e:
        print(f"전체 통계 계산 오류: {str(e)}")
        return {
            'variable_name': variable_name,
            'analysis_type': 'overall',
            'error': str(e),
            'mean': np.nan,
            'std': np.nan,
            'max': np.nan,
            'min': np.nan,
            'count': 0
        }

def analyze_variable_comprehensive(data_arr: np.ndarray,
                                 time_arr: Optional[np.ndarray] = None,
                                 variable_name: str = "",
                                 state_arr: Optional[np.ndarray] = None,
                                 state_column_name: str = "",
                                 threshold: float = 0.0) -> Dict[str, any]:
    """
    변수에 대한 종합적인 분석 (시간 기반)
    """
    analysis_result = {
        'variable_name': variable_name,
        'variable_preset': recommend_variable_preset(variable_name),
        'display_unit': get_display_unit(variable_name),
        'overall_statistics': {},
        'common_features': {},
        'scone_phase_statistics': {},
        'scone_phase_state_column': state_column_name,
        'symmetry_pair_variable': '',
        'symmetry_features': {},
        'data_quality': {},
        'analysis_notes': []
    }
    
    try:
        analysis_data_arr = convert_values_for_display(variable_name, data_arr)
        analysis_result['common_features'] = compute_common_features(
            analysis_data_arr,
            time_arr,
            threshold=threshold,
        )

        analysis_result['overall_statistics'] = calculate_overall_statistics(
            analysis_data_arr, variable_name
        )
        
        analysis_result['data_quality'] = evaluate_data_quality(analysis_data_arr)

        if state_arr is not None:
            try:
                scone_phase_stats = analyze_data_by_scone_phases(
                    analysis_data_arr,
                    state_arr,
                    time_arr,
                    ddof=STATISTICS_CONFIG.get('ddof', 0)
                )
                analysis_result['scone_phase_statistics'] = scone_phase_stats
                if scone_phase_stats:
                    analysis_result['analysis_notes'].append(
                        f"SCONE state 기반 phase 분석 완료: {len(scone_phase_stats)}개 phase"
                    )
                else:
                    analysis_result['analysis_notes'].append(
                        "SCONE state 기반 phase 분석 실패: 유효한 state 데이터가 없습니다."
                    )
            except Exception as e:
                analysis_result['analysis_notes'].append(
                    f"SCONE state 기반 phase 분석 오류: {str(e)}"
                )

        quality_score = assess_analysis_quality(analysis_result)
        analysis_result['quality_score'] = quality_score
        
        return analysis_result
        
    except Exception as e:
        analysis_result['error'] = str(e)
        analysis_result['analysis_notes'].append(f"분석 중 오류 발생: {str(e)}")
        return analysis_result

def evaluate_data_quality(data_arr: np.ndarray) -> Dict[str, any]:
    """
    데이터 품질 평가
    """
    try:
        data_clean = validate_data(data_arr)
        total_count = len(data_arr)
        valid_count = len(data_clean)
        
        quality_metrics = {
            'total_points': total_count,
            'valid_points': valid_count,
            'completeness': (valid_count / total_count * 100) if total_count > 0 else 0,
            'variability': np.std(data_clean) / np.mean(data_clean) if len(data_clean) > 0 and np.mean(data_clean) != 0 else np.nan,
            'range_ratio': (np.max(data_clean) - np.min(data_clean)) / np.mean(data_clean) if len(data_clean) > 0 and np.mean(data_clean) != 0 else np.nan
        }
        
        if quality_metrics['completeness'] >= 95:
            quality_metrics['grade'] = 'Excellent'
        elif quality_metrics['completeness'] >= 80:
            quality_metrics['grade'] = 'Good'
        elif quality_metrics['completeness'] >= 60:
            quality_metrics['grade'] = 'Fair'
        else:
            quality_metrics['grade'] = 'Poor'
        
        return quality_metrics
        
    except Exception as e:
        return {
            'error': str(e),
            'grade': 'Unknown',
            'total_points': len(data_arr),
            'valid_points': 0,
            'completeness': 0
        }

def assess_analysis_quality(analysis_result: Dict) -> float:
    """
    분석 품질 점수 계산 (0-100)
    """
    try:
        score = 0.0
        max_score = 100.0
        
        data_quality = analysis_result.get('data_quality', {})
        completeness = data_quality.get('completeness', 0)
        score += min(40, completeness * 0.4)
        
        overall_stats = analysis_result.get('overall_statistics', {})
        if overall_stats.get('count', 0) > 10:
            score += 15
        if not np.isnan(overall_stats.get('mean', np.nan)):
            score += 15
        
        scone_phase_stats = analysis_result.get('scone_phase_statistics', {})
        if scone_phase_stats:
            valid_phases = sum(1 for phase in scone_phase_stats.values()
                             if phase.get('count', 0) > 0)
            score += min(30, valid_phases * 5)
        elif analysis_result.get('common_features'):
            score += 10
        
        return min(max_score, score)
        
    except Exception:
        return 0.0
