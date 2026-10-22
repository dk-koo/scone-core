"""
개선된 보행 분석 관련 유틸리티 함수들
- 더 안전한 데이터 처리
- 향상된 보행 주기 검출
- 강화된 검증 로직
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from config.constants import RESAMPLE_POINTS, SCONE_STATE_PHASE_LABELS, STATE_COLS
from utils.math_utils import validate_data


def get_leg_and_state_col(var: str) -> Tuple[Optional[str], Optional[str]]:
    """
    변수명에서 다리와 state 컬럼명 반환
    
    Args:
        var: 변수명
        
    Returns:
        tuple: (다리 라벨, state 컬럼명)
    """
    var_lower = var.lower()
    
    # 더 정확한 패턴 매칭
    if '_r' in var_lower or 'right' in var_lower:
        return "오른쪽 다리", STATE_COLS['right']
    elif '_l' in var_lower or 'left' in var_lower:
        return "왼쪽 다리", STATE_COLS['left']
    
    return None, None


def get_state_name(state_value: int) -> str:
    """
    state 값에 해당하는 보행 단계명 반환
    
    Args:
        state_value: 보행 상태 값 (0-4)
        
    Returns:
        str: 보행 단계명
    """
    return SCONE_STATE_PHASE_LABELS.get(int(state_value), f'Unknown_{state_value}')


def validate_state_data(state_arr: np.ndarray) -> Tuple[bool, str, np.ndarray]:
    """
    보행 상태 데이터 검증
    
    Args:
        state_arr: 보행 상태 배열
        
    Returns:
        tuple: (유효성, 메시지, 정리된 배열)
    """
    try:
        # 기본 검증
        if len(state_arr) == 0:
            return False, "빈 상태 배열입니다.", np.array([])
        
        # numpy 배열로 변환
        state_clean = np.asarray(state_arr, dtype=float)
        
        # NaN 제거
        finite_mask = np.isfinite(state_clean)
        state_clean = state_clean[finite_mask]
        
        if len(state_clean) == 0:
            return False, "유효한 상태 값이 없습니다.", np.array([])
        
        # 정수로 변환
        state_clean = np.round(state_clean).astype(int)
        
        # 유효한 범위 확인 (0-4)
        valid_range = (state_clean >= 0) & (state_clean <= 4)
        if not valid_range.all():
            invalid_count = (~valid_range).sum()
            return False, f"유효하지 않은 상태 값 {invalid_count}개 발견 (0-4 범위 외)", state_clean
        
        # 상태 변화 확인
        unique_states = np.unique(state_clean)
        if len(unique_states) < 2:
            return False, f"상태 변화가 없습니다 (고유 상태: {unique_states})", state_clean
        
        return True, "상태 데이터가 유효합니다.", state_clean
        
    except Exception as e:
        return False, f"상태 데이터 검증 오류: {str(e)}", np.array([])


def find_state_transitions(normalized_state_cycle: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """
    정규화된 보행 주기에서 state 전환 지점들을 찾아서 각 단계의 구간 반환
    
    Args:
        normalized_state_cycle: 정규화된 상태 배열
        
    Returns:
        dict: {상태명: [(시작idx, 끝idx), ...]}
    """
    transitions = {}
    
    if len(normalized_state_cycle) == 0:
        return transitions
    
    try:
        # 상태 데이터 검증
        is_valid, msg, clean_state = validate_state_data(normalized_state_cycle)
        if not is_valid:
            return transitions
        
        current_state = clean_state[0]
        start_idx = 0
        
        for i in range(1, len(clean_state)):
            if clean_state[i] != current_state:
                # 현재 state 구간 종료
                state_name = get_state_name(current_state)
                if state_name not in transitions:
                    transitions[state_name] = []
                transitions[state_name].append((start_idx, i-1))
                
                # 새로운 state 구간 시작
                current_state = clean_state[i]
                start_idx = i
        
        # 마지막 구간 처리
        state_name = get_state_name(current_state)
        if state_name not in transitions:
            transitions[state_name] = []
        transitions[state_name].append((start_idx, len(clean_state)-1))
        
    except Exception as e:
        print(f"상태 전환 분석 오류: {str(e)}")
    
    return transitions


def find_gait_cycles(state_arr: np.ndarray, min_cycle_length: int = 10, 
                    max_cycle_length: int = 1000) -> List[int]:
    """
    개선된 보행 주기 검출
    state 배열에서 0(EarlyStance)으로 진입하는 인덱스(heel-strike)를 찾아 구간 리스트 반환
    
    Args:
        state_arr: 보행 상태 배열
        min_cycle_length: 최소 주기 길이
        max_cycle_length: 최대 주기 길이
        
    Returns:
        list: 보행 주기 시작점들
    """
    cycles = []
    
    try:
        # 상태 데이터 검증
        is_valid, msg, clean_state = validate_state_data(state_arr)
        if not is_valid:
            return cycles
        
        prev_state = clean_state[0]
        if prev_state == 0:
            cycles.append(0)
        last_cycle_idx = cycles[-1] if cycles else 0
        
        for i in range(1, len(clean_state)):
            current_state = clean_state[i]
            
            # 어떤 state에서든 EarlyStance(0)로 진입하는 전환 감지
            if prev_state != 0 and current_state == 0:
                cycle_length = i - last_cycle_idx
                
                # 주기 길이 검증
                if min_cycle_length <= cycle_length <= max_cycle_length:
                    cycles.append(i)
                    last_cycle_idx = i
                # 너무 짧거나 긴 주기는 무시하고 로그만 남김
                elif cycle_length < min_cycle_length:
                    print(f"너무 짧은 보행 주기 무시: {cycle_length} < {min_cycle_length}")
                elif cycle_length > max_cycle_length:
                    print(f"너무 긴 보행 주기 무시: {cycle_length} > {max_cycle_length}")
            
            prev_state = current_state
        
        # 최소 2개 이상의 주기가 필요
        if len(cycles) < 2:
            return []
            
    except Exception as e:
        print(f"보행 주기 검출 오류: {str(e)}")
        return []
    
    return cycles


def resample_cycle(data: np.ndarray, start_idx: int, end_idx: int, 
                  n_points: int = RESAMPLE_POINTS) -> np.ndarray:
    """
    개선된 보행 주기 리샘플링
    start_idx~end_idx 구간을 n_points(기본 101, 0~100%)로 리샘플링
    
    Args:
        data: 원본 데이터
        start_idx: 시작 인덱스
        end_idx: 끝 인덱스
        n_points: 리샘플링할 포인트 수
        
    Returns:
        np.ndarray: 리샘플링된 데이터
    """
    try:
        # 인덱스 검증
        if start_idx < 0 or end_idx >= len(data) or start_idx >= end_idx:
            return np.full(n_points, np.nan)
        
        # 구간 데이터 추출
        segment = data[start_idx:end_idx+1]
        
        # 데이터 검증
        clean_segment = validate_data(segment)
        if len(clean_segment) < 2:
            return np.full(n_points, np.nan)
        
        # 선형 보간으로 리샘플링
        original_indices = np.arange(len(segment))
        new_indices = np.linspace(0, len(segment)-1, n_points)
        
        # NaN이 포함된 경우 보간 처리
        if np.isnan(segment).any():
            # 유효한 데이터 포인트만 사용
            valid_mask = ~np.isnan(segment)
            if valid_mask.sum() < 2:
                return np.full(n_points, np.nan)
            
            resampled = np.interp(new_indices, 
                                original_indices[valid_mask], 
                                segment[valid_mask])
        else:
            resampled = np.interp(new_indices, original_indices, segment)
        
        return resampled
        
    except Exception as e:
        print(f"보행 주기 리샘플링 오류: {str(e)}")
        return np.full(n_points, np.nan)


def validate_gait_cycles(cycles: List[int], data_length: int, 
                        min_cycles: int = 2) -> Tuple[bool, str]:
    """
    보행 주기 유효성 검증
    
    Args:
        cycles: 보행 주기 시작점들
        data_length: 전체 데이터 길이
        min_cycles: 최소 필요 주기 수
        
    Returns:
        tuple: (유효성, 메시지)
    """
    if len(cycles) < min_cycles:
        return False, f"보행 주기가 부족합니다 ({len(cycles)} < {min_cycles})"
    
    # 인덱스 범위 확인
    for i, cycle_idx in enumerate(cycles):
        if cycle_idx < 0 or cycle_idx >= data_length:
            return False, f"유효하지 않은 주기 인덱스: {cycle_idx} (데이터 길이: {data_length})"
    
    # 주기 간격 확인
    intervals = np.diff(cycles)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # 주기 간격의 변동성이 너무 크면 경고
    if std_interval > mean_interval * 0.5:  # 50% 이상 변동
        return False, f"보행 주기 간격이 불규칙합니다 (평균: {mean_interval:.1f}, 표준편차: {std_interval:.1f})"
    
    return True, f"유효한 보행 주기 {len(cycles)}개"


def analyze_gait_pattern(state_arr: np.ndarray) -> Dict[str, any]:
    """
    보행 패턴 종합 분석
    
    Returns:
        dict: 보행 패턴 분석 결과
    """
    analysis = {
        'is_valid': False,
        'cycles_found': 0,
        'cycle_indices': [],
        'mean_cycle_length': 0,
        'cycle_variability': 0,
        'state_distribution': {},
        'errors': []
    }
    
    try:
        # 상태 데이터 검증
        is_valid, msg, clean_state = validate_state_data(state_arr)
        if not is_valid:
            analysis['errors'].append(msg)
            return analysis
        
        # 보행 주기 검출
        cycles = find_gait_cycles(clean_state)
        analysis['cycles_found'] = len(cycles)
        analysis['cycle_indices'] = cycles
        
        if len(cycles) >= 2:
            # 주기 길이 분석
            cycle_lengths = np.diff(cycles)
            analysis['mean_cycle_length'] = float(np.mean(cycle_lengths))
            analysis['cycle_variability'] = float(np.std(cycle_lengths) / np.mean(cycle_lengths))
            
            # 주기 유효성 검증
            is_valid_cycles, cycle_msg = validate_gait_cycles(cycles, len(clean_state))
            if is_valid_cycles:
                analysis['is_valid'] = True
            else:
                analysis['errors'].append(cycle_msg)
        else:
            analysis['errors'].append(f"충분한 보행 주기를 찾을 수 없습니다 ({len(cycles)}개)")
        
        # 상태 분포 분석
        unique_states, state_counts = np.unique(clean_state, return_counts=True)
        analysis['state_distribution'] = {
            get_state_name(state): int(count) 
            for state, count in zip(unique_states, state_counts)
        }
        
    except Exception as e:
        analysis['errors'].append(f"보행 패턴 분석 오류: {str(e)}")
    
    return analysis
