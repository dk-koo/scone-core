"""
개선된 상수 정의 모듈
- 검증 기능 추가
- 더 명확한 설정값들
- 사용자 설정 가능한 옵션들
"""

import os
from typing import Dict, List, Any


# 기본 설정
HEADER_DEFAULT_ROW = 7
RESAMPLE_POINTS = 101
SD_RANGE = 2

# 보행 분석 설정
STATE_COLS = {
    'right': 'leg1_r.state', 
    'left': 'leg0_l.state'
}

SCONE_PHASE_STATES = {
    "EarlyStance": (0,),
    "LateStance": (1,),
    "LiftOff": (2,),
    "Swing": (3,),
    "Landing": (4,),
    "stance": (0, 1, 2),
    "swing": (3, 4),
}

SCONE_STATE_PHASE_LABELS = {
    0: "EarlyStance",
    1: "LateStance",
    2: "LiftOff",
    3: "Swing",
    4: "Landing",
}

# 데이터 검증 설정
VALIDATION_CONFIG = {
    'min_data_points': 10,
    'max_file_size_mb': 100,
    'min_cycle_length': 10,
    'max_cycle_length': 1000,
    'min_cycles_required': 2,
    'enforce_min_time_coverage': True,
    'min_time_coverage_seconds': 5.0,
    'max_null_percentage': 90.0,
    'outlier_detection_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_factor': 1.5
}

# 통계 계산 설정
STATISTICS_CONFIG = {
    'ddof': 0,  # 자유도 (0=모집단, 1=표본)
    'remove_outliers': False,
    'confidence_level': 0.95,
    'percentiles': [25, 50, 75]
}

# 시각화 설정
PLOT_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'default',
    'color_palette': 'Set1',
    'alpha_fill': 0.2,
    'alpha_individual': 0.1,
    'line_width': 1.5,
    'grid_alpha': 0.3
}

# 파일 I/O 설정
FILE_CONFIG = {
    'supported_extensions': ['.csv', '.sto', '.txt', '.xlsx', '.xls'],
    'default_encoding': 'utf-8',
    'fallback_encodings': ['utf-8-sig', 'euc-kr', 'cp949', 'latin1'],
    'csv_separators': [',', '\t', ';', '|'],
    'max_header_search_rows': 20
}

# GUI 설정
GUI_CONFIG = {
    'window_size': (1200, 700),  # 높이 줄임
    'font_family': 'Malgun Gothic',
    'font_size': 9,  # 폰트 크기 줄임
    'theme': 'clam',
    'search_delay_ms': 300,
    'progress_update_interval': 100
}

# 에러 메시지
ERROR_MESSAGES = {
    'file_not_found': "파일을 찾을 수 없습니다.",
    'file_too_large': "파일이 너무 큽니다.",
    'invalid_format': "지원하지 않는 파일 형식입니다.",
    'no_numeric_data': "분석 가능한 숫자 데이터가 없습니다.",
    'insufficient_cycles': "충분한 보행 주기를 찾을 수 없습니다.",
    'invalid_state_data': "유효하지 않은 상태 데이터입니다.",
    'memory_error': "메모리가 부족합니다.",
    'calculation_error': "계산 중 오류가 발생했습니다."
}

# 성공 메시지
SUCCESS_MESSAGES = {
    'file_loaded': "파일이 성공적으로 로드되었습니다.",
    'analysis_complete': "분석이 완료되었습니다.",
    'file_saved': "파일이 저장되었습니다.",
    'plot_saved': "그래프가 저장되었습니다."
}


def validate_config() -> Dict[str, List[str]]:
    """
    설정값 검증
    
    Returns:
        dict: 검증 결과 {'warnings': [], 'errors': []}
    """
    warnings = []
    errors = []
    
    # 기본 설정 검증
    if HEADER_DEFAULT_ROW < 0:
        errors.append("HEADER_DEFAULT_ROW는 0 이상이어야 합니다.")
    
    if RESAMPLE_POINTS <= 0:
        errors.append("RESAMPLE_POINTS는 양수여야 합니다.")
    elif RESAMPLE_POINTS < 50:
        warnings.append("RESAMPLE_POINTS가 50 미만입니다. 정확도가 떨어질 수 있습니다.")
    
    if SD_RANGE <= 0:
        errors.append("SD_RANGE는 양수여야 합니다.")
    
    # 검증 설정 검증
    val_config = VALIDATION_CONFIG
    if val_config['min_data_points'] <= 0:
        errors.append("min_data_points는 양수여야 합니다.")
    
    if val_config['max_file_size_mb'] <= 0:
        errors.append("max_file_size_mb는 양수여야 합니다.")
    
    if val_config['min_cycle_length'] >= val_config['max_cycle_length']:
        errors.append("min_cycle_length는 max_cycle_length보다 작아야 합니다.")
    
    if not 0 <= val_config['max_null_percentage'] <= 100:
        errors.append("max_null_percentage는 0-100 사이여야 합니다.")
    
    # 통계 설정 검증
    stats_config = STATISTICS_CONFIG
    if stats_config['ddof'] not in [0, 1]:
        warnings.append("ddof는 일반적으로 0 또는 1입니다.")
    
    if not 0 < stats_config['confidence_level'] < 1:
        errors.append("confidence_level은 0과 1 사이여야 합니다.")
    
    # 파일 확장자 검증
    for ext in FILE_CONFIG['supported_extensions']:
        if not ext.startswith('.'):
            errors.append(f"파일 확장자는 '.'으로 시작해야 합니다: {ext}")
    
    return {'warnings': warnings, 'errors': errors}


def get_config_summary() -> Dict[str, Any]:
    """설정 요약 정보 반환"""
    return {
        'basic': {
            'header_row': HEADER_DEFAULT_ROW,
            'resample_points': RESAMPLE_POINTS,
            'sd_range': SD_RANGE
        },
        'validation': VALIDATION_CONFIG,
        'statistics': STATISTICS_CONFIG,
        'visualization': PLOT_CONFIG,
        'file_io': FILE_CONFIG,
        'gui': GUI_CONFIG
    }


def update_config(section: str, key: str, value: Any) -> bool:
    """
    설정값 동적 업데이트
    
    Args:
        section: 설정 섹션 이름
        key: 설정 키
        value: 새 값
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        config_map = {
            'validation': VALIDATION_CONFIG,
            'statistics': STATISTICS_CONFIG,
            'plot': PLOT_CONFIG,
            'file': FILE_CONFIG,
            'gui': GUI_CONFIG
        }
        
        if section in config_map and key in config_map[section]:
            config_map[section][key] = value
            return True
        return False
        
    except Exception:
        return False


# 설정 검증 실행
_validation_result = validate_config()
if _validation_result['errors']:
    print("설정 오류 발견:")
    for error in _validation_result['errors']:
        print(f"  - {error}")

if _validation_result['warnings']:
    print("설정 경고:")
    for warning in _validation_result['warnings']:
        print(f"  - {warning}")
