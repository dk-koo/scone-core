"""
개선된 수학 관련 유틸리티 함수들
- 더 안전한 데이터 검증
- 일관된 통계 계산
- 이상값 처리 기능 추가
"""

import numpy as np
import warnings


def validate_data(arr, remove_outliers=False, outlier_method='iqr', outlier_factor=1.5):
    """
    데이터 검증 및 정리
    
    Args:
        arr: 입력 배열
        remove_outliers: 이상값 제거 여부
        outlier_method: 이상값 검출 방법 ('iqr', 'zscore')
        outlier_factor: 이상값 판정 계수
    
    Returns:
        정리된 배열
    """
    if len(arr) == 0:
        return np.array([])
    
    # numpy 배열로 변환
    arr = np.asarray(arr, dtype=float)
    
    # 무한값과 NaN 제거
    finite_mask = np.isfinite(arr)
    arr_clean = arr[finite_mask]
    
    if len(arr_clean) == 0:
        return np.array([])
    
    # 이상값 제거 (옵션)
    if remove_outliers and len(arr_clean) > 4:  # 최소 4개 이상의 데이터가 있을 때만
        if outlier_method == 'iqr':
            q1, q3 = np.percentile(arr_clean, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - outlier_factor * iqr
            upper_bound = q3 + outlier_factor * iqr
            outlier_mask = (arr_clean >= lower_bound) & (arr_clean <= upper_bound)
            arr_clean = arr_clean[outlier_mask]
        elif outlier_method == 'zscore':
            std = np.std(arr_clean)
            if np.isclose(std, 0.0):
                return arr_clean
            z_scores = np.abs((arr_clean - np.mean(arr_clean)) / std)
            outlier_mask = z_scores < outlier_factor
            arr_clean = arr_clean[outlier_mask]
    
    return arr_clean


def safe_mean(arr, ddof=0, remove_outliers=False):
    """
    안전한 평균 계산
    
    Args:
        arr: 입력 배열
        ddof: 자유도 (미사용, 호환성용)
        remove_outliers: 이상값 제거 여부
    
    Returns:
        평균값 또는 NaN
    """
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    return np.mean(clean_arr) if len(clean_arr) > 0 else np.nan


def safe_std(arr, ddof=0, remove_outliers=False):
    """
    안전한 표준편차 계산 (일관된 ddof 사용)
    
    Args:
        arr: 입력 배열
        ddof: 자유도 (0=모집단, 1=표본)
        remove_outliers: 이상값 제거 여부
    
    Returns:
        표준편차 또는 NaN
    """
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    return np.std(clean_arr, ddof=ddof) if len(clean_arr) > 1 else np.nan


def safe_max(arr, remove_outliers=False):
    """안전한 최댓값 계산"""
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    return np.max(clean_arr) if len(clean_arr) > 0 else np.nan


def safe_min(arr, remove_outliers=False):
    """안전한 최솟값 계산"""
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    return np.min(clean_arr) if len(clean_arr) > 0 else np.nan


def safe_median(arr, remove_outliers=False):
    """안전한 중앙값 계산"""
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    return np.median(clean_arr) if len(clean_arr) > 0 else np.nan


def safe_percentile(arr, percentiles, remove_outliers=False):
    """
    안전한 백분위수 계산
    
    Args:
        arr: 입력 배열
        percentiles: 백분위수 리스트 [25, 50, 75]
        remove_outliers: 이상값 제거 여부
    
    Returns:
        백분위수 값들의 배열
    """
    clean_arr = validate_data(arr, remove_outliers=remove_outliers)
    if len(clean_arr) > 0:
        return np.percentile(clean_arr, percentiles)
    else:
        return np.full(len(percentiles), np.nan)


def get_data_quality_info(arr):
    """
    데이터 품질 정보 반환
    
    Returns:
        dict: 데이터 품질 정보
    """
    if len(arr) == 0:
        return {
            'total_count': 0,
            'valid_count': 0,
            'nan_count': 0,
            'inf_count': 0,
            'finite_percentage': 0.0
        }
    
    arr = np.asarray(arr)
    total_count = len(arr)
    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))
    valid_count = total_count - nan_count - inf_count
    
    return {
        'total_count': total_count,
        'valid_count': valid_count,
        'nan_count': nan_count,
        'inf_count': inf_count,
        'finite_percentage': (valid_count / total_count * 100) if total_count > 0 else 0.0
    }


def calculate_comprehensive_stats(arr, ddof=0, remove_outliers=False, include_percentiles=True):
    """
    포괄적인 통계 계산
    
    Returns:
        dict: 모든 통계값들
    """
    quality_info = get_data_quality_info(arr)
    clean_count = len(validate_data(arr, remove_outliers=remove_outliers))
    
    if quality_info['valid_count'] == 0:
        base_stats = {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'count': 0
        }
    else:
        base_stats = {
            'mean': safe_mean(arr, remove_outliers=remove_outliers),
            'std': safe_std(arr, ddof=ddof, remove_outliers=remove_outliers),
            'min': safe_min(arr, remove_outliers=remove_outliers),
            'max': safe_max(arr, remove_outliers=remove_outliers),
            'median': safe_median(arr, remove_outliers=remove_outliers),
            'count': clean_count
        }
    
    result = {**base_stats, **quality_info}
    
    if include_percentiles and clean_count > 0:
        percentiles = safe_percentile(arr, [25, 75], remove_outliers=remove_outliers)
        result.update({
            'q25': percentiles[0],
            'q75': percentiles[1],
            'iqr': percentiles[1] - percentiles[0] if not np.isnan(percentiles).any() else np.nan
        })
    
    return result
