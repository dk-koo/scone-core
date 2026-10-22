"""
math_utils 모듈 테스트
"""

import pytest
import numpy as np
import sys
import os

# 상위 디렉토리의 모듈을 import하기 위해 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import (
    safe_mean, safe_std, safe_max, safe_min, safe_median,
    validate_data, calculate_comprehensive_stats
)


class TestSafeMathFunctions:
    """안전한 수학 함수들 테스트"""
    
    def test_safe_mean(self):
        """safe_mean 함수 테스트"""
        # 정상 케이스
        assert abs(safe_mean([1, 2, 3, 4, 5]) - 3.0) < 1e-10
        
        # NaN 포함
        result = safe_mean([1, 2, np.nan, 4, 5])
        assert abs(result - 3.0) < 1e-10
        
        # 모든 값이 NaN
        assert np.isnan(safe_mean([np.nan, np.nan]))
        
        # 빈 배열
        assert np.isnan(safe_mean([]))
        
        # 단일 값
        assert safe_mean([42]) == 42
    
    def test_safe_std(self):
        """safe_std 함수 테스트"""
        # 정상 케이스 (모집단)
        data = [1, 2, 3, 4, 5]
        expected = np.std(data, ddof=0)
        assert abs(safe_std(data, ddof=0) - expected) < 1e-10
        
        # 표본 표준편차
        expected_sample = np.std(data, ddof=1)
        assert abs(safe_std(data, ddof=1) - expected_sample) < 1e-10
        
        # NaN 포함
        result = safe_std([1, 2, np.nan, 4, 5])
        assert not np.isnan(result)
        
        # 단일 값 (표준편차는 NaN이어야 함)
        assert np.isnan(safe_std([42]))
        
        # 빈 배열
        assert np.isnan(safe_std([]))
    
    def test_safe_max_min(self):
        """safe_max, safe_min 함수 테스트"""
        data = [1, 2, 3, 4, 5]
        
        assert safe_max(data) == 5
        assert safe_min(data) == 1
        
        # NaN 포함
        assert safe_max([1, np.nan, 3]) == 3
        assert safe_min([1, np.nan, 3]) == 1
        
        # 빈 배열
        assert np.isnan(safe_max([]))
        assert np.isnan(safe_min([]))
    
    def test_safe_median(self):
        """safe_median 함수 테스트"""
        # 홀수 개
        assert safe_median([1, 2, 3, 4, 5]) == 3
        
        # 짝수 개
        assert safe_median([1, 2, 3, 4]) == 2.5
        
        # NaN 포함
        assert safe_median([1, np.nan, 3]) == 2.0
        
        # 빈 배열
        assert np.isnan(safe_median([]))


class TestDataValidation:
    """데이터 검증 함수 테스트"""
    
    def test_validate_data_normal(self):
        """정상 데이터 검증"""
        data = [1, 2, 3, 4, 5]
        result = validate_data(data)
        np.testing.assert_array_equal(result, data)
    
    def test_validate_data_with_nan(self):
        """NaN 포함 데이터 검증"""
        data = [1, 2, np.nan, 4, 5]
        result = validate_data(data)
        expected = [1, 2, 4, 5]
        np.testing.assert_array_equal(result, expected)
    
    def test_validate_data_with_inf(self):
        """무한값 포함 데이터 검증"""
        data = [1, 2, np.inf, 4, -np.inf]
        result = validate_data(data)
        expected = [1, 2, 4]
        np.testing.assert_array_equal(result, expected)
    
    def test_validate_data_empty(self):
        """빈 데이터 검증"""
        result = validate_data([])
        assert len(result) == 0
    
    def test_validate_data_all_invalid(self):
        """모든 값이 유효하지 않은 데이터"""
        data = [np.nan, np.inf, -np.inf]
        result = validate_data(data)
        assert len(result) == 0

    def test_validate_data_zscore_keeps_constant_values(self):
        data = [1, 1, 1, 1, 1]

        result = validate_data(data, remove_outliers=True, outlier_method='zscore')

        np.testing.assert_array_equal(result, data)


class TestComprehensiveStats:
    """포괄적 통계 계산 테스트"""
    
    def test_comprehensive_stats_normal(self):
        """정상 데이터의 포괄적 통계"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = calculate_comprehensive_stats(data)
        
        # 기본 통계 확인
        assert result['mean'] == 5.5
        assert result['min'] == 1
        assert result['max'] == 10
        assert result['count'] == 10
        
        # 데이터 품질 확인
        assert result['total_count'] == 10
        assert result['valid_count'] == 10
        assert result['finite_percentage'] == 100.0
    
    def test_comprehensive_stats_with_nan(self):
        """NaN 포함 데이터의 포괄적 통계"""
        data = [1, 2, np.nan, 4, 5]
        result = calculate_comprehensive_stats(data)
        
        assert result['total_count'] == 5
        assert result['valid_count'] == 4
        assert result['nan_count'] == 1
        assert result['finite_percentage'] == 80.0

    def test_comprehensive_stats_count_matches_outlier_filtered_data(self):
        data = [1, 1, 1, 1, 100]
        result = calculate_comprehensive_stats(data, remove_outliers=True)

        assert result['count'] == 4
        assert result['valid_count'] == 5
        assert result['max'] == 1
    
    def test_comprehensive_stats_empty(self):
        """빈 데이터의 포괄적 통계"""
        result = calculate_comprehensive_stats([])
        
        assert result['total_count'] == 0
        assert result['valid_count'] == 0
        assert np.isnan(result['mean'])
        assert result['finite_percentage'] == 0.0


if __name__ == "__main__":
    # 간단한 테스트 실행
    test_class = TestSafeMathFunctions()
    test_class.test_safe_mean()
    test_class.test_safe_std()
    test_class.test_safe_max_min()
    test_class.test_safe_median()
    
    validation_test = TestDataValidation()
    validation_test.test_validate_data_normal()
    validation_test.test_validate_data_with_nan()
    validation_test.test_validate_data_with_inf()
    
    stats_test = TestComprehensiveStats()
    stats_test.test_comprehensive_stats_normal()
    stats_test.test_comprehensive_stats_with_nan()
    
    print("모든 테스트 통과!")
