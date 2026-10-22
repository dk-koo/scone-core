"""
파일 관련 유틸리티 함수들
"""

import os
import re
from typing import Optional

import pandas as pd


def extract_percentage_from_filename(filename: str) -> Optional[str]:
    """
    파일명에서 백분율 추출
    
    예시:
    - "0.2.csv" -> "20%"
    - "0.4.csv" -> "40%"
    - "0.6.csv" -> "60%"
    - "1.0.csv" -> "100%"
    - "normal.csv" -> "normal"
    """
    try:
        # 파일명에서 확장자 제거
        basename = os.path.splitext(filename)[0]

        # SCONE 조건명 축약: P2BW_W70_Gait10Slope -> P2BW_W70
        condition_match = re.match(r'^(.+?_[Ww]\d+(?:\.\d+)?)(?:_|$)', basename)
        if condition_match:
            return condition_match.group(1)
        
        # 소수점 숫자 패턴 찾기 (예: 0.2, 0.4, 1.0)
        decimal_pattern = r'^(\d+\.\d+)$'
        match = re.match(decimal_pattern, basename)
        
        if match:
            decimal_value = float(match.group(1))
            # 소수점을 백분율로 변환
            percentage = int(decimal_value * 100)
            return f"{percentage}%"
        
        # 정수 패턴 찾기 (예: 20, 40, 60)
        integer_pattern = r'^(\d+)$'
        match = re.match(integer_pattern, basename)
        
        if match:
            integer_value = int(match.group(1))
            # 이미 백분율이라고 가정
            if integer_value <= 100:
                return f"{integer_value}%"
        
        # 백분율이 명시적으로 포함된 경우 (예: "20%", "40%")
        percentage_pattern = r'(\d+)%'
        match = re.search(percentage_pattern, basename)
        
        if match:
            return f"{match.group(1)}%"
        
        # 패턴이 매치되지 않으면 원본 파일명 반환
        return basename
        
    except Exception:
        return os.path.splitext(filename)[0]


def format_legend_label(filename: str) -> str:
    """
    범례 라벨 포맷팅
    
    Args:
        filename: 파일명
        
    Returns:
        포맷팅된 범례 라벨
    """
    return extract_percentage_from_filename(filename) or os.path.splitext(filename)[0]


def get_clean_variable_name(var_name: str) -> str:
    """
    변수명을 그래프 제목에 적합하게 정리
    
    Args:
        var_name: 원본 변수명
        
    Returns:
        정리된 변수명 (예: 'knee_angle_r' -> 'Knee Angle_R')
    """
    # 언더스코어를 공백으로 변경
    clean_name = var_name.replace('_', ' ')
    
    # 각 단어의 첫 글자를 대문자화
    words = clean_name.split()
    formatted_words = []
    
    for word in words:
        # 마지막 글자가 단일 문자(L, R 등)인 경우 대문자로 처리
        if len(word) == 1:
            formatted_words.append(word.upper())
        else:
            # 첫 글자만 대문자화
            formatted_words.append(word.capitalize())
    
    # 언더스코어로 다시 연결
    return '_'.join(formatted_words)


def get_safe_filename_component(name: str, fallback: str = "variable") -> str:
    """Return a Windows-safe filename part while preserving readable labels."""
    if not name:
        return fallback

    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))
    safe_name = re.sub(r'_+', '_', safe_name).strip(' ._')
    return safe_name or fallback


def sanitize_dataframe_for_spreadsheet_export(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix risky text cells so spreadsheet apps do not evaluate formulas."""
    sanitized = df.copy()
    for column in sanitized.columns:
        sanitized[column] = sanitized[column].map(_sanitize_spreadsheet_cell)
    return sanitized


def _sanitize_spreadsheet_cell(value):
    if not isinstance(value, str) or not value:
        return value

    stripped = value.lstrip()
    if stripped and stripped[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value
