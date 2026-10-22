"""File loading helpers for motion analysis data."""

import os
import re
from typing import List, Tuple

import pandas as pd

from config.constants import FILE_CONFIG, HEADER_DEFAULT_ROW
from utils.data_validator import DataValidator
from utils.unit_utils import convert_values_from_display_to_source, requires_degree_scale_conversion


EXCEL_EXTENSIONS = {'.xlsx', '.xls'}
STO_EXTENSIONS = {'.sto'}
TEXT_EXTENSIONS = {'.txt'}
STO_SEPARATOR = r'\s*,\s*|\t+|\s+'


def _build_delimited_text_separator(include_whitespace: bool = False) -> str:
    separators = list(FILE_CONFIG.get('csv_separators', [',', '\t']))
    if include_whitespace and ' ' not in separators:
        separators.append(' ')

    patterns = []
    for separator in separators:
        if separator == '\t':
            patterns.append(r'\t+')
        elif separator == ' ':
            patterns.append(r'\s+')
        else:
            patterns.append(rf'\s*{re.escape(separator)}\s*')
    return '|'.join(patterns) or r'\s*,\s*|\t+'


DELIMITED_TEXT_SEPARATOR = _build_delimited_text_separator()
WHITESPACE_TEXT_SEPARATOR = _build_delimited_text_separator(include_whitespace=True)


def _split_sto_line(line: str) -> List[str]:
    return [part for part in re.split(STO_SEPARATOR, line.strip()) if part]


def _find_sto_header_row(lines: List[str], fallback_header_row: int) -> int:
    for index, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            for candidate in range(index + 1, len(lines)):
                if lines[candidate].strip():
                    return candidate
            break

    return fallback_header_row


def _sto_header_uses_degrees(lines: List[str], header_row: int) -> bool:
    header_lines = lines[:max(header_row, 0)]
    for line in header_lines:
        normalized = line.strip().lower().replace(' ', '')
        if not normalized.startswith('indegrees'):
            continue
        parts = re.split(r'[:=]', normalized, maxsplit=1)
        if len(parts) != 2:
            continue
        return parts[1] in {'yes', 'true', '1'}
    return False


def _normalize_in_degrees_sto_columns(df: pd.DataFrame) -> List[str]:
    converted_columns = []
    for column in df.columns:
        column_name = str(column)
        if not requires_degree_scale_conversion(column_name):
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        df[column] = convert_values_from_display_to_source(column_name, df[column].to_numpy(dtype=float))
        converted_columns.append(column_name)
    return converted_columns


def _read_sto_as_csv_dataframe(
    path: str,
    validator: DataValidator,
    default_header_row: int,
) -> Tuple[pd.DataFrame, int]:
    """Convert an STO text table to a DataFrame."""
    encoding = validator.get_file_encoding(path)
    suggested_header = validator.suggest_header_row(path)
    fallback_header_row = suggested_header if suggested_header is not None else default_header_row

    with open(path, 'r', encoding=encoding) as file:
        lines = file.readlines()

    header_row = _find_sto_header_row(lines, fallback_header_row)
    if header_row < 0 or header_row >= len(lines):
        raise ValueError("STO header row could not be found.")

    columns = _split_sto_line(lines[header_row])
    if not columns:
        raise ValueError("STO header row is empty.")

    rows = []
    for line in lines[header_row + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        values = _split_sto_line(stripped)
        if not values:
            continue

        if len(values) < len(columns):
            values = values + [''] * (len(columns) - len(values))
        rows.append(values[:len(columns)])

    df = pd.DataFrame(rows, columns=columns)
    for column in df.columns:
        numeric_values = pd.to_numeric(df[column], errors='coerce')
        if numeric_values.notna().sum() > 0:
            df[column] = numeric_values

    if _sto_header_uses_degrees(lines, header_row):
        converted_columns = _normalize_in_degrees_sto_columns(df)
        df.attrs['scone_sto_in_degrees'] = True
        df.attrs['degree_to_radian_columns'] = converted_columns

    return df, header_row


def _separator_for_extension(ext: str) -> str:
    if ext in TEXT_EXTENSIONS:
        return WHITESPACE_TEXT_SEPARATOR
    return DELIMITED_TEXT_SEPARATOR


def read_motion_data_file(
    path: str,
    validator: DataValidator,
    default_header_row: int = HEADER_DEFAULT_ROW - 1,
) -> Tuple[pd.DataFrame, int]:
    """Read a supported motion data file and return the dataframe plus header row."""
    ext = os.path.splitext(path)[1].lower()

    if ext in STO_EXTENSIONS:
        return _read_sto_as_csv_dataframe(path, validator, default_header_row)

    suggested_header = validator.suggest_header_row(path)
    header_row = suggested_header if suggested_header is not None else default_header_row

    if ext in EXCEL_EXTENSIONS:
        return pd.read_excel(path, header=header_row), header_row

    encoding = validator.get_file_encoding(path)
    return (
        pd.read_csv(path, header=header_row, sep=_separator_for_extension(ext), engine='python', encoding=encoding),
        header_row,
    )
