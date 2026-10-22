import pandas as pd

from utils.data_validator import DataValidator


def test_validate_file_path_accepts_excel_extensions(tmp_path):
    validator = DataValidator()
    xlsx_path = tmp_path / "sample.xlsx"
    xls_path = tmp_path / "sample.xls"
    xlsx_path.write_bytes(b"")
    xls_path.write_bytes(b"")

    assert validator.validate_file_path(str(xlsx_path))[0] is True
    assert validator.validate_file_path(str(xls_path))[0] is True


def test_validate_time_column_accepts_numeric_increasing_time():
    df = pd.DataFrame({"time": [0.0, 0.1, 0.2], "value": [1, 2, 3]})

    is_valid, message, info = DataValidator().validate_time_column(df)

    assert is_valid is True
    assert message == "시간 컬럼이 유효합니다."
    assert info["valid_count"] == 3
    assert info["duration"] == 0.2


def test_validate_time_column_rejects_unsorted_time():
    df = pd.DataFrame({"time": [0.0, 0.3, 0.2], "value": [1, 2, 3]})

    is_valid, message, info = DataValidator().validate_time_column(df)

    assert is_valid is False
    assert "오름차순" in message
    assert info["is_monotonic_increasing"] is False


def test_validate_time_column_rejects_duplicate_time_values():
    df = pd.DataFrame({"time": [0.0, 0.1, 0.1, 0.2], "value": [1, 2, 3, 4]})

    is_valid, message, info = DataValidator().validate_time_column(df)

    assert is_valid is False
    assert "중복" in message
    assert info["duplicate_time_count"] == 1
    assert info["is_monotonic_increasing"] is False


def test_validate_time_column_rejects_non_numeric_values():
    df = pd.DataFrame({"time": [0.0, "bad", 0.2], "value": [1, 2, 3]})

    is_valid, message, info = DataValidator().validate_time_column(df)

    assert is_valid is False
    assert "숫자로 변환" in message
    assert info["nan_count"] == 1
