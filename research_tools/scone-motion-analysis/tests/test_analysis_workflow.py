import numpy as np
import pandas as pd

from config.language import get_language
from src.analysis_workflow import run_analysis_workflow
from utils.data_validator import DataValidator


def test_run_analysis_workflow_returns_results_without_tkinter():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 30),
        "knee_angle_r": np.linspace(0.0, 1.0, 30),
        "knee_angle_l": np.linspace(1.0, 0.0, 30),
    })

    results_text, analysis_results = run_analysis_workflow(
        [df],
        ["sample.csv"],
        ["knee_angle_r"],
        0.5,
        get_language("ko"),
        DataValidator(),
        check_time_coverage=lambda _df: True,
        get_state_column_for_variable=lambda _df, _var: None,
    )

    result = analysis_results["0_sample.csv_knee_angle_r"]
    assert "변수: knee_angle_r" in results_text
    assert result["overall_statistics"]["count"] == 30
    assert result["symmetry_pair_variable"] == "knee_angle_l"
    assert result["symmetry_features"]


def test_run_analysis_workflow_records_time_coverage_skip():
    df = pd.DataFrame({
        "time": [0.0, 1.0, 2.0],
        "value": [1.0, 2.0, 3.0],
    })

    results_text, analysis_results = run_analysis_workflow(
        [df],
        ["short.csv"],
        ["value"],
        0.5,
        get_language("ko"),
        DataValidator(),
        check_time_coverage=lambda _df: False,
        get_state_column_for_variable=lambda _df, _var: None,
    )

    result = analysis_results["0_short.csv_value"]
    assert "분석 제외" in results_text
    assert result["time_coverage_failed"] is True


def test_run_analysis_workflow_can_skip_missing_variables_without_error_label():
    first = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 10),
        "only_first": np.linspace(0.0, 1.0, 10),
    })
    second = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 10),
        "only_second": np.linspace(1.0, 2.0, 10),
    })

    results_text, analysis_results = run_analysis_workflow(
        [first, second],
        ["first.csv", "second.csv"],
        ["only_first", "only_second"],
        0.5,
        get_language("ko"),
        DataValidator(),
        check_time_coverage=lambda _df: True,
        get_state_column_for_variable=lambda _df, _var: None,
        skip_missing_variables=True,
    )

    assert "건너뜀: 해당 파일에 변수가 없습니다." in results_text
    assert "0_first.csv_only_first" in analysis_results
    assert "1_second.csv_only_second" in analysis_results
