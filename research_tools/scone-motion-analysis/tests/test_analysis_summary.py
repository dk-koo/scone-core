import numpy as np
import pandas as pd

from src.analysis_engine import analyze_variable_comprehensive
from src.analysis_summary import (
    build_analysis_summary,
    build_condition_comparison_table,
    save_condition_comparison_table,
)


def make_analysis_results():
    state = np.array([0, 1, 2, 3, 4] * 4, dtype=float)
    time = np.linspace(0.0, 6.0, len(state))
    first = np.linspace(0.0, 1.0, len(state))
    second = np.linspace(0.0, 2.0, len(state))
    return {
        "0_trial_W70.sto_knee_angle_r": analyze_variable_comprehensive(
            first,
            time,
            "knee_angle_r",
            state_arr=state,
            state_column_name="leg1_r.state",
        ),
        "1_trial_W80.sto_knee_angle_r": analyze_variable_comprehensive(
            second,
            time,
            "knee_angle_r",
            state_arr=state,
            state_column_name="leg1_r.state",
        ),
    }


def test_build_analysis_summary_includes_quality_phase_and_comparison():
    file_paths = ["trial_W70.sto", "trial_W80.sto"]
    dataframes = [
        pd.DataFrame({"knee_angle_r": [0.0, 1.0]}),
        pd.DataFrame({"knee_angle_r": [0.0, 2.0]}),
    ]

    summary = build_analysis_summary(
        ["knee_angle_r"],
        file_paths,
        make_analysis_results(),
        dataframes=dataframes,
        analysis_purpose="보행 각도 분석",
    )

    assert "=== 자동 요약 ===" in summary
    assert "분석 목적: 보행 각도 분석" in summary
    assert "SCONE phase" in summary
    assert "조건 비교 하이라이트" in summary
    assert "W80" in summary


def test_condition_comparison_table_and_save(tmp_path):
    file_paths = ["trial_W70.sto", "trial_W80.sto"]
    table = build_condition_comparison_table(file_paths, make_analysis_results())

    assert not table.empty
    assert {"reference_condition", "condition", "variable", "metric", "delta_percent"} <= set(table.columns)
    assert "knee_angle_r" in set(table["variable"])

    saved_path = save_condition_comparison_table(file_paths, make_analysis_results(), str(tmp_path))

    assert saved_path.endswith("condition_comparison_summary.csv")
    assert (tmp_path / "condition_comparison_summary.csv").exists()


def test_condition_comparison_uses_absolute_reference_for_delta_percent():
    time = np.arange(10, dtype=float)
    analysis_results = {
        "0_ref.sto_value": analyze_variable_comprehensive(np.full(10, -2.0), time, "value"),
        "1_condition.sto_value": analyze_variable_comprehensive(np.full(10, -1.0), time, "value"),
    }

    table = build_condition_comparison_table(["ref.sto", "condition.sto"], analysis_results)
    mean_row = table[table["metric"] == "mean"].iloc[0]

    assert mean_row["delta"] == 1.0
    assert mean_row["delta_percent"] == 50.0
