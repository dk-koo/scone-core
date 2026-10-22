import numpy as np

from src.analysis_engine import analyze_variable_comprehensive, compute_symmetry_features
from src.result_export import build_feature_csv_tables, build_results_dataframe, save_feature_csv_bundle


def test_build_results_dataframe_includes_phase_rows_and_sanitizes_text():
    data = np.arange(10, dtype=float)
    time = np.arange(10, dtype=float)
    state = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    result = analyze_variable_comprehensive(
        data,
        time,
        "=unsafe",
        state_arr=state,
        state_column_name="leg1_r.state",
    )
    analysis_results = {"0_sample.csv_=unsafe": result}

    export_df = build_results_dataframe(["=unsafe"], ["sample.csv"], analysis_results)

    assert export_df.iloc[0]["변수명"] == "'=unsafe"
    assert "SCONE phase - EarlyStance" in set(export_df["분석유형"])
    assert "SCONE phase - stance" in set(export_df["분석유형"])
    assert "데이터수" in export_df.columns
    assert "데이터품질" not in export_df.columns
    assert "완성도" not in export_df.columns
    assert "품질점수" not in export_df.columns


def test_build_results_dataframe_exports_symmetry_without_duplicate_or_misleading_columns():
    time = np.arange(5, dtype=float)
    left_data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    right_data = np.array([2.0, 4.0, 6.0, 4.0, 2.0])
    left_result = analyze_variable_comprehensive(left_data, time, "knee_angle_l")
    right_result = analyze_variable_comprehensive(right_data, time, "knee_angle_r")
    left_result["symmetry_pair_variable"] = "knee_angle_r"
    left_result["symmetry_features"] = compute_symmetry_features(
        "knee_angle_l",
        "knee_angle_r",
        left_result["common_features"],
        right_result["common_features"],
    )
    right_result["symmetry_pair_variable"] = "knee_angle_l"
    right_result["symmetry_features"] = compute_symmetry_features(
        "knee_angle_r",
        "knee_angle_l",
        right_result["common_features"],
        left_result["common_features"],
    )
    analysis_results = {
        "0_trial.sto_knee_angle_l": left_result,
        "0_trial.sto_knee_angle_r": right_result,
    }

    export_df = build_results_dataframe(["knee_angle_l", "knee_angle_r"], ["trial.sto"], analysis_results)
    symmetry_rows = export_df[export_df["분석유형"] == "좌우 대칭"]

    assert len(symmetry_rows) == 8
    assert set(symmetry_rows["대칭 Feature"]) == {
        "mean",
        "max",
        "min",
        "rom",
        "rms",
        "auc",
        "positive_auc",
        "negative_auc",
    }
    max_row = symmetry_rows[symmetry_rows["대칭 Feature"] == "max"].iloc[0]
    assert max_row["최댓값"] == ""
    assert max_row["최솟값"] == ""
    assert max_row["오른쪽 값"] == np.rad2deg(6.0)
    assert max_row["왼쪽 값"] == np.rad2deg(3.0)


def test_build_feature_csv_tables_creates_standard_plot_inputs(tmp_path):
    data = np.arange(10, dtype=float)
    time = np.arange(10, dtype=float)
    state = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    result = analyze_variable_comprehensive(
        data,
        time,
        "knee_angle_r",
        state_arr=state,
        state_column_name="leg1_r.state",
    )
    analysis_results = {"0_trial_W70.sto_knee_angle_r": result}

    tables = build_feature_csv_tables(["knee_angle_r"], ["trial_W70.sto"], analysis_results)

    assert {"cycle_features.csv", "phase_features.csv", "comparison_features.csv"} == set(tables)
    assert {"condition", "variable", "phase", "feature", "value"} <= set(tables["cycle_features.csv"].columns)
    assert "max" in set(tables["cycle_features.csv"]["feature"])
    assert "EarlyStance" in set(tables["phase_features.csv"]["phase"])
    assert "unit" in tables["cycle_features.csv"].columns
    cycle_units = dict(zip(tables["cycle_features.csv"]["feature"], tables["cycle_features.csv"]["unit"]))
    phase_units = dict(zip(tables["phase_features.csv"]["feature"], tables["phase_features.csv"]["unit"]))
    assert cycle_units["max"] == "deg"
    assert cycle_units["auc"] == "deg*s"
    assert cycle_units["max_time"] == "s"
    assert cycle_units["max_percent"] == "%"
    assert cycle_units["threshold_exceed_count"] == "count"
    assert phase_units["max"] == "deg"
    assert phase_units["auc"] == "deg*s"
    assert phase_units["count"] == "count"

    saved_paths = save_feature_csv_bundle(["knee_angle_r"], ["trial_W70.sto"], analysis_results, str(tmp_path))

    assert str(tmp_path / "cycle_features.csv") in saved_paths
    assert str(tmp_path / "phase_features.csv") in saved_paths
    assert str(tmp_path / "comparison_features.csv") in saved_paths
    assert (tmp_path / "cycle_features.csv").exists()
    assert (tmp_path / "phase_features.csv").exists()
    assert (tmp_path / "comparison_features.csv").exists()


def test_build_feature_csv_tables_uses_integrated_units_for_angular_velocity():
    time = np.arange(3, dtype=float)
    result = analyze_variable_comprehensive(np.array([0.0, 1.0, 2.0]), time, "knee_angle_r_u")
    analysis_results = {"0_trial.sto_knee_angle_r_u": result}

    tables = build_feature_csv_tables(["knee_angle_r_u"], ["trial.sto"], analysis_results)
    cycle_units = dict(zip(tables["cycle_features.csv"]["feature"], tables["cycle_features.csv"]["unit"]))

    assert cycle_units["max"] == "deg/s"
    assert cycle_units["auc"] == "deg"


def test_build_feature_csv_tables_uses_absolute_reference_for_delta_percent():
    time = np.arange(10, dtype=float)
    reference = np.full(10, -2.0)
    condition = np.full(10, -1.0)
    analysis_results = {
        "0_ref.sto_value": analyze_variable_comprehensive(reference, time, "value"),
        "1_condition.sto_value": analyze_variable_comprehensive(condition, time, "value"),
    }

    tables = build_feature_csv_tables(["value"], ["ref.sto", "condition.sto"], analysis_results)
    comparison = tables["comparison_features.csv"]
    mean_row = comparison[comparison["feature"] == "mean"].iloc[0]

    assert mean_row["delta"] == 1.0
    assert mean_row["delta_percent"] == 50.0
    assert np.isnan(mean_row["delta_timing"])


def test_build_results_dataframe_includes_display_unit_and_degree_values():
    time = np.arange(3, dtype=float)
    result = analyze_variable_comprehensive(np.array([0.0, np.pi / 2.0, np.pi]), time, "knee_angle_r")
    analysis_results = {"0_trial.sto_knee_angle_r": result}

    export_df = build_results_dataframe(["knee_angle_r"], ["trial.sto"], analysis_results)
    overall_row = export_df[export_df["분석유형"] == "전체 데이터"].iloc[0]

    assert overall_row["단위"] == "deg"
    assert overall_row["최댓값"] == 180.0
