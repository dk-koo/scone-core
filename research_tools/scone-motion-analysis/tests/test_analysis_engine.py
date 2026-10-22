import numpy as np

from src.analysis_engine import (
    analyze_data_by_scone_phases,
    analyze_variable_comprehensive,
    classify_variable,
    compute_common_features,
    compute_symmetry_features,
    create_scone_phase_masks,
    find_side_pair_variable,
    recommend_variable_preset,
)


def test_create_scone_phase_masks_includes_state_and_composite_phases():
    state = np.array([0, 1, 2, 3, 4, 0])

    masks = create_scone_phase_masks(state)

    assert masks["EarlyStance"].tolist() == [True, False, False, False, False, True]
    assert masks["LateStance"].tolist() == [False, True, False, False, False, False]
    assert masks["LiftOff"].tolist() == [False, False, True, False, False, False]
    assert masks["Swing"].tolist() == [False, False, False, True, False, False]
    assert masks["Landing"].tolist() == [False, False, False, False, True, False]
    assert masks["stance"].tolist() == [True, True, True, False, False, True]
    assert masks["swing"].tolist() == [False, False, False, True, True, False]


def test_analyze_data_by_scone_phases_calculates_required_stats():
    data = np.arange(10, dtype=float)
    time = np.arange(10, dtype=float)
    state = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    result = analyze_data_by_scone_phases(data, state, time)

    assert result["EarlyStance"]["mean"] == 0.5
    assert result["EarlyStance"]["min"] == 0.0
    assert result["EarlyStance"]["max"] == 1.0
    assert result["EarlyStance"]["range"] == 1.0
    assert result["EarlyStance"]["rms"] == np.sqrt(0.5)
    assert result["EarlyStance"]["auc"] == 0.5
    assert result["stance"]["count"] == 6
    assert result["stance"]["auc"] == 12.5
    assert result["swing"]["count"] == 4
    assert result["swing"]["auc"] == 22.5


def test_analyze_variable_comprehensive_adds_scone_phase_without_time_phase():
    data = np.arange(30, dtype=float)
    time = np.linspace(0.0, 3.0, 30)
    state = np.array([0, 1, 2, 3, 4] * 6)

    result = analyze_variable_comprehensive(
        data,
        time,
        "knee_angle_r",
        state_arr=state,
        state_column_name="leg1_r.state",
    )

    assert "time_phase_statistics" not in result
    assert result["scone_phase_statistics"]["stance"]["count"] == 18
    assert result["scone_phase_state_column"] == "leg1_r.state"


def test_analyze_variable_comprehensive_converts_joint_angle_radians_to_degrees():
    data = np.array([0.0, np.pi / 2.0, np.pi])
    time = np.array([0.0, 1.0, 2.0])

    result = analyze_variable_comprehensive(data, time, "knee_angle_r")

    assert result["display_unit"] == "deg"
    assert result["common_features"]["max"] == 180.0
    assert result["overall_statistics"]["mean"] == 90.0


def test_analyze_variable_comprehensive_converts_orientation_radians_to_degrees():
    data = np.array([0.0, np.pi / 2.0, np.pi])
    time = np.array([0.0, 1.0, 2.0])

    result = analyze_variable_comprehensive(data, time, "pelvis.ori_x")

    assert result["display_unit"] == "deg"
    assert result["common_features"]["max"] == 180.0
    assert result["overall_statistics"]["mean"] == 90.0


def test_analyze_variable_comprehensive_keeps_normalized_mtu_force_dimensionless():
    data = np.array([0.1, 0.2, 0.3])

    result = analyze_variable_comprehensive(data, None, "soleus_r.mtu_force_norm")

    assert result["display_unit"] == "dimensionless"
    assert result["common_features"]["max"] == 0.3


def test_compute_common_features_calculates_peak_auc_and_threshold_features():
    data = np.array([-1.0, 0.0, 2.0, 4.0, -2.0])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    result = compute_common_features(data, time, threshold=1.0)

    assert result["max"] == 4.0
    assert result["max_time"] == 3.0
    assert result["max_percent"] == 75.0
    assert result["min"] == -2.0
    assert result["min_time"] == 4.0
    assert result["min_percent"] == 100.0
    assert result["rom"] == 6.0
    assert result["rms"] == np.sqrt(5.0)
    assert result["auc"] == 4.5
    assert result["positive_auc"] == 6.0
    assert result["negative_auc"] == -1.5
    assert result["threshold_exceed_count"] == 1
    assert result["onset_time"] == 2.0
    assert result["offset_time"] == 4.0
    assert result["active_duration"] == 2.0


def test_compute_common_features_uses_index_axis_for_duplicate_time_values():
    data = np.array([0.0, 1.0, 2.0])
    duplicated_time = np.array([0.0, 0.0, 1.0])

    result = compute_common_features(data, duplicated_time)

    assert result["axis_basis"] == "index"
    assert result["max_time"] == 2.0
    assert result["auc"] == 2.0


def test_find_side_pair_variable_detects_common_left_right_patterns():
    columns = ["knee_angle_l", "knee_angle_r", "hip_l.load", "hip_r.load", "leg0_l.grf_y", "leg1_r.grf_y"]

    assert find_side_pair_variable("knee_angle_l", columns) == "knee_angle_r"
    assert find_side_pair_variable("hip_r.load", columns) == "hip_l.load"
    assert find_side_pair_variable("leg0_l.grf_y", columns) == "leg1_r.grf_y"


def test_compute_symmetry_features_uses_right_minus_left_formula():
    left_features = {"mean": 2.0, "rms": 3.0}
    right_features = {"mean": 4.0, "rms": 6.0}

    result = compute_symmetry_features("knee_angle_l", "knee_angle_r", left_features, right_features)

    assert result["mean"]["right_value"] == 4.0
    assert result["mean"]["left_value"] == 2.0
    assert result["mean"]["symmetry_index"] == (4.0 - 2.0) / 3.0 * 100.0
    assert result["rms"]["symmetry_index"] == (6.0 - 3.0) / 4.5 * 100.0


def test_compute_symmetry_features_returns_nan_when_denominator_is_zero():
    left_features = {"mean": -1.0}
    right_features = {"mean": 1.0}

    result = compute_symmetry_features("knee_angle_l", "knee_angle_r", left_features, right_features)

    assert np.isnan(result["mean"]["symmetry_index"])


def test_recommend_variable_preset_matches_document_pattern_table():
    cases = {
        "knee_angle_r": ("joint_angle", "Joint kinematics"),
        "knee_angle_r_u": ("velocity", "Velocity"),
        "hip_r.load": ("joint_load", "Joint load/force"),
        "knee_r.moment": ("joint_load", "Joint torque/moment"),
        "back.torque": ("joint_load", "Joint torque/moment"),
        "ankle_r_force": ("joint_load", "Joint load/force"),
        "soleus_r.activation": ("muscle_activation", "Muscle activation"),
        "leg1_r_grf_norm_y": ("grf_cop_com", "GRF"),
        "leg1_r.grf_norm_y": ("grf_cop_com", "GRF"),
        "leg0_l_cop_x": ("grf_cop_com", "COP"),
        "leg0_l.cop_x": ("grf_cop_com", "COP"),
        "com_y": ("grf_cop_com", "COM/momentum"),
        "lin_mom.x": ("grf_cop_com", "COM/momentum"),
        "knee_l.load_penalty": ("penalty_quality", "Quality/objective"),
        "pelvis_tx": ("general", "General numeric"),
    }

    for var_name, (preset_key, variable_group) in cases.items():
        result = recommend_variable_preset(var_name)
        assert result["preset_key"] == preset_key
        assert result["variable_group"] == variable_group


def test_classify_variable_returns_recommended_group():
    assert classify_variable("ankle_angle_l") == "Joint kinematics"
    assert classify_variable("knee_angle_r_u") == "Velocity"
    assert classify_variable("pelvis_tx") == "General numeric"
    assert classify_variable("MuscleActivation.penalty") == "Quality/objective"


def test_common_features_ignore_nan_inf_without_failing():
    data = np.array([1.0, np.nan, 3.0, np.inf, -2.0])
    time = np.arange(len(data), dtype=float)

    result = compute_common_features(data, time, threshold=2.0)

    assert result["count"] == 3
    assert result["max"] == 3.0
    assert result["min"] == -2.0
    assert np.isfinite(result["rms"])
    assert result["threshold_exceed_count"] == 1


def test_scone_phase_empty_sections_return_count_zero():
    data = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0, 2.0])
    state = np.array([0, 0, 0])

    result = analyze_data_by_scone_phases(data, state, time)

    assert result["EarlyStance"]["count"] == 3
    assert result["Swing"]["count"] == 0
    assert np.isnan(result["Swing"]["mean"])


def test_analyze_variable_without_state_falls_back_to_non_phase_features():
    data = np.linspace(0.0, 1.0, 20)
    time = np.linspace(0.0, 6.0, 20)

    result = analyze_variable_comprehensive(data, time, "value")

    assert result["overall_statistics"]["count"] == 20
    assert "time_phase_statistics" not in result
    assert result["common_features"]["max"] == 1.0
    assert result["scone_phase_statistics"] == {}
