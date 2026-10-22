import numpy as np
import pandas as pd

from src.auto_analysis import build_auto_analysis_plan, feature_filter_for_purpose, format_auto_analysis_plan


def test_build_auto_analysis_plan_detects_possible_analysis_types():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "leg1_r.state": [0, 1, 2, 3, 4] * 4,
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
        "leg1_r.grf_norm_y": np.abs(np.sin(np.linspace(0.0, 2.0, 20))),
        "soleus_r.activation": np.linspace(0.1, 0.8, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"])

    assert "leg1_r.state" not in plan.selected_variables
    assert "knee_angle_r" in plan.selected_variables
    assert "knee_angle_l" in plan.selected_variables
    assert "leg1_r.grf_norm_y" in plan.selected_variables
    assert any("SCONE phase" in item for item in plan.available_analyses)
    assert any("symmetry" in item for item in plan.available_analyses)
    assert any("GRF" in item for item in plan.available_analyses)
    assert any("COP/COM" in item for item in plan.skipped_analyses)


def test_format_auto_analysis_plan_explains_skipped_items():
    df = pd.DataFrame({
        "time": [0.0, 1.0, 2.0],
        "value": [1.0, 2.0, 3.0],
    })

    text = format_auto_analysis_plan(build_auto_analysis_plan([df], ["trial.csv"]))

    assert "자동 분석 후보" in text
    assert "분석 대상 변수: 1개" in text
    assert "없는 변수군은 오류가 아니라 자동으로 건너뜁니다." in text


def test_feature_filter_for_purpose_supports_dynamics_alias():
    assert feature_filter_for_purpose("역학/관절 토크 분석") == "Joint load / torque / force"
    assert feature_filter_for_purpose("관절 부하 분석") == "Joint load / torque / force"
    assert feature_filter_for_purpose("GRF/균형 분석") == "GRF / COP / COM"


def test_build_auto_analysis_plan_filters_joint_angle_variables():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_r_u": np.linspace(0.0, 2.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
        "hip_flexion_l_u": np.linspace(0.0, 2.0, 20),
        "pelvis_tx": np.linspace(0.0, 0.1, 20),
        "hip_r.torque": np.linspace(0.0, 2.0, 20),
        "soleus_r.activation": np.linspace(0.1, 0.8, 20),
        "leg1_r.grf_norm_y": np.linspace(0.0, 1.0, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="Joint angle")

    assert plan.feature_filter == "Joint angle"
    assert plan.selected_variables == ["knee_angle_l", "knee_angle_r"]
    assert plan.graph_variables == ["knee_angle_l", "knee_angle_r"]
    assert "knee_angle_r_u" not in plan.selected_variables
    assert "hip_flexion_l_u" not in plan.selected_variables
    assert "pelvis_tx" not in plan.selected_variables


def test_build_auto_analysis_plan_filters_velocity_variables():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_r_u": np.linspace(0.0, 2.0, 20),
        "hip_flexion_l_u": np.linspace(0.0, 2.0, 20),
        "pelvis_tx": np.linspace(0.0, 0.1, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="Velocity")

    assert plan.selected_variables == ["hip_flexion_l_u", "knee_angle_r_u"]


def test_build_auto_analysis_plan_filters_joint_load_variables_strictly():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "hip_r.load": np.linspace(0.0, 2.0, 20),
        "knee_r.torque": np.linspace(0.0, 2.0, 20),
        "ankle_r_force": np.linspace(0.0, 2.0, 20),
        "back.torque": np.linspace(0.0, 2.0, 20),
        "knee_r.moment": np.linspace(0.0, 2.0, 20),
        "soleus_r.force": np.linspace(0.0, 2.0, 20),
        "leg1_r.grf_norm_y": np.linspace(0.0, 1.0, 20),
        "knee_l.load_penalty": np.linspace(0.0, 1.0, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="Joint load / torque / force")

    assert plan.selected_variables == ["ankle_r_force", "back.torque", "hip_r.load", "knee_r.moment", "knee_r.torque"]
    assert "soleus_r.force" not in plan.selected_variables
    assert "leg1_r.grf_norm_y" not in plan.selected_variables
    assert "knee_l.load_penalty" not in plan.selected_variables
    assert any("관절 부하 후보" in item for item in plan.warnings)

def test_build_auto_analysis_plan_filters_grf_balance_source_variables():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "leg1_r.grf_norm_x": np.linspace(0.0, 1.0, 20),
        "leg1_r.grf_norm_y": np.linspace(1.0, 2.0, 20),
        "leg1_r.cop_x": np.linspace(0.0, 0.2, 20),
        "leg1_r.cop_z": np.linspace(0.1, 0.3, 20),
        "com_x": np.linspace(0.0, 0.2, 20),
        "com_z": np.linspace(0.2, 0.4, 20),
        "lin_mom.x": np.linspace(0.0, 1.0, 20),
        "ankle_r.torque": np.linspace(0.0, 2.0, 20),
        "knee_r.moment": np.linspace(0.0, 2.0, 20),
        "hip_r.load": np.linspace(0.0, 2.0, 20),
        "soleus_r.activation": np.linspace(0.1, 0.8, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="GRF / COP / COM")

    assert plan.feature_filter == "GRF / COP / COM"
    assert plan.selected_variables == [
        "ankle_r.torque",
        "com_x",
        "com_z",
        "hip_r.load",
        "knee_r.moment",
        "leg1_r.cop_x",
        "leg1_r.cop_z",
        "leg1_r.grf_norm_x",
        "leg1_r.grf_norm_y",
        "lin_mom.x",
    ]
    assert "soleus_r.activation" not in plan.selected_variables


def test_build_auto_analysis_plan_filters_symmetry_pair_variables():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
        "value": np.linspace(0.0, 2.0, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="Symmetry pairs")

    assert plan.selected_variables == ["knee_angle_l", "knee_angle_r"]
    assert any("symmetry" in item for item in plan.available_analyses)


def test_build_auto_analysis_plan_unknown_filter_falls_back_to_all():
    df = pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "value": np.linspace(0.0, 1.0, 20),
    })

    plan = build_auto_analysis_plan([df], ["trial.sto"], feature_filter="unknown")

    assert plan.feature_filter == "All analyzable"
    assert plan.selected_variables == ["value"]
    assert plan.warnings
