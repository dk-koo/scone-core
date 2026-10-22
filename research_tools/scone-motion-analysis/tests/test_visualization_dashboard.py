import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.analysis_engine import analyze_variable_comprehensive, compute_common_features, compute_symmetry_features
from visualization.dashboard import _comparison_table, create_visualization_dashboard


def make_dashboard_dataframe(offset=0.0):
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 2 * np.pi, len(state))
    value_r = np.sin(x) + offset
    value_l = np.sin(x + 0.1) + offset * 0.8
    vertical = 0.7 + 0.3 * np.sin(2 * x) ** 2

    data = {
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "leg0_l.state": state,
        "knee_angle_r": value_r,
        "knee_angle_l": value_l,
        "hip_r.torque": np.sin(x) * 0.3,
        "knee_r.torque": np.cos(x) * 0.2,
        "ankle_r.torque": np.sin(x + 0.5) * 0.4,
        "leg1_r.grf_norm_x": 0.12 * np.sin(x),
        "leg1_r.grf_norm_y": vertical,
        "leg1_r.grf_norm_z": 0.02 * np.cos(x),
        "leg1_r.cop_x": np.linspace(0.0, 0.25, len(state)),
        "leg1_r.cop_y": np.zeros(len(state)),
        "leg1_r.cop_z": 0.02 * np.sin(x),
        "com_x": np.linspace(0.03, 0.28, len(state)),
        "com_y": 0.9 + 0.02 * np.sin(x),
        "com_z": 0.02 * np.cos(x),
        "lin_mom_x": np.sin(x),
        "ang_mom_z": np.cos(x) * 0.1,
        "soleus_r.activation": np.clip(0.45 + 0.35 * np.sin(x), 0.0, None),
        "soleus_r.excitation": np.clip(0.50 + 0.35 * np.sin(x), 0.0, None),
    }
    for idx, muscle in enumerate(["glut_max", "hamstrings", "iliopsoas", "vasti", "rect_fem", "bifemsh", "gastroc", "tib_ant"]):
        data[f"{muscle}_r.activation"] = np.clip(0.35 + 0.2 * np.sin(x + idx * 0.2), 0.0, None)
    return pd.DataFrame(data)


def make_dashboard_analysis_results(dataframes, file_paths):
    results = {}
    for file_idx, (df, path) in enumerate(zip(dataframes, file_paths)):
        time = df["time"].to_numpy(dtype=float)
        for variable in ("knee_angle_r", "knee_angle_l"):
            result = analyze_variable_comprehensive(
                df[variable].to_numpy(dtype=float),
                time,
                variable,
                state_arr=df["leg1_r.state"].to_numpy(dtype=float),
                state_column_name="leg1_r.state",
                threshold=0.05,
            )
            if variable == "knee_angle_r":
                pair_features = compute_common_features(df["knee_angle_l"].to_numpy(dtype=float), time, threshold=0.05)
                result["symmetry_pair_variable"] = "knee_angle_l"
                result["symmetry_features"] = compute_symmetry_features(
                    variable,
                    "knee_angle_l",
                    result["common_features"],
                    pair_features,
                )
            results[f"{file_idx}_{path}_{variable}"] = result
    return results


def test_create_visualization_dashboard_writes_html_and_summary(tmp_path):
    dataframes = [make_dashboard_dataframe(0.0), make_dashboard_dataframe(0.2)]
    file_paths = ["trial_W70.sto", "trial_W80.sto"]
    analysis_results = make_dashboard_analysis_results(dataframes, file_paths)

    dashboard_path, summary_path, warnings = create_visualization_dashboard(
        dataframes,
        file_paths,
        ["knee_angle_r", "knee_angle_l"],
        ["knee_angle_r"],
        analysis_results,
        output_root=str(tmp_path / "output"),
    )

    html = (tmp_path / "output" / "html" / "dashboard.html").read_text(encoding="utf-8")
    summary = (tmp_path / "output" / "reports" / "visualization_summary.md").read_text(encoding="utf-8")

    assert dashboard_path.endswith("dashboard.html")
    assert summary_path.endswith("visualization_summary.md")
    assert "Data quality summary" in html
    assert "Selected variable waveform" in html
    assert "Gait phase summary" in html
    assert "Multi-panel gait overview" in html
    assert "Feature summary" in html
    assert "Muscle activation" in html
    assert "CCI" in html
    assert "GRF/COP/COM" in html
    assert "Symmetry" in html
    assert "Condition comparison" in html
    assert "data:image/png;base64" in html
    assert "<table>" in html
    assert "Warnings and skipped items" in html
    assert "# Visualization Summary" in summary
    assert isinstance(warnings, list)


def test_selected_scope_dashboard_filters_unselected_visualization_variables(tmp_path):
    dataframes = [make_dashboard_dataframe(0.0)]
    file_paths = ["trial_W70.sto"]
    analysis_results = make_dashboard_analysis_results(dataframes, file_paths)

    create_visualization_dashboard(
        dataframes,
        file_paths,
        ["knee_angle_r"],
        ["knee_angle_r"],
        analysis_results,
        output_root=str(tmp_path / "output"),
        section_scope="selected",
    )

    html = (tmp_path / "output" / "html" / "dashboard.html").read_text(encoding="utf-8")
    summary = (tmp_path / "output" / "reports" / "visualization_summary.md").read_text(encoding="utf-8")

    assert "Scope: Selected variables" in html
    assert "leg1_r.grf_norm_y" not in html
    assert "Scope: selected variables" in summary


def test_dashboard_comparison_uses_absolute_reference_for_delta_percent():
    time = np.arange(10, dtype=float)
    analysis_results = {
        "0_ref.sto_value": analyze_variable_comprehensive(np.full(10, -2.0), time, "value"),
        "1_condition.sto_value": analyze_variable_comprehensive(np.full(10, -1.0), time, "value"),
    }

    table = _comparison_table(analysis_results)
    mean_row = table[table["metric"] == "mean"].iloc[0]

    assert mean_row["delta"] == 1.0
    assert mean_row["delta_percent"] == 50.0
