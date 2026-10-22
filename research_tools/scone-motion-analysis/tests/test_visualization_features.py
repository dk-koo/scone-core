import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from visualization.plot_features import (
    FEATURE_PLOT_TYPES,
    create_feature_plot,
    discover_feature_csvs,
    get_conditions_from_feature_tables,
)


def write_feature_csvs(tmp_path):
    cycle = pd.DataFrame([
        {"file": "a.sto", "condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "max", "value": 10.0},
        {"file": "b.sto", "condition": "W80", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "max", "value": 12.0},
        {"file": "a.sto", "condition": "W70", "variable": "ankle_angle_r", "phase": "whole_cycle", "feature": "max", "value": 5.0},
        {"file": "b.sto", "condition": "W80", "variable": "ankle_angle_r", "phase": "whole_cycle", "feature": "max", "value": 7.0},
        {"file": "a.sto", "condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "rom", "value": 20.0},
    ])
    phase = pd.DataFrame([
        {"file": "a.sto", "condition": "W70", "variable": "knee_angle_r", "phase": "stance", "feature": "mean", "value": 3.0},
        {"file": "a.sto", "condition": "W70", "variable": "knee_angle_r", "phase": "swing", "feature": "mean", "value": 6.0},
        {"file": "a.sto", "condition": "W70", "variable": "ankle_angle_r", "phase": "stance", "feature": "mean", "value": 2.0},
        {"file": "a.sto", "condition": "W70", "variable": "ankle_angle_r", "phase": "swing", "feature": "mean", "value": 4.0},
    ])
    comparison = pd.DataFrame([
        {"condition": "W80", "reference_condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "delta", "condition_value": 12.0, "reference_value": 10.0, "delta": 2.0, "delta_percent": 20.0, "delta_timing": 3.0},
        {"condition": "W80", "reference_condition": "W70", "variable": "ankle_angle_r", "phase": "whole_cycle", "feature": "delta", "condition_value": 7.0, "reference_value": 5.0, "delta": 2.0, "delta_percent": 40.0, "delta_timing": 4.0},
        {"condition": "W80", "reference_condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "rmse", "condition_value": 0.25, "reference_value": 0.0, "delta": 0.25, "delta_percent": 0.0, "delta_timing": 0.0},
        {"condition": "W80", "reference_condition": "W70", "variable": "ankle_angle_r", "phase": "whole_cycle", "feature": "ncc", "condition_value": 0.95, "reference_value": 1.0, "delta": -0.05, "delta_percent": -5.0, "delta_timing": 0.0},
    ])

    cycle_path = tmp_path / "cycle_features.csv"
    phase_path = tmp_path / "phase_features.csv"
    comparison_path = tmp_path / "comparison_features.csv"
    cycle.to_csv(cycle_path, index=False)
    phase.to_csv(phase_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return [str(cycle_path), str(phase_path), str(comparison_path)]


def test_discover_feature_csvs_finds_known_files(tmp_path):
    paths = write_feature_csvs(tmp_path)

    found = discover_feature_csvs([str(tmp_path)])

    assert set(found) == set(paths)


def test_get_conditions_from_feature_tables_returns_ordered_conditions(tmp_path):
    paths = write_feature_csvs(tmp_path)

    conditions = get_conditions_from_feature_tables(paths)

    assert conditions == ["W70", "W80"]


def test_create_feature_plot_supports_all_requested_plot_types(tmp_path):
    paths = write_feature_csvs(tmp_path)

    for plot_type in FEATURE_PLOT_TYPES:
        fig, has_data, warnings = create_feature_plot(paths, plot_type, reference_condition="W70")
        assert has_data is True, (plot_type, warnings)
        assert fig.axes
        plt.close(fig)


def test_create_feature_plot_missing_phase_file_warns_without_crash(tmp_path):
    paths = write_feature_csvs(tmp_path)
    paths = [path for path in paths if not path.endswith("phase_features.csv")]

    fig, has_data, warnings = create_feature_plot(paths, "Phase x variable heatmap")

    assert has_data is False
    assert any("phase column" in warning or "missing columns" in warning for warning in warnings)
    plt.close(fig)
