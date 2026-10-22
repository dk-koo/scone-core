import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.plot_gait_cycle import (
    create_normalized_phase_masks,
    draw_gait_cycle_waveform,
    normalize_to_gait_cycle,
)


def make_state_cycle_dataframe():
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    return pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "value_r": np.sin(np.linspace(0, 2 * np.pi, len(state))),
        "leg1_r.state": state,
    })


def test_normalize_to_gait_cycle_returns_101_point_cycles_and_states():
    df = make_state_cycle_dataframe()

    state_col, cycles, state_cycles = normalize_to_gait_cycle(df, "value_r")

    assert state_col == "leg1_r.state"
    assert cycles.shape[1] == 101
    assert state_cycles.shape[1] == 101
    assert cycles.shape[0] >= 1


def test_normalize_to_gait_cycle_uses_time_spacing_not_frame_spacing():
    state = [0] * 10 + [4] * 10 + [0] * 10 + [4] * 10 + [0] * 10
    time = np.arange(len(state), dtype=float) ** 2
    df = pd.DataFrame({
        "time": time,
        "value_r": time,
        "leg1_r.state": state,
    })

    _, cycles, _ = normalize_to_gait_cycle(df, "value_r")

    assert cycles.shape == (2, 101)
    assert cycles[0, 50] == (time[0] + time[20]) / 2.0


def test_normalize_to_gait_cycle_uses_frame_spacing_when_time_has_duplicates():
    state = [0] * 10 + [4] * 10 + [0] * 10
    time = np.arange(len(state), dtype=float) ** 2
    time[1] = time[0]
    df = pd.DataFrame({
        "time": time,
        "value_r": time,
        "leg1_r.state": state,
    })

    _, cycles, _ = normalize_to_gait_cycle(df, "value_r")

    assert cycles.shape == (1, 101)
    assert cycles[0, 50] == time[10]


def test_normalize_to_gait_cycle_converts_joint_angles_to_degrees():
    state = [0] * 10 + [4] * 10 + [0] * 10
    angle_values = np.concatenate([np.linspace(0.0, np.pi, 21), np.full(9, np.pi)])
    df = pd.DataFrame({
        "time": np.arange(len(state), dtype=float),
        "knee_angle_r": angle_values,
        "leg1_r.state": state,
    })

    _, cycles, _ = normalize_to_gait_cycle(df, "knee_angle_r")

    assert cycles.shape == (1, 101)
    assert cycles[0, -1] == 180.0


def test_normalize_to_gait_cycle_uses_any_entry_into_state_zero_as_boundary():
    state = [0] * 10 + [1] * 10 + [3] * 10 + [0] * 10 + [2] * 10 + [3] * 10 + [0] * 10
    df = pd.DataFrame({
        "time": np.arange(len(state), dtype=float),
        "value_r": np.arange(len(state), dtype=float),
        "leg1_r.state": state,
    })

    _, cycles, _ = normalize_to_gait_cycle(df, "value_r")

    assert cycles.shape == (2, 101)


def test_normalize_to_gait_cycle_tolerates_isolated_invalid_state_values():
    df = make_state_cycle_dataframe()
    df.loc[15, "leg1_r.state"] = np.nan
    df.loc[60, "leg1_r.state"] = 99

    state_col, cycles, state_cycles = normalize_to_gait_cycle(df, "value_r")

    assert state_col == "leg1_r.state"
    assert cycles.shape[1] == 101
    assert state_cycles.shape[1] == 101
    assert cycles.shape[0] >= 1
    assert np.isin(state_cycles, [0, 1, 2, 3, 4]).any()


def test_create_normalized_phase_masks_includes_composite_phases():
    state_cycle = np.array([0, 1, 2, 3, 4])

    masks = create_normalized_phase_masks(state_cycle)

    assert masks["EarlyStance"].tolist() == [True, False, False, False, False]
    assert masks["LiftOff"].tolist() == [False, False, True, False, False]
    assert masks["stance"].tolist() == [True, True, True, False, False]
    assert masks["swing"].tolist() == [False, False, False, True, True]


def test_draw_gait_cycle_waveform_adds_phase_shading_and_peak_markers_without_labels():
    df = make_state_cycle_dataframe()
    fig, ax = plt.subplots()

    has_data, warnings = draw_gait_cycle_waveform(ax, [df], ["sample.sto"], "value_r")

    assert has_data is True
    assert warnings == []
    assert len(ax.patches) > 0
    marker_values = [line.get_marker() for line in ax.lines]
    assert "^" in marker_values
    assert "v" in marker_values
    assert not any("max" in text.get_text() for text in ax.texts)
    assert not any("min" in text.get_text() for text in ax.texts)
    assert not any(text.get_text() in {"0%", "50%", "100%"} for text in ax.texts)
    assert ax.get_xlim() == (0.0, 100.0)

    plt.close(fig)


def test_draw_gait_cycle_waveform_without_state_returns_warning_not_error():
    df = pd.DataFrame({
        "time": [0.0, 0.1, 0.2],
        "value_r": [1.0, 2.0, 3.0],
    })
    fig, ax = plt.subplots()

    has_data, warnings = draw_gait_cycle_waveform(ax, [df], ["sample.sto"], "value_r")

    assert has_data is True
    assert len(warnings) == 2
    assert "state column not found" in warnings[0]
    assert "fell back" in warnings[1]
    assert len(ax.patches) == 0
    assert ax.get_xlabel() == "Time (s)"

    plt.close(fig)
