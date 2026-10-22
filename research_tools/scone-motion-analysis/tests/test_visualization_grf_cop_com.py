import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.plot_grf_cop_com import (
    create_grf_cop_com_figure,
    detect_com_variables,
    detect_cop_variables,
    detect_grf_variables,
    detect_momentum_variables,
)


def make_grf_cop_com_dataframe(include_cop=True, include_com=True):
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 4 * np.pi, len(state))
    gait = np.linspace(0, 2 * np.pi, len(state))
    vertical = 0.7 + 0.3 * np.sin(x) ** 2 + 0.12 * np.sin(2 * x)
    ap = 0.18 * np.sin(gait)

    data = {
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "leg0_l.state": state,
        "leg1_r.grf_norm_x": ap,
        "leg1_r.grf_norm_y": vertical,
        "leg1_r.grf_norm_z": 0.03 * np.cos(gait),
        "leg0_l.grf_norm_x": -ap * 0.9,
        "leg0_l.grf_norm_y": vertical * 0.95,
        "leg0_l.grf_norm_z": -0.02 * np.cos(gait),
        "lin_mom_x": np.sin(gait),
        "lin_mom_y": np.cos(gait) * 0.2,
        "ang_mom_z": np.sin(gait * 0.5) * 0.1,
        "ankle_r.torque": np.sin(gait) * 0.4,
        "knee_r.torque": np.sin(gait + 0.4) * 0.25,
        "hip_r.torque": np.sin(gait + 0.8) * 0.15,
    }
    if include_cop:
        data.update({
            "leg1_r.cop_x": np.linspace(0.0, 0.3, len(state)),
            "leg1_r.cop_y": np.zeros(len(state)),
            "leg1_r.cop_z": 0.03 * np.sin(gait),
            "leg0_l.cop_x": np.linspace(0.02, 0.28, len(state)),
            "leg0_l.cop_y": np.zeros(len(state)),
            "leg0_l.cop_z": -0.03 * np.sin(gait),
        })
    if include_com:
        data.update({
            "com_x": np.linspace(0.03, 0.33, len(state)),
            "com_y": 0.9 + 0.02 * np.sin(gait),
            "com_z": 0.02 * np.cos(gait),
        })
    return pd.DataFrame(data)


def test_detect_grf_cop_com_and_momentum_variables():
    df = make_grf_cop_com_dataframe()

    grf = detect_grf_variables(df.columns)
    cop = detect_cop_variables(df.columns)
    com = detect_com_variables(df.columns)
    momentum = detect_momentum_variables(df.columns)

    assert grf["Right"]["y"] == "leg1_r.grf_norm_y"
    assert grf["Left"]["x"] == "leg0_l.grf_norm_x"
    assert cop["Right"]["z"] == "leg1_r.cop_z"
    assert com["x"] == "com_x"
    assert momentum["linear"]["x"] == "lin_mom_x"
    assert momentum["angular"]["z"] == "ang_mom_z"


def test_create_grf_cop_com_figure_draws_requested_panels_and_markers():
    df = make_grf_cop_com_dataframe()

    fig, has_data, warnings = create_grf_cop_com_figure([df], ["trial_W70.sto"])

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    titles = [ax.get_title() for ax in visible_axes]
    vertical_ax = next(ax for ax in visible_axes if ax.get_title() == "Vertical GRF")
    ap_ax = next(ax for ax in visible_axes if ax.get_title() == "AP GRF braking / propulsion")

    assert has_data is True
    assert "COP trajectory" in titles
    assert "COM trajectory" in titles
    assert "COM-COP distance" in titles
    assert "COM position change rate" in titles
    assert "Linear / angular momentum" in titles
    assert "Balance strategy proxy" in titles
    assert any("1st" in text.get_text() for text in vertical_ax.texts)
    assert any("2nd" in text.get_text() for text in vertical_ax.texts)
    assert any("valley" in text.get_text() for text in vertical_ax.texts)
    assert any("brake" in text.get_text() and "prop" in text.get_text() for text in ap_ax.texts)
    assert len(ap_ax.collections) >= 2
    assert len(fig.legends) == 1
    assert not any(ax.get_legend() for ax in fig.axes)
    legend_labels = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
    assert "trial_W70 Right" in legend_labels
    assert "ankle" in legend_labels
    assert warnings == []

    plt.close(fig)


def test_create_grf_cop_com_figure_uses_y_axis_for_com_cop_distance_when_z_is_missing():
    df = make_grf_cop_com_dataframe()
    df = df.drop(columns=["com_z", "leg1_r.cop_z", "leg0_l.cop_z"])

    fig, has_data, warnings = create_grf_cop_com_figure([df], ["trial_W70.sto"])

    visible_titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert has_data is True
    assert "COM-COP distance" in visible_titles
    assert warnings == []

    plt.close(fig)


def test_create_grf_cop_com_figure_skips_missing_cop_or_com_without_error():
    df = make_grf_cop_com_dataframe(include_cop=False, include_com=False)

    fig, has_data, warnings = create_grf_cop_com_figure([df], ["trial_W70.sto"])

    visible_titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert has_data is True
    assert "Vertical GRF" in visible_titles
    assert "AP GRF braking / propulsion" in visible_titles
    assert "COP trajectory" not in visible_titles
    assert "COM trajectory" not in visible_titles
    assert any("COP variables" in warning for warning in warnings)
    assert any("com_x/com_y/com_z" in warning for warning in warnings)

    plt.close(fig)
