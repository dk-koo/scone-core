import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from visualization.plot_overview import (
    collect_overview_panels,
    create_overview_figure,
    create_overview_plotly_figure,
    get_condition_labels,
)


def make_overview_dataframe(offset=0.0):
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 2 * np.pi, len(state))
    return pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "leg0_l.state": state,
        "hip_flexion_r": np.sin(x) + offset,
        "hip_flexion_l": np.sin(x + 0.1) + offset,
        "knee_angle_r": np.cos(x) + offset,
        "ankle_angle_r": np.sin(x * 1.5) + offset,
        "hip_r.load": np.abs(np.sin(x)) + offset,
        "knee_r.torque": np.cos(x) * 0.2 + offset,
        "ankle_r.force": np.abs(np.cos(x)) + offset,
        "leg1_r.grf_norm_x": np.sin(x) * 0.1 + offset,
        "leg1_r.grf_norm_y": np.abs(np.sin(x)) + offset,
        "hip_r.power": np.sin(x) * np.cos(x) + offset,
    })


def test_collect_overview_panels_finds_default_preset_variables():
    panels, warnings = collect_overview_panels([make_overview_dataframe()])
    titles = [spec.title for spec, _ in panels]

    assert "Hip angle" in titles
    assert "Knee load / force / torque" in titles
    assert "Vertical GRF" in titles
    assert "Knee power: matching variable not found; skipped." in warnings


def test_collect_overview_panels_excludes_angular_velocity_from_angle_panel():
    df = pd.DataFrame({
        "time": [0.0, 1.0, 2.0],
        "knee_angle_r_u": [0.0, 1.0, 2.0],
    })

    panels, warnings = collect_overview_panels([df])
    titles = [spec.title for spec, _ in panels]

    assert "Knee angle" not in titles
    assert "Knee angle: matching variable not found; skipped." in warnings


def test_create_overview_figure_draws_subplot_grid_with_reference():
    dataframes = [make_overview_dataframe(0.0), make_overview_dataframe(0.2)]
    file_paths = ["trial_W70.sto", "trial_W80.sto"]

    fig, has_data, warnings = create_overview_figure(
        dataframes,
        file_paths,
        reference_condition="trial_W70",
        sd_multiplier=2.0,
    )

    assert has_data is True
    assert fig.axes
    assert any(ax.get_title() == "Hip angle" for ax in fig.axes)
    assert any(ax.lines for ax in fig.axes)
    assert len(fig.legends) == 1
    assert not any(ax.get_legend() for ax in fig.axes)
    assert any("matching variable not found" in warning for warning in warnings)

    plt.close(fig)


def test_create_overview_plotly_figure_returns_html_ready_figure():
    pytest.importorskip("plotly")
    dataframes = [make_overview_dataframe()]
    file_paths = ["trial_W70.sto"]

    fig, has_data, warnings = create_overview_plotly_figure(
        dataframes,
        file_paths,
        reference_condition="trial_W70",
    )

    assert has_data is True
    assert "Gait Overview" in fig.layout.title.text
    assert "reference: trial_W70" in fig.layout.title.text
    assert fig.to_html(include_plotlyjs=False)
    assert any("matching variable not found" in warning for warning in warnings)


def test_get_condition_labels_preserves_file_order():
    labels = get_condition_labels(["a_W70.sto", "b_W80.sto", "a_W70.sto"])

    assert labels == ["a_W70", "b_W80"]
