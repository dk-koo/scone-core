import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.common import save_figure
from visualization.plot_waveform import draw_raw_waveform


def test_draw_raw_waveform_uses_time_axis_and_overlays_files():
    dataframes = [
        pd.DataFrame({"time": [0.0, 0.5, 1.0], "knee_angle_r": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"time": [0.0, 0.5, 1.0], "knee_angle_r": [1.5, 2.5, 3.5]}),
    ]
    file_paths = ["trial_W70.sto", "trial_W75.sto"]
    fig, ax = plt.subplots()

    has_data, warnings = draw_raw_waveform(ax, dataframes, file_paths, "knee_angle_r")

    assert has_data is True
    assert warnings == []
    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.rad2deg([1.0, 2.0, 3.0]))
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Value (deg)"

    plt.close(fig)


def test_draw_raw_waveform_skips_missing_variable_without_error():
    dataframes = [
        pd.DataFrame({"time": [0.0, 0.5, 1.0], "value": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"time": [0.0, 0.5, 1.0], "other": [1.5, 2.5, 3.5]}),
    ]
    fig, ax = plt.subplots()

    has_data, warnings = draw_raw_waveform(ax, dataframes, ["a.sto", "b.sto"], "value")

    assert has_data is True
    assert len(warnings) == 1
    assert "variable not found" in warnings[0]
    assert len(ax.lines) == 1

    plt.close(fig)


def test_save_figure_writes_png_and_svg(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    saved_paths = save_figure(fig, tmp_path / "waveform", formats=("png", "svg"), dpi=80)

    assert len(saved_paths) == 2
    assert (tmp_path / "waveform.png").exists()
    assert (tmp_path / "waveform.svg").exists()
    assert (tmp_path / "waveform.png").stat().st_size > 0
    assert (tmp_path / "waveform.svg").stat().st_size > 0

    plt.close(fig)
