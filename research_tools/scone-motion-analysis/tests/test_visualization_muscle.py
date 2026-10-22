import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.plot_muscle import (
    _compute_cci,
    compute_activation_summary_rows,
    compute_cci_results,
    create_activation_summary_figure,
    create_cci_figure,
    create_muscle_activation_figure,
    detect_muscle_signal_variables,
    save_integrated_cci_csv,
)


def make_muscle_dataframe(offset=0.0):
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 2 * np.pi, len(state))
    base = np.clip(0.45 + 0.35 * np.sin(x) + offset, 0.0, None)

    data = {
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "leg0_l.state": state,
        "soleus_r.activation": base,
        "soleus_r.excitation": np.clip(base + 0.06, 0.0, None),
        "soleus_r.input": np.clip(base - 0.04, 0.0, None),
        "soleus_l.activation": np.clip(base * 0.9, 0.0, None),
    }

    cci_muscles = [
        "glut_max",
        "hamstrings",
        "iliopsoas",
        "vasti",
        "rect_fem",
        "bifemsh",
        "gastroc",
        "tib_ant",
    ]
    for idx, muscle in enumerate(cci_muscles):
        scale = 0.22 + idx * 0.025
        data[f"{muscle}_r.activation"] = np.clip(base + scale, 0.0, None)
        data[f"{muscle}_l.activation"] = np.clip(base * 0.85 + scale, 0.0, None)

    return pd.DataFrame(data)


def test_detect_muscle_signal_variables_parses_activation_excitation_input():
    records = detect_muscle_signal_variables([
        "soleus_r.activation",
        "soleus_r.excitation",
        "soleus_r.input",
        "leg0_l.gastroc.activation",
        "time",
    ])

    by_variable = {record.variable: record for record in records}
    assert by_variable["soleus_r.activation"].muscle == "soleus"
    assert by_variable["soleus_r.activation"].side == "r"
    assert by_variable["soleus_r.excitation"].signal == "excitation"
    assert by_variable["leg0_l.gastroc.activation"].side == "l"
    assert by_variable["leg0_l.gastroc.activation"].muscle == "gastroc"


def test_create_muscle_activation_figure_draws_overlay_and_onset_markers():
    df = make_muscle_dataframe()

    fig, has_data, warnings = create_muscle_activation_figure(
        [df],
        ["trial_W70.sto"],
        side_filter="right",
        threshold=0.3,
    )

    assert has_data is True
    assert fig.axes
    line_styles = [line.get_linestyle() for ax in fig.axes for line in ax.lines]
    assert "-" in line_styles
    assert "--" in line_styles
    assert ":" in line_styles
    assert any(ax.patches for ax in fig.axes)
    assert warnings == []

    plt.close(fig)


def test_create_muscle_activation_figure_respects_scope_and_uses_shared_legend():
    df = make_muscle_dataframe()

    fig, has_data, warnings = create_muscle_activation_figure(
        [df],
        ["trial_W70.sto"],
        side_filter="right",
        threshold=0.3,
        scope_variables=["soleus_r.activation"],
    )

    assert has_data is True
    assert warnings == []
    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == ["soleus (Right)"]
    assert len(fig.legends) == 1
    assert not any(ax.get_legend() for ax in fig.axes)
    legend_labels = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
    assert legend_labels == ["trial_W70 activation"]

    plt.close(fig)


def test_activation_summary_rows_and_figure_include_magnitude_timing_and_phase_metrics():
    df = make_muscle_dataframe()

    rows, warnings = compute_activation_summary_rows([df], ["trial_W70.sto"], side_filter="both", threshold=0.3)
    fig, has_data, figure_warnings, figure_rows = create_activation_summary_figure(
        [df],
        ["trial_W70.sto"],
        side_filter="both",
        threshold=0.3,
    )

    assert rows
    assert rows[0]["integrated_activation"] > 0
    assert rows[0]["peak_activation"] > 0
    assert 0 <= rows[0]["peak_timing_pct"] <= 100
    assert rows[0]["stance_integrated_activation"] > 0
    assert rows[0]["swing_integrated_activation"] > 0
    assert "active_duration_pct" not in rows[0]
    assert "duty_factor" not in rows[0]
    assert has_data is True
    assert len(figure_rows) == len(rows)
    assert warnings == []
    assert figure_warnings == []
    visible_titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert visible_titles == [
        "Integrated activation",
        "Peak activation",
        "Peak timing",
        "Stance activation",
        "Swing activation",
    ]

    plt.close(fig)


def test_activation_summary_metrics_do_not_depend_on_absolute_threshold():
    df = make_muscle_dataframe()
    df["soleus_r.activation"] = df["soleus_r.activation"] * 0.02

    rows, warnings = compute_activation_summary_rows(
        [df],
        ["trial_W70.sto"],
        side_filter="right",
        threshold=0.05,
        scope_variables=["soleus_r.activation"],
    )

    assert rows[0]["integrated_activation"] > 0
    assert rows[0]["peak_activation"] > 0
    assert 0 <= rows[0]["peak_timing_pct"] <= 100
    assert rows[0]["stance_integrated_activation"] > 0
    assert rows[0]["swing_integrated_activation"] > 0
    assert warnings == []


def test_activation_summary_uses_state_phase_masks_for_stance_and_swing():
    one_cycle = [0] * 20 + [1] * 20 + [2] * 40 + [3] * 10 + [4] * 10
    state = one_cycle + one_cycle + [0] * 20
    activation = np.where(np.asarray(state) == 2, 1.0, 0.0)
    df = pd.DataFrame({
        "time": np.arange(len(state), dtype=float) / 100.0,
        "leg1_r.state": state,
        "soleus_r.activation": activation,
    })

    rows, warnings = compute_activation_summary_rows(
        [df],
        ["trial_W70.sto"],
        side_filter="right",
        scope_variables=["soleus_r.activation"],
    )

    assert warnings == []
    assert rows[0]["stance_integrated_activation"] > 0
    assert rows[0]["swing_integrated_activation"] == 0.0


def test_activation_summary_figure_respects_scope_and_groups_by_muscle_side():
    df = make_muscle_dataframe()

    fig, has_data, warnings, rows = create_activation_summary_figure(
        [df],
        ["trial_W70.sto"],
        side_filter="right",
        threshold=0.3,
        scope_variables=["soleus_r.activation"],
    )

    assert has_data is True
    assert warnings == []
    assert {row["variable"] for row in rows} == {"soleus_r.activation"}
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["soleus (R)"]
    assert len(fig.legends) == 1

    plt.close(fig)


def test_create_cci_figure_computes_all_requested_pairs_and_csv(tmp_path):
    df = make_muscle_dataframe()

    fig, has_data, warnings, rows = create_cci_figure([df], ["trial_W70.sto"], side_filter="both")
    csv_path = save_integrated_cci_csv(rows, tmp_path / "integrated_cci.csv")
    saved = pd.read_csv(csv_path)

    assert has_data is True
    assert {"hip", "knee", "ankle"} <= set(saved["muscle_pair"])
    assert {"Right", "Left"} <= set(saved["side"])
    assert (saved["integrated_cci"] > 0).all()
    assert warnings == []
    assert (tmp_path / "integrated_cci.csv").exists()

    plt.close(fig)


def test_compute_cci_returns_zero_when_one_activation_group_is_zero():
    cci = _compute_cci(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.4, 0.8, 1.2]),
    )

    np.testing.assert_allclose(cci, np.zeros(3))


def test_compute_cci_results_skips_missing_pair_with_warning():
    df = make_muscle_dataframe().drop(columns=["tib_ant_r.activation"])

    series, rows, warnings = compute_cci_results([df], ["trial_W70.sto"], side_filter="right")

    assert series
    assert rows
    assert "ankle" not in {row["muscle_pair"] for row in rows}
    assert any("ankle Right CCI skipped" in warning for warning in warnings)
