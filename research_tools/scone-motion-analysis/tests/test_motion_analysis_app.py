from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.motion_analysis_app as motion_app_module
from src.auto_analysis import ANALYSIS_PURPOSE_OPTIONS
from config.language import get_language
from src.analysis_engine import analyze_variable_comprehensive, compute_symmetry_features
from src.motion_analysis_app import (
    MotionAnalysisApp,
    get_result_variable_help_text,
    get_variable_unit_help_text,
)
from utils.data_validator import DataValidator


@pytest.fixture(autouse=True)
def suppress_tk_message_boxes(monkeypatch):
    monkeypatch.setattr(motion_app_module.messagebox, "showerror", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)


class DummyWidget:
    def __init__(self):
        self.items = []
        self.config_kwargs = {}
        self.selected_indices = set()

    def delete(self, *args):
        self.items.clear()
        self.selected_indices.clear()

    def insert(self, *args):
        self.items.append(args[-1])

    def get(self, first, last=None):
        if last is None:
            return self.items[first]
        return tuple(self.items[first:])

    def config(self, **kwargs):
        self.config_kwargs.update(kwargs)

    def selection_set(self, first, last=None):
        first = self._normalize_index(first)
        last = first if last is None else self._normalize_index(last)
        for index in range(first, last + 1):
            if 0 <= index < len(self.items):
                self.selected_indices.add(index)

    def selection_clear(self, first=0, last=None):
        if last is None:
            self.selected_indices.clear()
            return
        first = self._normalize_index(first)
        last = self._normalize_index(last)
        for index in range(first, last + 1):
            self.selected_indices.discard(index)

    def curselection(self):
        return tuple(sorted(self.selected_indices))

    def _normalize_index(self, index):
        if index in ("end", "tk.END"):
            return len(self.items) - 1
        return int(index)


class DummyVar:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class DummySpine:
    def __init__(self):
        self.visible = True
        self.color = None

    def set_visible(self, value):
        self.visible = value

    def set_color(self, value):
        self.color = value


class DummyAxes:
    def __init__(self):
        self.plot_calls = []
        self.spans = []
        self.fill_between_calls = []
        self.text_calls = []
        self.vlines = []
        self.legend_calls = []
        self.title_calls = []
        self.xlim = None
        self.transAxes = object()
        self.spines = {name: DummySpine() for name in ("top", "right", "left", "bottom")}
        self.axisbelow = None
        self.tick_kwargs = {}

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))
        return [object()]

    def axvspan(self, *args, **kwargs):
        self.spans.append((args, kwargs))

    def axvline(self, *args, **kwargs):
        self.vlines.append((args, kwargs))

    def fill_between(self, *args, **kwargs):
        self.fill_between_calls.append((args, kwargs))

    def text(self, *args, **kwargs):
        self.text_calls.append((args, kwargs))

    def legend(self, *args, **kwargs):
        self.legend_calls.append((args, kwargs))

    def set_title(self, *args, **kwargs):
        self.title_calls.append((args, kwargs))

    def set_xlabel(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_xlim(self, *args, **kwargs):
        self.xlim = (args, kwargs)

    def grid(self, *args, **kwargs):
        pass

    def set_axisbelow(self, value):
        self.axisbelow = value

    def tick_params(self, **kwargs):
        self.tick_kwargs.update(kwargs)

    def get_xaxis_transform(self):
        return object()


class DummyFigure:
    def __init__(self):
        self.saved_paths = []

    def savefig(self, path, *args, **kwargs):
        self.saved_paths.append(path)


class DummyPlotlyScatter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyPlotlyFigure:
    def __init__(self):
        self.traces = []
        self.layout = {}
        self.xaxes = {}
        self.yaxes = {}
        self.annotations = []

    def add_trace(self, trace):
        self.traces.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)

    def update_xaxes(self, **kwargs):
        self.xaxes.update(kwargs)

    def update_yaxes(self, **kwargs):
        self.yaxes.update(kwargs)

    def add_annotation(self, **kwargs):
        self.annotations.append(kwargs)

    def to_html(self, *args, **kwargs):
        return "html"

    def write_html(self, path, *args, **kwargs):
        Path(path).write_text("html", encoding="utf-8")


class DummyPlotlyGo:
    Figure = DummyPlotlyFigure
    Scatter = DummyPlotlyScatter


def make_app_without_tk():
    app = object.__new__(MotionAnalysisApp)
    app.lang = get_language("ko")
    app.data_validator = DataValidator()
    app.dataframes = []
    app.file_paths = []
    app.analysis_results = {"stale": object()}
    app._update_progress = lambda *args, **kwargs: None
    app._set_text_content = lambda *args, **kwargs: None
    app._display_validation_results = lambda results: setattr(app, "validation_results", results)
    app.root = SimpleNamespace(after=lambda *args, **kwargs: None)
    app.file_label = DummyWidget()
    app.file_info_var = DummyVar()
    app.analyze_button = DummyWidget()
    app.save_button = DummyWidget()
    app.var_listbox = DummyWidget()
    app.select_listbox = DummyWidget()
    app.graph_listbox = DummyWidget()
    app.results_text = DummyWidget()
    return app


def make_gait_cycle_dataframe():
    state = [0] * 5 + [4] * 5 + [0] * 10 + [4] * 5 + [0] * 10 + [4] * 5 + [0] * 10
    return pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "value_r": np.arange(len(state), dtype=float),
        "leg1_r.state": state,
    })


def make_overview_dataframe():
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 2 * np.pi, len(state))
    return pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "hip_flexion_r": np.sin(x),
        "knee_angle_r": np.cos(x),
        "ankle_angle_r": np.sin(x * 1.5),
        "hip_r.load": np.abs(np.sin(x)),
        "leg1_r.grf_norm_y": np.abs(np.sin(x)),
    })


def make_muscle_cci_dataframe():
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 2 * np.pi, len(state))
    base = np.clip(0.45 + 0.35 * np.sin(x), 0.0, None)
    data = {
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "soleus_r.activation": base,
        "soleus_r.excitation": np.clip(base + 0.05, 0.0, None),
        "soleus_r.input": np.clip(base - 0.03, 0.0, None),
    }
    for idx, muscle in enumerate(["glut_max", "hamstrings", "iliopsoas", "vasti", "rect_fem", "bifemsh", "gastroc", "soleus", "tib_ant"]):
        data[f"{muscle}_r.activation"] = np.clip(base + 0.2 + idx * 0.02, 0.0, None)
    return pd.DataFrame(data)


def make_grf_cop_com_dataframe():
    state = [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8
    state = state + [0] * 12 + [1] * 12 + [2] * 8 + [3] * 12 + [4] * 8 + [0] * 12
    x = np.linspace(0, 4 * np.pi, len(state))
    gait = np.linspace(0, 2 * np.pi, len(state))
    vertical = 0.7 + 0.3 * np.sin(x) ** 2 + 0.12 * np.sin(2 * x)
    return pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "leg1_r.state": state,
        "leg0_l.state": state,
        "leg1_r.grf_norm_x": 0.18 * np.sin(gait),
        "leg1_r.grf_norm_y": vertical,
        "leg1_r.grf_norm_z": 0.03 * np.cos(gait),
        "leg0_l.grf_norm_x": -0.16 * np.sin(gait),
        "leg0_l.grf_norm_y": vertical * 0.95,
        "leg0_l.grf_norm_z": -0.02 * np.cos(gait),
        "leg1_r.cop_x": np.linspace(0.0, 0.3, len(state)),
        "leg1_r.cop_y": np.zeros(len(state)),
        "leg1_r.cop_z": 0.03 * np.sin(gait),
        "com_x": np.linspace(0.03, 0.33, len(state)),
        "com_y": 0.9 + 0.02 * np.sin(gait),
        "com_z": 0.02 * np.cos(gait),
        "lin_mom_x": np.sin(gait),
        "ang_mom_z": np.sin(gait * 0.5) * 0.1,
        "ankle_r.torque": np.sin(gait) * 0.4,
        "knee_r.torque": np.sin(gait + 0.4) * 0.25,
        "hip_r.torque": np.sin(gait + 0.8) * 0.15,
    })


def write_app_feature_csvs(tmp_path):
    pd.DataFrame([
        {"condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "max", "value": 10.0},
        {"condition": "W80", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "max", "value": 12.0},
    ]).to_csv(tmp_path / "cycle_features.csv", index=False)
    pd.DataFrame([
        {"condition": "W70", "variable": "knee_angle_r", "phase": "stance", "feature": "mean", "value": 3.0},
        {"condition": "W70", "variable": "knee_angle_r", "phase": "swing", "feature": "mean", "value": 6.0},
    ]).to_csv(tmp_path / "phase_features.csv", index=False)
    pd.DataFrame([
        {"condition": "W80", "reference_condition": "W70", "variable": "knee_angle_r", "phase": "whole_cycle", "feature": "rmse", "condition_value": 0.2, "delta": 0.2, "delta_percent": 2.0, "delta_timing": 1.0},
    ]).to_csv(tmp_path / "comparison_features.csv", index=False)


def test_check_time_coverage_uses_duration_not_absolute_max_time():
    app = make_app_without_tk()

    short_shifted_time = pd.DataFrame({"time": [10.0, 11.0, 12.0], "value": [1, 2, 3]})
    long_time = pd.DataFrame({"time": [10.0, 12.5, 15.0], "value": [1, 2, 3]})

    assert app._check_time_coverage(short_shifted_time) is False
    assert app._check_time_coverage(long_time) is True


def test_result_variable_help_text_documents_saved_outputs():
    help_text = get_result_variable_help_text()

    assert "데이터 저장 결과 변수 설명" in help_text
    assert "Symmetry Index" in help_text
    assert "cycle_features.csv" in help_text
    assert "condition_comparison_summary.csv" in help_text


def test_variable_unit_help_text_documents_display_units():
    help_text = get_variable_unit_help_text()

    assert "변수 단위 설명" in help_text
    assert "degree(deg)" in help_text
    assert "pelvis.ori_*" in help_text
    assert "torso.ori_*" in help_text
    assert "knee_angle_r 원본값 1.5708 rad" in help_text
    assert "*.load, *_load" in help_text
    assert "BW" in help_text
    assert "*.force, *_force" in help_text
    assert "N" in help_text
    assert "원본 부호를 유지" in help_text


def test_check_time_coverage_can_be_disabled_from_ui_setting():
    app = make_app_without_tk()
    app.min_time_filter_enabled_var = SimpleNamespace(get=lambda: False)
    short_time = pd.DataFrame({"time": [10.0, 11.0, 12.0], "value": [1, 2, 3]})

    assert app._check_time_coverage(short_time) is True


def test_check_time_coverage_uses_ui_minimum_seconds():
    app = make_app_without_tk()
    app.min_time_filter_enabled_var = SimpleNamespace(get=lambda: True)
    app.min_time_coverage_var = SimpleNamespace(get=lambda: "2.5")
    too_short = pd.DataFrame({"time": [0.0, 1.0, 2.0], "value": [1, 2, 3]})
    long_enough = pd.DataFrame({"time": [0.0, 1.5, 3.0], "value": [1, 2, 3]})

    assert app._check_time_coverage(too_short) is False
    assert app._check_time_coverage(long_enough) is True


def test_load_files_keeps_paths_aligned_with_loaded_dataframes(tmp_path, monkeypatch):
    invalid_path = tmp_path / "missing.csv"
    valid_path = tmp_path / "valid.csv"
    valid_path.write_text(
        "time,value\n" + "\n".join(f"{i / 10:.1f},{i}" for i in range(10)),
        encoding="utf-8",
    )

    app = make_app_without_tk()
    monkeypatch.setattr(
        motion_app_module.filedialog,
        "askopenfilenames",
        lambda **kwargs: (str(invalid_path), str(valid_path)),
    )
    monkeypatch.setattr(motion_app_module.messagebox, "showerror", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.load_files()

    assert app.file_paths == [str(valid_path)]
    assert len(app.dataframes) == 1
    assert app.analysis_results == {}
    assert app.var_listbox.items == ["value"]


def test_load_files_shows_union_of_variables_across_files(tmp_path, monkeypatch):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    first_path.write_text(
        "time,common,only_first\n" + "\n".join(f"{i / 10:.1f},{i},{i + 10}" for i in range(10)),
        encoding="utf-8",
    )
    second_path.write_text(
        "time,common,only_second\n" + "\n".join(f"{i / 10:.1f},{i},{i + 20}" for i in range(10)),
        encoding="utf-8",
    )

    app = make_app_without_tk()
    monkeypatch.setattr(
        motion_app_module.filedialog,
        "askopenfilenames",
        lambda **kwargs: (str(first_path), str(second_path)),
    )
    monkeypatch.setattr(motion_app_module.messagebox, "showerror", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.load_files()

    assert app.file_paths == [str(first_path), str(second_path)]
    assert app.var_listbox.items == ["common", "only_first", "only_second"]


def test_apply_auto_candidates_populates_selected_and_graph_variables():
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "leg1_r.state": [0, 1, 2, 3, 4] * 4,
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
    })]
    captured = {}
    app._set_text_content = lambda _widget, content: captured.setdefault("text", content)

    plan = app.apply_auto_candidates(show_message=False)

    assert plan is not None
    assert app.select_listbox.items == ["knee_angle_l", "knee_angle_r"]
    assert app.graph_listbox.items == ["knee_angle_l", "knee_angle_r"]
    assert "자동 분석 후보" in captured["text"]
    assert app.quick_candidate_listbox.items == ["knee_angle_l", "knee_angle_r"]
    assert app.quick_candidate_listbox.curselection() == (0, 1)


def test_apply_feature_candidates_uses_selected_feature_filter(monkeypatch):
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
        "hip_r.torque": np.linspace(0.0, 2.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint load / torque / force")
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    plan = app.apply_feature_candidates()

    assert plan.feature_filter == "Joint load / torque / force"
    assert app.select_listbox.items == ["hip_r.torque"]
    assert app.graph_listbox.items == ["hip_r.torque"]


def test_quick_workflow_uses_checked_candidate_subset(monkeypatch):
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint angle")
    app.quick_run_analysis_var = SimpleNamespace(get=lambda: False)
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: True)
    app.quick_show_overview_var = SimpleNamespace(get=lambda: False)
    app.quick_create_dashboard_var = SimpleNamespace(get=lambda: False)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {"plot_graph": 0}
    app.plot_graph = lambda: calls.__setitem__("plot_graph", calls["plot_graph"] + 1)

    app.apply_auto_candidates(show_message=False, feature_filter="Joint angle")
    app.quick_candidate_listbox.selection_clear(0, "end")
    app.quick_candidate_listbox.selection_set(1)
    app.run_quick_workflow_thread()

    assert app.select_listbox.items == ["knee_angle_r"]
    assert app.graph_listbox.items == ["knee_angle_r"]
    assert calls["plot_graph"] == 1


def test_quick_workflow_limits_waveform_variables(monkeypatch):
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    data = {"time": np.linspace(0.0, 6.0, 20)}
    for idx in range(35):
        data[f"value_{idx:02d}"] = np.linspace(0.0, 1.0, 20) + idx
    app.dataframes = [pd.DataFrame(data)]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "General numeric")
    app.quick_run_analysis_var = SimpleNamespace(get=lambda: False)
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: True)
    app.quick_show_overview_var = SimpleNamespace(get=lambda: False)
    app.quick_create_dashboard_var = SimpleNamespace(get=lambda: False)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    app.plot_graph = lambda: None

    app.run_quick_workflow_thread()

    assert len(app.select_listbox.items) == 35
    assert len(app.graph_listbox.items) == 30


def test_quick_recommended_visualization_runs_analysis_when_no_results(monkeypatch):
    app = make_app_without_tk()
    app.analysis_results = {}
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint angle")
    app.quick_run_analysis_var = SimpleNamespace(get=lambda: False)
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: False)
    app.quick_show_overview_var = SimpleNamespace(get=lambda: False)
    app.quick_create_dashboard_var = SimpleNamespace(get=lambda: True)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {}
    app._run_analysis_plan = lambda plan, title, selected_variables=None: calls.setdefault(
        "selected_variables",
        selected_variables,
    )

    app.run_quick_workflow_thread()

    assert calls["selected_variables"] == ["knee_angle_r"]


def test_feature_filter_change_updates_candidates_and_visual_defaults():
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "soleus_r.activation": np.linspace(0.0, 0.8, 20),
    })]
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("Muscle activation")
    app.quick_run_analysis_var = DummyVar()
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_overview_var = DummyVar()
    app.quick_create_dashboard_var = DummyVar()
    app.graph_x_axis_var = DummyVar()
    app._set_text_content = lambda *_args, **_kwargs: None

    app._on_feature_filter_changed()

    assert app.quick_candidate_listbox.items == ["soleus_r.activation"]
    assert app.quick_run_analysis_var.get() is True
    assert app.quick_show_waveform_var.get() is True
    assert app.quick_show_overview_var.get() is False
    assert app.graph_x_axis_var.get() == "percent gait cycle"


def test_analysis_purpose_change_maps_to_feature_filter_and_candidates():
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "soleus_r.activation": np.linspace(0.0, 0.8, 20),
    })]
    app.analysis_purpose_var = DummyVar()
    app.analysis_purpose_var.set("근활성 분석")
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("All analyzable")
    app.quick_run_analysis_var = DummyVar()
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_overview_var = DummyVar()
    app.quick_create_dashboard_var = DummyVar()
    app.graph_x_axis_var = DummyVar()
    app._set_text_content = lambda *_args, **_kwargs: None

    app._on_analysis_purpose_changed()

    assert app.feature_analysis_filter_var.get() == "Muscle activation"
    assert app.quick_candidate_listbox.items == ["soleus_r.activation"]
    assert app.quick_show_waveform_var.get() is True
    assert app.quick_show_overview_var.get() is False
    assert app.quick_create_dashboard_var.get() is False


def test_motion_app_uses_central_analysis_purpose_options():
    assert motion_app_module.ANALYSIS_PURPOSE_OPTIONS is ANALYSIS_PURPOSE_OPTIONS


def test_dynamics_purpose_maps_to_joint_load_filter_and_candidates():
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "hip_r.torque": np.linspace(0.0, 2.0, 20),
        "back.torque": np.linspace(0.0, 1.0, 20),
        "soleus_r.activation": np.linspace(0.0, 0.8, 20),
    })]
    app.analysis_purpose_var = DummyVar()
    app.analysis_purpose_var.set("역학/관절 토크 분석")
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("All analyzable")
    app.quick_run_analysis_var = DummyVar()
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_overview_var = DummyVar()
    app.quick_create_dashboard_var = DummyVar()
    app.graph_x_axis_var = DummyVar()
    app._set_text_content = lambda *_args, **_kwargs: None

    app._on_analysis_purpose_changed()

    assert app.feature_analysis_filter_var.get() == "Joint load / torque / force"
    assert app.quick_candidate_listbox.items == ["back.torque", "hip_r.torque"]
    assert app.quick_show_waveform_var.get() is True
    assert app.quick_show_overview_var.get() is False
    assert app.quick_create_dashboard_var.get() is False
    assert app.graph_x_axis_var.get() == "percent gait cycle"


def test_whole_report_purpose_disables_waveform_and_overview_windows():
    app = make_app_without_tk()
    app.analysis_purpose_var = DummyVar()
    app.analysis_purpose_var.set("전체 자동 리포트")
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("All analyzable")
    app.quick_run_analysis_var = DummyVar()
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_overview_var = DummyVar()
    app.quick_create_dashboard_var = DummyVar()
    app.graph_x_axis_var = DummyVar()

    app._apply_feature_visualization_defaults()

    assert app.quick_run_analysis_var.get() is True
    assert app.quick_show_waveform_var.get() is False
    assert app.quick_show_overview_var.get() is False
    assert app.quick_create_dashboard_var.get() is True


def test_recommended_run_whole_report_runs_analysis_without_overview(monkeypatch):
    app = make_app_without_tk()
    app.analysis_results = {}
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "soleus_r.activation": np.linspace(0.0, 0.8, 20),
    })]
    app.analysis_purpose_var = DummyVar()
    app.analysis_purpose_var.set("전체 자동 리포트")
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("All analyzable")
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_waveform_var.set(False)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {}
    app._run_analysis_plan = lambda plan, title, selected_variables=None: calls.update({
        "title": title,
        "selected_variables": selected_variables,
    })

    app.run_quick_workflow_thread()

    assert calls["title"] == "추천 실행"
    assert set(calls["selected_variables"]) == {"knee_angle_r", "soleus_r.activation"}
    assert app.pending_quick_actions.get("overview") is None
    assert app.pending_quick_actions.get("dashboard") is None
    assert app.pending_quick_actions.get("recommended_visualization") is None
    assert app.pending_quick_actions.get("waveform") is None
    assert app.graph_listbox.items == []


def test_recommended_run_muscle_uses_recommended_visualization(monkeypatch):
    app = make_app_without_tk()
    app.analysis_results = {}
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "soleus_r.activation": np.linspace(0.0, 0.8, 20),
    })]
    app.analysis_purpose_var = DummyVar()
    app.analysis_purpose_var.set("근활성 분석")
    app.feature_analysis_filter_var = DummyVar()
    app.feature_analysis_filter_var.set("Muscle activation")
    app.quick_show_waveform_var = DummyVar()
    app.quick_show_waveform_var.set(True)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {}
    app._run_analysis_plan = lambda plan, title, selected_variables=None: calls.update({
        "title": title,
        "selected_variables": selected_variables,
    })

    app.run_quick_workflow_thread()

    assert calls["title"] == "추천 실행"
    assert calls["selected_variables"] == ["soleus_r.activation"]
    assert app.pending_quick_actions["recommended_visualization"] is True
    assert app.pending_quick_actions.get("dashboard") is None


def test_save_selected_candidate_results_runs_analysis_when_needed(monkeypatch):
    app = make_app_without_tk()
    app.analysis_results = {}
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint angle")
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: True)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {}
    app._run_analysis_plan = lambda plan, title, selected_variables=None: calls.setdefault(
        "selected_variables",
        selected_variables,
    )

    app.save_selected_candidate_results()

    assert app.pending_quick_actions["save_results"] is True
    assert calls["selected_variables"] == ["knee_angle_r"]


def test_save_selected_candidate_visualizations_runs_analysis_when_needed(monkeypatch):
    app = make_app_without_tk()
    app.analysis_results = {}
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint angle")
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: True)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {}
    app._run_analysis_plan = lambda plan, title, selected_variables=None: calls.update({
        "title": title,
        "selected_variables": selected_variables,
    })

    app.save_selected_candidate_visualizations()

    assert app.pending_quick_actions["save_recommended_visualization"] is True
    assert calls["title"] == "추천 시각화 저장"
    assert calls["selected_variables"] == ["knee_angle_r"]


def test_visualize_selected_candidates_uses_waveform_without_overview(monkeypatch):
    app = make_app_without_tk()
    app.quick_candidate_listbox = DummyWidget()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 20),
        "knee_angle_r": np.linspace(0.0, 1.0, 20),
        "knee_angle_l": np.linspace(1.0, 0.0, 20),
    })]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Joint angle")
    app.quick_show_waveform_var = SimpleNamespace(get=lambda: True)
    app._set_text_content = lambda *_args, **_kwargs: None
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)
    calls = {"graph": 0, "overview_scope": None}
    app.plot_graph = lambda: calls.__setitem__("graph", calls["graph"] + 1)
    app.plot_overview_figure = lambda scope_variables=None: calls.__setitem__("overview_scope", scope_variables)

    app.apply_auto_candidates(show_message=False, feature_filter="Joint angle")
    app.quick_candidate_listbox.selection_clear(0, "end")
    app.quick_candidate_listbox.selection_set(1)
    app.visualize_selected_candidates()

    assert calls["graph"] == 1
    assert calls["overview_scope"] is None
    assert app.graph_listbox.items == ["knee_angle_r"]


def test_recommended_muscle_visualization_shows_summary_without_duplicate_activation_waveform():
    app = make_app_without_tk()
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Muscle activation")
    calls = {}
    app._select_quick_graph_variables = lambda selected, plan: []
    app.plot_graph = lambda: calls.__setitem__("graph", True)
    app.plot_overview_figure = lambda scope_variables=None: calls.__setitem__("overview", scope_variables)
    app.plot_muscle_activation_figure = (
        lambda scope_variables=None: calls.__setitem__("activation_scope", scope_variables)
    )
    app.plot_activation_summary_figure = (
        lambda scope_variables=None: calls.__setitem__("summary_scope", scope_variables)
    )

    app._run_recommended_visualization(["soleus_r.activation"])

    assert calls["summary_scope"] == ["soleus_r.activation"]
    assert "activation_scope" not in calls
    assert "overview" not in calls
    assert "graph" not in calls


def test_analyze_data_without_state_or_pair_keeps_existing_analysis():
    app = make_app_without_tk()
    app.root = SimpleNamespace(after=lambda delay, callback: callback())
    app.file_paths = ["sample.csv"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 6.0, 30),
        "value": np.linspace(0.0, 1.0, 30),
    })]

    app.analyze_data(["value"], threshold_value=0.5)

    result = app.analysis_results["0_sample.csv_value"]
    assert result["overall_statistics"]["count"] == 30
    assert "time_phase_statistics" not in result
    assert result["scone_phase_statistics"] == {}
    assert result["symmetry_features"] == {}


def test_finish_analysis_prepends_summary_and_writes_run_outputs(tmp_path):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto", "trial_W80.sto"]
    app.dataframes = [
        pd.DataFrame({"time": np.linspace(0.0, 6.0, 20), "knee_angle_r": np.linspace(0.0, 1.0, 20)}),
        pd.DataFrame({"time": np.linspace(0.0, 6.0, 20), "knee_angle_r": np.linspace(0.0, 2.0, 20)}),
    ]
    app.select_listbox.items = ["knee_angle_r"]
    app.current_run_output_dir = str(tmp_path / "run")
    app.auto_feature_csv_enabled = True
    app.analysis_purpose_var = SimpleNamespace(get=lambda: "보행 각도 분석")
    app.feature_csv_var = DummyVar()
    app.feature_reference_var = DummyVar()
    app.feature_csv_combo = DummyWidget()
    app.feature_reference_combo = DummyWidget()

    first = analyze_variable_comprehensive(
        app.dataframes[0]["knee_angle_r"].to_numpy(dtype=float),
        app.dataframes[0]["time"].to_numpy(dtype=float),
        "knee_angle_r",
    )
    second = analyze_variable_comprehensive(
        app.dataframes[1]["knee_angle_r"].to_numpy(dtype=float),
        app.dataframes[1]["time"].to_numpy(dtype=float),
        "knee_angle_r",
    )

    app._finish_analysis(
        "상세 결과",
        {
            "0_trial_W70.sto_knee_angle_r": first,
            "1_trial_W80.sto_knee_angle_r": second,
        },
    )

    assert app.results_data.startswith("=== 자동 요약 ===")
    assert "조건 비교 하이라이트" in app.results_data
    assert (tmp_path / "run" / "features" / "cycle_features.csv").exists()
    assert (tmp_path / "run" / "features" / "condition_comparison_summary.csv").exists()
    assert (tmp_path / "run" / "reports" / "analysis_summary.md").exists()


def test_save_results_writes_scone_phase_statistics(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.select_listbox.items = ["value_r"]
    app.dataframes = [pd.DataFrame({"value_r": [0.0, 1.0]})]
    app.file_paths = ["sample.csv"]

    data = np.arange(10, dtype=float)
    time = np.arange(10, dtype=float)
    state = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    app.analysis_results = {
        "0_sample.csv_value_r": analyze_variable_comprehensive(
            data,
            time,
            "value_r",
            state_arr=state,
            state_column_name="leg1_r.state",
        )
    }

    save_path = tmp_path / "results.csv"
    monkeypatch.setattr(
        motion_app_module.filedialog,
        "asksaveasfilename",
        lambda **kwargs: str(save_path),
    )
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_results()

    saved = pd.read_csv(save_path)
    overall_row = saved[saved["분석유형"] == "전체 데이터"].iloc[0]
    assert overall_row["변수군"] == "General numeric"
    assert overall_row["추천 Preset"] == "basic statistics, peak/timing, AUC, threshold"
    scone_rows = saved[saved["분석유형"].str.startswith("SCONE phase")]
    assert "SCONE phase - EarlyStance" in set(scone_rows["분석유형"])
    assert "SCONE phase - stance" in set(scone_rows["분석유형"])
    stance_row = scone_rows[scone_rows["분석유형"] == "SCONE phase - stance"].iloc[0]
    assert stance_row["RMS"] > 0
    assert stance_row["AUC"] == 12.5
    assert stance_row["상태컬럼"] == "leg1_r.state"
    assert (tmp_path / "cycle_features.csv").exists()
    assert (tmp_path / "phase_features.csv").exists()
    assert (tmp_path / "comparison_features.csv").exists()


def test_save_graphs_writes_raw_graph_file(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.graph_listbox.items = ["value"]
    app.file_paths = ["sample.csv"]
    app.dataframes = [pd.DataFrame({"value": [1.0, 2.0, 3.0, np.nan]})]

    fig = DummyFigure()
    ax = DummyAxes()
    monkeypatch.setattr(motion_app_module.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(motion_app_module.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    app.save_graphs()

    assert len(fig.saved_paths) == 2
    assert fig.saved_paths[0].endswith("_graph.png")
    assert fig.saved_paths[1].endswith("_graph.svg")


def test_save_overview_figure_writes_png(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto"]
    app.dataframes = [make_overview_dataframe()]
    app.overview_preset_var = SimpleNamespace(get=lambda: "Gait overview")
    app.overview_band_var = SimpleNamespace(get=lambda: "mean ± SD")
    app.reference_condition_var = SimpleNamespace(get=lambda: "first file")

    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_overview_figure()

    assert (tmp_path / "gait_overview.png").exists()
    assert (tmp_path / "gait_overview.png").stat().st_size > 0


def test_save_overview_html_writes_html(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto"]
    app.dataframes = [make_overview_dataframe()]
    app.overview_preset_var = SimpleNamespace(get=lambda: "Gait overview")
    app.overview_band_var = SimpleNamespace(get=lambda: "mean ± SD")
    app.reference_condition_var = SimpleNamespace(get=lambda: "first file")

    monkeypatch.setattr(app, "_get_plotly_go", lambda: DummyPlotlyGo)
    monkeypatch.setattr(
        motion_app_module,
        "create_overview_plotly_figure",
        lambda *args, **kwargs: (DummyPlotlyFigure(), True, []),
    )
    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_overview_html()

    assert (tmp_path / "gait_overview.html").exists()


def test_refresh_feature_csv_candidates_finds_feature_results(tmp_path):
    write_app_feature_csvs(tmp_path)
    app = make_app_without_tk()
    app.file_paths = [str(tmp_path / "sample.sto")]
    app.feature_csv_var = DummyVar()
    app.feature_reference_var = DummyVar()
    app.feature_csv_var.set("")
    app.feature_reference_var.set("first condition")
    app.feature_csv_combo = DummyWidget()
    app.feature_reference_combo = DummyWidget()

    paths = app.refresh_feature_csv_candidates()

    assert len(paths) == 3
    assert app.feature_csv_var.get() == "auto: all feature CSVs"
    assert "W70" in app.feature_reference_combo.config_kwargs["values"]


def test_save_feature_csv_figure_writes_png(tmp_path, monkeypatch):
    write_app_feature_csvs(tmp_path)
    app = make_app_without_tk()
    app.file_paths = [str(tmp_path / "sample.sto")]
    app.feature_csv_paths = [str(tmp_path / "cycle_features.csv"), str(tmp_path / "phase_features.csv"), str(tmp_path / "comparison_features.csv")]
    app.feature_csv_var = SimpleNamespace(get=lambda: "auto: all feature CSVs")
    app.feature_plot_type_var = SimpleNamespace(get=lambda: "Variable x condition heatmap")
    app.feature_reference_var = SimpleNamespace(get=lambda: "W70")

    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_feature_csv_figure()

    assert (tmp_path / "variable_x_condition_heatmap.png").exists()


def test_save_cci_results_writes_png_and_integrated_csv(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto"]
    app.dataframes = [make_muscle_cci_dataframe()]
    app.muscle_side_var = SimpleNamespace(get=lambda: "right")

    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_cci_results()

    assert (tmp_path / "cci_summary.png").exists()
    assert (tmp_path / "integrated_cci.csv").exists()
    saved = pd.read_csv(tmp_path / "integrated_cci.csv")
    assert {"hip", "knee", "ankle"} <= set(saved["muscle_pair"])


def test_save_grf_cop_com_figure_writes_png(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto"]
    app.dataframes = [make_grf_cop_com_dataframe()]

    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_grf_cop_com_figure()

    assert (tmp_path / "grf_cop_com_overview.png").exists()
    assert (tmp_path / "grf_cop_com_overview.png").stat().st_size > 0


def test_save_recommended_visualization_files_writes_symmetry_outputs(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 4.0, 5),
        "knee_angle_l": [1.0, 2.0, 3.0, 2.0, 1.0],
        "knee_angle_r": [2.0, 4.0, 6.0, 4.0, 2.0],
    })]
    app.select_listbox.items = ["knee_angle_l", "knee_angle_r"]
    app.feature_analysis_filter_var = SimpleNamespace(get=lambda: "Symmetry pairs")

    left_result = analyze_variable_comprehensive(
        app.dataframes[0]["knee_angle_l"].to_numpy(dtype=float),
        app.dataframes[0]["time"].to_numpy(dtype=float),
        "knee_angle_l",
    )
    right_result = analyze_variable_comprehensive(
        app.dataframes[0]["knee_angle_r"].to_numpy(dtype=float),
        app.dataframes[0]["time"].to_numpy(dtype=float),
        "knee_angle_r",
    )
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
    app.analysis_results = {
        "0_trial.sto_knee_angle_l": left_result,
        "0_trial.sto_knee_angle_r": right_result,
    }

    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_recommended_visualization_files(
        selected_variables=["knee_angle_l", "knee_angle_r"],
        feature_filter="Symmetry pairs",
    )

    assert (tmp_path / "recommended_symmetry_index.csv").exists()
    assert (tmp_path / "recommended_symmetry_index.png").exists()
    saved_symmetry = pd.read_csv(tmp_path / "recommended_symmetry_index.csv")
    assert len(saved_symmetry) == 8


def test_save_dashboard_writes_html_and_summary(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.file_paths = ["trial_W70.sto"]
    app.dataframes = [make_grf_cop_com_dataframe()]
    app.select_listbox.items = ["leg1_r.grf_norm_y"]
    app.graph_listbox.items = ["leg1_r.grf_norm_y"]
    result = analyze_variable_comprehensive(
        app.dataframes[0]["leg1_r.grf_norm_y"].to_numpy(dtype=float),
        app.dataframes[0]["time"].to_numpy(dtype=float),
        "leg1_r.grf_norm_y",
        state_arr=app.dataframes[0]["leg1_r.state"].to_numpy(dtype=float),
        state_column_name="leg1_r.state",
    )
    app.analysis_results = {"0_trial_W70.sto_leg1_r.grf_norm_y": result}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.messagebox, "showwarning", lambda *args, **kwargs: None)

    app.save_dashboard()

    dashboard_path = tmp_path / "output" / "html" / "dashboard.html"
    summary_path = tmp_path / "output" / "reports" / "visualization_summary.md"
    assert dashboard_path.exists()
    assert summary_path.exists()
    html = dashboard_path.read_text(encoding="utf-8")
    assert "GRF/COP/COM" in html
    assert "Warnings and skipped items" in html


def test_get_normalized_gait_cycles_uses_state_boundaries():
    app = make_app_without_tk()
    df = make_gait_cycle_dataframe()

    state_col, normalized_cycles = app._get_normalized_gait_cycles(df, "value_r")

    assert state_col == "leg1_r.state"
    assert normalized_cycles.shape == (3, 101)


def test_get_normalized_gait_cycles_tolerates_invalid_state_values():
    app = make_app_without_tk()
    df = make_gait_cycle_dataframe()
    df.loc[12, "leg1_r.state"] = np.nan
    df.loc[20, "leg1_r.state"] = 99

    state_col, normalized_cycles = app._get_normalized_gait_cycles(df, "value_r")

    assert state_col == "leg1_r.state"
    assert normalized_cycles.shape == (3, 101)


def test_get_normalized_gait_cycles_includes_initial_early_stance_boundary():
    app = make_app_without_tk()
    state = [0] * 11 + [1] * 26 + [2] * 9 + [3] * 25 + [4] * 23 + [0] * 35 + [1] * 20
    df = pd.DataFrame({
        "time": np.arange(len(state)) / 100,
        "knee_angle_r": np.arange(len(state), dtype=float),
        "leg1_r.state": state,
    })

    state_col, normalized_cycles = app._get_normalized_gait_cycles(df, "knee_angle_r")

    assert state_col == "leg1_r.state"
    assert normalized_cycles.shape == (1, 101)


def test_get_normalized_gait_cycles_rejects_incomplete_cycle_without_transitions():
    app = make_app_without_tk()
    df = pd.DataFrame({
        "time": np.arange(11) / 100,
        "value_r": np.arange(11, dtype=float),
        "leg1_r.state": [0] * 11,
    })

    state_col, normalized_cycles = app._get_normalized_gait_cycles(df, "value_r")

    assert state_col == "leg1_r.state"
    assert normalized_cycles.shape == (0, 101)


def test_plot_time_phase_graph_draws_normalized_gait_cycle(monkeypatch):
    app = make_app_without_tk()
    app.graph_listbox.items = ["value_r"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [make_gait_cycle_dataframe()]
    app._show_plot_window = lambda *args, **kwargs: None

    fig = DummyFigure()
    ax = DummyAxes()
    monkeypatch.setattr(motion_app_module.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(motion_app_module.plt, "tight_layout", lambda *args, **kwargs: None)

    app.plot_time_phase_graph()

    assert len(ax.plot_calls) == 6
    x_percent = ax.plot_calls[3][0][0]
    assert x_percent[0] == 0
    assert x_percent[-1] == 100
    assert ax.xlim[0] == (0, 100)
    assert ax.plot_calls[3][1]["label"] == "sample  n=3"
    assert len(ax.fill_between_calls) == 1
    assert len(ax.vlines) == 3
    assert len(ax.spans) > 0
    assert any(call[1].get("marker") == "^" for call in ax.plot_calls)
    assert any(call[1].get("marker") == "v" for call in ax.plot_calls)
    assert not any("max" in call[0][2] for call in ax.text_calls if len(call[0]) >= 3)
    assert not any("min" in call[0][2] for call in ax.text_calls if len(call[0]) >= 3)
    assert ax.spines["top"].visible is False
    assert ax.axisbelow is True
    assert ax.title_calls[-1][1]["loc"] == "center"
    assert ax.legend_calls[-1][1]["loc"] == "center left"
    assert ax.legend_calls[-1][1]["bbox_to_anchor"] == (1.02, 0.5)


def test_save_time_phase_graphs_draws_normalized_gait_cycle_and_saves(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.graph_listbox.items = ["value_r"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [make_gait_cycle_dataframe()]

    fig = DummyFigure()
    ax = DummyAxes()
    monkeypatch.setattr(motion_app_module.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(motion_app_module.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    app.save_time_phase_graphs()

    assert len(ax.plot_calls) == 6
    assert len(fig.saved_paths) == 1
    assert fig.saved_paths[0].endswith("_normalized_gait_cycle.png")


def test_plot_graph_can_use_percent_gait_cycle_axis(monkeypatch):
    app = make_app_without_tk()
    app.graph_x_axis_var = SimpleNamespace(get=lambda: "percent gait cycle")
    app.graph_listbox.items = ["value_r"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [make_gait_cycle_dataframe()]
    app._show_plot_window = lambda *args, **kwargs: None

    fig = DummyFigure()
    ax = DummyAxes()
    monkeypatch.setattr(motion_app_module.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(motion_app_module.plt, "tight_layout", lambda *args, **kwargs: None)

    app.plot_graph()

    assert ax.xlim[0] == (0, 100)
    assert len(ax.spans) > 0
    assert any(call[1].get("marker") == "^" for call in ax.plot_calls)
    assert not any("max" in call[0][2] for call in ax.text_calls if len(call[0]) >= 3)


def test_plot_graph_time_axis_places_legend_on_right(monkeypatch):
    app = make_app_without_tk()
    app.graph_x_axis_var = SimpleNamespace(get=lambda: "time")
    app.graph_listbox.items = ["value"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [pd.DataFrame({
        "time": np.linspace(0.0, 1.0, 5),
        "value": np.linspace(0.0, 1.0, 5),
    })]
    app._show_plot_window = lambda *args, **kwargs: None

    fig = DummyFigure()
    ax = DummyAxes()
    monkeypatch.setattr(motion_app_module.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(motion_app_module.plt, "tight_layout", lambda *args, **kwargs: None)

    app.plot_graph()

    assert ax.legend_calls[-1][1]["loc"] == "center left"
    assert ax.legend_calls[-1][1]["bbox_to_anchor"] == (1.02, 0.5)
    assert ax.legend_calls[-1][1]["frameon"] is False


def test_create_interactive_graph_figure_adds_plotly_line_and_sd_band():
    app = make_app_without_tk()
    app.file_paths = ["sample.sto"]
    app.dataframes = [pd.DataFrame({"value": [1.0, 2.0, 3.0, np.nan]})]

    fig, has_data = app._create_interactive_graph_figure("value", DummyPlotlyGo)

    assert has_data is True
    assert len(fig.traces) == 2
    assert fig.traces[-1].kwargs["name"] == "sample"
    assert fig.layout["hovermode"] == "x unified"
    assert fig.layout["template"] == "simple_white"
    assert fig.layout["font"]["family"] == "Times New Roman"
    assert fig.layout["xaxis_title"] == "Frame / Index"
    assert fig.layout["xaxis"]["rangeslider"]["visible"] is True


def test_create_interactive_graph_figure_converts_joint_angles_to_degrees():
    app = make_app_without_tk()
    app.file_paths = ["P2BW_W70_Gait10Slope.sto"]
    app.dataframes = [pd.DataFrame({"knee_angle_r": [0.0, np.pi / 2.0, np.pi]})]

    fig, has_data = app._create_interactive_graph_figure("knee_angle_r", DummyPlotlyGo)

    assert has_data is True
    np.testing.assert_allclose(fig.traces[-1].kwargs["y"], [0.0, 90.0, 180.0])
    assert fig.traces[-1].kwargs["name"] == "P2BW_W70"
    assert fig.layout["yaxis_title"] == "Value (deg)"


def test_create_interactive_normalized_gait_cycle_figure_adds_plotly_line_and_sd_band():
    app = make_app_without_tk()
    app.file_paths = ["sample.sto"]
    app.dataframes = [make_gait_cycle_dataframe()]

    fig, has_data = app._create_interactive_normalized_gait_cycle_figure("value_r", DummyPlotlyGo)

    assert has_data is True
    assert len(fig.traces) > 2
    assert fig.traces[-1].kwargs["name"] == "sample"
    assert fig.xaxes["range"] == [0, 100]
    assert fig.layout["xaxis_title"] == "Gait Cycle (%)"
    assert len(fig.layout["shapes"]) == 3


def test_save_interactive_graphs_writes_html(tmp_path, monkeypatch):
    app = make_app_without_tk()
    app.graph_listbox.items = ["value"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [pd.DataFrame({"value": [1.0, 2.0, 3.0]})]

    monkeypatch.setattr(app, "_get_plotly_go", lambda: DummyPlotlyGo)
    monkeypatch.setattr(motion_app_module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(motion_app_module.messagebox, "showinfo", lambda *args, **kwargs: None)

    app.save_interactive_graphs()

    assert (tmp_path / "Value_interactive_graph.html").exists()


def test_plot_interactive_graph_opens_plotly_html(monkeypatch):
    app = make_app_without_tk()
    app.graph_listbox.items = ["value"]
    app.file_paths = ["sample.sto"]
    app.dataframes = [pd.DataFrame({"value": [1.0, 2.0, 3.0]})]
    opened = []

    monkeypatch.setattr(app, "_get_plotly_go", lambda: DummyPlotlyGo)
    monkeypatch.setattr(app, "_open_plotly_figure", lambda fig, filename_hint: opened.append(filename_hint))

    app.plot_interactive_graph()

    assert opened == ["Value_interactive_graph"]
