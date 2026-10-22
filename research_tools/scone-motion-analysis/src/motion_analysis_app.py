"""
동작 데이터 분석기 GUI 클래스 v3.0.

Tkinter 화면 구성, 사용자 입력 처리, 분석 실행, 그래프 표시와 저장 동작을
연결합니다. 계산 로직은 src/analysis_engine.py와 src/analysis_workflow.py로
분리되어 있습니다.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 디렉터리를 import 경로에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.constants import (
    VALIDATION_CONFIG, STATISTICS_CONFIG, GUI_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES,
    RESAMPLE_POINTS, SD_RANGE, STATE_COLS
)
from config.language import get_language
from utils.data_validator import DataValidator
from utils.math_utils import calculate_comprehensive_stats
from utils.plot_utils import setup_plot_style, get_font_family, get_visualization_font_family
from utils.file_utils import (
    format_legend_label,
    get_clean_variable_name,
    get_safe_filename_component,
)
from utils.unit_utils import convert_values_for_display, get_axis_label
from src.analysis_summary import (
    build_analysis_summary,
    save_condition_comparison_table,
)
from src.analysis_workflow import format_analysis_result_text, run_analysis_workflow
from src.auto_analysis import (
    ANALYSIS_PURPOSE_OPTIONS,
    FEATURE_ANALYSIS_OPTIONS,
    build_auto_analysis_plan,
    feature_filter_for_purpose,
    format_auto_analysis_plan,
    is_known_analysis_purpose,
)
from src.data_loading import read_motion_data_file
from src.result_export import (
    build_results_dataframe,
    save_feature_csv_bundle,
    save_results_dataframe,
)
from visualization.common import save_figure
from visualization.dashboard import create_visualization_dashboard
from visualization.plot_features import (
    FEATURE_PLOT_TYPES,
    create_feature_plot,
    discover_feature_csvs,
    get_conditions_from_feature_tables,
)
from visualization.plot_gait_cycle import draw_gait_cycle_waveform, normalize_to_gait_cycle
from visualization.plot_grf_cop_com import create_grf_cop_com_figure
from visualization.plot_muscle import (
    create_activation_summary_figure,
    create_cci_figure,
    create_muscle_activation_figure,
    save_integrated_cci_csv,
)
from visualization.plot_overview import (
    PANEL_PRESETS,
    create_overview_figure,
    create_overview_plotly_figure,
    get_condition_labels,
)
from visualization.plot_symmetry import build_symmetry_table, create_symmetry_figure
from visualization.plot_waveform import draw_raw_waveform


PLOTLY_COLORS = [
    '#0072B2', '#D55E00', '#009E73', '#CC79A7',
    '#E69F00', '#56B4E9', '#6A4C93', '#4D4D4D',
]
PLOTLY_TEMPLATE = 'simple_white'
PLOTLY_GRID_COLOR = 'rgba(30, 41, 59, 0.12)'
PLOTLY_ZERO_LINE_COLOR = 'rgba(30, 41, 59, 0.28)'
MAX_INDIVIDUAL_CYCLES = 25


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f'rgba({red}, {green}, {blue}, {alpha})'


class MotionAnalysisApp:
    def __init__(self, root, lang=None):
        self.root = root
        self.lang = lang or get_language('ko')
        
        self.root.title(self.lang['app_title'])
        window_size = GUI_CONFIG['window_size']
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")
        
        self.dataframes: List[pd.DataFrame] = []
        self.file_paths: List[str] = []
        self.results_data = ""
        self.selected_vars: List[str] = []
        self.graph_vars: List[str] = []
        self.last_fig = None
        self.all_columns: List[str] = []
        self.analysis_results: Dict = {}
        self.auto_analysis_plan = None
        self.auto_analysis_mode = False
        self.pending_quick_actions = None
        self.latest_feature_csv_dir = ""
        self.current_run_output_dir = ""
        self.auto_feature_csv_enabled = True
        
        self.data_validator = DataValidator()
        
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar()
        self.status_var.set("준비됨")
        self.threshold_var = tk.StringVar(value="0.05")
        self.min_time_filter_enabled_var = tk.BooleanVar(
            value=VALIDATION_CONFIG.get('enforce_min_time_coverage', True)
        )
        self.min_time_coverage_var = tk.StringVar(
            value=str(VALIDATION_CONFIG.get('min_time_coverage_seconds', 5.0))
        )
        self.graph_x_axis_var = tk.StringVar(value="time")
        self.overview_preset_var = tk.StringVar(value=PANEL_PRESETS[0])
        self.overview_band_var = tk.StringVar(value="mean ± SD")
        self.reference_condition_var = tk.StringVar(value="first file")
        self.feature_csv_var = tk.StringVar(value="")
        self.feature_plot_type_var = tk.StringVar(value=FEATURE_PLOT_TYPES[0])
        self.feature_reference_var = tk.StringVar(value="first condition")
        self.analysis_purpose_var = tk.StringVar(value=ANALYSIS_PURPOSE_OPTIONS[0])
        self.feature_analysis_filter_var = tk.StringVar(
            value=feature_filter_for_purpose(ANALYSIS_PURPOSE_OPTIONS[0])
        )
        self.quick_run_analysis_var = tk.BooleanVar(value=True)
        self.quick_show_waveform_var = tk.BooleanVar(value=True)
        self.quick_show_overview_var = tk.BooleanVar(value=False)
        self.quick_create_dashboard_var = tk.BooleanVar(value=False)
        self.feature_csv_paths: List[str] = []
        self.muscle_side_var = tk.StringVar(value="both")
        self.muscle_threshold_var = tk.StringVar(value="0.05")
        self._apply_feature_visualization_defaults()
        
        self._setup_styles()
        self._create_gui()
        setup_plot_style()
    
    def _is_min_time_filter_enabled(self) -> bool:
        source_var = getattr(self, "min_time_filter_enabled_var", None)
        if source_var is None:
            return bool(VALIDATION_CONFIG.get('enforce_min_time_coverage', True))
        try:
            return bool(source_var.get())
        except Exception:
            return bool(VALIDATION_CONFIG.get('enforce_min_time_coverage', True))

    def _get_min_time_coverage_seconds(self):
        source_var = getattr(self, "min_time_coverage_var", None)
        if source_var is None:
            return float(VALIDATION_CONFIG.get('min_time_coverage_seconds', 5.0))
        try:
            value = float(source_var.get())
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value >= 0 else None

    def _check_time_coverage(self, df: pd.DataFrame, min_time: float = None) -> bool:
        """사용자가 설정한 최소 시간 기준을 만족하는지 확인합니다."""
        try:
            if not self._is_min_time_filter_enabled():
                return True

            if min_time is None:
                min_time = self._get_min_time_coverage_seconds()
            if min_time is None:
                return False

            if 'time' not in df.columns:
                return True

            is_valid_time, _, time_info = self.data_validator.validate_time_column(df)
            if not is_valid_time:
                return False

            duration = time_info.get('duration', np.nan)
            if np.isnan(duration) or float(duration) < min_time:
                return False
            
            return True
        except Exception:
            return True  # 검증 실패 시 기존 흐름을 막지 않습니다.

    def _get_state_column_for_variable(self, df: pd.DataFrame, var: str):
        var_lower = var.lower()
        side_order = []

        if '_r' in var_lower or 'right' in var_lower or 'leg1_r' in var_lower:
            side_order.append('right')
        if '_l' in var_lower or 'left' in var_lower or 'leg0_l' in var_lower:
            side_order.append('left')

        for side in ('right', 'left'):
            if side not in side_order:
                side_order.append(side)

        for side in side_order:
            state_col = STATE_COLS.get(side)
            if state_col in df.columns:
                return state_col
        return None

    def _get_normalized_gait_cycles(self, df: pd.DataFrame, var: str):
        state_col, normalized_cycles, _ = normalize_to_gait_cycle(df, var, n_points=RESAMPLE_POINTS)
        return state_col, normalized_cycles

    def _get_gait_cycle_boundaries(self, state_arr: np.ndarray):
        if len(state_arr) == 0:
            return []

        try:
            raw_state = np.asarray(state_arr, dtype=float)
        except (TypeError, ValueError):
            return []

        state_values = np.full(raw_state.shape, -1, dtype=int)
        finite_indices = np.where(np.isfinite(raw_state))[0]
        if len(finite_indices) == 0:
            return []
        rounded = np.rint(raw_state[finite_indices]).astype(int)
        valid_mask = np.isin(rounded, [0, 1, 2, 3, 4])
        state_values[finite_indices[valid_mask]] = rounded[valid_mask]
        valid_indices = np.where(np.isin(state_values, [0, 1, 2, 3, 4]))[0]
        if len(valid_indices) == 0:
            return []

        boundaries = []
        first_idx = int(valid_indices[0])
        if state_values[first_idx] == 0:
            boundaries.append(first_idx)

        previous_state = int(state_values[first_idx])

        for i in valid_indices[1:]:
            current_state = int(state_values[i])
            if previous_state != 0 and current_state == 0:
                if not boundaries:
                    boundaries.append(int(i))
                    previous_state = current_state
                    continue

                cycle_length = i - (boundaries[-1] if boundaries else 0)
                if (
                    VALIDATION_CONFIG.get('min_cycle_length', 10)
                    <= cycle_length
                    <= VALIDATION_CONFIG.get('max_cycle_length', 1000)
                ):
                    boundaries.append(int(i))
            previous_state = current_state

        return boundaries

    def _draw_normalized_gait_cycle_graph(self, ax, var: str) -> bool:
        has_data, _ = draw_gait_cycle_waveform(
            ax,
            self.dataframes,
            self.file_paths,
            var,
            show_mean_sd=True,
            show_phase_shading=True,
            show_peak_markers=True,
        )
        return has_data
    
    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use(GUI_CONFIG.get('theme', 'clam'))
        
        # 시스템에 설치된 한글 지원 폰트를 선택합니다.
        font_family = get_font_family()
        
        font_size = GUI_CONFIG.get('font_size', 10)
        style.configure('Title.TLabel', font=(font_family, font_size + 2, 'bold'))
        style.configure('Status.TLabel', font=(font_family, font_size - 1))
    
    def _create_gui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_header_section(main_container)
        self._create_main_work_area(main_container)
        self._create_compact_button_section(main_container)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def _create_header_section(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        file_frame = ttk.LabelFrame(header_frame, text=self.lang['select_file'], padding="5")
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        file_top_frame = ttk.Frame(file_frame)
        file_top_frame.pack(fill=tk.X, pady=(0, 3))
        
        self.load_button = ttk.Button(file_top_frame, text=self.lang['open_file'], command=self.load_files)
        self.load_button.pack(side=tk.LEFT)
        
        self.file_info_var = tk.StringVar()
        self.file_info_label = ttk.Label(file_top_frame, textvariable=self.file_info_var, style='Status.TLabel')
        self.file_info_label.pack(side=tk.RIGHT)
        
        self.file_label = ttk.Label(file_frame, text=self.lang['file_not_selected'], foreground="grey", style='Status.TLabel')
        self.file_label.pack(anchor=tk.W)
        
        progress_frame = ttk.LabelFrame(header_frame, text="진행 상황", padding="5")
        progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        progress_top_frame = ttk.Frame(progress_frame)
        progress_top_frame.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_top_frame, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.status_label = ttk.Label(progress_top_frame, textvariable=self.status_var, style='Status.TLabel', width=15)
        self.status_label.pack(side=tk.RIGHT)
    
    def _create_main_work_area(self, parent):
        work_frame = ttk.Frame(parent)
        work_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self._create_variable_section(work_frame)
        self._create_results_section(work_frame)
    
    def _create_variable_section(self, parent):
        scroll_outer = ttk.Frame(parent)
        scroll_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        variable_canvas = tk.Canvas(scroll_outer, width=305, highlightthickness=0)
        variable_scrollbar = ttk.Scrollbar(scroll_outer, orient=tk.VERTICAL, command=variable_canvas.yview)
        variable_canvas.configure(yscrollcommand=variable_scrollbar.set)
        variable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        variable_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        var_container = ttk.Frame(variable_canvas)
        canvas_window = variable_canvas.create_window((0, 0), window=var_container, anchor=tk.NW)

        def _update_scroll_region(_event=None):
            variable_canvas.configure(scrollregion=variable_canvas.bbox("all"))

        def _resize_inner_frame(event):
            variable_canvas.itemconfigure(canvas_window, width=event.width)

        def _has_vertical_overflow():
            bbox = variable_canvas.bbox("all")
            if not bbox:
                return False
            content_height = bbox[3] - bbox[1]
            viewport_height = variable_canvas.winfo_height()
            return content_height > viewport_height + 1

        def _on_mousewheel(event):
            if not _has_vertical_overflow():
                return None
            variable_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _bind_mousewheel(_event):
            variable_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event):
            variable_canvas.unbind_all("<MouseWheel>")

        var_container.bind("<Configure>", _update_scroll_region)
        variable_canvas.bind("<Configure>", _resize_inner_frame)
        variable_canvas.bind("<Enter>", _bind_mousewheel)
        variable_canvas.bind("<Leave>", _unbind_mousewheel)

        self._create_quick_analysis_section(var_container)
        self.advanced_tools_frame = ttk.Frame(var_container)
        var_container = self.advanced_tools_frame
        
        vars_frame = ttk.LabelFrame(var_container, text="변수 관리", padding="5")
        vars_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 3))
        
        search_frame = ttk.Frame(vars_frame)
        search_frame.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(search_frame, text="검색").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', self._on_search_change)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=18)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        
        list_frame = ttk.Frame(vars_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.var_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, exportselection=False, width=22, height=8)
        self.var_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        var_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.var_listbox.yview)
        var_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.var_listbox.config(yscrollcommand=var_scrollbar.set)
        
        var_btn_frame = ttk.Frame(vars_frame)
        var_btn_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(var_btn_frame, text="선택 추가", command=self.add_to_select_list).pack(fill=tk.X)
        
        select_frame = ttk.LabelFrame(var_container, text="선택된 변수", padding="5")
        select_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 3))
        self.select_listbox = tk.Listbox(select_frame, selectmode=tk.MULTIPLE, exportselection=False, width=22, height=5)
        self.select_listbox.pack(fill=tk.BOTH, expand=True)
        select_btn_frame = ttk.Frame(select_frame)
        select_btn_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(select_btn_frame, text="삭제", command=self.remove_from_select_list).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(select_btn_frame, text="그래프 추가", command=self.add_to_graph_list).pack(side=tk.LEFT, fill=tk.X, expand=True)

        feature_frame = ttk.LabelFrame(var_container, text="Feature 설정", padding="5")
        feature_frame.pack(fill=tk.X, pady=(0, 3))
        threshold_row = ttk.Frame(feature_frame)
        threshold_row.pack(fill=tk.X)
        ttk.Label(threshold_row, text="Threshold").pack(side=tk.LEFT)
        threshold_entry = ttk.Entry(threshold_row, textvariable=self.threshold_var, width=8)
        threshold_entry.pack(side=tk.LEFT, padx=(4, 0))
        time_filter_row = ttk.Frame(feature_frame)
        time_filter_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Checkbutton(
            time_filter_row,
            text="짧은 데이터 제외",
            variable=self.min_time_filter_enabled_var,
        ).pack(side=tk.LEFT)
        min_time_entry = ttk.Entry(time_filter_row, textvariable=self.min_time_coverage_var, width=6)
        min_time_entry.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(time_filter_row, text="초 미만").pack(side=tk.LEFT)
        feature_filter_row = ttk.Frame(feature_frame)
        feature_filter_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(feature_filter_row, text="Feature").pack(side=tk.LEFT)
        feature_filter_combo = ttk.Combobox(
            feature_filter_row,
            textvariable=self.feature_analysis_filter_var,
            values=FEATURE_ANALYSIS_OPTIONS,
            state="readonly",
            width=20,
        )
        feature_filter_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        feature_action_row = ttk.Frame(feature_frame)
        feature_action_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(feature_action_row, text="Feature 선택", command=self.apply_feature_candidates).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(feature_action_row, text="Feature 분석", command=self.run_feature_analysis_thread).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        
        graph_frame = ttk.LabelFrame(var_container, text="그래프 변수", padding="5")
        graph_frame.pack(fill=tk.BOTH, expand=True)
        self.graph_listbox = tk.Listbox(graph_frame, selectmode=tk.MULTIPLE, exportselection=False, width=22, height=4)
        self.graph_listbox.pack(fill=tk.BOTH, expand=True)
        graph_axis_frame = ttk.Frame(graph_frame)
        graph_axis_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(graph_axis_frame, text="X축").pack(side=tk.LEFT)
        graph_axis_combo = ttk.Combobox(
            graph_axis_frame,
            textvariable=self.graph_x_axis_var,
            values=("time", "percent gait cycle"),
            state="readonly",
            width=18,
        )
        graph_axis_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        graph_btn_frame = ttk.Frame(graph_frame)
        graph_btn_frame.pack(fill=tk.X, pady=(3, 0))
        btn_grid = ttk.Frame(graph_btn_frame)
        btn_grid.pack(fill=tk.X)
        ttk.Button(btn_grid, text="삭제", command=self.remove_from_graph_list).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn_grid, text="그래프", command=self.plot_graph).grid(row=0, column=1, sticky="ew")
        ttk.Button(btn_grid, text="시간 구간 그래프", command=self.plot_time_phase_graph).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Button(btn_grid, text="HTML 보기", command=self.plot_interactive_graph).grid(row=2, column=0, sticky="ew", padx=(0, 2), pady=(2, 0))
        ttk.Button(btn_grid, text="HTML 시간", command=self.plot_interactive_time_phase_graph).grid(row=2, column=1, sticky="ew", pady=(2, 0))
        btn_grid.grid_columnconfigure(0, weight=1)
        btn_grid.grid_columnconfigure(1, weight=1)

        feature_plot_frame = ttk.LabelFrame(var_container, text="Feature Plot", padding="5")
        feature_plot_frame.pack(fill=tk.X, pady=(3, 0))
        feature_csv_row = ttk.Frame(feature_plot_frame)
        feature_csv_row.pack(fill=tk.X)
        ttk.Label(feature_csv_row, text="CSV").pack(side=tk.LEFT)
        self.feature_csv_combo = ttk.Combobox(
            feature_csv_row,
            textvariable=self.feature_csv_var,
            values=(),
            state="readonly",
            width=16,
        )
        self.feature_csv_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        feature_type_row = ttk.Frame(feature_plot_frame)
        feature_type_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(feature_type_row, text="Plot").pack(side=tk.LEFT)
        feature_type_combo = ttk.Combobox(
            feature_type_row,
            textvariable=self.feature_plot_type_var,
            values=FEATURE_PLOT_TYPES,
            state="readonly",
            width=16,
        )
        feature_type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        feature_ref_row = ttk.Frame(feature_plot_frame)
        feature_ref_row.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(feature_ref_row, text="Ref").pack(side=tk.LEFT)
        self.feature_reference_combo = ttk.Combobox(
            feature_ref_row,
            textvariable=self.feature_reference_var,
            values=("first condition",),
            state="readonly",
            width=12,
        )
        self.feature_reference_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        feature_btns = ttk.Frame(feature_plot_frame)
        feature_btns.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(feature_btns, text="새로고침", command=self.refresh_feature_csv_candidates).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(feature_btns, text="보기", command=self.plot_feature_csv_figure).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(feature_btns, text="PNG", command=self.save_feature_csv_figure).pack(side=tk.LEFT, fill=tk.X, expand=True)

        muscle_frame = ttk.LabelFrame(var_container, text="Muscle / CCI", padding="5")
        muscle_frame.pack(fill=tk.X, pady=(3, 0))
        muscle_option_row = ttk.Frame(muscle_frame)
        muscle_option_row.pack(fill=tk.X)
        ttk.Label(muscle_option_row, text="Side").pack(side=tk.LEFT)
        muscle_side_combo = ttk.Combobox(
            muscle_option_row,
            textvariable=self.muscle_side_var,
            values=("both", "right", "left"),
            state="readonly",
            width=8,
        )
        muscle_side_combo.pack(side=tk.LEFT, padx=(4, 6))
        ttk.Label(muscle_option_row, text="Thr").pack(side=tk.LEFT)
        muscle_threshold_entry = ttk.Entry(muscle_option_row, textvariable=self.muscle_threshold_var, width=7)
        muscle_threshold_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        muscle_btns = ttk.Frame(muscle_frame)
        muscle_btns.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(muscle_btns, text="Activation", command=self.plot_muscle_activation_figure).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(muscle_btns, text="Act 요약", command=self.plot_activation_summary_figure).grid(row=0, column=1, sticky="ew")
        ttk.Button(muscle_btns, text="CCI", command=self.plot_cci_figure).grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(2, 0))
        ttk.Button(muscle_btns, text="CCI 저장", command=self.save_cci_results).grid(row=1, column=1, sticky="ew", pady=(2, 0))
        muscle_btns.grid_columnconfigure(0, weight=1)
        muscle_btns.grid_columnconfigure(1, weight=1)

        grf_frame = ttk.LabelFrame(var_container, text="GRF / COP / COM", padding="5")
        grf_frame.pack(fill=tk.X, pady=(3, 0))
        grf_btns = ttk.Frame(grf_frame)
        grf_btns.pack(fill=tk.X)
        ttk.Button(grf_btns, text="보기", command=self.plot_grf_cop_com_figure).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(grf_btns, text="PNG", command=self.save_grf_cop_com_figure).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_quick_analysis_section(self, parent):
        quick_frame = ttk.LabelFrame(parent, text="간편 분석", padding="6")
        quick_frame.pack(fill=tk.X, pady=(0, 4))

        purpose_row = ttk.Frame(quick_frame)
        purpose_row.pack(fill=tk.X)
        ttk.Label(purpose_row, text="분석 목적").pack(side=tk.LEFT)
        purpose_combo = ttk.Combobox(
            purpose_row,
            textvariable=self.analysis_purpose_var,
            values=ANALYSIS_PURPOSE_OPTIONS,
            state="readonly",
            width=22,
        )
        purpose_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        purpose_combo.bind("<<ComboboxSelected>>", self._on_analysis_purpose_changed)

        action_row = ttk.Frame(quick_frame)
        action_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(action_row, text="후보 찾기", command=self.apply_feature_candidates).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        self.quick_run_button = ttk.Button(
            action_row,
            text="추천 실행",
            command=self.run_quick_workflow_thread,
            state=tk.DISABLED,
        )
        self.quick_run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        candidate_frame = ttk.LabelFrame(quick_frame, text="후보 변수", padding="4")
        candidate_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.quick_candidate_listbox = tk.Listbox(
            candidate_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            width=24,
            height=7,
        )
        self.quick_candidate_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        candidate_scrollbar = ttk.Scrollbar(
            candidate_frame,
            orient=tk.VERTICAL,
            command=self.quick_candidate_listbox.yview,
        )
        candidate_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.quick_candidate_listbox.config(yscrollcommand=candidate_scrollbar.set)

        candidate_btns = ttk.Frame(quick_frame)
        candidate_btns.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(candidate_btns, text="전체 선택", command=self.select_all_quick_candidates).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2)
        )
        ttk.Button(candidate_btns, text="선택 해제", command=self.clear_quick_candidates).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        output_btns = ttk.Frame(quick_frame)
        output_btns.pack(fill=tk.X, pady=(4, 0))
        self.quick_save_button = ttk.Button(
            output_btns,
            text="데이터 저장",
            command=self.save_selected_candidate_results,
            state=tk.DISABLED,
        )
        self.quick_save_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.quick_visualize_button = ttk.Button(
            output_btns,
            text="추천 시각화",
            command=self.visualize_selected_candidates,
            state=tk.DISABLED,
        )
        self.quick_visualize_button.grid(row=0, column=1, sticky="ew", padx=(0, 2))
        self.quick_save_visualization_button = ttk.Button(
            output_btns,
            text="시각화 저장",
            command=self.save_selected_candidate_visualizations,
            state=tk.DISABLED,
        )
        self.quick_save_visualization_button.grid(row=0, column=2, sticky="ew")
        for column_idx in range(3):
            output_btns.grid_columnconfigure(column_idx, weight=1)

        ttk.Button(
            quick_frame,
            text="결과 변수 설명",
            command=self.show_result_variable_help,
        ).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            quick_frame,
            text="변수 단위",
            command=self.show_variable_unit_help,
        ).pack(fill=tk.X, pady=(4, 0))

    def _get_graph_x_axis_mode(self) -> str:
        graph_axis_var = getattr(self, "graph_x_axis_var", None)
        if graph_axis_var is None:
            return "time"
        try:
            value = graph_axis_var.get()
        except Exception:
            return "time"
        return "percent gait cycle" if value == "percent gait cycle" else "time"

    def _refresh_reference_conditions(self):
        if not hasattr(self, "reference_condition_var"):
            return
        labels = ["first file"] + get_condition_labels(self.file_paths)
        combo = getattr(self, "reference_condition_combo", None)
        if combo is not None:
            combo.config(values=tuple(labels))
        if self.reference_condition_var.get() not in labels:
            self.reference_condition_var.set("first file")

    def _get_reference_condition(self):
        labels = get_condition_labels(self.file_paths)
        if not labels:
            return None
        if not hasattr(self, "reference_condition_var"):
            return labels[0]
        selected = self.reference_condition_var.get()
        if selected == "first file" or selected not in labels:
            return labels[0]
        return selected

    def _get_overview_sd_multiplier(self) -> float:
        if not hasattr(self, "overview_band_var"):
            return 1.0
        return 2.0 if "2SD" in self.overview_band_var.get() else 1.0

    def _feature_search_roots(self) -> List[str]:
        roots = []
        for file_path in self.file_paths:
            directory = os.path.dirname(file_path)
            if directory and directory not in roots:
                roots.append(directory)
        last_result_dir = getattr(self, "last_result_save_dir", "")
        if last_result_dir and last_result_dir not in roots:
            roots.append(last_result_dir)
        latest_feature_dir = getattr(self, "latest_feature_csv_dir", "")
        if latest_feature_dir and os.path.isdir(latest_feature_dir) and latest_feature_dir not in roots:
            roots.append(latest_feature_dir)
        if not roots:
            roots.append(os.getcwd())
        return roots

    def refresh_feature_csv_candidates(self):
        paths = discover_feature_csvs(self._feature_search_roots())
        self.feature_csv_paths = paths
        labels = ["auto: all feature CSVs"] + [os.path.basename(path) for path in paths]
        combo = getattr(self, "feature_csv_combo", None)
        if combo is not None:
            combo.config(values=tuple(labels))
        if not hasattr(self, "feature_csv_var"):
            return paths
        if paths:
            if self.feature_csv_var.get() not in labels:
                self.feature_csv_var.set(labels[0])
        else:
            self.feature_csv_var.set("")

        self._refresh_feature_reference_conditions()
        return paths

    def _save_latest_feature_csvs(self):
        if not getattr(self, "auto_feature_csv_enabled", False):
            return []
        if not getattr(self, "analysis_results", None) or not self.file_paths:
            return []
        selected_vars = list(self.select_listbox.get(0, tk.END)) if hasattr(self, "select_listbox") else []
        if not selected_vars:
            selected_vars = _variables_from_analysis_results(self.analysis_results)
        if not selected_vars:
            return []

        output_dir = os.path.join(self._ensure_run_output_dir(), "features")
        try:
            saved_paths = save_feature_csv_bundle(
                selected_vars,
                self.file_paths,
                self.analysis_results,
                output_dir,
            )
            comparison_path = save_condition_comparison_table(
                self.file_paths,
                self.analysis_results,
                output_dir,
            )
            if comparison_path:
                saved_paths = list(saved_paths) + [comparison_path]
            if saved_paths:
                self.latest_feature_csv_dir = output_dir
                self.refresh_feature_csv_candidates()
            return saved_paths
        except Exception as exc:
            self.results_data += f"\n[Feature CSV 자동 저장 경고] {exc}\n"
            return []

    def _ensure_run_output_dir(self) -> str:
        current = getattr(self, "current_run_output_dir", "")
        if current:
            os.makedirs(current, exist_ok=True)
            return current

        purpose = self._get_analysis_purpose() if hasattr(self, "_get_analysis_purpose") else "analysis"
        purpose_part = get_safe_filename_component(purpose) or "analysis"
        condition_part = get_safe_filename_component("_".join(get_condition_labels(self.file_paths)[:4])) or "run"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(os.getcwd(), "output", "runs", f"{timestamp}_{purpose_part}_{condition_part}")
        os.makedirs(run_dir, exist_ok=True)
        self.current_run_output_dir = run_dir
        return run_dir

    def _refresh_feature_reference_conditions(self):
        paths = self._get_selected_feature_csv_paths()
        conditions = get_conditions_from_feature_tables(paths)
        labels = ["first condition"] + conditions
        combo = getattr(self, "feature_reference_combo", None)
        if combo is not None:
            combo.config(values=tuple(labels))
        if not hasattr(self, "feature_reference_var"):
            return
        if self.feature_reference_var.get() not in labels:
            self.feature_reference_var.set("first condition")

    def _get_selected_feature_csv_paths(self) -> List[str]:
        if not getattr(self, "feature_csv_paths", None):
            return []
        selected = self.feature_csv_var.get() if hasattr(self, "feature_csv_var") else ""
        if selected in ("", "auto: all feature CSVs"):
            return list(self.feature_csv_paths)
        for path in self.feature_csv_paths:
            if os.path.basename(path) == selected:
                return [path]
        return list(self.feature_csv_paths)

    def _get_feature_reference_condition(self):
        selected = self.feature_reference_var.get() if hasattr(self, "feature_reference_var") else "first condition"
        if selected != "first condition":
            return selected
        conditions = get_conditions_from_feature_tables(self._get_selected_feature_csv_paths())
        return conditions[0] if conditions else None

    def _get_analysis_purpose(self) -> str:
        source_var = getattr(self, "analysis_purpose_var", None)
        if source_var is None:
            return ANALYSIS_PURPOSE_OPTIONS[0]
        try:
            selected = source_var.get()
        except Exception:
            return ANALYSIS_PURPOSE_OPTIONS[0]
        return selected if is_known_analysis_purpose(selected) else ANALYSIS_PURPOSE_OPTIONS[0]

    def _get_muscle_side_filter(self) -> str:
        if not hasattr(self, "muscle_side_var"):
            return "both"
        try:
            selected = self.muscle_side_var.get()
        except Exception:
            return "both"
        return selected if selected in ("both", "right", "left") else "both"

    def _get_muscle_threshold(self) -> float:
        source_var = getattr(self, "muscle_threshold_var", getattr(self, "threshold_var", None))
        if source_var is None:
            return 0.05
        try:
            value = float(source_var.get())
        except Exception:
            return 0.05
        return value if np.isfinite(value) else 0.05

    def _on_feature_filter_changed(self, _event=None):
        self._apply_feature_visualization_defaults()
        if self.dataframes:
            self.apply_auto_candidates(
                show_message=False,
                feature_filter=self._get_feature_analysis_filter(),
            )

    def _on_analysis_purpose_changed(self, _event=None):
        analysis_purpose = self._get_analysis_purpose()
        self._set_string_var(
            "feature_analysis_filter_var",
            feature_filter_for_purpose(analysis_purpose),
        )
        self._apply_feature_visualization_defaults()
        if self.dataframes:
            self.apply_auto_candidates(
                show_message=False,
                feature_filter=self._get_feature_analysis_filter(),
            )

    def _apply_feature_visualization_defaults(self):
        feature_filter = self._get_feature_analysis_filter()
        analysis_purpose = self._get_analysis_purpose()
        whole_report = getattr(self, "analysis_purpose_var", None) is not None and analysis_purpose == "전체 자동 리포트"
        self._set_boolean_var("quick_run_analysis_var", True)
        self._set_boolean_var("quick_show_waveform_var", not whole_report)
        self._set_boolean_var(
            "quick_show_overview_var",
            False,
        )
        self._set_boolean_var(
            "quick_create_dashboard_var",
            whole_report or feature_filter == "Symmetry pairs",
        )
        gait_axis_features = {
            "Joint angle",
            "Joint load / torque / force",
            "Muscle activation",
            "Muscle mechanics / force",
            "GRF / COP / COM",
            "Symmetry pairs",
        }
        self._set_string_var(
            "graph_x_axis_var",
            "percent gait cycle" if feature_filter in gait_axis_features else "time",
        )

    def _set_boolean_var(self, attr_name: str, value: bool):
        source_var = getattr(self, attr_name, None)
        if source_var is not None and hasattr(source_var, "set"):
            source_var.set(bool(value))

    def _set_string_var(self, attr_name: str, value: str):
        source_var = getattr(self, attr_name, None)
        if source_var is not None and hasattr(source_var, "set"):
            source_var.set(value)

    def _set_quick_action_buttons_state(self, state):
        for attr_name in (
            "quick_run_button",
            "quick_save_button",
            "quick_visualize_button",
            "quick_save_visualization_button",
        ):
            button = getattr(self, attr_name, None)
            if button is not None:
                button.config(state=state)

    def show_result_variable_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("결과 변수 설명")
        help_window.geometry("760x620")
        help_window.minsize(560, 420)

        container = ttk.Frame(help_window, padding="8")
        container.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(container, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert(tk.END, get_result_variable_help_text())
        text_widget.config(state=tk.DISABLED)
        help_window.focus_set()

    def show_variable_unit_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("변수 단위")
        help_window.geometry("760x620")
        help_window.minsize(560, 420)

        container = ttk.Frame(help_window, padding="8")
        container.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(container, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert(tk.END, get_variable_unit_help_text())
        text_widget.config(state=tk.DISABLED)
        help_window.focus_set()
    
    def _create_results_section(self, parent):
        results_container = ttk.Frame(parent)
        results_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(results_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text=self.lang['analysis_result'])
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, state=tk.DISABLED, font=('Consolas', 9))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        quality_frame = ttk.Frame(notebook)
        notebook.add(quality_frame, text=self.lang['data_quality'])
        self.quality_text = tk.Text(quality_frame, wrap=tk.WORD, state=tk.DISABLED, font=('Consolas', 9))
        self.quality_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        quality_scrollbar = ttk.Scrollbar(quality_frame, orient=tk.VERTICAL, command=self.quality_text.yview)
        quality_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.quality_text.config(yscrollcommand=quality_scrollbar.set)
    
    def _create_compact_button_section(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        self.bottom_tools_frame = ttk.Frame(button_frame)
        self.analyze_button = ttk.Button(self.bottom_tools_frame, text=self.lang['run_analysis'], command=self.run_analysis_thread, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 5))
        self.auto_candidate_button = ttk.Button(self.bottom_tools_frame, text="자동 후보", command=self.apply_auto_candidates, state=tk.DISABLED)
        self.auto_candidate_button.pack(side=tk.LEFT, padx=(0, 5))
        self.auto_analyze_button = ttk.Button(self.bottom_tools_frame, text="자동 분석", command=self.run_auto_analysis_thread, state=tk.DISABLED)
        self.auto_analyze_button.pack(side=tk.LEFT, padx=(0, 8))
        self.save_button = ttk.Button(self.bottom_tools_frame, text=self.lang['save_result'], command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))
        self.recommended_visualization_save_button = ttk.Button(
            self.bottom_tools_frame,
            text="추천 시각화 저장",
            command=self.save_recommended_visualization_files,
            state=tk.DISABLED,
        )
        self.recommended_visualization_save_button.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_tools_frame, text="그래프 저장", command=self.save_graphs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_tools_frame, text="시간 구간 저장", command=self.save_time_phase_graphs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_tools_frame, text="HTML 그래프 저장", command=self.save_interactive_graphs).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_tools_frame, text="HTML 시간 저장", command=self.save_interactive_time_phase_graphs).pack(side=tk.LEFT, padx=(0, 5))
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT)
        self.status_bar_var = tk.StringVar()
        self.status_bar_var.set("준비됨")
        ttk.Label(right_frame, textvariable=self.status_bar_var, style='Status.TLabel').pack(side=tk.RIGHT, padx=(10, 0))
    
    def _on_search_change(self, *args):
        search_text = self.search_var.get().lower()
        self.var_listbox.delete(0, tk.END)
        for col in self.all_columns:
            if search_text in col.lower():
                self.var_listbox.insert(tk.END, col)
    
    def _update_progress(self, value: float, message: str = ""):
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
            self.status_bar_var.set(message)
        self.root.update_idletasks()

    def _queue_progress(self, value: float, message: str = ""):
        self.root.after(0, lambda: self._update_progress(value, message))

    def _get_threshold_value(self):
        try:
            threshold = float(self.threshold_var.get())
        except (TypeError, ValueError):
            messagebox.showwarning("Threshold 오류", "Threshold는 숫자로 입력해 주세요.")
            return None
        if not np.isfinite(threshold):
            messagebox.showwarning("Threshold 오류", "Threshold는 유한한 숫자여야 합니다.")
            return None
        return threshold
    
    def _set_text_content(self, text_widget, content: str):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    def _set_listbox_items(self, listbox, values: List[str]):
        listbox.delete(0, tk.END)
        for value in values:
            listbox.insert(tk.END, value)

    def _set_quick_candidate_items(self, values: List[str]):
        listbox = getattr(self, "quick_candidate_listbox", None)
        if listbox is None:
            return
        self._set_listbox_items(listbox, values)
        if values:
            listbox.selection_set(0, tk.END)

    def select_all_quick_candidates(self):
        listbox = getattr(self, "quick_candidate_listbox", None)
        if listbox is not None:
            listbox.selection_set(0, tk.END)

    def clear_quick_candidates(self):
        listbox = getattr(self, "quick_candidate_listbox", None)
        if listbox is not None:
            listbox.selection_clear(0, tk.END)

    def _get_quick_selected_variables(self) -> List[str]:
        listbox = getattr(self, "quick_candidate_listbox", None)
        if listbox is None:
            return list(self.select_listbox.get(0, tk.END))
        items = list(listbox.get(0, tk.END))
        selected_indices = list(listbox.curselection())
        return [items[index] for index in selected_indices if 0 <= index < len(items)]

    def _get_feature_analysis_filter(self) -> str:
        source_var = getattr(self, "feature_analysis_filter_var", None)
        if source_var is None:
            return FEATURE_ANALYSIS_OPTIONS[0]
        try:
            selected = source_var.get()
        except Exception:
            return FEATURE_ANALYSIS_OPTIONS[0]
        return selected if selected in FEATURE_ANALYSIS_OPTIONS else FEATURE_ANALYSIS_OPTIONS[0]

    def apply_auto_candidates(self, show_message: bool = True, feature_filter: str = None):
        if not self.dataframes:
            if show_message:
                messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return None

        selected_filter = feature_filter or self._get_feature_analysis_filter()
        plan = build_auto_analysis_plan(
            self.dataframes,
            self.file_paths,
            feature_filter=selected_filter,
        )
        self.auto_analysis_plan = plan
        self._set_listbox_items(self.select_listbox, plan.selected_variables)
        self._set_listbox_items(self.graph_listbox, plan.graph_variables)
        self._set_quick_candidate_items(plan.selected_variables)
        self._set_text_content(self.results_text, format_auto_analysis_plan(plan))

        status_message = (
            f"{plan.feature_filter}: 분석 변수 {len(plan.selected_variables)}개, "
            f"그래프 변수 {len(plan.graph_variables)}개"
        )
        self._update_progress(100, status_message)
        self._set_quick_action_buttons_state(tk.NORMAL if plan.selected_variables else tk.DISABLED)
        if show_message:
            messagebox.showinfo("자동 후보", status_message)
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        return plan

    def apply_feature_candidates(self):
        return self.apply_auto_candidates(
            show_message=True,
            feature_filter=self._get_feature_analysis_filter(),
        )
    
    def load_files(self):
        lang = self.lang
        file_types = [
            ("STO 파일", "*.sto"),
            (lang['csv_file'], "*.csv"),
            ("Excel 파일", "*.xlsx;*.xls"),
            ("텍스트 파일", "*.txt"),
            (lang['all_file'], "*.*")
        ]
        selected_paths = filedialog.askopenfilenames(title=lang['open_file'], filetypes=file_types)
        if not selected_paths:
            return
        self._update_progress(0, self.lang['loading_file'])
        try:
            self.dataframes = []
            self.file_paths = []
            self.analysis_results = {}
            columns_set = set()
            validation_results = []
            total_files = len(selected_paths)
            for i, path in enumerate(selected_paths):
                progress = (i / total_files) * 50
                self._update_progress(progress, f"파일 로딩 중: {os.path.basename(path)}")
                is_valid_path, path_msg = self.data_validator.validate_file_path(path)
                if not is_valid_path:
                    messagebox.showerror(lang['file_error'], path_msg)
                    continue
                try:
                    df, header_row = read_motion_data_file(path, self.data_validator)
                    is_valid_df, df_msg, df_info = self.data_validator.validate_dataframe(df, path)
                    time_valid, time_msg, time_info = self.data_validator.validate_time_column(df)
                    df_info['time_column_valid'] = time_valid
                    df_info['time_column_message'] = time_msg
                    df_info['time_column_info'] = time_info
                    validation_results.append({'file': os.path.basename(path), 'valid': is_valid_df, 'message': df_msg, 'info': df_info})
                    if is_valid_df:
                        self.dataframes.append(df)
                        self.file_paths.append(path)
                        current_cols = set([col for col in df.columns if col.lower() != 'time'])
                        columns_set |= current_cols
                    else:
                        messagebox.showwarning(lang['file_error'], f"{os.path.basename(path)}: {df_msg}")
                except Exception as e:
                    error_msg = f"파일 '{os.path.basename(path)}' 로딩 실패: {str(e)}"
                    validation_results.append({'file': os.path.basename(path), 'valid': False, 'message': error_msg, 'info': {}})
                    messagebox.showerror(lang['file_error'], error_msg)
            self._update_progress(75, self.lang['validating_data'])
            if not self.dataframes:
                messagebox.showerror(lang['file_error'], "로딩된 파일이 없습니다.")
                self._update_progress(0, "오류 발생")
                if hasattr(self, "auto_candidate_button"):
                    self.auto_candidate_button.config(state=tk.DISABLED)
                if hasattr(self, "auto_analyze_button"):
                    self.auto_analyze_button.config(state=tk.DISABLED)
                if hasattr(self, "quick_run_button"):
                    self.quick_run_button.config(state=tk.DISABLED)
                self._set_quick_action_buttons_state(tk.DISABLED)
                return
            self.all_columns = sorted(list(columns_set)) if columns_set else []
            self.var_listbox.delete(0, tk.END)
            for col in self.all_columns:
                self.var_listbox.insert(tk.END, col)
            file_names = [os.path.basename(p) for p in self.file_paths]
            self.file_label.config(text=f"{len(file_names)}개 파일: {', '.join(file_names[:3])}{'...' if len(file_names) > 3 else ''}")
            self.file_info_var.set(f"총 {len(self.all_columns)}개 변수")
            self.analyze_button.config(state=tk.NORMAL)
            if hasattr(self, "auto_candidate_button"):
                self.auto_candidate_button.config(state=tk.NORMAL)
            if hasattr(self, "auto_analyze_button"):
                self.auto_analyze_button.config(state=tk.NORMAL)
            if hasattr(self, "quick_run_button"):
                self.quick_run_button.config(state=tk.NORMAL)
            self._set_quick_action_buttons_state(tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            if hasattr(self, "recommended_visualization_save_button"):
                self.recommended_visualization_save_button.config(state=tk.DISABLED)
            self._set_text_content(self.results_text, self.lang['file_loaded'])
            self._display_validation_results(validation_results)
            self.select_listbox.delete(0, tk.END)
            self.graph_listbox.delete(0, tk.END)
            self._refresh_reference_conditions()
            self.refresh_feature_csv_candidates()
            self.apply_auto_candidates(
                show_message=False,
                feature_filter=FEATURE_ANALYSIS_OPTIONS[0],
            )
            self._update_progress(100, f"{len(self.dataframes)}개 파일 로딩 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            error_msg = f"파일 로딩 중 오류: {str(e)}"
            messagebox.showerror(lang['file_error'], error_msg)
            self._update_progress(0, "오류 발생")
            self.dataframes = []
            self.file_paths = []
            self.analyze_button.config(state=tk.DISABLED)
            if hasattr(self, "auto_candidate_button"):
                self.auto_candidate_button.config(state=tk.DISABLED)
            if hasattr(self, "auto_analyze_button"):
                self.auto_analyze_button.config(state=tk.DISABLED)
            if hasattr(self, "quick_run_button"):
                self.quick_run_button.config(state=tk.DISABLED)
            self._set_quick_action_buttons_state(tk.DISABLED)
    
    def _display_validation_results(self, validation_results: List[Dict]):
        quality_content = "=== 데이터 검증 결과 ===\n\n"
        for result in validation_results:
            quality_content += f"파일: {result['file']}\n"
            quality_content += f"상태: {'유효' if result['valid'] else '오류'}\n"
            quality_content += f"메시지: {result['message']}\n"
            if result['info']:
                info = result['info']
                quality_content += f"  - 행 수: {info.get('row_count', 'N/A')}\n"
                quality_content += f"  - 컬럼 수: {info.get('column_count', 'N/A')}\n"
                quality_content += f"  - 숫자 컬럼: {len(info.get('numeric_columns', []))}\n"
                quality_content += f"  - 메모리 사용: {info.get('memory_usage_mb', 0):.1f}MB\n"
                if info.get('time_column_info', {}).get('exists', False):
                    time_status = "유효" if info.get('time_column_valid', False) else "확인 필요"
                    quality_content += f"  - time 컬럼: {time_status} ({info.get('time_column_message', '')})\n"
            quality_content += "-" * 50 + "\n\n"
        self._set_text_content(self.quality_text, quality_content)
    
    def add_to_select_list(self):
        selected = self.var_listbox.curselection()
        for i in selected:
            var = self.var_listbox.get(i)
            if var not in self.select_listbox.get(0, tk.END):
                self.select_listbox.insert(tk.END, var)
        self.var_listbox.selection_clear(0, tk.END)
    
    def remove_from_select_list(self):
        selected = list(self.select_listbox.curselection())[::-1]
        for i in selected:
            self.select_listbox.delete(i)
    
    def add_to_graph_list(self):
        selected = self.select_listbox.curselection()
        for i in selected:
            var = self.select_listbox.get(i)
            if var not in self.graph_listbox.get(0, tk.END):
                self.graph_listbox.insert(tk.END, var)
    
    def remove_from_graph_list(self):
        selected = list(self.graph_listbox.curselection())[::-1]
        for i in selected:
            self.graph_listbox.delete(i)
    
    def run_analysis_thread(self):
        selected_vars = list(self.select_listbox.get(0, tk.END))
        if not selected_vars:
            messagebox.showwarning(self.lang['select_error'], self.lang['select_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        threshold_value = self._get_threshold_value()
        if threshold_value is None:
            return
        if self._is_min_time_filter_enabled() and self._get_min_time_coverage_seconds() is None:
            messagebox.showwarning("시간 필터 오류", "최소 시간은 0 이상의 숫자로 입력해 주세요.")
            return

        self.analyze_button.config(state=tk.DISABLED)
        if hasattr(self, "auto_analyze_button"):
            self.auto_analyze_button.config(state=tk.DISABLED)
        if hasattr(self, "quick_run_button"):
            self.quick_run_button.config(state=tk.DISABLED)
        self._set_quick_action_buttons_state(tk.DISABLED)
        if hasattr(self, "recommended_visualization_save_button"):
            self.recommended_visualization_save_button.config(state=tk.DISABLED)
        self.auto_analysis_mode = False
        threading.Thread(target=self.analyze_data, args=(selected_vars, threshold_value), daemon=True).start()

    def run_auto_analysis_thread(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        plan = self.apply_auto_candidates(
            show_message=False,
            feature_filter=FEATURE_ANALYSIS_OPTIONS[0],
        )
        self._run_analysis_plan(plan, "자동 분석")

    def run_feature_analysis_thread(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        plan = self.apply_auto_candidates(
            show_message=False,
            feature_filter=self._get_feature_analysis_filter(),
        )
        self._run_analysis_plan(plan, "Feature 분석")

    def run_quick_workflow_thread(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        selected_vars, plan = self._prepare_selected_candidate_context("추천 실행")
        if not selected_vars:
            return

        self.pending_quick_actions = self._recommended_actions_for_current_purpose()
        self.pending_quick_actions["selected_variables"] = selected_vars

        run_analysis = self.pending_quick_actions.get("analysis", True)
        if self.pending_quick_actions.get("recommended_visualization") and not getattr(self, "analysis_results", None):
            run_analysis = True

        if run_analysis:
            self._run_analysis_plan(plan, "추천 실행", selected_variables=selected_vars)
        else:
            self._run_pending_quick_actions()

    def save_selected_candidate_results(self):
        selected_vars, plan = self._prepare_selected_candidate_context("데이터 저장")
        if not selected_vars:
            return
        if self._selected_candidate_results_ready(selected_vars):
            self.save_results()
            return
        self.pending_quick_actions = {
            "save_results": True,
            "selected_variables": selected_vars,
        }
        self._run_analysis_plan(plan, "데이터 저장", selected_variables=selected_vars)

    def save_selected_candidate_visualizations(self):
        selected_vars, plan = self._prepare_selected_candidate_context("추천 시각화 저장")
        if not selected_vars:
            return
        if self._selected_candidate_results_ready(selected_vars):
            self.save_recommended_visualization_files(selected_variables=selected_vars, plan=plan)
            return
        self.pending_quick_actions = {
            "save_recommended_visualization": True,
            "selected_variables": selected_vars,
            "feature_filter": getattr(plan, "feature_filter", ""),
        }
        self._run_analysis_plan(plan, "추천 시각화 저장", selected_variables=selected_vars)

    def visualize_selected_candidates(self):
        selected_vars, plan = self._prepare_selected_candidate_context("추천 시각화")
        if not selected_vars:
            return
        if self._get_feature_analysis_filter() == "Symmetry pairs" and not self._selected_candidate_results_ready(selected_vars):
            self.pending_quick_actions = {
                "recommended_visualization": True,
                "selected_variables": selected_vars,
            }
            self._run_analysis_plan(plan, "추천 시각화", selected_variables=selected_vars)
            return
        self._run_recommended_visualization(selected_vars, plan)

    def _prepare_selected_candidate_context(self, dialog_title: str):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return [], None

        current_plan = getattr(self, "auto_analysis_plan", None)
        if current_plan is None or current_plan.feature_filter != self._get_feature_analysis_filter():
            plan = self.apply_auto_candidates(
                show_message=False,
                feature_filter=self._get_feature_analysis_filter(),
            )
        else:
            plan = current_plan

        selected_vars = self._get_quick_selected_variables()
        if not selected_vars:
            messagebox.showwarning(dialog_title, "후보 변수에서 하나 이상 선택해 주세요.")
            return [], plan

        self._set_listbox_items(self.select_listbox, selected_vars)
        graph_vars = []
        if self._quick_option_enabled("quick_show_waveform_var", True):
            graph_vars = self._select_quick_graph_variables(selected_vars, plan)
        self._set_listbox_items(self.graph_listbox, graph_vars)
        return selected_vars, plan

    def _selected_candidate_results_ready(self, selected_vars: List[str]) -> bool:
        if not getattr(self, "analysis_results", None):
            return False
        expected_count = 0
        for file_idx, (path, df) in enumerate(zip(self.file_paths, self.dataframes)):
            file_name = os.path.basename(path)
            for variable in selected_vars:
                if variable not in df.columns or not pd.api.types.is_numeric_dtype(df[variable]):
                    continue
                expected_count += 1
                analysis_key = f"{file_idx}_{file_name}_{variable}"
                if analysis_key not in self.analysis_results:
                    return False
        return expected_count > 0

    def _run_recommended_visualization(self, selected_vars: List[str], plan=None):
        feature_filter = self._get_feature_analysis_filter()
        graph_vars = self._select_quick_graph_variables(selected_vars, plan)
        self._set_listbox_items(self.graph_listbox, graph_vars)
        if graph_vars:
            self.plot_graph()

        if feature_filter == "Muscle activation":
            self.plot_activation_summary_figure(scope_variables=selected_vars)
        elif feature_filter == "GRF / COP / COM":
            self.plot_grf_cop_com_figure()
        elif feature_filter == "Symmetry pairs":
            self.plot_symmetry_figure(selected_variables=selected_vars)

    def _recommended_actions_for_current_purpose(self) -> Dict:
        if getattr(self, "analysis_purpose_var", None) is None:
            return {
                "analysis": self._quick_option_enabled("quick_run_analysis_var", True),
                "waveform": self._quick_option_enabled("quick_show_waveform_var", True),
                "recommended_visualization": self._quick_option_enabled("quick_create_dashboard_var", False),
            }

        purpose = self._get_analysis_purpose()
        if purpose == "전체 자동 리포트":
            return {"analysis": True}
        if purpose in ("보행 각도 분석", "관절 부하 분석", "관절 부하/토크 분석", "역학/관절 토크 분석"):
            return {"analysis": True, "waveform": True}
        if purpose in ("근활성 분석", "GRF/균형 분석", "GRF/COP/COM 균형 분석"):
            return {"analysis": True, "recommended_visualization": True}
        if purpose == "좌우 대칭 분석":
            return {"analysis": True, "recommended_visualization": True}
        if purpose in ("품질/패널티 분석", "일반 숫자 분석"):
            return {"analysis": True, "waveform": True}
        return {"analysis": True, "waveform": True}

    def _quick_option_enabled(self, attr_name: str, default: bool) -> bool:
        source_var = getattr(self, attr_name, None)
        if source_var is None:
            return default
        try:
            return bool(source_var.get())
        except Exception:
            return default

    def _select_quick_graph_variables(self, selected_vars: List[str], plan) -> List[str]:
        selected_set = set(selected_vars)
        planned_graph_vars = list(getattr(plan, "graph_variables", []) or [])
        graph_vars = [var for var in planned_graph_vars if var in selected_set]
        if not graph_vars:
            graph_vars = list(selected_vars)
        return graph_vars[:30]

    def _run_analysis_plan(self, plan, dialog_title: str, selected_variables: List[str] = None):
        variables_to_analyze = list(selected_variables or (plan.selected_variables if plan is not None else []))
        if plan is None or not variables_to_analyze:
            messagebox.showwarning(dialog_title, "선택한 Feature에서 분석할 숫자 변수를 찾지 못했습니다.")
            return

        threshold_value = self._get_threshold_value()
        if threshold_value is None:
            return
        if self._is_min_time_filter_enabled() and self._get_min_time_coverage_seconds() is None:
            messagebox.showwarning("시간 필터 오류", "최소 시간은 0 이상의 숫자로 입력해 주세요.")
            return

        self.analyze_button.config(state=tk.DISABLED)
        if hasattr(self, "auto_analyze_button"):
            self.auto_analyze_button.config(state=tk.DISABLED)
        if hasattr(self, "quick_run_button"):
            self.quick_run_button.config(state=tk.DISABLED)
        self._set_quick_action_buttons_state(tk.DISABLED)
        if hasattr(self, "recommended_visualization_save_button"):
            self.recommended_visualization_save_button.config(state=tk.DISABLED)
        self.auto_analysis_mode = True
        threading.Thread(
            target=self.analyze_data,
            args=(variables_to_analyze, threshold_value, True),
            daemon=True,
        ).start()
    
    def analyze_data(
        self,
        selected_vars: List[str],
        threshold_value: float = 0.05,
        skip_missing_variables: bool = False,
    ):
        try:
            results_data, analysis_results = run_analysis_workflow(
                self.dataframes,
                self.file_paths,
                selected_vars,
                threshold_value,
                self.lang,
                self.data_validator,
                self._check_time_coverage,
                self._get_state_column_for_variable,
                progress_callback=self._queue_progress,
                skip_missing_variables=skip_missing_variables,
            )
            self.root.after(0, lambda: self._finish_analysis(results_data, analysis_results))
        except Exception as e:
            error_msg = f"분석 중 오류: {str(e)}"
            self.root.after(0, lambda: self._handle_analysis_error(error_msg))

    def _finish_analysis(self, results_data: str, analysis_results: Dict):
        self.analysis_results = analysis_results
        self._save_latest_feature_csvs()
        selected_vars = list(self.select_listbox.get(0, tk.END)) if hasattr(self, "select_listbox") else []
        summary_text = build_analysis_summary(
            selected_vars,
            self.file_paths,
            self.analysis_results,
            dataframes=self.dataframes,
            analysis_purpose=self._get_analysis_purpose(),
        )
        self._save_analysis_summary_text(summary_text)
        self.results_data = f"{summary_text}\n\n{results_data}"
        self._update_progress(100, self.lang['analysis_done'])
        self._set_text_content(self.results_text, self.results_data)
        self.save_button.config(state=tk.NORMAL)
        if hasattr(self, "recommended_visualization_save_button"):
            self.recommended_visualization_save_button.config(state=tk.NORMAL)
        self.analyze_button.config(state=tk.NORMAL)
        if hasattr(self, "auto_analyze_button"):
            self.auto_analyze_button.config(state=tk.NORMAL)
        if hasattr(self, "quick_run_button"):
            self.quick_run_button.config(state=tk.NORMAL)
        self._set_quick_action_buttons_state(tk.NORMAL)
        if getattr(self, "pending_quick_actions", None):
            self.root.after(50, self._run_pending_quick_actions)
        self.root.after(2000, lambda: self._update_progress(0, "분석 완료"))

    def _save_analysis_summary_text(self, summary_text: str):
        if not summary_text:
            return ""
        try:
            reports_dir = os.path.join(self._ensure_run_output_dir(), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            summary_path = os.path.join(reports_dir, "analysis_summary.md")
            with open(summary_path, "w", encoding="utf-8") as file:
                file.write("# Analysis Summary\n\n")
                file.write(summary_text)
                file.write("\n")
            return summary_path
        except Exception:
            return ""

    def _handle_analysis_error(self, error_msg: str):
        messagebox.showerror("분석 오류", error_msg)
        self._update_progress(0, "오류 발생")
        self.analyze_button.config(state=tk.NORMAL)
        if hasattr(self, "auto_analyze_button"):
            self.auto_analyze_button.config(state=tk.NORMAL)
        if hasattr(self, "quick_run_button"):
            self.quick_run_button.config(state=tk.NORMAL)
        self._set_quick_action_buttons_state(tk.NORMAL)
        if hasattr(self, "recommended_visualization_save_button") and getattr(self, "analysis_results", None):
            self.recommended_visualization_save_button.config(state=tk.NORMAL)

    def _run_pending_quick_actions(self):
        actions = self.pending_quick_actions or {}
        self.pending_quick_actions = None
        if actions.get("waveform"):
            self.plot_graph()
        if actions.get("recommended_visualization"):
            self._run_recommended_visualization(actions.get("selected_variables") or [])
        if actions.get("save_recommended_visualization"):
            self.save_recommended_visualization_files(
                selected_variables=actions.get("selected_variables") or [],
                feature_filter=actions.get("feature_filter", ""),
            )
        if actions.get("save_results"):
            self.save_results()
    
    def _format_analysis_result(self, analysis_result: Dict):
        self.results_data += self._format_analysis_result_text(analysis_result)

    def _format_analysis_result_text(self, analysis_result: Dict) -> str:
        return format_analysis_result_text(analysis_result)
    
    def plot_graph(self):
        vars_to_plot = list(self.graph_listbox.get(0, tk.END))
        if not vars_to_plot:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        self._update_progress(0, self.lang['generating_plots'])
        try:
            plotted_count = 0
            warnings = []
            x_axis_mode = self._get_graph_x_axis_mode()
            for i, var in enumerate(vars_to_plot):
                progress = (i / len(vars_to_plot)) * 100
                self._update_progress(progress, f"그래프 생성 중: {var}")
                fig, ax = plt.subplots(figsize=(12, 7) if x_axis_mode == "percent gait cycle" else (10, 6))
                if x_axis_mode == "percent gait cycle":
                    has_data, plot_warnings = draw_gait_cycle_waveform(
                        ax,
                        self.dataframes,
                        self.file_paths,
                        var,
                        show_mean_sd=True,
                        show_phase_shading=True,
                        show_peak_markers=True,
                    )
                else:
                    has_data, plot_warnings = draw_raw_waveform(
                        ax,
                        self.dataframes,
                        self.file_paths,
                        var,
                        show_mean_sd=True,
                        prefer_time_axis=True,
                    )
                warnings.extend(plot_warnings)
                clean_var_name = get_clean_variable_name(var)
                plt.tight_layout()
                if has_data:
                    self._show_plot_window(fig, f"Graph: {clean_var_name}")
                    plotted_count += 1
                else:
                    plt.close(fig)
            if plotted_count == 0:
                messagebox.showwarning("그래프", "표시할 수 있는 숫자 데이터가 없습니다.")
            elif warnings:
                messagebox.showwarning("그래프 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "그래프 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("그래프 오류", f"그래프 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")
    
    def plot_time_phase_graph(self):
        vars_to_plot = list(self.graph_listbox.get(0, tk.END))
        if not vars_to_plot:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        self._update_progress(0, "시간 구간 그래프 생성 중...")
        try:
            for i, var in enumerate(vars_to_plot):
                progress = (i / len(vars_to_plot)) * 100
                self._update_progress(progress, f"시간 구간 그래프: {var}")
                fig, ax = plt.subplots(figsize=(12, 7))
                self._draw_normalized_gait_cycle_graph(ax, var)
                clean_var_name = get_clean_variable_name(var)
                plt.tight_layout()
                self._show_plot_window(fig, f"Normalized Gait Cycle: {clean_var_name}")
            self._update_progress(100, "시간 구간 그래프 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("그래프 오류", f"시간 구간 그래프 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_overview_figure(self, scope_variables: List[str] = None):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        self._update_progress(0, "Overview 생성 중...")
        try:
            fig, has_data, warnings = create_overview_figure(
                self.dataframes,
                self.file_paths,
                preset_name=self.overview_preset_var.get() if hasattr(self, "overview_preset_var") else PANEL_PRESETS[0],
                reference_condition=self._get_reference_condition(),
                sd_multiplier=self._get_overview_sd_multiplier(),
                show_sd_band=True,
                scope_variables=scope_variables,
            )
            if has_data:
                self._show_plot_window(fig, "Gait Overview")
            else:
                plt.close(fig)
                messagebox.showwarning("Overview", "표시할 수 있는 overview 변수가 없습니다.")
            if warnings:
                messagebox.showwarning("Overview 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Overview 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("Overview 오류", f"Overview 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_feature_csv_figure(self):
        if not self._get_selected_feature_csv_paths():
            self.refresh_feature_csv_candidates()
        feature_paths = self._get_selected_feature_csv_paths()
        if not feature_paths:
            messagebox.showwarning("Feature Plot", "cycle_features.csv, phase_features.csv, comparison_features.csv를 찾지 못했습니다.")
            return

        self._update_progress(0, "Feature plot 생성 중...")
        try:
            fig, has_data, warnings = create_feature_plot(
                feature_paths,
                self.feature_plot_type_var.get() if hasattr(self, "feature_plot_type_var") else FEATURE_PLOT_TYPES[0],
                reference_condition=self._get_feature_reference_condition(),
            )
            if has_data:
                self._show_plot_window(fig, "Feature Plot")
            else:
                plt.close(fig)
                messagebox.showwarning("Feature Plot", "표시할 수 있는 feature 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("Feature Plot 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Feature plot 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("Feature Plot 오류", f"Feature plot 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_muscle_activation_figure(self, scope_variables: List[str] = None):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        self._update_progress(0, "Muscle activation 생성 중...")
        try:
            fig, has_data, warnings = create_muscle_activation_figure(
                self.dataframes,
                self.file_paths,
                side_filter=self._get_muscle_side_filter(),
                threshold=self._get_muscle_threshold(),
                scope_variables=scope_variables,
            )
            if has_data:
                self._show_plot_window(fig, "Muscle Activation")
            else:
                plt.close(fig)
                messagebox.showwarning("Muscle Activation", "표시할 수 있는 muscle activation 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("Muscle Activation 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Muscle activation 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("Muscle Activation 오류", f"Muscle activation 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_activation_summary_figure(self, scope_variables: List[str] = None):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        self._update_progress(0, "Activation 요약 생성 중...")
        try:
            fig, has_data, warnings, _ = create_activation_summary_figure(
                self.dataframes,
                self.file_paths,
                side_filter=self._get_muscle_side_filter(),
                threshold=self._get_muscle_threshold(),
                scope_variables=scope_variables,
            )
            if has_data:
                self._show_plot_window(fig, "Activation Summary")
            else:
                plt.close(fig)
                messagebox.showwarning("Activation Summary", "표시할 수 있는 activation 요약 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("Activation Summary 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Activation 요약 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("Activation Summary 오류", f"Activation 요약 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_cci_figure(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        self._update_progress(0, "CCI 생성 중...")
        try:
            fig, has_data, warnings, _ = create_cci_figure(
                self.dataframes,
                self.file_paths,
                side_filter=self._get_muscle_side_filter(),
            )
            if has_data:
                self._show_plot_window(fig, "Co-contraction Index")
            else:
                plt.close(fig)
                messagebox.showwarning("CCI", "표시할 수 있는 CCI 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("CCI 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "CCI 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("CCI 오류", f"CCI 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_grf_cop_com_figure(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        self._update_progress(0, "GRF/COP/COM 생성 중...")
        try:
            fig, has_data, warnings = create_grf_cop_com_figure(self.dataframes, self.file_paths)
            if has_data:
                self._show_plot_window(fig, "GRF / COP / COM")
            else:
                plt.close(fig)
                messagebox.showwarning("GRF/COP/COM", "표시할 수 있는 GRF/COP/COM 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("GRF/COP/COM 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "GRF/COP/COM 생성 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("GRF/COP/COM 오류", f"GRF/COP/COM 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_symmetry_figure(self, selected_variables: List[str] = None):
        if not getattr(self, "analysis_results", None):
            messagebox.showwarning("좌우 대칭", "먼저 분석 실행을 완료해 주세요.")
            return

        selected_vars = selected_variables or list(self.select_listbox.get(0, tk.END))
        self._update_progress(0, "좌우 대칭 시각화 생성 중...")
        try:
            symmetry_table = build_symmetry_table(self.analysis_results, selected_vars)
            fig, has_data, warnings = create_symmetry_figure(symmetry_table)
            if has_data:
                self._show_plot_window(fig, "Symmetry Index")
            else:
                plt.close(fig)
                messagebox.showwarning("좌우 대칭", "표시할 수 있는 symmetry 결과가 없습니다.")
            if warnings:
                messagebox.showwarning("좌우 대칭 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "좌우 대칭 시각화 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("좌우 대칭 오류", f"좌우 대칭 시각화 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def _show_plot_window(self, fig, title):
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("800x600")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        self.last_fig = fig
        def on_close():
            plt.close(fig)
            top.destroy()
        top.protocol("WM_DELETE_WINDOW", on_close)

    def _write_plotly_html(self, fig, html_path):
        html = fig.to_html(include_plotlyjs=True, full_html=True)
        Path(html_path).write_text(html, encoding='utf-8')

    def _open_plotly_figure(self, fig, filename_hint):
        safe_name = get_safe_filename_component(filename_hint)
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_{safe_name}.html',
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(fig.to_html(include_plotlyjs=True, full_html=True))
            html_path = temp_file.name

        webbrowser.open_new_tab(Path(html_path).resolve().as_uri())

    def _get_plotly_go(self):
        try:
            importlib.invalidate_caches()
            import plotly.graph_objects as go
            return go
        except ImportError as e:
            install_command = f'"{sys.executable}" -m pip install plotly'
            messagebox.showerror(
                "Plotly 설치 필요",
                "현재 앱을 실행한 Python에서 Plotly를 찾지 못했습니다.\n\n"
                f"앱 실행 Python:\n{sys.executable}\n\n"
                "아래 명령어로 같은 Python에 설치한 뒤 앱을 다시 시작하세요:\n\n"
                f"{install_command}\n\n"
                f"원래 오류:\n{e}"
            )
            return None

    def _add_plotly_sd_band(self, fig, go, x, center, spread, color):
        x_values = np.asarray(x)
        center_values = np.asarray(center, dtype=float)
        spread_values = np.asarray(spread, dtype=float)
        upper = center_values + spread_values
        lower = center_values - spread_values

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor=_hex_to_rgba(color, 0.13),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            mode='lines',
            showlegend=False
        ))

    def _get_summary_stats(self, data):
        valid_data = np.asarray(data, dtype=float)
        valid_data = valid_data[np.isfinite(valid_data)]
        if len(valid_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0}

        return {
            'mean': np.nanmean(valid_data),
            'std': np.nanstd(valid_data, ddof=STATISTICS_CONFIG.get('ddof', 0)) if len(valid_data) > 1 else np.nan,
            'min': np.nanmin(valid_data),
            'max': np.nanmax(valid_data),
            'count': len(valid_data),
        }

    def _apply_plotly_report_style(self, fig, title, x_title, y_title):
        fig.update_layout(
            title=dict(text=title, x=0.02, xanchor='left', font=dict(size=20)),
            template=PLOTLY_TEMPLATE,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title=x_title,
            yaxis_title=y_title,
            font=dict(family=get_visualization_font_family(), size=13, color='#111827'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.04,
                xanchor='right',
                x=1,
                bgcolor='rgba(255,255,255,0.78)',
                bordercolor='rgba(148,163,184,0.35)',
                borderwidth=1,
            ),
            margin=dict(l=70, r=35, t=90, b=65),
            hoverlabel=dict(bgcolor='white', bordercolor='rgba(30,41,59,0.22)', font_size=12),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=PLOTLY_ZERO_LINE_COLOR,
            ticks='outside',
            tickcolor=PLOTLY_ZERO_LINE_COLOR,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=PLOTLY_GRID_COLOR,
            zeroline=True,
            zerolinecolor=PLOTLY_ZERO_LINE_COLOR,
            showline=True,
            linewidth=1,
            linecolor=PLOTLY_ZERO_LINE_COLOR,
            ticks='outside',
            tickcolor=PLOTLY_ZERO_LINE_COLOR,
        )

    def _create_interactive_graph_figure(self, var, go):
        clean_var_name = get_clean_variable_name(var)
        fig = go.Figure()
        has_data = False

        for color_idx, (path, df) in enumerate(zip(self.file_paths, self.dataframes)):
            if var not in df.columns:
                continue

            data = convert_values_for_display(
                var,
                pd.to_numeric(df[var], errors='coerce').to_numpy(dtype=float),
            )
            if data.size == 0 or not np.isfinite(data).any():
                continue

            x = np.arange(len(data))
            color = PLOTLY_COLORS[color_idx % len(PLOTLY_COLORS)]
            legend_label = format_legend_label(os.path.basename(path))
            stats = self._get_summary_stats(data)

            if stats['count'] > 1 and np.isfinite(stats['std']):
                self._add_plotly_sd_band(fig, go, x, data, SD_RANGE * stats['std'], color)

            fig.add_trace(go.Scatter(
                x=x,
                y=data,
                mode='lines',
                name=legend_label,
                line=dict(color=color, width=2.6),
                hovertemplate=(
                    "<b>" + legend_label + "</b><br>"
                    "Frame: %{x}<br>"
                    "Value: %{y:.4f}<br>"
                    f"Mean: {stats['mean']:.4f}<br>"
                    f"SD: {stats['std']:.4f}<br>"
                    f"N: {stats['count']}<extra></extra>"
                )
            ))
            has_data = True

        if not has_data:
            fig.add_annotation(
                text="저장할 수 있는 숫자 데이터가 없습니다.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
            )

        self._apply_plotly_report_style(fig, clean_var_name, "Frame / Index", get_axis_label(var))
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.08)
            )
        )
        return fig, has_data

    def _create_interactive_normalized_gait_cycle_figure(self, var, go):
        clean_var_name = get_clean_variable_name(var)
        fig = go.Figure()
        x_percent = np.linspace(0, 100, RESAMPLE_POINTS)
        has_data = False

        for color_idx, (path, df) in enumerate(zip(self.file_paths, self.dataframes)):
            _, normalized_cycles = self._get_normalized_gait_cycles(df, var)
            if normalized_cycles.size == 0:
                continue

            mean_cycle = np.nanmean(normalized_cycles, axis=0)
            std_cycle = np.nanstd(normalized_cycles, axis=0)
            color = PLOTLY_COLORS[color_idx % len(PLOTLY_COLORS)]
            legend_label = format_legend_label(os.path.basename(path))
            cycle_count = len(normalized_cycles)

            cycle_indices = np.arange(cycle_count)
            if cycle_count > MAX_INDIVIDUAL_CYCLES:
                cycle_indices = np.unique(np.linspace(0, cycle_count - 1, MAX_INDIVIDUAL_CYCLES).astype(int))

            for cycle_idx in cycle_indices:
                fig.add_trace(go.Scatter(
                    x=x_percent,
                    y=normalized_cycles[cycle_idx],
                    mode='lines',
                    name=f"{legend_label} cycle",
                    line=dict(color=_hex_to_rgba(color, 0.22), width=1),
                    hoverinfo='skip',
                    showlegend=False
                ))

            if len(normalized_cycles) > 1:
                self._add_plotly_sd_band(fig, go, x_percent, mean_cycle, SD_RANGE * std_cycle, color)

            fig.add_trace(go.Scatter(
                x=x_percent,
                y=mean_cycle,
                mode='lines',
                name=legend_label,
                line=dict(color=color, width=3.2, shape='spline', smoothing=0.45),
                hovertemplate=(
                    "<b>" + legend_label + "</b><br>"
                    "Gait Cycle: %{x:.1f}%<br>"
                    "Mean value: %{y:.4f}<br>"
                    f"Cycles: {cycle_count}<extra></extra>"
                )
            ))
            has_data = True

        if not has_data:
            fig.add_annotation(
                text="유효한 보행 주기를 찾지 못했습니다.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False
            )

        self._apply_plotly_report_style(fig, clean_var_name, "Gait Cycle (%)", get_axis_label(var))
        fig.update_xaxes(range=[0, 100])
        fig.update_layout(
            shapes=[
                dict(type='line', x0=0, x1=0, y0=0, y1=1, xref='x', yref='paper',
                     line=dict(color='rgba(30,41,59,0.32)', width=1.2, dash='dot')),
                dict(type='line', x0=50, x1=50, y0=0, y1=1, xref='x', yref='paper',
                     line=dict(color='rgba(30,41,59,0.24)', width=1, dash='dash')),
                dict(type='line', x0=100, x1=100, y0=0, y1=1, xref='x', yref='paper',
                     line=dict(color='rgba(30,41,59,0.32)', width=1.2, dash='dot')),
            ]
        )
        return fig, has_data

    def plot_interactive_graph(self):
        vars_to_plot = list(self.graph_listbox.get(0, tk.END))
        if not vars_to_plot:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        go = self._get_plotly_go()
        if go is None:
            return

        self._update_progress(0, "HTML 그래프 여는 중...")
        try:
            opened_count = 0
            for i, var in enumerate(vars_to_plot):
                progress = (i / len(vars_to_plot)) * 100
                self._update_progress(progress, f"HTML 그래프 보기: {var}")
                fig, has_data = self._create_interactive_graph_figure(var, go)
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    self._open_plotly_figure(fig, f"{clean_var_name}_interactive_graph")
                    opened_count += 1

            if opened_count == 0:
                messagebox.showwarning("HTML 그래프", "열 수 있는 숫자 데이터가 없습니다.")
            self._update_progress(100, "HTML 그래프 열기 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("HTML 그래프 오류", f"HTML 그래프를 여는 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def plot_interactive_time_phase_graph(self):
        vars_to_plot = list(self.graph_listbox.get(0, tk.END))
        if not vars_to_plot:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        go = self._get_plotly_go()
        if go is None:
            return

        self._update_progress(0, "HTML 시간 구간 그래프 여는 중...")
        try:
            opened_count = 0
            for i, var in enumerate(vars_to_plot):
                progress = (i / len(vars_to_plot)) * 100
                self._update_progress(progress, f"HTML 시간 구간 보기: {var}")
                fig, has_data = self._create_interactive_normalized_gait_cycle_figure(var, go)
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    self._open_plotly_figure(fig, f"{clean_var_name}_interactive_normalized_gait_cycle")
                    opened_count += 1

            if opened_count == 0:
                messagebox.showwarning("HTML 시간 구간 그래프", "유효한 보행 주기를 찾지 못했습니다.")
            self._update_progress(100, "HTML 시간 구간 그래프 열기 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror("HTML 시간 구간 그래프 오류", f"HTML 시간 구간 그래프를 여는 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")
    
    def save_results(self):
        selected_vars = list(self.select_listbox.get(0, tk.END))
        if not selected_vars:
            messagebox.showwarning(self.lang['save_error'], self.lang['save_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_path = filedialog.asksaveasfilename(title=self.lang['save_title'],defaultextension=".csv", filetypes=[(self.lang['csv_file'], "*.csv"), ("Excel 파일", "*.xlsx"), (self.lang['all_file'], "*.*")] )
        if not save_path:
            return
        self._update_progress(0, self.lang['saving_results'])
        try:
            df_results = build_results_dataframe(selected_vars, self.file_paths, self.analysis_results)
            save_results_dataframe(df_results, save_path)
            feature_csv_dir = os.path.dirname(save_path)
            feature_csv_paths = save_feature_csv_bundle(
                selected_vars,
                self.file_paths,
                self.analysis_results,
                feature_csv_dir,
            )
            self._update_progress(100, "저장 완료")
            extra_message = ""
            if feature_csv_paths:
                extra_message = f"\n\nFeature CSV {len(feature_csv_paths)}개도 같은 폴더에 저장되었습니다."
            messagebox.showinfo(self.lang['save_success'], f"{self.lang['save_success']}\n{save_path}{extra_message}")
            self.last_result_save_dir = os.path.dirname(save_path)
            self.refresh_feature_csv_candidates()
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror(self.lang['save_fail'], f"{self.lang['save_fail']}\n{str(e)}")
            self._update_progress(0, "오류 발생")

    def save_overview_figure(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_dir = filedialog.askdirectory(title='Overview PNG 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "Overview PNG 저장 중...")
        try:
            fig, has_data, warnings = create_overview_figure(
                self.dataframes,
                self.file_paths,
                preset_name=self.overview_preset_var.get() if hasattr(self, "overview_preset_var") else PANEL_PRESETS[0],
                reference_condition=self._get_reference_condition(),
                sd_multiplier=self._get_overview_sd_multiplier(),
                show_sd_band=True,
            )
            if has_data:
                output_base = os.path.join(save_dir, "gait_overview")
                save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight')
                messagebox.showinfo('저장 완료', f'Overview PNG가 저장되었습니다: {save_dir}')
            else:
                messagebox.showwarning("Overview 저장", "저장할 수 있는 overview 변수가 없습니다.")
            plt.close(fig)
            if warnings:
                messagebox.showwarning("Overview 저장 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Overview PNG 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"Overview PNG 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_overview_html(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        go = self._get_plotly_go()
        if go is None:
            return

        save_dir = filedialog.askdirectory(title='Overview HTML 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "Overview HTML 저장 중...")
        try:
            fig, has_data, warnings = create_overview_plotly_figure(
                self.dataframes,
                self.file_paths,
                preset_name=self.overview_preset_var.get() if hasattr(self, "overview_preset_var") else PANEL_PRESETS[0],
                reference_condition=self._get_reference_condition(),
                sd_multiplier=self._get_overview_sd_multiplier(),
                show_sd_band=True,
            )
            if has_data:
                html_path = os.path.join(save_dir, "gait_overview.html")
                self._write_plotly_html(fig, html_path)
                messagebox.showinfo('저장 완료', f'Overview HTML이 저장되었습니다: {html_path}')
            else:
                messagebox.showwarning("Overview HTML", "저장할 수 있는 overview 변수가 없습니다.")
            if warnings:
                messagebox.showwarning("Overview HTML 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Overview HTML 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"Overview HTML 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_feature_csv_figure(self):
        if not self._get_selected_feature_csv_paths():
            self.refresh_feature_csv_candidates()
        feature_paths = self._get_selected_feature_csv_paths()
        if not feature_paths:
            messagebox.showwarning("Feature Plot", "cycle_features.csv, phase_features.csv, comparison_features.csv를 찾지 못했습니다.")
            return
        save_dir = filedialog.askdirectory(title='Feature Plot PNG 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "Feature plot PNG 저장 중...")
        try:
            plot_type = self.feature_plot_type_var.get() if hasattr(self, "feature_plot_type_var") else FEATURE_PLOT_TYPES[0]
            fig, has_data, warnings = create_feature_plot(
                feature_paths,
                plot_type,
                reference_condition=self._get_feature_reference_condition(),
            )
            if has_data:
                safe_plot_name = get_safe_filename_component(plot_type.lower().replace(" ", "_"))
                output_base = os.path.join(save_dir, safe_plot_name)
                save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight')
                messagebox.showinfo('저장 완료', f'Feature plot PNG가 저장되었습니다: {save_dir}')
            else:
                messagebox.showwarning("Feature Plot 저장", "저장할 수 있는 feature 데이터가 없습니다.")
            plt.close(fig)
            if warnings:
                messagebox.showwarning("Feature Plot 저장 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "Feature plot PNG 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"Feature plot PNG 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_cci_results(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_dir = filedialog.askdirectory(title='CCI 결과 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "CCI 결과 저장 중...")
        try:
            fig, has_data, warnings, rows = create_cci_figure(
                self.dataframes,
                self.file_paths,
                side_filter=self._get_muscle_side_filter(),
            )
            if has_data:
                output_base = os.path.join(save_dir, "cci_summary")
                save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight')
                csv_path = os.path.join(save_dir, "integrated_cci.csv")
                save_integrated_cci_csv(rows, csv_path)
                messagebox.showinfo('저장 완료', f'CCI PNG와 CSV가 저장되었습니다:\n{save_dir}')
            else:
                messagebox.showwarning("CCI 저장", "저장할 수 있는 CCI 데이터가 없습니다.")
            plt.close(fig)
            if warnings:
                messagebox.showwarning("CCI 저장 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "CCI 결과 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"CCI 결과 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_grf_cop_com_figure(self):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_dir = filedialog.askdirectory(title='GRF/COP/COM PNG 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "GRF/COP/COM PNG 저장 중...")
        try:
            fig, has_data, warnings = create_grf_cop_com_figure(self.dataframes, self.file_paths)
            if has_data:
                output_base = os.path.join(save_dir, "grf_cop_com_overview")
                save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight')
                messagebox.showinfo('저장 완료', f'GRF/COP/COM PNG가 저장되었습니다: {save_dir}')
            else:
                messagebox.showwarning("GRF/COP/COM 저장", "저장할 수 있는 GRF/COP/COM 데이터가 없습니다.")
            plt.close(fig)
            if warnings:
                messagebox.showwarning("GRF/COP/COM 저장 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "GRF/COP/COM PNG 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"GRF/COP/COM PNG 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_recommended_visualization_files(
        self,
        selected_variables: List[str] = None,
        plan=None,
        feature_filter: str = "",
    ):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        selected_vars = list(selected_variables or self.select_listbox.get(0, tk.END))
        if not selected_vars:
            messagebox.showwarning("추천 시각화 저장", "저장할 변수를 먼저 선택해 주세요.")
            return

        save_dir = filedialog.askdirectory(title='추천 시각화 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "추천 시각화 저장 중...")
        try:
            saved_paths, warnings = self._write_recommended_visualization_files(
                save_dir,
                selected_vars,
                plan=plan,
                feature_filter=feature_filter,
            )
            if saved_paths:
                preview = "\n".join(saved_paths[:8])
                if len(saved_paths) > 8:
                    preview += f"\n... 외 {len(saved_paths) - 8}개"
                messagebox.showinfo(
                    "저장 완료",
                    f"추천 시각화 파일 {len(saved_paths)}개가 저장되었습니다:\n{preview}",
                )
            else:
                messagebox.showwarning("추천 시각화 저장", "저장할 수 있는 추천 시각화 데이터가 없습니다.")
            if warnings:
                messagebox.showwarning("추천 시각화 저장 경고", "\n".join(warnings[:10]))
            self._update_progress(100, "추천 시각화 저장 완료")
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"추천 시각화 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def _write_recommended_visualization_files(
        self,
        save_dir: str,
        selected_vars: List[str],
        plan=None,
        feature_filter: str = "",
    ):
        os.makedirs(save_dir, exist_ok=True)
        if plan is None:
            plan = getattr(self, "auto_analysis_plan", None)
        active_filter = feature_filter or getattr(plan, "feature_filter", "") or self._get_feature_analysis_filter()

        saved_paths = []
        warnings = []
        graph_vars = self._select_quick_graph_variables(selected_vars, plan)
        self._set_listbox_items(self.graph_listbox, graph_vars)

        saved, section_warnings = self._save_recommended_waveform_files(save_dir, graph_vars)
        saved_paths.extend(saved)
        warnings.extend(section_warnings)

        if active_filter in ("All analyzable", "Muscle activation"):
            saved, section_warnings = self._save_recommended_muscle_files(save_dir, selected_vars)
            saved_paths.extend(saved)
            warnings.extend(section_warnings)

        if active_filter in ("All analyzable", "GRF / COP / COM"):
            saved, section_warnings = self._save_recommended_grf_file(save_dir)
            saved_paths.extend(saved)
            warnings.extend(section_warnings)

        if active_filter in ("All analyzable", "Symmetry pairs"):
            saved, section_warnings = self._save_recommended_symmetry_files(save_dir, selected_vars)
            saved_paths.extend(saved)
            warnings.extend(section_warnings)

        return saved_paths, warnings

    def _save_recommended_waveform_files(self, save_dir: str, graph_vars: List[str]):
        saved_paths = []
        warnings = []
        if not graph_vars:
            return saved_paths, warnings

        x_axis_mode = self._get_graph_x_axis_mode()
        waveform_dir = os.path.join(save_dir, "waveforms")
        for i, var in enumerate(graph_vars):
            progress = (i / max(len(graph_vars), 1)) * 60
            self._update_progress(progress, f"추천 waveform 저장: {var}")
            fig, ax = plt.subplots(figsize=(12, 7) if x_axis_mode == "percent gait cycle" else (10, 6))
            if x_axis_mode == "percent gait cycle":
                has_data, plot_warnings = draw_gait_cycle_waveform(
                    ax,
                    self.dataframes,
                    self.file_paths,
                    var,
                    show_mean_sd=True,
                    show_phase_shading=True,
                    show_peak_markers=True,
                )
            else:
                has_data, plot_warnings = draw_raw_waveform(
                    ax,
                    self.dataframes,
                    self.file_paths,
                    var,
                    show_mean_sd=True,
                    prefer_time_axis=True,
                )
            warnings.extend(plot_warnings)
            plt.tight_layout()
            if has_data:
                safe_var_name = get_safe_filename_component(get_clean_variable_name(var))
                output_base = os.path.join(waveform_dir, f"{safe_var_name}_recommended_waveform")
                saved_paths.extend(save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight'))
            plt.close(fig)
        return saved_paths, warnings

    def _save_recommended_muscle_files(self, save_dir: str, selected_vars: List[str] = None):
        saved_paths = []
        warnings = []

        fig, has_data, section_warnings, _ = create_activation_summary_figure(
            self.dataframes,
            self.file_paths,
            side_filter=self._get_muscle_side_filter(),
            threshold=self._get_muscle_threshold(),
            scope_variables=selected_vars,
        )
        warnings.extend(section_warnings)
        if has_data:
            output_base = os.path.join(save_dir, "recommended_activation_summary")
            saved_paths.extend(save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight'))
        plt.close(fig)

        fig, has_data, section_warnings, rows = create_cci_figure(
            self.dataframes,
            self.file_paths,
            side_filter=self._get_muscle_side_filter(),
        )
        warnings.extend(section_warnings)
        if has_data:
            output_base = os.path.join(save_dir, "recommended_cci_summary")
            saved_paths.extend(save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight'))
            csv_path = os.path.join(save_dir, "recommended_integrated_cci.csv")
            save_integrated_cci_csv(rows, csv_path)
            saved_paths.append(csv_path)
        plt.close(fig)
        return saved_paths, warnings

    def _save_recommended_grf_file(self, save_dir: str):
        fig, has_data, warnings = create_grf_cop_com_figure(self.dataframes, self.file_paths)
        saved_paths = []
        if has_data:
            output_base = os.path.join(save_dir, "recommended_grf_cop_com_overview")
            saved_paths.extend(save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight'))
        plt.close(fig)
        return saved_paths, warnings

    def _save_recommended_symmetry_files(self, save_dir: str, selected_vars: List[str]):
        if not getattr(self, "analysis_results", None):
            return [], ["Symmetry 저장: 먼저 분석 결과가 필요합니다."]

        symmetry_table = build_symmetry_table(self.analysis_results, selected_vars)
        if symmetry_table.empty:
            return [], ["Symmetry 저장: 저장할 좌우 대칭 결과가 없습니다."]

        saved_paths = []
        csv_path = os.path.join(save_dir, "recommended_symmetry_index.csv")
        symmetry_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        saved_paths.append(csv_path)

        fig, has_data, warnings = create_symmetry_figure(symmetry_table)
        if has_data:
            output_base = os.path.join(save_dir, "recommended_symmetry_index")
            saved_paths.extend(save_figure(fig, output_base, formats=("png",), dpi=300, bbox_inches='tight'))
        plt.close(fig)
        return saved_paths, warnings

    def save_dashboard(self, section_scope: str = "all"):
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        if not getattr(self, "analysis_results", None):
            messagebox.showwarning("Dashboard", "먼저 분석 실행을 완료해 주세요.")
            return

        self._update_progress(0, "Dashboard 생성 중...")
        try:
            selected_vars = list(self.select_listbox.get(0, tk.END)) if hasattr(self, "select_listbox") else []
            graph_vars = list(self.graph_listbox.get(0, tk.END)) if hasattr(self, "graph_listbox") else []
            if not getattr(self, "feature_csv_paths", None):
                self.refresh_feature_csv_candidates()

            output_root = getattr(self, "current_run_output_dir", "") or os.path.join(os.getcwd(), "output")
            dashboard_path, summary_path, warnings = create_visualization_dashboard(
                self.dataframes,
                self.file_paths,
                selected_vars,
                graph_vars,
                self.analysis_results,
                feature_csv_paths=getattr(self, "feature_csv_paths", []),
                output_root=output_root,
                section_scope=section_scope,
            )
            self._update_progress(100, "Dashboard 생성 완료")
            messagebox.showinfo(
                "Dashboard 저장 완료",
                f"dashboard.html이 생성되었습니다:\n{dashboard_path}\n\n요약 파일:\n{summary_path}"
            )
            if warnings:
                messagebox.showwarning("Dashboard 경고", "\n".join(warnings[:10]))
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('Dashboard 저장 실패', f"Dashboard 생성 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")
    
    def save_graphs(self):
        graph_vars = list(self.graph_listbox.get(0, tk.END))
        if not graph_vars:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_dir = filedialog.askdirectory(title='그래프 저장 폴더 선택')
        if not save_dir:
            return
        self._update_progress(0, "그래프 저장 중...")
        try:
            saved_count = 0
            warnings = []
            x_axis_mode = self._get_graph_x_axis_mode()
            for i, var in enumerate(graph_vars):
                progress = (i / len(graph_vars)) * 100
                self._update_progress(progress, f"그래프 저장: {var}")
                fig, ax = plt.subplots(figsize=(12, 7) if x_axis_mode == "percent gait cycle" else (10, 6))
                if x_axis_mode == "percent gait cycle":
                    has_data, plot_warnings = draw_gait_cycle_waveform(
                        ax,
                        self.dataframes,
                        self.file_paths,
                        var,
                        show_mean_sd=True,
                        show_phase_shading=True,
                        show_peak_markers=True,
                    )
                else:
                    has_data, plot_warnings = draw_raw_waveform(
                        ax,
                        self.dataframes,
                        self.file_paths,
                        var,
                        show_mean_sd=True,
                        prefer_time_axis=True,
                    )
                warnings.extend(plot_warnings)
                plt.tight_layout()
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    safe_var_name = get_safe_filename_component(clean_var_name)
                    output_base = os.path.join(save_dir, f"{safe_var_name}_graph")
                    save_figure(fig, output_base, formats=("png", "svg"), dpi=300, bbox_inches='tight')
                    saved_count += 1
                plt.close(fig)
            self._update_progress(100, "그래프 저장 완료")
            if saved_count == 0:
                messagebox.showwarning("그래프 저장", "저장할 수 있는 숫자 데이터가 없습니다.")
            else:
                messagebox.showinfo('저장 완료', f'그래프 {saved_count}개가 PNG/SVG로 저장되었습니다: {save_dir}')
            if warnings:
                messagebox.showwarning("그래프 저장 경고", "\n".join(warnings[:10]))
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"그래프 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")
    
    def save_time_phase_graphs(self):
        graph_vars = list(self.graph_listbox.get(0, tk.END))
        if not graph_vars:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return
        save_dir = filedialog.askdirectory(title='시간 구간 그래프 저장 폴더 선택')
        if not save_dir:
            return
        self._update_progress(0, "시간 구간 그래프 저장 중...")
        try:
            saved_count = 0
            for i, var in enumerate(graph_vars):
                progress = (i / len(graph_vars)) * 100
                self._update_progress(progress, f"시간 구간 그래프 저장: {var}")
                fig, ax = plt.subplots(figsize=(12, 7))
                has_data = self._draw_normalized_gait_cycle_graph(ax, var)
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    plt.tight_layout()
                    safe_var_name = get_safe_filename_component(clean_var_name)
                    img_path = os.path.join(save_dir, f"{safe_var_name}_normalized_gait_cycle.png")
                    fig.savefig(img_path, dpi=300, bbox_inches='tight')
                    saved_count += 1
                plt.close(fig)
            self._update_progress(100, "시간 구간 그래프 저장 완료")
            messagebox.showinfo('저장 완료', f'시간 구간 그래프 {saved_count}개가 저장되었습니다: {save_dir}')
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"시간 구간 그래프 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_interactive_graphs(self):
        graph_vars = list(self.graph_listbox.get(0, tk.END))
        if not graph_vars:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        go = self._get_plotly_go()
        if go is None:
            return

        save_dir = filedialog.askdirectory(title='HTML 그래프 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "HTML 그래프 저장 중...")
        try:
            saved_count = 0
            for i, var in enumerate(graph_vars):
                progress = (i / len(graph_vars)) * 100
                self._update_progress(progress, f"HTML 그래프 저장: {var}")
                fig, has_data = self._create_interactive_graph_figure(var, go)
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    safe_var_name = get_safe_filename_component(clean_var_name)
                    html_path = os.path.join(save_dir, f"{safe_var_name}_interactive_graph.html")
                    self._write_plotly_html(fig, html_path)
                    saved_count += 1

            self._update_progress(100, "HTML 그래프 저장 완료")
            messagebox.showinfo('저장 완료', f'HTML 그래프 {saved_count}개가 저장되었습니다: {save_dir}')
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"HTML 그래프 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")

    def save_interactive_time_phase_graphs(self):
        graph_vars = list(self.graph_listbox.get(0, tk.END))
        if not graph_vars:
            messagebox.showwarning(self.lang['graph_error'], self.lang['graph_error'])
            return
        if not self.dataframes:
            messagebox.showwarning(self.lang['data_error'], self.lang['data_error'])
            return

        go = self._get_plotly_go()
        if go is None:
            return

        save_dir = filedialog.askdirectory(title='HTML 시간 구간 그래프 저장 폴더 선택')
        if not save_dir:
            return

        self._update_progress(0, "HTML 시간 구간 그래프 저장 중...")
        try:
            saved_count = 0
            for i, var in enumerate(graph_vars):
                progress = (i / len(graph_vars)) * 100
                self._update_progress(progress, f"HTML 시간 구간 그래프 저장: {var}")
                fig, has_data = self._create_interactive_normalized_gait_cycle_figure(var, go)
                if has_data:
                    clean_var_name = get_clean_variable_name(var)
                    safe_var_name = get_safe_filename_component(clean_var_name)
                    html_path = os.path.join(save_dir, f"{safe_var_name}_interactive_normalized_gait_cycle.html")
                    self._write_plotly_html(fig, html_path)
                    saved_count += 1

            self._update_progress(100, "HTML 시간 구간 그래프 저장 완료")
            messagebox.showinfo('저장 완료', f'HTML 시간 구간 그래프 {saved_count}개가 저장되었습니다: {save_dir}')
            self.root.after(2000, lambda: self._update_progress(0, "준비됨"))
        except Exception as e:
            messagebox.showerror('저장 실패', f"HTML 시간 구간 그래프 저장 중 오류: {str(e)}")
            self._update_progress(0, "오류 발생")
    
    def on_close(self):
        try:
            plt.close('all')
            self.root.destroy()
        except Exception as e:
            print(f"종료 중 오류: {e}")


def _variables_from_analysis_results(analysis_results: Dict) -> List[str]:
    variables = []
    for result in analysis_results.values():
        if not isinstance(result, dict):
            continue
        variable = str(result.get("variable_name", "")).strip()
        if variable and variable not in variables:
            variables.append(variable)
    return variables


def get_result_variable_help_text() -> str:
    return """데이터 저장 결과 변수 설명

[일반 결과 파일: results.csv / results.xlsx]

파일명
  어떤 파일에서 계산된 결과인지 표시합니다.

변수명
  분석한 원본 변수 이름입니다. 예: knee_angle_r, hip_flexion_l

분석유형
  결과가 계산된 범위입니다.
  - 전체 데이터: 변수 전체 구간 통계
  - SCONE phase - EarlyStance/LateStance/LiftOff/Swing/Landing: state 값별 phase 통계
  - SCONE phase - stance/swing: composite phase 통계
  - 좌우 대칭: 좌우 pair 대칭성 결과

변수군 / 추천 Preset / 매칭 패턴
  변수명 패턴을 기준으로 앱이 자동 분류한 정보입니다.
  예: Joint kinematics, Joint torque/moment, Muscle activation, GRF

평균 / 표준편차 / 최댓값 / 최솟값 / 중앙값
  기본 통계값입니다.

범위
  최댓값 - 최솟값입니다. joint angle에서는 ROM처럼 해석할 수 있습니다.

RMS
  값의 전체적인 크기를 나타냅니다. 신호 크기 비교에 유용합니다.

AUC
  곡선 아래 면적입니다. 시간 동안 누적된 양으로 보면 됩니다.

Positive AUC / Negative AUC
  양수 구간과 음수 구간을 나눠 계산한 AUC입니다.
  torque, load, force 분석에서 방향성 있는 누적량을 볼 때 유용합니다.

Max Time / Max Percent
  최댓값이 나온 시간과 전체 구간 기준 위치입니다.

Min Time / Min Percent
  최솟값이 나온 시간과 전체 구간 기준 위치입니다.

Threshold
  활성/초과 여부를 판단하는 기준값입니다.

Threshold 초과구간
  값이 threshold보다 큰 연속 구간 개수입니다.

Onset / Offset
  threshold를 처음 넘은 시점과 마지막으로 벗어난 시점입니다.

Active Duration
  threshold를 넘은 총 지속 시간입니다.
  muscle activation, penalty, controller signal 분석에서 특히 유용합니다.

Pair 변수
  좌우 대칭 분석에 사용된 상대 변수입니다.
  예: knee_angle_r의 pair는 knee_angle_l

대칭 Feature
  좌우 비교에 사용한 대표값입니다.
  예: mean은 오른쪽 평균과 왼쪽 평균 비교, max는 오른쪽 최댓값과 왼쪽 최댓값 비교입니다.

오른쪽 값 / 왼쪽 값
  대칭 Feature에 해당하는 오른쪽/왼쪽 값입니다.
  예: 대칭 Feature가 max라면 오른쪽 값은 오른쪽 변수의 max, 왼쪽 값은 왼쪽 변수의 max입니다.

Symmetry Index
  좌우 차이를 백분율로 나타낸 값입니다.
  0에 가까울수록 좌우가 비슷합니다.

데이터수
  통계 계산에 사용된 유효 데이터 포인트 수입니다.

상태컬럼
  SCONE phase 분석에 사용한 state 컬럼입니다.
  예: leg1_r.state, leg0_l.state


[자동 feature CSV]

cycle_features.csv
  전체 gait cycle 또는 전체 구간 기준 feature를 긴 형식으로 저장합니다.
  주요 컬럼: file, condition, variable, phase, feature, value

phase_features.csv
  SCONE phase별 feature를 저장합니다.
  EarlyStance, LateStance, LiftOff, Swing, Landing, stance, swing 결과가 들어갑니다.

comparison_features.csv
  조건 간 차이값을 저장합니다.
  주요 값: condition_value, reference_value, delta, delta_percent, delta_timing

condition_comparison_summary.csv
  기준 조건 대비 다른 조건의 mean/max/rom/rms/auc 변화를 보기 쉽게 요약한 파일입니다.


[해석 팁]

joint angle
  ROM, max/min timing, phase별 평균을 주로 봅니다.

joint load / torque / force
  max, AUC, positive/negative AUC, 조건 간 delta를 주로 봅니다.

muscle activation
  onset, offset, active duration, RMS, AUC를 주로 봅니다.

GRF / COP / COM
  peak, impulse 계열 feature와 조건 비교를 주로 봅니다.

symmetry
  Symmetry Index가 0에서 얼마나 벗어나는지 확인합니다.
"""


def get_variable_unit_help_text() -> str:
    return """변수 단위 설명

이 도움말은 현재 앱에서 그래프/결과 저장에 표시하는 단위 기준입니다.
프로젝트에 포함된 config/scone_variables_unit.csv 단위표를 먼저 사용하고,
단위표의 unit은 표시/저장 단위, source_unit은 원본 SCONE 단위입니다.
단위표에 없는 특수한 사용자 정의 변수는 변수명 패턴으로 단위를 추정합니다.

[중요: 관절각 표시 단위]

SCONE/OpenSim 관절각/회전 성분 원본값은 radian(rad) 기준입니다.
현재 앱은 단위표에서 source_unit이 rad인 변수를 그래프와 저장 결과에서 degree(deg)로 변환해서 표시합니다.

  pelvis_tilt
  hip_flexion_*
  knee_angle_*
  ankle_angle_*
  pelvis.ori_*
  torso.ori_*
  femur/tibia/calcn/ground 등의 *.ori_x/y/z
  hip/knee/ankle/back 등의 *.rot_x/y/z
  hip/knee/ankle 이름과 angle 패턴이 함께 들어간 변수

예:
  knee_angle_r 원본값 1.5708 rad -> 표시/저장값 약 90 deg

주의:
  hip_flexion_r_u, knee_angle_r_u처럼 원본 단위가 rad/s인 각속도 변수는
  그래프와 저장 결과에서 deg/s로 변환해서 표시합니다.
  pelvis_tilt.DA처럼 원본 단위가 rad/s^2인 각가속도 변수는 deg/s^2로 표시합니다.


[시간 / 상태]

time
  s

*.state, *_state
  state index입니다. 실제 물리 단위는 아니며, 보행 phase 구분값으로 사용합니다.


[좌표 / 위치 / 속도]

COP, COM position, *.pos.x/y/z
  m

linear velocity, com_lin_vel
  m/s

angular velocity
  deg/s


[힘 / 하중 / 모멘트]

*.load, *_load
  BW
  체중으로 정규화된 하중으로 표시합니다.

*.force, *_force
  N

*.torque, *_torque, *.moment, *_moment, grm
  N*m

grf_norm
  BW

grf
  N


[근육 / 제어 신호]

activation, excitation, input
  dimensionless

reflex/controller 계열 신호(.rl, .rf, .rdp, .rdv, .rda)
  dimensionless

fiber_length, tendon_length
  m

fiber_length_norm, length_norm, fiber_velocity_norm, ce_vel_norm, mtu_force_norm
  dimensionless

ce_vel
  m/s


[운동량 / 기타]

lin_mom
  kg*m/s

ang_mom
  kg*m^2/s

penalty
  scalar


[부호 해석 주의]

양수/음수 방향은 모델의 좌표계와 DOF 정의에 따라 달라질 수 있습니다.
앱은 부호를 임의로 flexion/extension, medial/lateral 같은 해부학적 의미로 바꾸지 않고,
원본 부호를 유지한 상태에서 단위 변환만 적용합니다.
"""


# 이 파일을 직접 실행하면 애플리케이션을 시작합니다.
if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        
        # 폰트 설정
        selected_font = get_font_family()
        plt.rcParams['font.family'] = selected_font
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"폰트 설정 성공: {selected_font}")
        
        root = tk.Tk()
        app = MotionAnalysisApp(root)
        
        print("애플리케이션 초기화 완료")
        
        root.mainloop()
        
    except Exception as e:
        print(f"애플리케이션 시작 중 오류 발생: {e}")
        input("Enter 키를 눌러 종료...")


