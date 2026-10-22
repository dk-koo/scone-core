import pandas as pd

from utils.file_utils import format_legend_label, get_safe_filename_component, sanitize_dataframe_for_spreadsheet_export


def test_format_legend_label_shortens_scone_condition_name():
    assert format_legend_label("P2BW_W70_Gait10Slope.sto") == "P2BW_W70"
    assert format_legend_label("P2BW_W70.sto") == "P2BW_W70"


def test_get_safe_filename_component_replaces_windows_reserved_characters():
    assert get_safe_filename_component("hip/flexion:angle*R?") == "hip_flexion_angle_R"


def test_get_safe_filename_component_uses_fallback_for_empty_result():
    assert get_safe_filename_component("...") == "variable"


def test_sanitize_dataframe_for_spreadsheet_export_prefixes_formula_like_text():
    df = pd.DataFrame({
        "file": ["=trial.csv", "+plus.csv", "-minus.csv", "@cmd.csv", "safe.csv"],
        "value": [-1.0, 2.0, 3.0, 4.0, 5.0],
    })

    sanitized = sanitize_dataframe_for_spreadsheet_export(df)

    assert sanitized["file"].tolist() == ["'=trial.csv", "'+plus.csv", "'-minus.csv", "'@cmd.csv", "safe.csv"]
    assert sanitized["value"].tolist() == [-1.0, 2.0, 3.0, 4.0, 5.0]
