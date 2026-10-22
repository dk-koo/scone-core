import numpy as np

from src.data_loading import read_motion_data_file
from src.analysis_engine import analyze_variable_comprehensive
from utils.data_validator import DataValidator


def test_read_motion_data_file_uses_detected_tab_header(tmp_path):
    data_path = tmp_path / "sample.sto"
    data_path.write_text(
        "\n".join([
            "comment 1",
            "comment 2",
            "comment 3",
            "comment 4",
            "comment 5",
            "comment 6",
            "time\tknee_angle",
            "0.0\t10",
            "0.1\t11",
            "0.2\t12",
            "0.3\t13",
            "0.4\t14",
        ]),
        encoding="utf-8",
    )

    df, header_row = read_motion_data_file(str(data_path), DataValidator())

    assert header_row == 6
    assert list(df.columns) == ["time", "knee_angle"]
    assert df["knee_angle"].tolist() == [10, 11, 12, 13, 14]


def test_read_motion_data_file_converts_opensim_sto_to_internal_source_units(tmp_path):
    data_path = tmp_path / "opensim_sample.sto"
    data_path.write_text(
        "\n".join([
            "name opensim_sample",
            "version=1",
            "nRows=3",
            "nColumns=3",
            "inDegrees=yes",
            "endheader",
            "time knee_angle ankle_angle",
            "0.0 10.5 5.0",
            "0.1 11.0 5.5",
            "0.2 12.0 6.0",
        ]),
        encoding="utf-8",
    )

    df, header_row = read_motion_data_file(str(data_path), DataValidator())

    assert header_row == 6
    assert list(df.columns) == ["time", "knee_angle", "ankle_angle"]
    assert df["time"].tolist() == [0.0, 0.1, 0.2]
    np.testing.assert_allclose(df["knee_angle"].to_numpy(), np.deg2rad([10.5, 11.0, 12.0]))
    np.testing.assert_allclose(df["ankle_angle"].to_numpy(), np.deg2rad([5.0, 5.5, 6.0]))
    assert df.attrs["scone_sto_in_degrees"] is True
    assert df.attrs["degree_to_radian_columns"] == ["knee_angle", "ankle_angle"]


def test_in_degrees_sto_analysis_does_not_double_convert_degrees(tmp_path):
    data_path = tmp_path / "opensim_degrees.sto"
    data_path.write_text(
        "\n".join([
            "name opensim_degrees",
            "version=1",
            "nRows=3",
            "nColumns=2",
            "inDegrees=yes",
            "endheader",
            "time knee_angle_r",
            "0.0 0",
            "0.1 90",
            "0.2 180",
        ]),
        encoding="utf-8",
    )

    df, _ = read_motion_data_file(str(data_path), DataValidator())
    result = analyze_variable_comprehensive(df["knee_angle_r"].values, df["time"].values, "knee_angle_r")

    assert result["display_unit"] == "deg"
    assert result["common_features"]["max"] == 180.0


def test_suggest_header_row_returns_none_when_no_numeric_data(tmp_path):
    data_path = tmp_path / "notes.csv"
    data_path.write_text("a,b\nx,y\nm,n\n", encoding="utf-8")

    assert DataValidator().suggest_header_row(str(data_path)) is None


def test_read_motion_data_file_uses_configured_semicolon_separator(tmp_path):
    data_path = tmp_path / "semicolon.csv"
    data_path.write_text(
        "time;knee_angle;ankle_angle\n"
        "0.0;10;5\n"
        "0.1;11;6\n"
        "0.2;12;7\n"
        "0.3;13;8\n"
        "0.4;14;9\n"
        "0.5;15;10\n"
        "0.6;16;11\n"
        "0.7;17;12\n"
        "0.8;18;13\n"
        "0.9;19;14\n",
        encoding="utf-8",
    )

    df, header_row = read_motion_data_file(str(data_path), DataValidator())

    assert header_row == 0
    assert list(df.columns) == ["time", "knee_angle", "ankle_angle"]
    assert df["ankle_angle"].tolist()[-1] == 14


def test_read_motion_data_file_uses_whitespace_separator_for_txt(tmp_path):
    data_path = tmp_path / "whitespace.txt"
    data_path.write_text(
        "time knee_angle ankle_angle\n"
        "0.0 10 5\n"
        "0.1 11 6\n"
        "0.2 12 7\n"
        "0.3 13 8\n"
        "0.4 14 9\n"
        "0.5 15 10\n"
        "0.6 16 11\n"
        "0.7 17 12\n"
        "0.8 18 13\n"
        "0.9 19 14\n",
        encoding="utf-8",
    )

    df, header_row = read_motion_data_file(str(data_path), DataValidator())

    assert header_row == 0
    assert list(df.columns) == ["time", "knee_angle", "ankle_angle"]
    assert df["knee_angle"].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_suggest_header_row_uses_configured_twenty_row_search(tmp_path):
    data_path = tmp_path / "metadata_then_header.csv"
    metadata_lines = [f"metadata {idx}" for idx in range(12)]
    data_lines = [
        "time,knee_angle",
        "0.0,10",
        "0.1,11",
        "0.2,12",
        "0.3,13",
        "0.4,14",
        "0.5,15",
        "0.6,16",
        "0.7,17",
        "0.8,18",
        "0.9,19",
    ]
    data_path.write_text("\n".join(metadata_lines + data_lines), encoding="utf-8")

    df, header_row = read_motion_data_file(str(data_path), DataValidator())

    assert header_row == 12
    assert list(df.columns) == ["time", "knee_angle"]
    assert df["knee_angle"].tolist()[0] == 10
