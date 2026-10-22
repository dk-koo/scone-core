"""Variable unit reference and display-scale conversion helpers."""

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ANGLE_TO_DEG = 180.0 / np.pi
DEG_TO_ANGLE = np.pi / 180.0
UNIT_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "config" / "scone_variables_unit.csv"
DEGREE_CONVERSION_SOURCE_UNITS = {"rad", "rad/s", "rad/s^2"}


def _normalize_variable_name(variable_name: str) -> str:
    return str(variable_name or "").strip().lower()


@lru_cache(maxsize=1)
def _load_unit_reference() -> dict[str, dict[str, str]]:
    """Load the curated SCONE variable-unit table bundled with the project."""
    if not UNIT_REFERENCE_PATH.exists():
        return {}

    for encoding in ("utf-8-sig", "cp949"):
        try:
            return _read_unit_reference_csv(encoding)
        except UnicodeDecodeError:
            continue
    return {}


def _read_unit_reference_csv(encoding: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with UNIT_REFERENCE_PATH.open("r", encoding=encoding, newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            variable = _normalize_variable_name(row.get("variable", ""))
            if not variable:
                continue
            rows[variable] = {key: str(value or "").strip() for key, value in row.items()}
    return rows


def _get_unit_reference_row(variable_name: str) -> dict[str, str]:
    return _load_unit_reference().get(_normalize_variable_name(variable_name), {})


def get_source_unit(variable_name: str) -> str:
    """Return the raw unit from the bundled SCONE unit table, if known."""
    row = _get_unit_reference_row(variable_name)
    return row.get("source_unit") or row.get("unit", "")


def _get_reference_display_unit(variable_name: str) -> str:
    row = _get_unit_reference_row(variable_name)
    return row.get("unit", "") if row else ""


def _canonical_display_unit(source_unit: str) -> str:
    """Map verbose source-unit descriptions to concise plot/export labels."""
    unit = str(source_unit or "").strip()
    lower = unit.lower()

    if lower == "rad":
        return "deg"
    if lower == "rad/s":
        return "deg/s"
    if lower == "rad/s^2":
        return "deg/s^2"
    if lower == "dimensionless state code":
        return "state"
    if lower == "dimensionless control signal":
        return "dimensionless"
    if lower == "dimensionless / bw-normalized scalar":
        return "dimensionless/BW"
    if lower == "cost scalar":
        return "scalar"
    return unit


def _matches_angle_position_pattern(variable_name: str) -> bool:
    """Fallback for radian angle variables not listed in the unit table."""
    lower = _normalize_variable_name(variable_name)
    if lower.endswith(("_u", ".u")):
        return False
    if lower == "pelvis_tilt":
        return True
    if re.search(r"[._]ori_[xyz]$", lower):
        return True
    if re.search(r"[._]rot_[xyz]$", lower):
        return True
    if re.fullmatch(r"(hip_flexion|knee_angle|ankle_angle)_[lr]", lower):
        return True
    if re.search(r"(^|[._-])(hip|knee|ankle)[._-].*angle", lower):
        return True
    return False


def _matches_angular_velocity_pattern(variable_name: str) -> bool:
    """Fallback for angular velocity variables not listed in the unit table."""
    lower = _normalize_variable_name(variable_name)
    if "ang_vel" in lower:
        return True
    if lower.endswith("_u"):
        return _matches_angle_position_pattern(lower[:-2])
    if lower.endswith(".u"):
        return _matches_angle_position_pattern(lower[:-2])
    return False


def is_joint_angle_variable(variable_name: str) -> bool:
    """Return True for angle/orientation variables stored in radians."""
    source_unit = get_source_unit(variable_name)
    if source_unit:
        return source_unit.lower() == "rad"
    return _matches_angle_position_pattern(variable_name)


def _requires_degree_scale_conversion(variable_name: str) -> bool:
    source_unit = get_source_unit(variable_name).lower()
    if source_unit:
        return source_unit in DEGREE_CONVERSION_SOURCE_UNITS
    return _matches_angle_position_pattern(variable_name) or _matches_angular_velocity_pattern(variable_name)


def requires_degree_scale_conversion(variable_name: str) -> bool:
    """Return True when raw values are expected in radians and displayed in degrees."""
    return _requires_degree_scale_conversion(variable_name)


def convert_values_for_display(variable_name: str, values: Any) -> np.ndarray:
    """Convert raw values to the app's display/export unit for a variable."""
    array = np.asarray(values, dtype=float)
    if _requires_degree_scale_conversion(variable_name):
        return array * ANGLE_TO_DEG
    return array


def convert_values_from_display_to_source(variable_name: str, values: Any) -> np.ndarray:
    """Convert degree-based display values back to the app's expected raw source unit."""
    array = np.asarray(values, dtype=float)
    if _requires_degree_scale_conversion(variable_name):
        return array * DEG_TO_ANGLE
    return array


def get_display_unit(variable_name: str) -> str:
    """Infer the unit shown in results/plots after display conversion."""
    lower = _normalize_variable_name(variable_name)
    reference_display_unit = _get_reference_display_unit(lower)

    if lower == "time":
        return "s"
    if reference_display_unit:
        return _canonical_display_unit(reference_display_unit)
    if is_joint_angle_variable(lower):
        return "deg"
    if _matches_angular_velocity_pattern(lower):
        return "deg/s"
    if lower.endswith(".state") or lower.endswith("_state"):
        return "state"
    if lower.endswith((".activation", ".excitation", ".input")) or lower.endswith(("_activation", "_excitation", "_input")):
        return "dimensionless"
    if "fiber_velocity_norm" in lower or "ce_vel_norm" in lower or "mtu_force_norm" in lower:
        return "dimensionless"
    if "grf_norm" in lower:
        return "BW"
    if "grf" in lower and "grm" not in lower:
        return "N"
    if "grm" in lower:
        return "N*m"
    if "cop" in lower:
        return "m"
    if lower.endswith(".load") or "_load" in lower or lower.endswith("load"):
        return "BW"
    if ".force" in lower or "_force" in lower:
        return "N"
    if ".torque" in lower or "_torque" in lower or ".moment" in lower or "_moment" in lower:
        return "N*m"
    if lower.startswith(("com_", "com.")) or "com_pos" in lower or lower.endswith((".pos.x", ".pos.y", ".pos.z")):
        return "m"
    if "lin_vel" in lower or "com_lin_vel" in lower:
        return "m/s"
    if "ang_vel" in lower:
        return "deg/s"
    if lower.startswith("lin_mom"):
        return "kg*m/s"
    if lower.startswith("ang_mom"):
        return "kg*m^2/s"
    if "fiber_length" in lower or "tendon_length" in lower:
        return "dimensionless" if lower.endswith("_norm") or "length_norm" in lower else "m"
    if "ce_vel" in lower:
        return "m/s"
    if "penalty" in lower:
        return "scalar"
    if lower.endswith((".rl", ".rf", ".rdp", ".rdv", ".rda")):
        return "dimensionless"
    return ""


def get_axis_label(variable_name: str, default: str = "Value") -> str:
    unit = get_display_unit(variable_name)
    return f"Value ({unit})" if unit else default
