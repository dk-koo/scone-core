import csv

import numpy as np

from utils.unit_utils import (
    UNIT_REFERENCE_PATH,
    convert_values_for_display,
    get_display_unit,
    get_source_unit,
    is_joint_angle_variable,
)


def test_unit_reference_uses_csv_units_for_known_variables():
    assert get_source_unit("pelvis_tx") == "m"
    assert get_display_unit("pelvis_tx") == "m"
    assert get_display_unit("leg1_r.grf_norm_y") == "BW"
    assert get_display_unit("leg1_r.state") == "state"
    assert get_display_unit("soleus_r.mtu_force_norm") == "dimensionless"


def test_unit_reference_csv_uses_degree_display_unit_for_angles():
    with UNIT_REFERENCE_PATH.open("r", encoding="utf-8-sig", newline="") as file_obj:
        rows = {row["variable"]: row for row in csv.DictReader(file_obj)}

    assert rows["knee_angle_r"]["unit"] == "deg"
    assert rows["knee_angle_r"]["source_unit"] == "rad"
    assert rows["femur_r.ori_x"]["unit"] == "deg"
    assert rows["femur_r.ori_x"]["source_unit"] == "rad"
    assert rows["knee_angle_r_u"]["unit"] == "deg/s"
    assert rows["knee_angle_r_u"]["source_unit"] == "rad/s"
    assert rows["pelvis_tilt.DA"]["unit"] == "deg/s^2"
    assert rows["pelvis_tilt.DA"]["source_unit"] == "rad/s^2"


def test_rad_angle_variables_are_displayed_in_degrees():
    values = np.array([0.0, np.pi / 2.0, np.pi])

    converted = convert_values_for_display("femur_r.ori_x", values)

    assert is_joint_angle_variable("femur_r.ori_x") is True
    assert get_source_unit("femur_r.ori_x") == "rad"
    assert get_display_unit("femur_r.ori_x") == "deg"
    np.testing.assert_allclose(converted, [0.0, 90.0, 180.0])


def test_rotational_velocity_variables_are_displayed_in_degrees_per_second():
    values = np.array([0.0, np.pi / 2.0, np.pi])

    converted = convert_values_for_display("knee_angle_r_u", values)

    assert is_joint_angle_variable("knee_angle_r_u") is False
    assert get_source_unit("knee_angle_r_u") == "rad/s"
    assert get_display_unit("knee_angle_r_u") == "deg/s"
    np.testing.assert_allclose(converted, [0.0, 90.0, 180.0])


def test_body_angular_velocity_variables_are_displayed_in_degrees_per_second():
    values = np.array([0.0, np.pi])

    converted = convert_values_for_display("pelvis.ang_vel.x", values)

    assert get_source_unit("pelvis.ang_vel.x") == "rad/s"
    assert get_display_unit("pelvis.ang_vel.x") == "deg/s"
    np.testing.assert_allclose(converted, [0.0, 180.0])


def test_angular_acceleration_variables_are_displayed_in_degrees_per_second_squared():
    assert get_source_unit("pelvis_tilt.DA") == "rad/s^2"
    assert get_display_unit("pelvis_tilt.DA") == "deg/s^2"
    np.testing.assert_allclose(convert_values_for_display("pelvis_tilt.DA", [np.pi]), [180.0])


def test_joint_rotation_components_from_csv_are_degrees():
    assert get_source_unit("hip_r.rot_x") == "rad"
    assert get_display_unit("hip_r.rot_x") == "deg"
    np.testing.assert_allclose(convert_values_for_display("hip_r.rot_x", [np.pi]), [180.0])
