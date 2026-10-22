from utils.gait_analysis import get_state_name


def test_get_state_name_uses_canonical_liftoff_label():
    assert get_state_name(2) == "LiftOff"
