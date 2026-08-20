from gait_aqa.data.build_transfer_manifests import (
    _cmu_subject_group,
    _disabled_recording_group,
)


def test_cmu_group_uses_original_subject_directory() -> None:
    assert _cmu_subject_group("cmuconvert-max-01-09/02/02_01.bvh") == "cmu:02"


def test_disabled_group_keeps_recording_series_together() -> None:
    assert _disabled_recording_group("c03") == "disabled_gait:series_c"
