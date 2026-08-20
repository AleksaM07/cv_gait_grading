from pathlib import Path

import pytest

from gait_aqa.data.organize_walking_datasets import (
    _disabled_clip_id,
    _gahu_clip_id,
)


def test_disabled_clip_id_includes_category() -> None:
    clip_id, category = _disabled_clip_id(Path("Non Assistive/C03.mp4"))

    assert clip_id == "disabled_gait__non_assistive__c03"
    assert category == "non_assistive"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Originals/S001.avi", ("gahu__s001__original", "s001", "original")),
        (
            "Sx_Track 2_Left/S044_T2_L.avi",
            ("gahu__s044__track2_left", "s044", "track2_left"),
        ),
        (
            "Sx_Track 3/S010_T3.avi",
            ("gahu__s010__track3_center", "s010", "track3_center"),
        ),
    ],
)
def test_gahu_clip_id_preserves_subject_track_and_view(
    path: str, expected: tuple[str, str, str]
) -> None:
    assert _gahu_clip_id(Path(path)) == expected
