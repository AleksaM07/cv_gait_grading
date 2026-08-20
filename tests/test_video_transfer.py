import numpy as np

from gait_aqa.training.train_video_transfer import (
    _aggregate_policy_scores,
    _fit_whitening_basis,
    _group_average_metrics,
    _leave_one_group_mean_predictions,
    _nested_group_predictions,
    _sample_window_indices,
)


def test_sample_windows_are_bounded_and_deterministic() -> None:
    first = _sample_window_indices(90, 30.0, 16, 2, 2.0)
    second = _sample_window_indices(90, 30.0, 16, 2, 2.0)

    assert len(first) == 2
    assert all(len(indices) == 16 for indices in first)
    assert all(indices.min() >= 0 and indices.max() < 90 for indices in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_short_clip_sampling_repeats_without_out_of_bounds() -> None:
    windows = _sample_window_indices(8, 30.0, 16, 1, 2.0)

    assert windows[0].shape == (16,)
    assert windows[0][0] == 0
    assert windows[0][-1] == 7


def test_whitening_basis_has_expected_shape_and_scale() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 12))
    basis = _fit_whitening_basis(features, components=5)
    transformed = basis.transform(features)

    assert basis.components.shape == (5, 12)
    assert transformed.shape == (40, 5)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(transformed.std(axis=0, ddof=1), 1.0, atol=1e-10)


def test_nested_group_predictions_cover_every_sample() -> None:
    groups = np.repeat(["a", "b", "c", "d"], 3)
    target = np.linspace(10.0, 90.0, len(groups))
    features = np.c_[target, np.sin(target)]

    predictions, selected = _nested_group_predictions(
        features, target, groups, (0.01, 1.0, 100.0)
    )

    assert predictions.shape == target.shape
    assert np.isfinite(predictions).all()
    assert set(selected) == set(groups)


def test_policy_aggregation_excludes_invalid_scenarios() -> None:
    results = [
        {
            "scenario": "forward",
            "predicted_research_score": 80.0,
            "score_valid": True,
            "distribution_warning": False,
        },
        {
            "scenario": "turn",
            "predicted_research_score": 60.0,
            "score_valid": True,
            "distribution_warning": False,
        },
        {
            "scenario": "lateral",
            "predicted_research_score": 99.0,
            "score_valid": False,
            "distribution_warning": True,
        },
    ]

    aggregate = _aggregate_policy_scores(results, expected_video_count=3)

    assert aggregate["policy_research_score"] is None
    assert aggregate["diagnostic_mean_of_valid_scenarios"] == 70.0
    assert aggregate["valid_videos"] == 2
    assert aggregate["scenario_set_complete"] is True
    assert aggregate["policy_score_valid"] is False
    assert aggregate["distribution_warning"] is True


def test_policy_aggregation_returns_score_for_complete_valid_set() -> None:
    results = [
        {
            "scenario": scenario,
            "predicted_research_score": score,
            "score_valid": True,
            "distribution_warning": False,
        }
        for scenario, score in (
            ("forward", 60.0),
            ("turn", 70.0),
            ("lateral", 80.0),
        )
    ]

    aggregate = _aggregate_policy_scores(results, expected_video_count=3)

    assert aggregate["policy_research_score"] == 70.0
    assert aggregate["policy_score_valid"] is True
    assert aggregate["scenario_set_complete"] is True


def test_group_average_metrics_evaluate_policy_means() -> None:
    target = np.asarray([10.0, 20.0, 70.0, 80.0])
    predictions = np.asarray([15.0, 15.0, 75.0, 75.0])
    groups = np.asarray(["a", "a", "b", "b"])

    metrics, rows = _group_average_metrics(target, predictions, groups)

    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert [row["scenario_count"] for row in rows] == [2.0, 2.0]


def test_leave_one_group_mean_never_uses_held_out_target() -> None:
    target = np.asarray([10.0, 20.0, 80.0, 100.0])
    groups = np.asarray(["a", "a", "b", "b"])

    predictions = _leave_one_group_mean_predictions(target, groups)

    assert np.allclose(predictions[:2], 90.0)
    assert np.allclose(predictions[2:], 15.0)
