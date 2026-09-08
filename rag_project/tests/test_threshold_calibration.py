import pytest

from rag_project.calibrate_threshold import (
    build_thresholds,
    calculate_metrics,
    choose_best_result,
)


SAMPLE_CASES = [
    {
        "id": "positive_1",
        "expected_title": "RAG基础",
        "max_score": 0.8,
    },
    {
        "id": "positive_2",
        "expected_title": "RAG基础",
        "max_score": 0.6,
    },
    {
        "id": "negative_1",
        "expected_title": None,
        "max_score": 0.4,
    },
    {
        "id": "negative_2",
        "expected_title": None,
        "max_score": 0.2,
    },
]


def test_build_thresholds_includes_boundaries():
    thresholds = build_thresholds(
        start=0.20,
        stop=0.70,
        step=0.01,
    )

    assert thresholds[0] == 0.20
    assert thresholds[-1] == 0.70
    assert len(thresholds) == 51


def test_metrics_are_perfect_at_clear_threshold():
    result = calculate_metrics(
        SAMPLE_CASES,
        threshold=0.5,
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["true_negative"] == 2
    assert result["false_negative"] == 0
    assert result["f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0


def test_low_threshold_creates_false_positive():
    result = calculate_metrics(
        SAMPLE_CASES,
        threshold=0.3,
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 0
    assert result["precision"] == pytest.approx(
        2 / 3
    )


def test_high_threshold_creates_false_negative():
    result = calculate_metrics(
        SAMPLE_CASES,
        threshold=0.7,
    )

    assert result["true_positive"] == 1
    assert result["false_positive"] == 0
    assert result["true_negative"] == 2
    assert result["false_negative"] == 1
    assert result["recall"] == 0.5


def test_best_result_prefers_current_threshold():
    results = [
        {
            "threshold": 0.38,
            "balanced_accuracy": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
        },
        {
            "threshold": 0.40,
            "balanced_accuracy": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
        },
        {
            "threshold": 0.44,
            "balanced_accuracy": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
        },
    ]

    selected = choose_best_result(
        results,
        baseline_threshold=0.40,
    )

    assert selected["threshold"] == 0.40