from rag_project.rag_gate_calibrate import (
    calculate_gate_metrics,
    select_threshold,
)


def make_records():
    return [
        {"answerable": True, "max_score": 0.90, "category": "fact"},
        {"answerable": True, "max_score": 0.70, "category": "code"},
        {
            "answerable": False,
            "max_score": 0.60,
            "category": "negative_boundary",
        },
        {
            "answerable": False,
            "max_score": 0.30,
            "category": "negative_unrelated",
        },
    ]


def test_metrics_use_answerable_not_hit():
    result = calculate_gate_metrics(
        make_records(),
        threshold=0.65,
        score_field="max_score",
    )
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["boundary_rejection_rate"] == 1.0


def test_selection_minimizes_false_accepts_under_recall_constraint():
    results = [
        calculate_gate_metrics(make_records(), threshold, "max_score")
        for threshold in (0.50, 0.65, 0.75)
    ]
    selected = select_threshold(results, minimum_recall=1.0)
    assert selected["threshold"] == 0.65
    assert selected["fp"] == 0
