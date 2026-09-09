import json

from rag_project.config import (
    DEFAULT_MIN_SIMILARITY,
    PROJECT_DIR,
)


INPUT_REPORT_PATH = (
    PROJECT_DIR / "retrieval_evaluation_report.json"
)

OUTPUT_REPORT_PATH = (
    PROJECT_DIR / "threshold_calibration_report.json"
)


def load_evaluation_cases() -> list:
    with open(
        INPUT_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    cases = report.get("cases", [])

    if not cases:
        raise ValueError(
            "评估报告中没有样例，请先运行 "
            "python -m rag_project.evaluate_retrieval"
        )

    return cases


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator

def get_expected_titles(
    case: dict,
) -> list[str]:
    """同时兼容新旧评估报告格式。"""

    if "expected_titles" in case:
        expected_titles = case["expected_titles"]
    else:
        expected_title = case.get(
            "expected_title"
        )
        expected_titles = (
            [expected_title]
            if expected_title is not None
            else []
        )

    if not isinstance(expected_titles, list):
        raise ValueError(
            "expected_titles 必须是列表"
        )

    return expected_titles


def is_relevant_case(case: dict) -> bool:
    """判断评估样例是否属于知识库相关问题。"""

    return bool(get_expected_titles(case))

def build_thresholds(
    start: float,
    stop: float,
    step: float,
) -> list[float]:
    if step <= 0:
        raise ValueError("step 必须大于 0")

    count = round((stop - start) / step)

    return [
        round(start + index * step, 2)
        for index in range(count + 1)
    ]


def calculate_metrics(
    cases: list,
    threshold: float,
) -> dict:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for case in cases:
        is_relevant = is_relevant_case(case)
        accepted = (
            case["max_score"] >= threshold
        )

        if is_relevant and accepted:
            true_positive += 1
        elif not is_relevant and accepted:
            false_positive += 1
        elif not is_relevant and not accepted:
            true_negative += 1
        else:
            false_negative += 1

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        len(cases),
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


def choose_best_result(
    results: list,
    baseline_threshold: float,
) -> dict:
    return max(
        results,
        key=lambda result: (
            result["balanced_accuracy"],
            result["f1"],
            result["accuracy"],
            -abs(
                result["threshold"]
                - baseline_threshold
            ),
        ),
    )


def summarize_scores(cases: list) -> dict:
    positive_scores = [
        case["max_score"]
        for case in cases
        if is_relevant_case(case)
    ]

    negative_scores = [
        case["max_score"]
        for case in cases
        if not is_relevant_case(case)
    ]

    if not positive_scores or not negative_scores:
        raise ValueError(
            "门槛校准必须同时包含正例和负例"
        )

    minimum_positive = min(positive_scores)
    maximum_negative = max(negative_scores)

    separation_margin = (
        minimum_positive - maximum_negative
    )

    midpoint_threshold = (
        minimum_positive + maximum_negative
    ) / 2

    return {
        "minimum_positive_score": minimum_positive,
        "maximum_positive_score": max(
            positive_scores
        ),
        "minimum_negative_score": min(
            negative_scores
        ),
        "maximum_negative_score": maximum_negative,
        "separation_margin": separation_margin,
        "midpoint_threshold": midpoint_threshold,
    }


def find_mistakes(
    cases: list,
    threshold: float,
) -> list:
    mistakes = []

    for case in cases:
        is_relevant = is_relevant_case(case)
        accepted = (
            case["max_score"] >= threshold
        )

        if is_relevant != accepted:
            mistakes.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_titles": (
                        get_expected_titles(case)
                    ),
                    "max_score": case["max_score"],
                    "accepted": accepted,
                }
            )

    return mistakes


def save_report(report: dict) -> None:
    with open(
        OUTPUT_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_metrics(
    title: str,
    result: dict,
) -> None:
    print("\n" + title)
    print("-" * 50)
    print("门槛：", result["threshold"])
    print("TP：", result["true_positive"])
    print("FP：", result["false_positive"])
    print("TN：", result["true_negative"])
    print("FN：", result["false_negative"])
    print(
        "Precision：",
        f"{result['precision']:.2%}",
    )
    print(
        "Recall：",
        f"{result['recall']:.2%}",
    )
    print(
        "Specificity：",
        f"{result['specificity']:.2%}",
    )
    print(
        "F1：",
        f"{result['f1']:.2%}",
    )
    print(
        "Balanced Accuracy：",
        f"{result['balanced_accuracy']:.2%}",
    )


def main() -> None:
    cases = load_evaluation_cases()

    thresholds = build_thresholds(
        start=0.20,
        stop=0.70,
        step=0.01,
    )

    results = [
        calculate_metrics(cases, threshold)
        for threshold in thresholds
    ]

    baseline_result = calculate_metrics(
        cases,
        DEFAULT_MIN_SIMILARITY,
    )

    selected_result = choose_best_result(
        results,
        DEFAULT_MIN_SIMILARITY,
    )

    score_summary = summarize_scores(cases)

    perfect_thresholds = [
        result["threshold"]
        for result in results
        if (
            result["balanced_accuracy"] == 1.0
            and result["accuracy"] == 1.0
        )
    ]

    report = {
        "settings": {
            "source_report": (
                INPUT_REPORT_PATH.name
            ),
            "candidate_start": 0.20,
            "candidate_stop": 0.70,
            "candidate_step": 0.01,
            "baseline_threshold": (
                DEFAULT_MIN_SIMILARITY
            ),
        },
        "score_summary": score_summary,
        "baseline": baseline_result,
        "selected": selected_result,
        "perfect_thresholds": perfect_thresholds,
        "selected_mistakes": find_mistakes(
            cases,
            selected_result["threshold"],
        ),
        "all_thresholds": results,
    }

    save_report(report)

    print_metrics(
        "当前门槛结果",
        baseline_result,
    )

    print_metrics(
        "自动选择结果",
        selected_result,
    )

    print("\n分数分布")
    print("-" * 50)
    print(
        "最低正例分数：",
        f"{score_summary['minimum_positive_score']:.4f}",
    )
    print(
        "最高负例分数：",
        f"{score_summary['maximum_negative_score']:.4f}",
    )
    print(
        "正负例间隔：",
        f"{score_summary['separation_margin']:.4f}",
    )
    print(
        "分数中点：",
        f"{score_summary['midpoint_threshold']:.4f}",
    )

    if perfect_thresholds:
        print(
            "满分门槛范围：",
            perfect_thresholds[0],
            "到",
            perfect_thresholds[-1],
        )
    else:
        print("当前候选范围内没有满分门槛")

    if (
        selected_result["threshold"]
        == DEFAULT_MIN_SIMILARITY
    ):
        print("\n当前默认门槛无需修改。")
    else:
        print(
            "\n候选门槛：",
            selected_result["threshold"],
        )
        print(
            "评估集较小，暂时不要自动修改 config.py。"
        )

    print(
        "\n校准报告已保存：",
        OUTPUT_REPORT_PATH,
    )


if __name__ == "__main__":
    main()