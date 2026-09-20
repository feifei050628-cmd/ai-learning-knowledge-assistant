"""按业务成本标定 RAG 双阈值门控，而不是手调一个余弦数字。"""

import argparse
import json
from pathlib import Path


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_gate_metrics(
    records: list[dict],
    threshold: float,
    score_field: str,
) -> dict:
    tp = fp = tn = fn = 0
    boundary_total = boundary_rejected = 0
    for record in records:
        answerable = bool(record["answerable"])
        accepted = float(record[score_field]) >= threshold
        if answerable and accepted:
            tp += 1
        elif answerable:
            fn += 1
        elif accepted:
            fp += 1
        else:
            tn += 1
        if record.get("category") == "negative_boundary":
            boundary_total += 1
            boundary_rejected += int(not accepted)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "accuracy": safe_divide(tp + tn, len(records)),
        "boundary_rejection_rate": safe_divide(
            boundary_rejected,
            boundary_total,
        ),
    }


def select_threshold(
    results: list[dict],
    minimum_recall: float = 0.95,
) -> dict:
    eligible = [
        result
        for result in results
        if result["recall"] >= minimum_recall
    ]
    if not eligible:
        raise ValueError("没有候选阈值满足最低正样本召回约束")
    return max(
        eligible,
        key=lambda result: (
            -result["fp"],
            result["boundary_rejection_rate"],
            result["precision"],
            result["threshold"],
        ),
    )


def calibrate(
    input_path: Path,
    output_path: Path,
    baseline: float,
    score_field: str,
    minimum_recall: float,
) -> dict:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    records = report["cases"]
    if not records:
        raise ValueError("评测记录不能为空")
    if any("answerable" not in record for record in records):
        raise ValueError("每条记录必须显式包含 answerable 标签")
    if any(score_field not in record for record in records):
        raise ValueError(f"记录中缺少分数字段：{score_field}")

    thresholds = [round(0.20 + index * 0.01, 2) for index in range(71)]
    results = [
        calculate_gate_metrics(records, threshold, score_field)
        for threshold in thresholds
    ]
    baseline_result = calculate_gate_metrics(
        records,
        baseline,
        score_field,
    )
    selected = select_threshold(results, minimum_recall)
    # 旧单阈值保留为灰区下界：低于它时证据明显不足；高于它但未达到
    # 数据标定出的高阈值时，仅允许摘录原文。这样既保留灰区，也避免被
    # 极少数高分离群负例（例如主题词偶然重合）把下界错误抬到高阈值。
    low_threshold = min(baseline, selected["threshold"] - 0.01)
    result = {
        "settings": {
            "input": input_path.name,
            "score_field": score_field,
            "minimum_recall": minimum_recall,
            "baseline_threshold": baseline,
        },
        "baseline": baseline_result,
        "recommended": selected,
        "dual_threshold": {
            "low": low_threshold,
            "high": selected["threshold"],
            "gray_zone_behavior": "extractive_only_with_incomplete_coverage",
        },
        "all_thresholds": results,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("retrieval_evaluation_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("gate_calibration_report.json"),
    )
    parser.add_argument("--baseline", type=float, default=0.48)
    parser.add_argument("--score-field", default="max_score")
    parser.add_argument("--minimum-recall", type=float, default=0.95)
    args = parser.parse_args()
    report = calibrate(
        args.input,
        args.output,
        args.baseline,
        args.score_field,
        args.minimum_recall,
    )
    print(json.dumps(report["baseline"], ensure_ascii=False, indent=2))
    print(json.dumps(report["recommended"], ensure_ascii=False, indent=2))
    print(json.dumps(report["dual_threshold"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
