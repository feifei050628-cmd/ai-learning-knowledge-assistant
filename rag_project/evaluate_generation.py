import json
import re

from rag_project.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    PROJECT_DIR,
)


CASES_PATH = (
    PROJECT_DIR
    / "generation_evaluation_cases.json"
)

REPORT_PATH = (
    PROJECT_DIR
    / "generation_evaluation_report.json"
)

EVALUATION_MAX_NEW_TOKENS = 120

CITATION_PATTERN = re.compile(
    r"【资料(\d+)】"
)


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_evaluation_cases() -> list:
    with open(
        CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    if not cases:
        raise ValueError("生成评估数据集不能为空")

    return cases


def check_keyword_groups(
    answer: str,
    keyword_groups: list,
) -> tuple[list, float]:
    normalized_answer = answer.lower()
    details = []

    for alternatives in keyword_groups:
        matched_keyword = None

        for keyword in alternatives:
            if keyword.lower() in normalized_answer:
                matched_keyword = keyword
                break

        details.append(
            {
                "alternatives": alternatives,
                "matched": matched_keyword is not None,
                "matched_keyword": matched_keyword,
            }
        )

    matched_count = sum(
        detail["matched"]
        for detail in details
    )

    coverage = safe_divide(
        matched_count,
        len(keyword_groups),
    )

    return details, coverage


def extract_citations(answer: str) -> list[int]:
    return [
        int(number)
        for number in CITATION_PATTERN.findall(
            answer
        )
    ]


def citations_are_valid(
    citation_numbers: list[int],
    source_count: int,
) -> bool:
    if not citation_numbers:
        return False

    return all(
        1 <= number <= source_count
        for number in citation_numbers
    )


def evaluate_case(
    pipeline,
    case: dict,
) -> dict:
    result = pipeline.answer(
        query=case["query"],
        top_k=DEFAULT_TOP_K,
        min_similarity=DEFAULT_MIN_SIMILARITY,
        max_new_tokens=EVALUATION_MAX_NEW_TOKENS,
    )

    expected_passed = case["expected_passed"]
    pass_match = (
        result["passed"] == expected_passed
    )

    citation_numbers = extract_citations(
        result["answer"]
    )

    citation_present = bool(citation_numbers)

    citation_valid = citations_are_valid(
        citation_numbers,
        len(result["sources"]),
    )

    if expected_passed:
        source_title_hit = any(
            source["title"]
            == case["expected_title"]
            for source in result["sources"]
        )

        (
            keyword_details,
            keyword_coverage,
        ) = check_keyword_groups(
            result["answer"],
            case["keyword_groups"],
        )

        refusal_correct = None

        case_success = (
            pass_match
            and source_title_hit
            and keyword_coverage == 1.0
            and citation_valid
        )
    else:
        source_title_hit = None
        keyword_details = []
        keyword_coverage = None

        refusal_correct = (
            result["passed"] is False
            and result["answer"].strip()
            == "现有资料不足"
            and result["sources"] == []
        )

        case_success = (
            pass_match and refusal_correct
        )

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_passed": expected_passed,
        "actual_passed": result["passed"],
        "pass_match": pass_match,
        "expected_title": case["expected_title"],
        "source_title_hit": source_title_hit,
        "max_score": result["max_score"],
        "answer": result["answer"],
        "answer_length": len(result["answer"]),
        "keyword_details": keyword_details,
        "keyword_coverage": keyword_coverage,
        "citation_numbers": citation_numbers,
        "citation_present": citation_present,
        "citation_valid": citation_valid,
        "refusal_correct": refusal_correct,
        "case_success": case_success,
        "sources": result["sources"],
    }


def summarize_results(
    case_results: list,
) -> dict:
    positive_results = [
        result
        for result in case_results
        if result["expected_passed"]
    ]

    negative_results = [
        result
        for result in case_results
        if not result["expected_passed"]
    ]

    gate_correct = sum(
        result["pass_match"]
        for result in case_results
    )

    source_hits = sum(
        result["source_title_hit"] is True
        for result in positive_results
    )

    keyword_coverage_sum = sum(
        result["keyword_coverage"] or 0.0
        for result in positive_results
    )

    citation_present_count = sum(
        result["citation_present"]
        for result in positive_results
    )

    citation_valid_count = sum(
        result["citation_valid"]
        for result in positive_results
    )

    refusal_correct_count = sum(
        result["refusal_correct"] is True
        for result in negative_results
    )

    successful_cases = sum(
        result["case_success"]
        for result in case_results
    )

    return {
        "case_count": len(case_results),
        "positive_count": len(positive_results),
        "negative_count": len(negative_results),
        "gate_accuracy": safe_divide(
            gate_correct,
            len(case_results),
        ),
        "source_title_hit_rate": safe_divide(
            source_hits,
            len(positive_results),
        ),
        "average_keyword_coverage": (
            safe_divide(
                keyword_coverage_sum,
                len(positive_results),
            )
        ),
        "citation_presence_rate": safe_divide(
            citation_present_count,
            len(positive_results),
        ),
        "citation_validity_rate": safe_divide(
            citation_valid_count,
            len(positive_results),
        ),
        "refusal_accuracy": safe_divide(
            refusal_correct_count,
            len(negative_results),
        ),
        "overall_case_pass_rate": safe_divide(
            successful_cases,
            len(case_results),
        ),
    }


def print_case_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("用例：", result["id"])
    print("问题：", result["query"])
    print("通过门槛：", result["actual_passed"])
    print(
        "最高相似度：",
        f"{result['max_score']:.4f}",
    )
    print("回答：")
    print(result["answer"])

    if result["expected_passed"]:
        print(
            "正确来源命中：",
            result["source_title_hit"],
        )
        print(
            "关键词覆盖率：",
            f"{result['keyword_coverage']:.2%}",
        )
        print(
            "引用编号：",
            result["citation_numbers"],
        )
        print(
            "引用有效：",
            result["citation_valid"],
        )
    else:
        print(
            "拒答正确：",
            result["refusal_correct"],
        )

    print(
        "用例整体通过：",
        result["case_success"],
    )


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("生成质量评估汇总")
    print("=" * 60)

    print("样例总数：", metrics["case_count"])
    print("正例数量：", metrics["positive_count"])
    print("负例数量：", metrics["negative_count"])

    metric_names = {
        "gate_accuracy": "门槛判断准确率",
        "source_title_hit_rate": "正确来源命中率",
        "average_keyword_coverage": "平均关键词覆盖率",
        "citation_presence_rate": "引用出现率",
        "citation_validity_rate": "引用有效率",
        "refusal_accuracy": "拒答准确率",
        "overall_case_pass_rate": "整体用例通过率",
    }

    for key, name in metric_names.items():
        print(
            f"{name}："
            f"{metrics[key]:.2%}"
        )


def save_report(report: dict) -> None:
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n评估报告已保存：", REPORT_PATH)


def main() -> None:
    from rag_project.rag_pipeline import (
        RAGPipeline,
    )

    cases = load_evaluation_cases()

    print("正在加载完整 RAG Pipeline……")
    pipeline = RAGPipeline()

    case_results = []

    for case in cases:
        result = evaluate_case(
            pipeline,
            case,
        )
        case_results.append(result)
        print_case_result(result)

    metrics = summarize_results(case_results)

    report = {
        "settings": {
            "top_k": DEFAULT_TOP_K,
            "min_similarity": (
                DEFAULT_MIN_SIMILARITY
            ),
            "max_new_tokens": (
                EVALUATION_MAX_NEW_TOKENS
            ),
        },
        "metrics": metrics,
        "cases": case_results,
    }

    print_summary(metrics)
    save_report(report)


if __name__ == "__main__":
    main()