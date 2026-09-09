import pytest

from rag_project.rag_pipeline import RAGPipeline
from rag_project.evaluate_generation import (
    check_keyword_groups,
    citations_are_valid,
    extract_citations,
    summarize_results,
    get_expected_titles,
)


def test_keyword_groups_allow_alternatives():
    details, coverage = check_keyword_groups(
        answer=(
            "训练时先进行前向传播，"
            "然后调用 backward。"
        ),
        keyword_groups=[
            ["前向传播"],
            ["反向传播", "backward"],
        ],
    )

    assert details[0]["matched"] is True
    assert details[1]["matched_keyword"] == "backward"
    assert coverage == 1.0


def test_keyword_coverage_detects_missing_fact():
    _, coverage = check_keyword_groups(
        answer="Docker可以打包代码和依赖。",
        keyword_groups=[
            ["代码"],
            ["依赖"],
            ["运行环境", "环境"],
            ["镜像"],
        ],
    )

    assert coverage == 0.5


def test_extracts_and_validates_citations():
    answer = "答案来自【资料1】和【资料2】。"

    citation_numbers = extract_citations(answer)

    assert citation_numbers == [1, 2]
    assert citations_are_valid(
        citation_numbers,
        source_count=2,
    ) is True

    assert citations_are_valid(
        [1, 3],
        source_count=2,
    ) is False

    assert citations_are_valid(
        [],
        source_count=2,
    ) is False


def test_summarizes_positive_and_negative_cases():
    case_results = [
        {
            "expected_passed": True,
            "pass_match": True,
            "source_title_hit": True,
            "keyword_coverage": 1.0,
            "citation_present": True,
            "citation_valid": True,
            "refusal_correct": None,
            "case_success": True,
        },
        {
            "expected_passed": False,
            "pass_match": True,
            "source_title_hit": None,
            "keyword_coverage": None,
            "citation_present": False,
            "citation_valid": False,
            "refusal_correct": True,
            "case_success": True,
        },
    ]

    metrics = summarize_results(case_results)

    assert metrics["gate_accuracy"] == 1.0
    assert metrics["source_title_hit_rate"] == 1.0
    assert (
        metrics["average_keyword_coverage"]
        == 1.0
    )
    assert metrics["refusal_accuracy"] == 1.0
    assert (
        metrics["overall_case_pass_rate"]
        == 1.0
    )

def test_get_expected_titles_supports_new_and_old_formats():
    assert get_expected_titles(
        {
            "expected_titles": [
                "资料A",
                "资料B",
            ]
        }
    ) == ["资料A", "资料B"]

    assert get_expected_titles(
        {
            "expected_title": "资料A",
        }
    ) == ["资料A"]

    assert get_expected_titles(
        {
            "expected_title": None,
        }
    ) == []


def test_append_source_citations():
    answer = RAGPipeline.append_source_citations(
        answer="这是生成的回答。",
        source_count=2,
    )

    assert answer == (
        "这是生成的回答。\n\n"
        "参考资料：【资料1】 【资料2】"
    )

    refusal = RAGPipeline.append_source_citations(
        answer="现有资料不足",
        source_count=2,
    )

    assert refusal == "现有资料不足"