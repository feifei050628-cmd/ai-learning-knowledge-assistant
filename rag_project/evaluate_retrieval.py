import json

from rag_project.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    PROJECT_DIR,
)
from rag_project.model_manager import (
    load_retrieval_components,
)
from rag_project.retrieval import (
    load_knowledge_base,
    retrieve_chunks,
)


CASES_PATH = PROJECT_DIR / "evaluation_cases.json"
REPORT_PATH = PROJECT_DIR / "retrieval_evaluation_report.json"


def load_evaluation_cases() -> list:
    with open(
        CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    if not cases:
        raise ValueError("评估数据集不能为空")

    return cases


def find_expected_rank(
    retrieved_chunks: list,
    expected_title: str | None,
) -> int | None:
    if expected_title is None:
        return None

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if chunk["title"] == expected_title:
            return rank

    return None


def safe_divide(
    numerator: int | float,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_retrieval(
    cases: list,
    tokenizer,
    model,
    chunks: list,
    embeddings,
    top_k: int,
    min_similarity: float,
) -> dict:
    positive_total = 0
    positive_hits = 0
    reciprocal_rank_sum = 0.0
    relevant_accepted = 0

    negative_total = 0
    negative_rejected = 0

    correct_total = 0
    case_results = []

    for case in cases:
        retrieval_result = retrieve_chunks(
            query=case["query"],
            tokenizer=tokenizer,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
            min_similarity=-1.0,
        )

        retrieved_chunks = retrieval_result[
            "retrieved_chunks"
        ]
        max_score = retrieval_result["max_score"]
        expected_title = case["expected_title"]

        is_relevant = expected_title is not None
        accepted = max_score >= min_similarity

        expected_rank = find_expected_rank(
            retrieved_chunks,
            expected_title,
        )

        hit = expected_rank is not None

        if is_relevant:
            positive_total += 1

            if hit:
                positive_hits += 1
                reciprocal_rank_sum += 1 / expected_rank

            if accepted:
                relevant_accepted += 1

            case_correct = hit and accepted
        else:
            negative_total += 1

            if not accepted:
                negative_rejected += 1

            case_correct = not accepted

        if case_correct:
            correct_total += 1

        retrieved_titles = [
            chunk["title"]
            for chunk in retrieved_chunks
        ]

        case_result = {
            "id": case["id"],
            "query": case["query"],
            "expected_title": expected_title,
            "retrieved_titles": retrieved_titles,
            "expected_rank": expected_rank,
            "max_score": max_score,
            "accepted": accepted,
            "correct": case_correct,
        }

        case_results.append(case_result)

        expected_text = (
            expected_title
            if expected_title is not None
            else "应被门槛拒绝"
        )
        rank_text = (
            str(expected_rank)
            if expected_rank is not None
            else "-"
        )
        status_text = (
            "正确"
            if case_correct
            else "需要检查"
        )

        print("\n" + "=" * 60)
        print("用例：", case["id"])
        print("问题：", case["query"])
        print("期望：", expected_text)
        print("Top-K标题：", retrieved_titles)
        print("正确资料排名：", rank_text)
        print("最高相似度：", f"{max_score:.4f}")
        print("是否通过门槛：", accepted)
        print("评估结果：", status_text)

    metrics = {
        f"hit_at_{top_k}": safe_divide(
            positive_hits,
            positive_total,
        ),
        f"mrr_at_{top_k}": safe_divide(
            reciprocal_rank_sum,
            positive_total,
        ),
        "relevant_accept_rate": safe_divide(
            relevant_accepted,
            positive_total,
        ),
        "irrelevant_rejection_rate": safe_divide(
            negative_rejected,
            negative_total,
        ),
        "overall_accuracy": safe_divide(
            correct_total,
            len(cases),
        ),
    }

    return {
        "settings": {
            "top_k": top_k,
            "min_similarity": min_similarity,
            "case_count": len(cases),
            "positive_count": positive_total,
            "negative_count": negative_total,
        },
        "metrics": metrics,
        "cases": case_results,
    }


def print_summary(report: dict) -> None:
    settings = report["settings"]
    metrics = report["metrics"]

    print("\n" + "=" * 60)
    print("检索评估汇总")
    print("=" * 60)
    print("评估问题数量：", settings["case_count"])
    print("相关问题数量：", settings["positive_count"])
    print("无关问题数量：", settings["negative_count"])
    print("Top-K：", settings["top_k"])
    print(
        "相关性门槛：",
        settings["min_similarity"],
    )

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name}："
            f"{metric_value:.2%}"
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
    cases = load_evaluation_cases()

    metadata, chunks, embeddings = (
        load_knowledge_base()
    )

    tokenizer, model = load_retrieval_components(
        metadata["model_id"]
    )

    report = evaluate_retrieval(
        cases=cases,
        tokenizer=tokenizer,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=DEFAULT_TOP_K,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )

    print_summary(report)
    save_report(report)


if __name__ == "__main__":
    main()