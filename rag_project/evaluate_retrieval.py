import json

from rag_project.config import (
    DEFAULT_GATE_HIGH,
    DEFAULT_GATE_LOW,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_RERANKER,
    PROJECT_DIR,
)
from rag_project.model_manager import (
    load_retrieval_components,
    load_reranker_components,
)
from rag_project.retrieval import (
    load_knowledge_base,
    retrieve_chunks,
)


CASES_PATH = (
    PROJECT_DIR
    / "evaluation_cases_day27.json"
)
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
    expected_titles: list[str],
) -> int | None:
    if not expected_titles:
        return None

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if chunk["title"] in expected_titles:
            return rank

    return None


def safe_divide(
    numerator: int | float,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator

def build_retrieved_details(
    retrieved_chunks: list,
) -> list[dict]:
    """整理 Top-K 文本块，便于分析检索错误。"""

    details = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        text_preview = (
            chunk["text"]
            .replace("\n", " ")
            [:160]
        )

        details.append(
            {
                "rank": rank,
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"],
                "category": chunk.get("category"),
                "day": chunk.get("day"),
                "text_preview": text_preview,
            }
        )

    return details

def evaluate_retrieval(
    cases: list,
    tokenizer,
    model,
    chunks: list,
    embeddings,
    top_k: int,
    min_similarity: float,
    reranker_tokenizer=None,
    reranker_model=None,
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
            min_similarity=min_similarity,
            use_hybrid=ENABLE_HYBRID_RETRIEVAL,
            reranker_tokenizer=reranker_tokenizer,
            reranker_model=reranker_model,
            gate_low=DEFAULT_GATE_LOW,
            gate_high=DEFAULT_GATE_HIGH,
        )

        retrieved_chunks = retrieval_result[
            "reference_chunks"
        ]
        max_score = retrieval_result["gate_score"]
        expected_titles = case["expected_titles"]

        is_relevant = case.get(
            "answerable",
            bool(expected_titles),
        )
        accepted = retrieval_result["gate_status"] == "answer"

        expected_rank = find_expected_rank(
            retrieved_chunks,
            expected_titles,
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
        
        retrieved_details = build_retrieved_details(
            retrieved_chunks
        )

        retrieved_titles = [
            item["title"]
            for item in retrieved_details
        ]

        case_result = {
            "id": case["id"],
            "category": case.get("category", "unknown"),
            "query": case["query"],
            "expected_titles": expected_titles,
            "retrieved_titles": retrieved_titles,
            "retrieved_details": retrieved_details,
            "expected_rank": expected_rank,
            "max_score": max_score,
            "vector_max_score": retrieval_result["max_score"],
            "gate_status": retrieval_result["gate_status"],
            "score_type": retrieval_result["score_type"],
            "answerable": is_relevant,
            "accepted": accepted,
            "correct": case_correct,
        }

        case_results.append(case_result)

        expected_text = (
            "、".join(expected_titles)
            if expected_titles
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
        for item in retrieved_details:
            print(
                f"  排名 {item['rank']} | "
                f"分数 {item['score']:.4f} | "
                f"{item['title']} | "
                f"{item['chunk_id']}"
            )
            print(
                "  内容预览：",
                item["text_preview"],
            )
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
            "gate_low": DEFAULT_GATE_LOW,
            "gate_high": DEFAULT_GATE_HIGH,
            "hybrid_retrieval": ENABLE_HYBRID_RETRIEVAL,
            "reranker_enabled": reranker_model is not None,
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

    reranker_tokenizer = None
    reranker_model = None
    if ENABLE_RERANKER:
        reranker_tokenizer, reranker_model = (
            load_reranker_components()
        )

    report = evaluate_retrieval(
        cases=cases,
        tokenizer=tokenizer,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=DEFAULT_TOP_K,
        min_similarity=DEFAULT_MIN_SIMILARITY,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
    )

    print_summary(report)
    save_report(report)


if __name__ == "__main__":
    main()
