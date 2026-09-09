from rag_project.evaluate_retrieval import (
    find_expected_rank,
    load_evaluation_cases,
)
from rag_project.model_manager import (
    load_retrieval_components,
)
from rag_project.retrieval import (
    load_knowledge_base,
    retrieve_chunks,
)


TOP_K_VALUES = (1, 3, 5, 10)
MAX_TOP_K = max(TOP_K_VALUES)


def safe_divide(
    numerator: int | float,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    cases = load_evaluation_cases()

    metadata, chunks, embeddings = (
        load_knowledge_base()
    )

    tokenizer, model = load_retrieval_components(
        metadata["model_id"]
    )

    positive_cases = [
        case
        for case in cases
        if case["expected_titles"]
    ]

    hit_counts = {
        top_k: 0
        for top_k in TOP_K_VALUES
    }

    reciprocal_rank_sum = 0.0

    for case in positive_cases:
        result = retrieve_chunks(
            query=case["query"],
            tokenizer=tokenizer,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=MAX_TOP_K,
            min_similarity=-1.0,
        )

        retrieved_chunks = result[
            "retrieved_chunks"
        ]

        expected_rank = find_expected_rank(
            retrieved_chunks,
            case["expected_titles"],
        )

        if expected_rank is not None:
            reciprocal_rank_sum += (
                1 / expected_rank
            )

            for top_k in TOP_K_VALUES:
                if expected_rank <= top_k:
                    hit_counts[top_k] += 1

        print("\n" + "=" * 60)
        print("用例：", case["id"])
        print("问题：", case["query"])
        print(
            "正确资料排名：",
            expected_rank
            if expected_rank is not None
            else "Top-10 未命中",
        )

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            marker = (
                "[OK]"
                if chunk["title"]
                in case["expected_titles"]
                else "    "
            )

            print(
                f"{marker} {rank:>2} | "
                f"{chunk['score']:.4f} | "
                f"{chunk['title']}"
            )

    positive_count = len(positive_cases)

    print("\n" + "=" * 60)
    print("不同检索深度的排名指标")
    print("=" * 60)
    print("正例数量：", positive_count)

    for top_k in TOP_K_VALUES:
        hit_rate = safe_divide(
            hit_counts[top_k],
            positive_count,
        )

        print(
            f"Hit@{top_k}："
            f"{hit_rate:.2%}"
        )

    mrr_at_10 = safe_divide(
        reciprocal_rank_sum,
        positive_count,
    )

    print(f"MRR@10：{mrr_at_10:.2%}")


if __name__ == "__main__":
    main()