import json

import torch
import torch.nn.functional as F

from rag_project.config import (
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TOP_K,
    DEVICE,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    RETRIEVAL_MAX_LENGTH,
    RETRIEVAL_QUERY_INSTRUCTION,
)


def load_knowledge_base():
    """加载文本块信息和文本块向量矩阵。"""

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    chunks = metadata["chunks"]

    embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("知识库向量必须是Tensor")

    if embeddings.ndim != 2:
        raise ValueError("知识库向量必须是二维矩阵")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "文本块数量与向量矩阵行数不一致"
        )

    expected_dimension = metadata["embedding_dimension"]

    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            "向量实际维度与元数据记录不一致"
        )

    return metadata, chunks, embeddings


def encode_query(
    query: str,
    tokenizer,
    model,
) -> torch.Tensor:
    """将用户问题转换成归一化查询向量。"""

    if not query.strip():
        raise ValueError("用户问题不能为空")

    query_text = RETRIEVAL_QUERY_INSTRUCTION + query

    model_inputs = tokenizer(
        query_text,
        padding=True,
        truncation=True,
        max_length=RETRIEVAL_MAX_LENGTH,
        return_tensors="pt",
    )

    # tokenizer返回的是字典，需要把字典中的每个Tensor都移动到模型设备。
    model_inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in model_inputs.items()
    }

    with torch.inference_mode():
        model_outputs = model(**model_inputs)

        # BGE使用第一个Token的上下文向量作为句子向量。
        query_embedding = (
            model_outputs.last_hidden_state[:, 0]
        )

        query_embedding = F.normalize(
            query_embedding,
            p=2,
            dim=1,
        )

    # 文档向量保存在CPU，因此查询向量也移回CPU。
    return query_embedding.cpu()


def retrieve_chunks(
    query: str,
    tokenizer,
    model,
    chunks: list,
    embeddings: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> dict:
    """检索与用户问题最相关的文本块。"""

    if top_k <= 0:
        raise ValueError("top_k必须大于0")

    if len(chunks) == 0:
        raise ValueError("知识库中没有文本块")

    query_embedding = encode_query(
        query,
        tokenizer,
        model,
    )

    # (1, 512) @ (512, 文本块数量) → (1, 文本块数量)
    similarities = (
        query_embedding @ embeddings.T
    ).squeeze(0)

    # 防止要求返回的数量超过知识库实际文本块数量。
    actual_k = min(top_k, len(chunks))

    top_scores, top_indices = torch.topk(
        similarities,
        k=actual_k,
    )

    max_score = top_scores[0].item()
    passed = max_score >= min_similarity

    retrieved_chunks = []

    if passed:
        for score_tensor, index_tensor in zip(
            top_scores,
            top_indices,
        ):
            score = score_tensor.item()
            index = index_tensor.item()

            # 使用副本，避免把score永久写入原始知识库数据。
            chunk_result = chunks[index].copy()
            chunk_result["score"] = score

            retrieved_chunks.append(chunk_result)

    return {
        "query": query,
        "passed": passed,
        "max_score": max_score,
        "retrieved_chunks": retrieved_chunks,
    }


if __name__ == "__main__":
    from rag_project.model_manager import (
        load_retrieval_components,
    )

    metadata, chunks, embeddings = load_knowledge_base()

    tokenizer, model = load_retrieval_components(
        metadata["model_id"]
    )

    result = retrieve_chunks(
        query="RAG由哪三个阶段组成？",
        tokenizer=tokenizer,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\n===== 检索结果 =====")
    print("用户问题：", result["query"])
    print("最高相似度：", result["max_score"])
    print("是否通过门槛：", result["passed"])

    for rank, chunk in enumerate(
        result["retrieved_chunks"],
        start=1,
    ):
        print(f"\n资料{rank}")
        print("标题：", chunk["title"])
        print("文本块编号：", chunk["chunk_id"])
        print("相似度：", chunk["score"])
        print("正文：", chunk["text"])


