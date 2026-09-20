import json
import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from rag_project.config import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_GATE_HIGH,
    DEFAULT_GATE_LOW,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DEVICE,
    EMBEDDINGS_PATH,
    METADATA_PATH,
    RETRIEVAL_MAX_LENGTH,
    RETRIEVAL_QUERY_INSTRUCTION,
)


def load_knowledge_base():
    """加载文本块信息和文本块向量矩阵。"""
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
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
        raise ValueError("文本块数量与向量矩阵行数不一致")
    if embeddings.shape[1] != metadata["embedding_dimension"]:
        raise ValueError("向量实际维度与元数据记录不一致")
    return metadata, chunks, embeddings


def encode_query(query: str, tokenizer, model) -> torch.Tensor:
    """将用户问题转换成归一化查询向量。"""
    if not query.strip():
        raise ValueError("用户问题不能为空")
    model_inputs = tokenizer(
        RETRIEVAL_QUERY_INSTRUCTION + query,
        padding=True,
        truncation=True,
        max_length=RETRIEVAL_MAX_LENGTH,
        return_tensors="pt",
    )
    model_inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in model_inputs.items()
    }
    with torch.inference_mode():
        outputs = model(**model_inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu()


def tokenize_for_bm25(text: str) -> list[str]:
    """零依赖的中英文切分：英文词、中文单字和中文二元组。"""
    normalized = text.lower()
    latin_terms = re.findall(r"[a-z0-9_+#.-]+", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_terms = []
    for run in chinese_runs:
        chinese_terms.extend(run)
        chinese_terms.extend(
            run[index:index + 2]
            for index in range(len(run) - 1)
        )
    return latin_terms + chinese_terms


def calculate_bm25_scores(
    query: str,
    chunks: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """计算 BM25 分数，用关键词缺失拉开边界诱导样本。"""
    documents = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
    query_terms = set(tokenize_for_bm25(query))
    if not query_terms:
        return [0.0] * len(chunks)
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    document_frequency = Counter()
    for terms in documents:
        document_frequency.update(set(terms) & query_terms)
    scores = []
    total = len(documents)
    for terms in documents:
        frequencies = Counter(terms)
        score = 0.0
        length_factor = 1 - b + b * len(terms) / max(average_length, 1.0)
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = math.log(
                1 + (total - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += idf * frequency * (k1 + 1) / (
                frequency + k1 * length_factor
            )
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    dense_scores: list[float],
    sparse_scores: list[float],
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """用 RRF 融合向量与 BM25 排名，避免直接比较异构分数。"""
    limit = min(candidate_k, len(dense_scores))
    dense_order = sorted(
        range(len(dense_scores)),
        key=dense_scores.__getitem__,
        reverse=True,
    )[:limit]
    sparse_order = sorted(
        range(len(sparse_scores)),
        key=sparse_scores.__getitem__,
        reverse=True,
    )[:limit]
    fused = Counter()
    for order in (dense_order, sparse_order):
        for rank, index in enumerate(order, start=1):
            fused[index] += 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def score_with_reranker(
    query: str,
    candidate_chunks: list[dict],
    tokenizer,
    model,
) -> list[float]:
    """对 query-chunk 对进行 Cross-Encoder 打分并压缩到 0~1。"""
    if not candidate_chunks:
        return []
    inputs = tokenizer(
        [query] * len(candidate_chunks),
        [chunk["text"] for chunk in candidate_chunks],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits.reshape(-1)
        return torch.sigmoid(logits).cpu().tolist()


def retrieve_chunks(
    query: str,
    tokenizer,
    model,
    chunks: list,
    embeddings: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    *,
    use_hybrid: bool = False,
    reranker_tokenizer=None,
    reranker_model=None,
    gate_low: float = DEFAULT_GATE_LOW,
    gate_high: float = DEFAULT_GATE_HIGH,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> dict:
    """混合召回、可选重排，并输出 answer/gray/reject 门控状态。"""
    if top_k <= 0:
        raise ValueError("top_k必须大于0")
    if not chunks:
        raise ValueError("知识库中没有文本块")
    if gate_low > gate_high:
        raise ValueError("gate_low 不能大于 gate_high")

    query_embedding = encode_query(query, tokenizer, model)
    similarities = (query_embedding @ embeddings.T).squeeze(0)
    dense_scores = similarities.tolist()
    sparse_scores = calculate_bm25_scores(query, chunks)

    if use_hybrid:
        fused = reciprocal_rank_fusion(
            dense_scores,
            sparse_scores,
            candidate_k=candidate_k,
        )
    else:
        fused = [
            (index, 0.0)
            for index in sorted(
                range(len(dense_scores)),
                key=dense_scores.__getitem__,
                reverse=True,
            )[:min(candidate_k, len(chunks))]
        ]

    candidate_chunks = []
    for index, rrf_score in fused:
        item = chunks[index].copy()
        item["vector_score"] = dense_scores[index]
        item["bm25_score"] = sparse_scores[index]
        item["rrf_score"] = rrf_score
        candidate_chunks.append(item)

    reranker_enabled = (
        reranker_tokenizer is not None
        and reranker_model is not None
    )
    if reranker_enabled:
        rerank_scores = score_with_reranker(
            query,
            candidate_chunks,
            reranker_tokenizer,
            reranker_model,
        )
        for item, score in zip(candidate_chunks, rerank_scores):
            item["rerank_score"] = score
            item["score"] = score
        candidate_chunks.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )
        gate_score = candidate_chunks[0]["rerank_score"]
        if gate_score >= gate_high:
            gate_status = "answer"
        elif gate_score >= gate_low:
            gate_status = "gray"
        else:
            gate_status = "reject"
    else:
        candidate_chunks.sort(
            key=lambda item: item["vector_score"],
            reverse=True,
        )
        for item in candidate_chunks:
            item["score"] = item["vector_score"]
        gate_score = candidate_chunks[0]["vector_score"]
        gate_status = (
            "answer"
            if gate_score >= min_similarity
            else "reject"
        )

    references = candidate_chunks[:min(top_k, len(candidate_chunks))]
    if reranker_enabled:
        selected = references
    else:
        selected = [
            item
            for item in references
            if item["vector_score"] >= min_similarity
        ]
    passed = gate_status != "reject"
    return {
        "query": query,
        "passed": passed,
        "gate_status": gate_status,
        "gate_score": gate_score,
        "max_score": max(dense_scores),
        "score_type": "rerank" if reranker_enabled else "cosine",
        "retrieved_chunks": selected if passed else [],
        "reference_chunks": references,
    }


if __name__ == "__main__":
    from rag_project.model_manager import load_retrieval_components

    metadata, chunks, embeddings = load_knowledge_base()
    tokenizer, model = load_retrieval_components(metadata["model_id"])
    result = retrieve_chunks(
        query="RAG由哪三个阶段组成？",
        tokenizer=tokenizer,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
