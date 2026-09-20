import pytest
import torch

import rag_project.retrieval as retrieval_module
from rag_project.retrieval import (
    calculate_bm25_scores,
    reciprocal_rank_fusion,
    retrieve_chunks,
)


def make_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chunk_1",
            "title": "资料一",
            "text": "第一段资料",
        },
        {
            "chunk_id": "chunk_2",
            "title": "资料二",
            "text": "第二段资料",
        },
        {
            "chunk_id": "chunk_3",
            "title": "资料三",
            "text": "第三段资料",
        },
    ]


def fake_encode_query(
    query,
    tokenizer,
    model,
):
    return torch.tensor(
        [[1.0, 0.0]],
        dtype=torch.float32,
    )


def test_retrieve_filters_chunks_below_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval_module,
        "encode_query",
        fake_encode_query,
    )

    embeddings = torch.tensor(
        [
            [0.90, 0.0],
            [0.50, 0.0],
            [0.30, 0.0],
        ],
        dtype=torch.float32,
    )

    result = retrieve_chunks(
        query="测试问题",
        tokenizer=None,
        model=None,
        chunks=make_chunks(),
        embeddings=embeddings,
        top_k=3,
        min_similarity=0.60,
    )

    assert result["passed"] is True
    assert result["max_score"] == pytest.approx(
        0.90
    )
    assert [
        chunk["chunk_id"]
        for chunk in result["retrieved_chunks"]
    ] == ["chunk_1"]


def test_retrieve_returns_empty_when_best_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval_module,
        "encode_query",
        fake_encode_query,
    )

    embeddings = torch.tensor(
        [
            [0.90, 0.0],
            [0.50, 0.0],
            [0.30, 0.0],
        ],
        dtype=torch.float32,
    )

    result = retrieve_chunks(
        query="测试问题",
        tokenizer=None,
        model=None,
        chunks=make_chunks(),
        embeddings=embeddings,
        top_k=3,
        min_similarity=0.95,
    )

    assert result["passed"] is False
    assert result["retrieved_chunks"] == []
    assert result["gate_status"] == "reject"


def test_bm25_rewards_exact_terms():
    chunks = make_chunks()
    chunks[0]["text"] = "FastAPI OAuth2 登录"
    chunks[1]["text"] = "FastAPI 路由与请求模型"
    scores = calculate_bm25_scores(
        "FastAPI OAuth2 登录",
        chunks,
    )
    assert scores[0] > scores[1]


def test_rrf_combines_dense_and_sparse_rankings():
    fused = reciprocal_rank_fusion(
        dense_scores=[0.9, 0.8, 0.1],
        sparse_scores=[0.0, 2.0, 0.1],
        candidate_k=3,
    )
    assert fused[0][0] == 1


def test_reranker_drives_three_way_gate(
    monkeypatch,
):
    monkeypatch.setattr(
        retrieval_module,
        "encode_query",
        fake_encode_query,
    )
    monkeypatch.setattr(
        retrieval_module,
        "score_with_reranker",
        lambda *args, **kwargs: [0.55, 0.20, 0.10],
    )
    embeddings = torch.tensor(
        [[0.9, 0.0], [0.5, 0.0], [0.3, 0.0]],
        dtype=torch.float32,
    )
    result = retrieve_chunks(
        query="测试问题",
        tokenizer=None,
        model=None,
        chunks=make_chunks(),
        embeddings=embeddings,
        reranker_tokenizer=object(),
        reranker_model=object(),
        gate_low=0.45,
        gate_high=0.62,
    )
    assert result["passed"] is True
    assert result["gate_status"] == "gray"
    assert result["gate_score"] == pytest.approx(0.55)
    assert result["score_type"] == "rerank"
