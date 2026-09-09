import pytest
import torch

import rag_project.retrieval as retrieval_module
from rag_project.retrieval import retrieve_chunks


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