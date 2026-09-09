import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rag_project.knowledge_base_builder as builder
from rag_project.knowledge_base_builder import (
    encode_all_chunks,
    encode_chunk_batch,
    load_chunk_preview,
    save_knowledge_base,
    validate_chunks,
)


def make_chunk(index: int) -> dict:
    text = f"测试文本块{index}"

    return {
        "document_id": "doc_test",
        "chunk_id": f"doc_test_chunk_{index:04d}",
        "source": "learning_notes/day25笔记.txt",
        "title": "Day 25 学习笔记",
        "text": text,
        "char_count": len(text),
        "category": "learning_note",
        "day": 25,
    }


def make_preview(chunks: list[dict]) -> dict:
    return {
        "version": "test-preview-v1",
        "document_count": 1,
        "reference_count": 0,
        "learning_note_count": 1,
        "total_character_count": sum(
            chunk["char_count"]
            for chunk in chunks
        ),
        "chunk_count": len(chunks),
        "documents": [
            {
                "document_id": "doc_test",
                "source": (
                    "learning_notes/day25笔记.txt"
                ),
                "title": "Day 25 学习笔记",
                "char_count": 100,
                "category": "learning_note",
                "day": 25,
            }
        ],
        "chunks": chunks,
    }


def test_load_and_validate_chunk_preview(
    tmp_path: Path,
):
    chunks = [
        make_chunk(1),
        make_chunk(2),
    ]
    preview = make_preview(chunks)
    preview_path = tmp_path / "preview.json"

    preview_path.write_text(
        json.dumps(
            preview,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_chunk_preview(preview_path)
    validated_chunks = validate_chunks(loaded)

    assert len(validated_chunks) == 2
    assert validated_chunks[0]["chunk_id"] == (
        "doc_test_chunk_0001"
    )


def test_validate_chunks_rejects_wrong_count():
    chunks = [make_chunk(1)]
    preview = make_preview(chunks)
    preview["chunk_count"] = 2

    with pytest.raises(
        ValueError,
        match="数量与实际数量不一致",
    ):
        validate_chunks(preview)


def test_validate_chunks_rejects_missing_field():
    chunk = make_chunk(1)
    del chunk["category"]
    preview = make_preview([chunk])

    with pytest.raises(
        ValueError,
        match="缺少字段",
    ):
        validate_chunks(preview)


def test_validate_chunks_rejects_duplicate_id():
    first_chunk = make_chunk(1)
    second_chunk = make_chunk(1)
    preview = make_preview(
        [first_chunk, second_chunk]
    )

    with pytest.raises(
        ValueError,
        match="重复chunk_id",
    ):
        validate_chunks(preview)


class FakeTokenizer:
    def __call__(
        self,
        texts,
        **kwargs,
    ):
        batch_size = len(texts)

        return {
            "input_ids": torch.ones(
                (batch_size, 3),
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (batch_size, 3),
                dtype=torch.long,
            ),
        }


class FakeModel:
    def __call__(self, **model_inputs):
        batch_size = (
            model_inputs["input_ids"].shape[0]
        )
        device = model_inputs["input_ids"].device

        hidden_state = torch.zeros(
            (batch_size, 3, 2),
            dtype=torch.float32,
            device=device,
        )

        hidden_state[:, 0, 0] = 3.0
        hidden_state[:, 0, 1] = 4.0

        return SimpleNamespace(
            last_hidden_state=hidden_state
        )


def test_encode_chunk_batch_normalizes_vectors(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        builder,
        "DEVICE",
        torch.device("cpu"),
    )

    embeddings = encode_chunk_batch(
        texts=["文本一", "文本二"],
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    expected = torch.tensor(
        [
            [0.6, 0.8],
            [0.6, 0.8],
        ]
    )

    assert embeddings.shape == (2, 2)
    assert embeddings.device.type == "cpu"
    assert torch.allclose(
        embeddings,
        expected,
        atol=1e-6,
    )


def test_encode_all_chunks_uses_batches(
    monkeypatch: pytest.MonkeyPatch,
):
    chunks = [
        make_chunk(index)
        for index in range(1, 6)
    ]
    batch_sizes = []

    def fake_encode_batch(
        texts,
        tokenizer,
        model,
    ):
        batch_sizes.append(len(texts))

        return torch.full(
            (len(texts), 2),
            fill_value=2 ** -0.5,
        )

    monkeypatch.setattr(
        builder,
        "encode_chunk_batch",
        fake_encode_batch,
    )

    embeddings = encode_all_chunks(
        chunks=chunks,
        tokenizer=object(),
        model=object(),
        batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    assert embeddings.shape == (5, 2)
    assert torch.isfinite(embeddings).all()


def test_encode_all_chunks_rejects_invalid_batch():
    with pytest.raises(
        ValueError,
        match="batch_size必须大于0",
    ):
        encode_all_chunks(
            chunks=[make_chunk(1)],
            tokenizer=object(),
            model=object(),
            batch_size=0,
        )


def test_save_knowledge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    chunks = [
        make_chunk(1),
        make_chunk(2),
    ]
    preview = make_preview(chunks)

    metadata_path = tmp_path / "metadata.json"
    embeddings_path = tmp_path / "embeddings.pth"

    monkeypatch.setattr(
        builder,
        "EXPANDED_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        builder,
        "EXPANDED_EMBEDDINGS_PATH",
        embeddings_path,
    )

    embeddings = torch.tensor(
        [
            [0.6, 0.8],
            [0.8, 0.6],
        ],
        dtype=torch.float32,
    )

    metadata = save_knowledge_base(
        preview=preview,
        chunks=chunks,
        embeddings=embeddings,
    )

    saved_embeddings = torch.load(
        embeddings_path,
        map_location="cpu",
        weights_only=True,
    )
    saved_metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata_path.exists()
    assert embeddings_path.exists()
    assert metadata["chunk_count"] == 2
    assert metadata["embedding_dimension"] == 2
    assert saved_metadata["model_id"] == (
        builder.RETRIEVAL_MODEL_ID
    )
    assert torch.equal(
        saved_embeddings,
        embeddings,
    )