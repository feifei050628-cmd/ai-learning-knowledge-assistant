import json

import torch

import rag_project.update_knowledge_base as updater


def write_json(path, data) -> None:
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def prepare_valid_knowledge_base(
    tmp_path,
    monkeypatch,
):
    preview_path = tmp_path / "preview.json"
    metadata_path = tmp_path / "metadata.json"
    embeddings_path = tmp_path / "embeddings.pth"

    monkeypatch.setattr(
        updater,
        "EXPANDED_CHUNK_PREVIEW_PATH",
        preview_path,
    )
    monkeypatch.setattr(
        updater,
        "EXPANDED_METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        updater,
        "EXPANDED_EMBEDDINGS_PATH",
        embeddings_path,
    )

    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "RAG 包含检索、增强和生成。",
        },
        {
            "chunk_id": "chunk_2",
            "text": "检索结果需要经过相关性门槛。",
        },
    ]

    write_json(
        preview_path,
        {
            "chunk_count": 2,
            "chunks": chunks,
        },
    )

    write_json(
        metadata_path,
        {
            "model_id": updater.RETRIEVAL_MODEL_ID,
            "embedding_dimension": 512,
            "chunk_count": 2,
            "chunks": chunks,
        },
    )

    torch.save(
        torch.zeros((2, 512)),
        embeddings_path,
    )

    return preview_path


def test_current_knowledge_base_skips_rebuild(
    tmp_path,
    monkeypatch,
):
    prepare_valid_knowledge_base(
        tmp_path,
        monkeypatch,
    )

    assert updater.knowledge_base_is_current()


def test_changed_content_requires_rebuild(
    tmp_path,
    monkeypatch,
):
    preview_path = prepare_valid_knowledge_base(
        tmp_path,
        monkeypatch,
    )

    write_json(
        preview_path,
        {
            "chunk_count": 2,
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                    "text": "学习笔记内容已经发生变化。",
                },
                {
                    "chunk_id": "chunk_2",
                    "text": "检索结果需要经过相关性门槛。",
                },
            ],
        },
    )

    assert not updater.knowledge_base_is_current()