import json
from pathlib import Path

import pytest

import rag_project.note_importer as note_importer
from rag_project.note_importer import (
    attach_source_metadata,
    discover_learning_notes,
    extract_day_number,
    merge_documents,
    read_learning_note,
)


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("day01笔记.txt", 1),
        ("day25笔记.txt", 25),
    ],
)
def test_extract_day_number(
    file_name: str,
    expected: int,
):
    assert extract_day_number(Path(file_name)) == expected


def test_extract_day_number_rejects_invalid_name():
    with pytest.raises(ValueError):
        extract_day_number(Path("第25天笔记.txt"))


def test_discover_learning_notes_filters_and_sorts(
    tmp_path: Path,
):
    (tmp_path / "day02笔记.txt").write_text(
        "第二天",
        encoding="utf-8",
    )
    (tmp_path / "day01笔记.txt").write_text(
        "第一天",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "不是学习笔记",
        encoding="utf-8",
    )

    paths = discover_learning_notes(
        notes_directory=tmp_path,
        pattern="day[0-9][0-9]笔记.txt",
    )

    assert [path.name for path in paths] == [
        "day01笔记.txt",
        "day02笔记.txt",
    ]


def test_read_learning_note_builds_metadata(
    tmp_path: Path,
):
    note_path = tmp_path / "day25笔记.txt"
    note_path.write_text(
        "\ufeff# Day 25 学习笔记\n\n知识库扩充。",
        encoding="utf-8",
    )

    document = read_learning_note(
        note_path,
        notes_directory=tmp_path,
    )

    assert document["source"] == (
        "learning_notes/day25笔记.txt"
    )
    assert document["title"] == "# Day 25 学习笔记"
    assert document["category"] == "learning_note"
    assert document["day"] == 25
    assert document["char_count"] == len(document["text"])

    second_document = read_learning_note(
        note_path,
        notes_directory=tmp_path,
    )
    assert (
        document["document_id"]
        == second_document["document_id"]
    )


def test_merge_documents_adds_reference_metadata():
    reference = {
        "document_id": "doc_reference",
        "source": "source_documents/rag.md",
        "title": "RAG 基础",
        "text": "RAG 内容",
        "char_count": 6,
    }
    learning_note = {
        "document_id": "doc_note",
        "source": "learning_notes/day25笔记.txt",
        "title": "Day 25",
        "text": "学习内容",
        "char_count": 4,
        "category": "learning_note",
        "day": 25,
    }

    documents = merge_documents(
        [reference],
        [learning_note],
    )

    assert len(documents) == 2
    assert documents[0]["category"] == "reference"
    assert documents[0]["day"] is None
    assert documents[1]["category"] == "learning_note"


def test_merge_documents_rejects_duplicate_ids():
    document = {
        "document_id": "duplicate_id",
        "source": "source_documents/rag.md",
        "title": "RAG",
        "text": "内容",
        "char_count": 2,
    }
    learning_note = {
        **document,
        "source": "learning_notes/day25笔记.txt",
        "category": "learning_note",
        "day": 25,
    }

    with pytest.raises(
        ValueError,
        match="重复的 document_id",
    ):
        merge_documents(
            [document],
            [learning_note],
        )


def test_attach_source_metadata():
    documents = [
        {
            "document_id": "doc_note",
            "category": "learning_note",
            "day": 25,
        }
    ]
    chunks = [
        {
            "document_id": "doc_note",
            "chunk_id": "doc_note_chunk_0001",
            "text": "测试文本块",
        }
    ]

    result = attach_source_metadata(
        chunks=chunks,
        documents=documents,
    )

    assert result[0]["category"] == "learning_note"
    assert result[0]["day"] == 25


def test_build_expanded_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reference_text = "RAG 检索增强生成基础知识。"
    note_text = "第25天学习了如何扩充知识库。"

    documents = [
        {
            "document_id": "doc_reference",
            "source": "source_documents/rag.md",
            "title": "RAG 基础",
            "text": reference_text,
            "char_count": len(reference_text),
            "category": "reference",
            "day": None,
        },
        {
            "document_id": "doc_note",
            "source": "learning_notes/day25笔记.txt",
            "title": "Day 25 学习笔记",
            "text": note_text,
            "char_count": len(note_text),
            "category": "learning_note",
            "day": 25,
        },
    ]

    output_path = tmp_path / "expanded_preview.json"

    monkeypatch.setattr(
        note_importer,
        "load_all_source_documents",
        lambda: documents,
    )
    monkeypatch.setattr(
        note_importer,
        "EXPANDED_CHUNK_PREVIEW_PATH",
        output_path,
    )

    preview = note_importer.build_expanded_preview()

    assert preview["document_count"] == 2
    assert preview["reference_count"] == 1
    assert preview["learning_note_count"] == 1
    assert preview["chunk_count"] == 2
    assert output_path.exists()

    saved_preview = json.loads(
        output_path.read_text(encoding="utf-8")
    )
    assert saved_preview["version"] == (
        "day25-expanded-preview-v1"
    )
    assert saved_preview["document_count"] == 2