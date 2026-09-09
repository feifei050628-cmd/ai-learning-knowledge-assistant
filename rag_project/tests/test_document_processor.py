from pathlib import Path

import pytest

from rag_project.document_processor import (
    build_chunks,
    chunk_text,
    normalize_text,
    read_document,
    split_text_into_units,
)


def test_normalize_text_unifies_newlines():
    text = "第一行  \r\n\r\n\r\n第二行\t"

    assert normalize_text(text) == "第一行\n\n第二行"


def test_chunk_text_uses_sentence_boundaries():
    text = "第一句内容完整。第二句内容完整。第三句内容完整。"

    chunks = chunk_text(
        text=text,
        chunk_size=17,
        overlap=0,
    )

    assert "".join(chunks) == text
    assert all(chunk.endswith("。") for chunk in chunks)


def test_chunk_text_keeps_semantic_overlap():
    first = "第一句内容完整。"
    second = "第二句内容完整。"
    third = "第三句内容完整。"

    chunks = chunk_text(
        text=first + second + third,
        chunk_size=18,
        overlap=8,
    )

    assert len(chunks) == 2
    assert second in chunks[0]
    assert second in chunks[1]


def test_long_unit_uses_character_fallback():
    chunks = chunk_text(
        text="A" * 25,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 3
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert chunks[0][-2:] == chunks[1][:2]
    assert chunks[1][-2:] == chunks[2][:2]


def test_markdown_code_block_stays_together():
    fence = "`" * 3

    code_block = (
        f"{fence}python\n"
        "value = 1\n"
        "print(value)\n"
        f"{fence}"
    )

    text = (
        "说明文字。\n\n"
        f"{code_block}\n\n"
        "结束文字。"
    )

    units = split_text_into_units(
        normalize_text(text)
    )

    assert code_block in units


def test_document_metadata_is_stable(
    tmp_path: Path,
):
    source_root = tmp_path / "documents"
    source_root.mkdir()

    document_path = source_root / "rag.md"
    document_path.write_text(
        "# RAG基础\n\n第一句。第二句。",
        encoding="utf-8",
    )

    first_document = read_document(
        document_path,
        source_root,
    )
    second_document = read_document(
        document_path,
        source_root,
    )

    chunks = build_chunks(
        [first_document],
        chunk_size=20,
        overlap=0,
    )

    assert first_document["title"] == "RAG基础"
    assert first_document["source"] == "rag.md"

    assert (
        first_document["document_id"]
        == second_document["document_id"]
    )

    assert len(chunks) == 1
    assert chunks[0]["title"] == "RAG基础"
    assert chunks[0]["chunk_id"].endswith(
        "_chunk_0001"
    )


def test_chunk_text_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text(
            "测试",
            chunk_size=0,
            overlap=0,
        )

    with pytest.raises(ValueError):
        chunk_text(
            "测试",
            chunk_size=100,
            overlap=100,
        )