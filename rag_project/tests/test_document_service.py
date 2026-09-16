import json

import pytest

from rag_project import document_service


@pytest.fixture
def document_paths(tmp_path, monkeypatch):
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()
    metadata_path = tmp_path / "knowledge_base.json"
    monkeypatch.setattr(document_service, "SOURCE_DOCUMENTS_DIR", source_dir)
    monkeypatch.setattr(document_service, "EXPANDED_METADATA_PATH", metadata_path)
    monkeypatch.setattr(document_service, "LEARNING_NOTES_DIR", tmp_path)
    document_service._failures.clear()
    return source_dir, metadata_path


def test_save_list_and_delete_document(document_paths):
    source_dir, metadata_path = document_paths
    path = document_service.save_upload(
        "RAG 指南.md",
        "# RAG 指南\n\n检索、增强、生成。".encode("utf-8"),
    )
    document_id = document_service._document_id(path)
    metadata_path.write_text(
        json.dumps({"chunks": [
            {"document_id": document_id},
            {"document_id": document_id},
        ]}),
        encoding="utf-8",
    )

    items = document_service.list_documents()

    assert len(items) == 1
    assert items[0]["name"] == "RAG 指南.md"
    assert items[0]["chunks"] == 2
    assert items[0]["status"] == "ready"
    assert items[0]["category"] == "reference"
    assert items[0]["read_only"] is False

    deleted = document_service.delete_document(document_id)
    assert deleted == source_dir / "RAG 指南.md"
    assert not deleted.exists()


def test_upload_rejects_unsupported_and_duplicate_files(document_paths):
    document_service.save_upload("notes.txt", "有效内容".encode("utf-8"))

    with pytest.raises(document_service.DocumentConflictError):
        document_service.save_upload("notes.txt", "另一份内容".encode("utf-8"))

    with pytest.raises(ValueError, match="仅支持"):
        document_service.save_upload("malware.exe", b"not allowed")


def test_upload_rejects_empty_document(document_paths):
    with pytest.raises(ValueError, match="文档内容为空"):
        document_service.save_upload("empty.md", b"   \n")


def test_list_documents_includes_read_only_learning_notes(document_paths):
    _, metadata_path = document_paths
    note_path = document_service.LEARNING_NOTES_DIR / "day01笔记.txt"
    note_path.write_text("第一天学习内容", encoding="utf-8")
    note_id = document_service._learning_note_id(note_path)
    metadata_path.write_text(
        json.dumps({"chunks": [{"document_id": note_id}]}),
        encoding="utf-8",
    )

    items = document_service.list_documents()

    assert len(items) == 1
    assert items[0]["name"] == "day01笔记.txt"
    assert items[0]["category"] == "learning_note"
    assert items[0]["read_only"] is True
    assert items[0]["chunks"] == 1
