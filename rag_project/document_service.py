"""真实知识库文档的查询与文件操作。"""

import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag_project.config import (
    EXPANDED_METADATA_PATH,
    LEARNING_NOTES_DIR,
    LEARNING_NOTE_PATTERN,
    SOURCE_DOCUMENTS_DIR,
)
from rag_project.document_processor import (
    SUPPORTED_SUFFIXES,
    create_document_id,
    read_document,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DOCUMENT_OPERATION_LOCK = threading.Lock()
_failures: dict[str, str] = {}


class DocumentConflictError(RuntimeError):
    pass


class DocumentNotFoundError(FileNotFoundError):
    pass


def _safe_filename(filename: str) -> str:
    clean_name = Path(filename).name.strip()
    if not clean_name or clean_name in {".", ".."}:
        raise ValueError("文件名无效")
    if Path(clean_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("仅支持 PDF、TXT 和 Markdown 文件")
    return clean_name


def _document_id(path: Path) -> str:
    relative_path = path.relative_to(SOURCE_DOCUMENTS_DIR).as_posix()
    return create_document_id(relative_path)


def _learning_note_id(path: Path) -> str:
    return create_document_id(f"learning_notes/{path.name}")


def _chunk_counts() -> Counter:
    if not EXPANDED_METADATA_PATH.exists():
        return Counter()
    try:
        payload = json.loads(EXPANDED_METADATA_PATH.read_text(encoding="utf-8"))
        return Counter(chunk["document_id"] for chunk in payload.get("chunks", []))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return Counter()


def list_documents() -> list[dict]:
    SOURCE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_counts = _chunk_counts()
    items = []
    for path in sorted(SOURCE_DOCUMENTS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document_id = _document_id(path)
        stat = path.stat()
        error = _failures.get(document_id)
        items.append({
            "id": document_id,
            "name": path.name,
            "type": path.suffix.lstrip(".").upper(),
            "size_bytes": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "status": "failed" if error else "ready",
            "chunks": chunk_counts[document_id],
            "error": error,
            "category": "reference",
            "read_only": False,
        })

    for path in sorted(LEARNING_NOTES_DIR.glob(LEARNING_NOTE_PATTERN)):
        if not path.is_file():
            continue
        document_id = _learning_note_id(path)
        stat = path.stat()
        error = _failures.get(document_id)
        items.append({
            "id": document_id,
            "name": path.name,
            "type": "TXT",
            "size_bytes": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "status": "failed" if error else "ready",
            "chunks": chunk_counts[document_id],
            "error": error,
            "category": "learning_note",
            "read_only": True,
        })
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def resolve_document(document_id: str) -> Path:
    for path in SOURCE_DOCUMENTS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            if _document_id(path) == document_id:
                return path
    raise DocumentNotFoundError("文档不存在或已被删除")


def save_upload(filename: str, content: bytes) -> Path:
    clean_name = _safe_filename(filename)
    if not content:
        raise ValueError("上传文件不能为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("单个文件不能超过 20 MB")

    SOURCE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    target = SOURCE_DOCUMENTS_DIR / clean_name
    if target.exists():
        raise DocumentConflictError("同名文档已存在，请先删除或修改文件名")

    temporary = target.with_name(
        f".{target.stem}.uploading{target.suffix}"
    )
    temporary.write_bytes(content)
    try:
        # 保存前先验证编码、PDF 结构和可提取文本，避免污染启动流程。
        read_document(temporary, SOURCE_DOCUMENTS_DIR)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def delete_document(document_id: str) -> Path:
    path = resolve_document(document_id)
    path.unlink()
    _failures.pop(document_id, None)
    return path


def validate_document(document_id: str) -> Path:
    path = resolve_document(document_id)
    read_document(path, SOURCE_DOCUMENTS_DIR)
    _failures.pop(document_id, None)
    return path


def mark_failed(path: Path, error: Exception) -> None:
    if path.exists():
        _failures[_document_id(path)] = str(error)
