import re
from pathlib import Path
import json

from rag_project.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EXPANDED_CHUNK_PREVIEW_PATH,
    LEARNING_NOTE_PATTERN,
    LEARNING_NOTES_DIR,
)
from rag_project.document_processor import (
    build_chunks,
    create_document_id,
    load_documents,
    normalize_text,
)

def discover_learning_notes(
    notes_directory: Path = LEARNING_NOTES_DIR,
    pattern: str = LEARNING_NOTE_PATTERN,
) -> list[Path]:
    """按照文件名规则发现学习笔记。"""

    if not notes_directory.exists():
        raise FileNotFoundError(
            f"学习笔记目录不存在：{notes_directory}"
        )

    paths = sorted(
        path
        for path in notes_directory.glob(pattern)
        if path.is_file()
    )

    if not paths:
        raise FileNotFoundError(
            f"没有找到学习笔记：{pattern}"
        )

    return paths


def extract_day_number(path: Path) -> int:
    """从 dayXX笔记.txt 中提取学习天数。"""

    match = re.fullmatch(
        r"day(\d{2})笔记\.txt",
        path.name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"学习笔记文件名不符合规则：{path.name}"
        )

    return int(match.group(1))


def read_learning_note(
    path: Path,
    notes_directory: Path = LEARNING_NOTES_DIR,
) -> dict:
    """把一篇学习笔记转换为知识文档。"""

    text = path.read_text(encoding="utf-8")
    text = normalize_text(text.lstrip("\ufeff"))

    if not text:
        raise ValueError(f"学习笔记内容为空：{path}")

    day_number = extract_day_number(path)
    source = f"learning_notes/{path.name}"

    first_line = text.splitlines()[0].strip()
    title = first_line or f"Day {day_number} 学习笔记"

    return {
        "document_id": create_document_id(source),
        "source": source,
        "title": title,
        "text": text,
        "char_count": len(text),
        "category": "learning_note",
        "day": day_number,
    }


def load_learning_notes(
    notes_directory: Path = LEARNING_NOTES_DIR,
) -> list[dict]:
    """读取全部符合命名规则的学习笔记。"""

    return [
        read_learning_note(path, notes_directory)
        for path in discover_learning_notes(
            notes_directory=notes_directory
        )
    ]


def merge_documents(
    reference_documents: list[dict],
    learning_notes: list[dict],
) -> list[dict]:
    """合并基础资料和学习笔记，并检查编号冲突。"""

    for document in reference_documents:
        document.setdefault("category", "reference")
        document.setdefault("day", None)

    documents = reference_documents + learning_notes

    document_ids = [
        document["document_id"]
        for document in documents
    ]

    sources = [
        document["source"]
        for document in documents
    ]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("检测到重复的 document_id")

    if len(sources) != len(set(sources)):
        raise ValueError("检测到重复的 source")

    return documents


def load_all_source_documents() -> list[dict]:
    """读取4篇基础资料和全部学习笔记。"""

    reference_documents = load_documents()
    learning_notes = load_learning_notes()

    return merge_documents(
        reference_documents,
        learning_notes,
    )


def attach_source_metadata(
    chunks: list[dict],
    documents: list[dict],
) -> list[dict]:
    """把文档类别和学习天数加入文本块。"""

    documents_by_id = {
        document["document_id"]: document
        for document in documents
    }

    for chunk in chunks:
        document = documents_by_id[
            chunk["document_id"]
        ]

        chunk["category"] = document["category"]
        chunk["day"] = document["day"]

    return chunks


def build_expanded_preview() -> dict:
    """生成包含基础资料和学习笔记的切块预览。"""

    documents = load_all_source_documents()

    chunks = build_chunks(
        documents=documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )

    chunks = attach_source_metadata(
        chunks=chunks,
        documents=documents,
    )

    reference_count = sum(
        document["category"] == "reference"
        for document in documents
    )

    learning_note_count = sum(
        document["category"] == "learning_note"
        for document in documents
    )

    chunk_lengths = [
        chunk["char_count"]
        for chunk in chunks
    ]

    preview = {
        "version": "day25-expanded-preview-v1",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "document_count": len(documents),
        "reference_count": reference_count,
        "learning_note_count": learning_note_count,
        "total_character_count": sum(
            document["char_count"]
            for document in documents
        ),
        "chunk_count": len(chunks),
        "minimum_chunk_length": min(chunk_lengths),
        "maximum_chunk_length": max(chunk_lengths),
        "average_chunk_length": round(
            sum(chunk_lengths) / len(chunk_lengths),
            2,
        ),
        "documents": [
            {
                "document_id": document["document_id"],
                "source": document["source"],
                "title": document["title"],
                "char_count": document["char_count"],
                "category": document["category"],
                "day": document["day"],
            }
            for document in documents
        ],
        "chunks": chunks,
    }

    EXPANDED_CHUNK_PREVIEW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPANDED_CHUNK_PREVIEW_PATH.write_text(
        json.dumps(
            preview,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return preview


def main() -> None:
    preview = build_expanded_preview()

    print("扩展知识库预览生成完成")
    print(f"基础资料：{preview['reference_count']}")
    print(f"学习笔记：{preview['learning_note_count']}")
    print(f"知识源总数：{preview['document_count']}")
    print(f"总字符数：{preview['total_character_count']}")
    print(f"文本块总数：{preview['chunk_count']}")

    print(
        "文本块长度："
        f"{preview['minimum_chunk_length']}～"
        f"{preview['maximum_chunk_length']}"
    )

    print(
        "平均文本块长度："
        f"{preview['average_chunk_length']}"
    )

    print(
        "扩展预览文件："
        f"{EXPANDED_CHUNK_PREVIEW_PATH}"
    )


if __name__ == "__main__":
    main()