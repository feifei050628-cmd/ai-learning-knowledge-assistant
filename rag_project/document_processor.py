import hashlib
import re
from pathlib import Path
import json

from pypdf import PdfReader

from rag_project.config import (
    CHUNK_PREVIEW_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SOURCE_DOCUMENTS_DIR,
)

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

def normalize_text(text: str) -> str:
    """统一换行符、删除行尾空白、压缩多余空行。"""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return re.sub(r"\n{3,}", "\n\n", text)


def split_text_into_units(text: str) -> list[str]:
    """
    将文本拆成语义单元。

    普通文本按段落和句子拆分，
    Markdown代码块尽量保持完整。
    """

    units: list[str] = []
    parts = re.split(r"(```[\s\S]*?```)", text)

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if part.startswith("```") and part.endswith("```"):
            units.append(part)
            continue

        paragraphs = re.split(r"\n\s*\n", part)

        for paragraph in paragraphs:
            paragraph = " ".join(
                line.strip()
                for line in paragraph.splitlines()
                if line.strip()
            )

            if not paragraph:
                continue

            sentences = re.findall(
                r".+?(?:[。！？!?；;]|$)",
                paragraph,
            )

            units.extend(
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            )

    return units


def split_long_unit(
    unit: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    对没有标点的超长内容使用字符窗口切分。
    """

    pieces: list[str] = []
    start = 0

    while start < len(unit):
        end = min(start + chunk_size, len(unit))
        pieces.append(unit[start:end])

        if end == len(unit):
            break

        start = end - overlap

    return pieces


def select_overlap_units(
    units: list[str],
    overlap: int,
) -> list[str]:
    """从上一块末尾选择完整句子作为重叠内容。"""

    selected: list[str] = []
    selected_length = 0

    for unit in reversed(units):
        separator_length = 2 if selected else 0
        added_length = len(unit) + separator_length

        if selected_length + added_length > overlap:
            break

        selected.insert(0, unit)
        selected_length += added_length

    return selected


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """把一篇文档切分成带语义重叠的文本块。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap 必须大于等于0，并且小于chunk_size"
        )

    text = normalize_text(text)

    if not text:
        return []

    original_units = split_text_into_units(text)
    units: list[str] = []

    for unit in original_units:
        if len(unit) <= chunk_size:
            units.append(unit)
        else:
            units.extend(
                split_long_unit(
                    unit=unit,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate = "\n\n".join(current_units + [unit])

        if len(candidate) <= chunk_size:
            current_units.append(unit)
            continue

        if current_units:
            chunks.append("\n\n".join(current_units))

        current_units = select_overlap_units(
            current_units,
            overlap,
        )

        while (
            current_units
            and len("\n\n".join(current_units + [unit]))
            > chunk_size
        ):
            current_units.pop(0)

        current_units.append(unit)

    if current_units:
        final_chunk = "\n\n".join(current_units)

        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks

def create_document_id(relative_path: str) -> str:
    """根据文档相对路径生成稳定编号。"""

    normalized_path = relative_path.replace("\\", "/").lower()

    digest = hashlib.sha1(
        normalized_path.encode("utf-8")
    ).hexdigest()[:10]

    return f"doc_{digest}"


def read_document(
    path: Path,
    source_root: Path,
) -> dict:
    """读取一个 TXT、Markdown 或 PDF 文档。"""

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        text = normalize_text(
            "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        )
    else:
        text = normalize_text(
            path.read_text(encoding="utf-8-sig")
        )

    if not text:
        raise ValueError(f"文档内容为空：{path}")

    relative_path = path.relative_to(
        source_root
    ).as_posix()

    title = path.stem
    body = text
    lines = text.splitlines()

    # Markdown一级标题作为文档标题，不重复放进正文。
    if (
        path.suffix.lower() == ".md"
        and lines[0].startswith("# ")
    ):
        title = lines[0][2:].strip() or path.stem
        body = "\n".join(lines[1:]).strip()

    return {
        "document_id": create_document_id(relative_path),
        "source": relative_path,
        "title": title,
        "text": body,
        "char_count": len(body),
    }


def load_documents(
    source_directory: Path = SOURCE_DOCUMENTS_DIR,
) -> list[dict]:
    """递归读取目录中的TXT和Markdown文档。"""

    if not source_directory.exists():
        raise FileNotFoundError(
            f"原始文档目录不存在：{source_directory}"
        )

    paths = sorted(
        path
        for path in source_directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"目录中没有 PDF、TXT 或 Markdown 文档："
            f"{source_directory}"
        )

    return [
        read_document(path, source_directory)
        for path in paths
    ]


def build_chunks(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """把全部文档转换成带来源信息的文本块。"""

    chunks: list[dict] = []

    for document in documents:
        document_chunks = chunk_text(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, text in enumerate(
            document_chunks,
            start=1,
        ):
            chunks.append(
                {
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "title": document["title"],
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"_chunk_{chunk_index:04d}"
                    ),
                    "chunk_index": chunk_index,
                    "text": text,
                    "char_count": len(text),
                }
            )

    return chunks

def build_preview() -> dict:
    """生成不包含向量的知识库切块预览。"""

    documents = load_documents()
    chunks = build_chunks(documents)

    preview = {
        "version": "day24-preview-v1",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": [
            {
                "document_id": document["document_id"],
                "source": document["source"],
                "title": document["title"],
                "char_count": document["char_count"],
            }
            for document in documents
        ],
        "chunks": chunks,
    }

    CHUNK_PREVIEW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHUNK_PREVIEW_PATH.write_text(
        json.dumps(
            preview,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return preview


def main() -> None:
    preview = build_preview()

    print("文档导入和切块完成")
    print(f"文档数量：{preview['document_count']}")
    print(f"文本块数量：{preview['chunk_count']}")
    print(f"最大块长度：{preview['chunk_size']}")
    print(f"重叠长度：{preview['overlap']}")
    print(f"预览文件：{CHUNK_PREVIEW_PATH}")

    for chunk in preview["chunks"]:
        print(
            f"- {chunk['chunk_id']} | "
            f"{chunk['title']} | "
            f"{chunk['char_count']}字符"
        )


if __name__ == "__main__":
    main()
