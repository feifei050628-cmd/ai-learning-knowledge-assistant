import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rag_project.config import (
    DEVICE,     #DEVICE 是 PyTorch 设备对象，表示使用的计算设备（CPU 或 GPU）。
    EXPANDED_CHUNK_PREVIEW_PATH,    #EXPANDED_CHUNK_PREVIEW_PATH 是扩展知识库切块预览的路径
    RETRIEVAL_MAX_LENGTH,       #RETRIEVAL_MAX_LENGTH 是检索模型的最大输入长度
    RETRIEVAL_MODEL_ID,         #RETRIEVAL_MODEL_ID 是检索模型的标识符
    EMBEDDING_BATCH_SIZE,       #EMBEDDING_BATCH_SIZE 是批量处理文本块时的大小
    EXPANDED_EMBEDDINGS_PATH,   #EXPANDED_EMBEDDINGS_PATH 是扩展知识库切块向量的保存路径
    EXPANDED_METADATA_PATH,     #EXPANDED_METADATA_PATH 是扩展知识库元数据的保存路径
)
from rag_project.model_manager import (
    load_retrieval_components,      #load_retrieval_model,          #load_retrieval_model 函数用于加载检索模型
)

REQUIRED_CHUNK_FIELDS = {       #REQUIRED_CHUNK_FIELDS 是一个集合，定义了每个文本块必须包含的字段
    "document_id",
    "chunk_id",
    "source",
    "title",
    "text",
    "char_count",
    "category",
    "day",
}


def load_chunk_preview(
    path: Path = EXPANDED_CHUNK_PREVIEW_PATH,
) -> dict:
    """读取扩展知识库切块预览。"""

    if not path.exists():
        raise FileNotFoundError(
            f"扩展切块预览不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        preview = json.load(file)

    if not isinstance(preview, dict):
        raise TypeError("切块预览必须是JSON对象")

    chunks = preview.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("切块预览中没有有效文本块")

    return preview


def validate_chunks(
    preview: dict,
) -> list[dict]:
    """验证向量化之前的文本块结构。"""

    chunks = preview["chunks"]
    expected_count = preview.get("chunk_count")

    if expected_count != len(chunks):
        raise ValueError(
            "预览记录的文本块数量与实际数量不一致"
        )

    chunk_ids = set()

    for position, chunk in enumerate(
        chunks,
        start=1,
    ):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"第{position}个文本块不是JSON对象"
            )

        missing_fields = (
            REQUIRED_CHUNK_FIELDS
            - chunk.keys()
        )

        if missing_fields:
            raise ValueError(
                f"第{position}个文本块缺少字段："
                f"{sorted(missing_fields)}"
            )

        text = chunk["text"]

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"第{position}个文本块正文为空"
            )

        if chunk["char_count"] != len(text):
            raise ValueError(
                f"第{position}个文本块字符数不一致"
            )

        chunk_id = chunk["chunk_id"]

        if chunk_id in chunk_ids:
            raise ValueError(
                f"检测到重复chunk_id：{chunk_id}"
            )

        chunk_ids.add(chunk_id)

    return chunks


def encode_chunk_batch(
    texts: list[str],
    tokenizer,
    model,
) -> torch.Tensor:
    """把一批文本块转换成归一化向量。"""

    if not texts:
        raise ValueError("待编码文本不能为空")

    model_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=RETRIEVAL_MAX_LENGTH,
        return_tensors="pt",
    )

    model_inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in model_inputs.items()
    }

    with torch.inference_mode():
        model_outputs = model(**model_inputs)

        embeddings = (
            model_outputs.last_hidden_state[:, 0]
        )

        embeddings = F.normalize(   #F.normalize 是 PyTorch 中的一个函数，用于对张量进行归一化处理。它将输入张量沿指定维度进行归一化，使得每个向量的长度为 1，从而便于后续的相似度计算。
            embeddings,
            p=2,    #p=2 表示使用 L2 范数进行归一化，即将向量的长度缩放为 1。
            dim=1,
        )

    return embeddings.cpu()

def encode_all_chunks(  #encode_all_chunks 函数用于将所有文本块分批转换为向量表示，并返回一个包含所有向量的张量。
    chunks: list[dict],
    tokenizer,
    model,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> torch.Tensor:
    """分批生成全部文本块向量。"""

    if batch_size <= 0:
        raise ValueError("batch_size必须大于0")

    embedding_batches = []
    total_count = len(chunks)

    for start in range(
        0,
        total_count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total_count,
        )

        batch_texts = [
            chunk["text"]
            for chunk in chunks[start:end]
        ]

        batch_embeddings = encode_chunk_batch(
            texts=batch_texts,
            tokenizer=tokenizer,
            model=model,
        )

        embedding_batches.append(
            batch_embeddings
        )

        print(
            f"向量生成进度：{end}/{total_count}"
        )

    embeddings = torch.cat(
        embedding_batches,
        dim=0,
    )

    if embeddings.shape[0] != total_count:
        raise ValueError(
            "生成的向量数量与文本块数量不一致"
        )

    if not torch.isfinite(embeddings).all():
        raise ValueError("向量中包含无效数值")

    return embeddings

def save_knowledge_base(        #save_knowledge_base 函数用于保存正式的知识库元数据和向量矩阵。
    preview: dict,
    chunks: list[dict],
    embeddings: torch.Tensor,
) -> dict:
    """保存正式元数据和向量矩阵。"""

    if embeddings.ndim != 2:
        raise ValueError("向量矩阵必须是二维Tensor")

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "文本块数量与向量矩阵行数不一致"
        )

    metadata = {
        "version": "day26-knowledge-base-v1",
        "source_preview_version": preview["version"],
        "model_id": RETRIEVAL_MODEL_ID,
        "embedding_dimension": embeddings.shape[1],
        "document_count": preview["document_count"],
        "reference_count": preview["reference_count"],
        "learning_note_count": (
            preview["learning_note_count"]
        ),
        "total_character_count": (
            preview["total_character_count"]
        ),
        "chunk_count": len(chunks),
        "documents": preview["documents"],
        "chunks": chunks,
    }

    EXPANDED_METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EXPANDED_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    torch.save(
        embeddings.cpu(),
        EXPANDED_EMBEDDINGS_PATH,
    )

    return metadata

def main() -> None:
    preview = load_chunk_preview()
    chunks = validate_chunks(preview)

    tokenizer, model = load_retrieval_components(
        RETRIEVAL_MODEL_ID
    )

    embeddings = encode_all_chunks(
        chunks=chunks,
        tokenizer=tokenizer,
        model=model,
    )

    vector_norms = embeddings.norm(
        p=2,
        dim=1,
    )

    if not torch.allclose(
        vector_norms,
        torch.ones_like(vector_norms),
        atol=1e-5,
    ):
        raise ValueError(
            "存在没有正确归一化的文本块向量"
        )

    metadata = save_knowledge_base(
        preview=preview,
        chunks=chunks,
        embeddings=embeddings,
    )

    print("\n正式知识库生成完成")
    print(f"知识源数量：{metadata['document_count']}")
    print(f"文本块数量：{metadata['chunk_count']}")
    print(
        "向量矩阵形状："
        f"{tuple(embeddings.shape)}"
    )
    print(
        "向量维度："
        f"{metadata['embedding_dimension']}"
    )
    print(f"元数据文件：{EXPANDED_METADATA_PATH}")
    print(f"向量文件：{EXPANDED_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()