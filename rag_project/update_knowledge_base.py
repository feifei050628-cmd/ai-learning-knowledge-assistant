import json

import torch

from rag_project.config import (
    EXPANDED_CHUNK_PREVIEW_PATH,
    EXPANDED_EMBEDDINGS_PATH,
    EXPANDED_METADATA_PATH,
    RETRIEVAL_MODEL_ID,
)
from rag_project.knowledge_base_builder import (
    main as build_knowledge_base,
)
from rag_project.note_importer import (
    main as import_learning_notes,
)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def knowledge_base_is_current() -> bool:
    """判断当前向量知识库是否与最新资料一致。"""

    required_paths = [
        EXPANDED_CHUNK_PREVIEW_PATH,
        EXPANDED_METADATA_PATH,
        EXPANDED_EMBEDDINGS_PATH,
    ]

    if not all(path.exists() for path in required_paths):
        return False

    try:
        preview = load_json(
            EXPANDED_CHUNK_PREVIEW_PATH
        )
        metadata = load_json(
            EXPANDED_METADATA_PATH
        )

        # 检索模型改变后必须重建向量。
        if metadata.get("model_id") != RETRIEVAL_MODEL_ID:
            return False

        # 文本内容或来源发生变化后必须重建。
        if preview.get("chunks") != metadata.get("chunks"):
            return False

        embeddings = torch.load(
            EXPANDED_EMBEDDINGS_PATH,
            map_location="cpu",
            weights_only=True,
        )

        if embeddings.ndim != 2:
            return False

        if embeddings.shape[0] != preview.get("chunk_count"):
            return False

        if embeddings.shape[1] != metadata.get(
            "embedding_dimension"
        ):
            return False

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return False

    return True


def verify_knowledge_base() -> None:
    metadata = load_json(
        EXPANDED_METADATA_PATH
    )

    embeddings = torch.load(
        EXPANDED_EMBEDDINGS_PATH,
        map_location="cpu",
        weights_only=True,
    )

    metadata_chunk_count = metadata["chunk_count"]
    vector_row_count = embeddings.shape[0]

    if metadata_chunk_count != vector_row_count:
        raise ValueError(
            "文本块数量与向量行数不一致"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "向量矩阵必须是二维 Tensor"
        )

    print("知识库验证通过")
    print(f"知识源数量：{metadata['document_count']}")
    print(f"文本块数量：{metadata_chunk_count}")
    print(f"向量矩阵形状：{tuple(embeddings.shape)}")
    print(f"元数据文件：{EXPANDED_METADATA_PATH}")
    print(f"向量文件：{EXPANDED_EMBEDDINGS_PATH}")


def main() -> None:
    print("第一步：扫描并导入最新学习笔记")
    import_learning_notes()

    if knowledge_base_is_current():
        print("\n知识库内容没有变化，跳过向量重建")
        verify_knowledge_base()
        print("\n知识库已经是最新状态")
        return

    print("\n第二步：检测到资料变化，重建向量知识库")
    build_knowledge_base()

    print("\n第三步：验证知识库")
    verify_knowledge_base()

    print("\n知识库更新完成")


if __name__ == "__main__":
    main()